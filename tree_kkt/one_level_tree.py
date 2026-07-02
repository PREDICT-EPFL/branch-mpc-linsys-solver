"""One-level tree KKT solver built on top of the batched SOCU tail solver.

Problem structure (one shared root ``x0`` connected to ``B`` independent tails
of equal length ``T``, all blocks of size ``n``)::

    K = [ K_1              C_1 ]
        [      K_2         C_2 ]
        [           .       .  ]
        [              K_B  C_B]
        [ C_1^T C_2^T .. C_B^T D0]

Each ``K_i`` is block tridiagonal (solved by SOCU); each ``C_i`` couples the
root to the *first physical node* of tail ``i`` through a single block ``G_i``.

Algorithm (block elimination over the tails / Schur complement on the root)::

    U_i   = K_i^{-1} C_i                 (matrix-dependent, done in factorize)
    S0    = D0 - sum_i C_i^T U_i         (root Schur complement)
    v_i   = K_i^{-1} r_i                 (rhs-dependent, done in solve)
    r0_h  = r0 - sum_i C_i^T v_i
    x0    = S0^{-1} r0_h
    w_i   = v_i - U_i x0

Because ``C_i`` has a single non-zero block at physical node 0 and SOCU stores
data in physical order (see :mod:`tree_kkt.permutation`), ``C_i^T U_i`` reduces
to ``G_i^T U_i[interface_pos]`` with ``interface_pos == 0``.

No dense global ``K`` and no dense permutation matrix is ever constructed.
"""

import numpy as np
import warp as wp

from socu.block_tridiag_solver import (
    create_cholesky_factor_launch,
    create_cholesky_solve_launch,
    calculate_off_diag_storage_len,
)
from tree_kkt.permutation import interface_position
from tree_kkt import kernels as _k


_SUPPORTED_DTYPES = (wp.float32, wp.float64)


class OneLevelTreeCholesky:
    """Batched-tail Cholesky solver for a one-level (single-branch) tree system.

    Parameters
    ----------
    num_tails : int
        Number of tails ``B`` (>= 1). Used as the SOCU batch dimension.
    tail_length : int
        Blocks per tail ``T`` (>= 1).
    block_size : int
        Block size ``n`` (>= 1), shared by every node and the root.
    dtype :
        ``wp.float32`` or ``wp.float64``.
    device : str
        Warp device, e.g. ``"cuda:0"``.
    root_block_dim : int
        Threads per tile-block for the (dense) root and auxiliary kernels.
    """

    def __init__(self, num_tails, tail_length, block_size,
                 dtype=wp.float64, device="cuda", root_block_dim=128):
        if num_tails < 1:
            raise ValueError(f"num_tails must be >= 1, got {num_tails}")
        if tail_length < 1:
            raise ValueError(f"tail_length must be >= 1, got {tail_length}")
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        if dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"dtype must be one of {_SUPPORTED_DTYPES}, got {dtype}")

        # Validate device eagerly so construction fails fast on a bad device.
        try:
            self._device = wp.get_device(device)
        except Exception as exc:  # noqa: BLE001 - re-raise with clearer message
            raise ValueError(f"invalid device {device!r}: {exc}") from exc
        if not self._device.is_cuda:
            raise ValueError(
                f"OneLevelTreeCholesky requires a CUDA device, got {self._device}")

        self.num_tails = int(num_tails)
        self.tail_length = int(tail_length)
        self.block_size = int(block_size)
        self.dtype = dtype
        self.root_block_dim = int(root_block_dim)

        self.interface_pos = interface_position(self.tail_length)
        self._n_off = calculate_off_diag_storage_len(self.tail_length)

        self._allocate_persistent_buffers()

        # rhs-dependent buffers/launches, cached by number of right-hand sides q
        self._v = {}                 # q -> tail solution buffer (B, T, n, q)
        self._reduced_rhs = {}       # q -> (1, n, q)
        self._x_tail = {}            # q -> (B, T, n, q)
        self._tail_solve_launch = {} # q -> cached SOCU solve launch

        self._factored = False
        self._tail_ok = None
        self._root_ok = None
        self._status_read = False

    # ------------------------------------------------------------------ setup
    def _allocate_persistent_buffers(self):
        B, T, n = self.num_tails, self.tail_length, self.block_size
        dev, dt = self._device, self.dtype

        # tail matrices (overwritten in place by the factorization)
        self._L = wp.zeros((B, T, n, n), dtype=dt, device=dev)
        self._E = wp.zeros((B, self._n_off, n, n), dtype=dt, device=dev)
        # coupling blocks and root diagonal
        self._G = wp.zeros((B, n, n), dtype=dt, device=dev)
        self._root_diag = wp.zeros((1, n, n), dtype=dt, device=dev)
        # U_i = K_i^{-1} C_i  (n right-hand-side columns), reused across solves
        self._U = wp.zeros((B, T, n, n), dtype=dt, device=dev)
        # root Schur complement and its Cholesky factor
        self._root_schur = wp.zeros((1, n, n), dtype=dt, device=dev)
        self._root_factor = wp.zeros((1, n, n), dtype=dt, device=dev)
        # factorization status flags (1 == ok)
        self._tail_status = wp.zeros(1, dtype=wp.int32, device=dev)
        self._root_status = wp.zeros(1, dtype=wp.int32, device=dev)

        # Persistent launches bound to the persistent tail buffers.
        self._tail_factor_launch = create_cholesky_factor_launch(
            self._L, self._E, device=dev, dtype=dt)
        # coupling solve: n right-hand sides (the columns of C_i / G_i)
        self._coupling_solve_launch = create_cholesky_solve_launch(
            self._L, self._E, self._U, device=dev, dtype=dt)

    # ------------------------------------------------------------- staging I/O
    def _stage(self, dst, src, name):
        """Copy ``src`` (numpy or warp) into the persistent buffer ``dst``."""
        if isinstance(src, np.ndarray):
            tmp = wp.from_numpy(np.ascontiguousarray(src), dtype=self.dtype,
                                device=self._device)
        elif isinstance(src, wp.array):
            if src.dtype != self.dtype:
                raise TypeError(
                    f"{name}: expected dtype {self.dtype}, got {src.dtype}")
            tmp = src
        else:
            raise TypeError(f"{name}: expected numpy or warp array, got {type(src)}")
        if tuple(tmp.shape) != tuple(dst.shape):
            raise ValueError(
                f"{name}: expected shape {tuple(dst.shape)}, got {tuple(tmp.shape)}")
        wp.copy(dst, tmp)

    def _stage_root_diag(self, root_diag):
        arr = root_diag
        if isinstance(arr, np.ndarray):
            if arr.shape == (self.block_size, self.block_size):
                arr = arr.reshape(1, self.block_size, self.block_size)
            self._stage(self._root_diag, arr, "root_diag")
        elif isinstance(arr, wp.array):
            if arr.ndim == 2:
                arr = arr.reshape((1, *arr.shape))
            self._stage(self._root_diag, arr, "root_diag")
        else:
            raise TypeError(f"root_diag: expected numpy or warp array, got {type(arr)}")

    # ------------------------------------------------------------- factorize
    def factorize(self, root_diag, tail_diag, tail_offdiag, root_tail_coupling,
                  check=True):
        """Perform all matrix-dependent work (reusable across right-hand sides).

        Shapes
        ------
        root_diag           : ``(n, n)`` or ``(1, n, n)``
        tail_diag           : ``(B, T, n, n)``
        tail_offdiag        : ``(B, T-1, n, n)``  (ignored/empty when ``T == 1``)
        root_tail_coupling  : ``(B, n, n)``  (the blocks ``G_i``)
        """
        self._stage_matrix(root_diag, tail_diag, tail_offdiag, root_tail_coupling)
        self._factorize_numeric()
        self._update_status(check=check)
        return self

    def _stage_matrix(self, root_diag, tail_diag, tail_offdiag,
                      root_tail_coupling):
        """Copy the matrix blocks into the persistent device buffers.

        Separated from :meth:`_factorize_numeric` so callers (e.g. benchmarks)
        can pay the host->device staging cost once and re-run only the numeric
        factorization.
        """
        B, T, n = self.num_tails, self.tail_length, self.block_size
        self._stage_root_diag(root_diag)
        self._stage(self._L, tail_diag, "tail_diag")
        self._stage(self._G, root_tail_coupling, "root_tail_coupling")

        self._E.zero_()
        if T > 1:
            # physical off-diagonals live in the first T-1 storage slots;
            # the remaining slots are fill-in workspace (zeroed above).
            expected = (B, T - 1, n, n)
            if isinstance(tail_offdiag, np.ndarray):
                if tuple(tail_offdiag.shape) != expected:
                    raise ValueError(
                        f"tail_offdiag: expected shape {expected}, "
                        f"got {tuple(tail_offdiag.shape)}")
                tmp = wp.from_numpy(np.ascontiguousarray(tail_offdiag),
                                    dtype=self.dtype, device=self._device)
            elif isinstance(tail_offdiag, wp.array):
                if tuple(tail_offdiag.shape) != expected:
                    raise ValueError(
                        f"tail_offdiag: expected shape {expected}, "
                        f"got {tuple(tail_offdiag.shape)}")
                if tail_offdiag.dtype != self.dtype:
                    raise TypeError(
                        f"tail_offdiag: expected dtype {self.dtype}, "
                        f"got {tail_offdiag.dtype}")
                tmp = tail_offdiag
            else:
                raise TypeError("tail_offdiag: expected numpy or warp array")
            wp.copy(self._E[:, :T - 1], tmp)

    def _factorize_numeric(self):
        """Run the matrix-dependent numeric factorization (no staging, no host
        sync).  Assumes :meth:`_stage_matrix` populated the buffers."""
        B, T, n = self.num_tails, self.tail_length, self.block_size

        # 1. Factorize all B block-tridiagonal tails in one batched SOCU call.
        self._tail_factor_launch()

        # 2. Build the coupling right-hand side C_i (one non-zero block at
        #    interface_pos) directly in SOCU's physical ordering.
        self._U.zero_()
        wp.launch_tiled(
            _k.create_build_coupling_rhs_kernel(n, self.dtype),
            dim=[B],
            inputs=[self.interface_pos, self._G, self._U],
            block_dim=self.root_block_dim,
            device=self._device,
        )

        # 3. Solve K_i U_i = C_i for every tail (batched), reusing the factor.
        self._coupling_solve_launch()

        # 4. Form the root Schur complement S0 = D0 - sum_i G_i^T U_i[interface].
        wp.launch_tiled(
            _k.create_build_root_schur_kernel(n, self.dtype),
            dim=[1],
            inputs=[B, self.interface_pos, self._root_diag, self._G,
                    self._U, self._root_schur],
            block_dim=self.root_block_dim,
            device=self._device,
        )

        # 5. Factorize the root Schur complement on the GPU.
        wp.copy(self._root_factor, self._root_schur)
        wp.launch_tiled(
            _k.create_root_factor_kernel(n, self.dtype),
            dim=[1],
            inputs=[self._root_factor],
            block_dim=self.root_block_dim,
            device=self._device,
        )
        self._factored = True

    def _update_status(self, check):
        """Launch the Cholesky-breakdown detection kernels.

        The host readback (a device sync) is deferred: it happens on demand when
        a status property is read, or eagerly here when ``check=True`` so a
        failed factorization raises immediately.
        """
        B, T, n = self.num_tails, self.tail_length, self.block_size
        self._tail_status.fill_(1)
        wp.launch(
            _k.create_check_spd_diag_kernel(n, self.dtype),
            dim=[B * T],
            inputs=[self._L.reshape((B * T, n, n)), self._tail_status],
            device=self._device,
        )
        self._root_status.fill_(1)
        wp.launch(
            _k.create_check_spd_diag_kernel(n, self.dtype),
            dim=[1],
            inputs=[self._root_factor.reshape((1, n, n)), self._root_status],
            device=self._device,
        )
        self._status_read = False
        self._tail_ok = None
        self._root_ok = None

        if check:
            self._read_status()
            if not self._tail_ok:
                raise RuntimeError(
                    "Cholesky factorization failed for at least one tail matrix "
                    "(non-SPD or numerically indefinite). Not regularizing; "
                    "inspect solver.tail_factor_status.")
            if not self._root_ok:
                raise RuntimeError(
                    "Cholesky factorization of the root Schur complement failed "
                    "(non-SPD). Not regularizing; inspect solver.root_factor_status.")

    def _read_status(self):
        """Read the status flags back to the host once (cached)."""
        if not self._status_read:
            self._tail_ok = bool(self._tail_status.numpy()[0])
            self._root_ok = bool(self._root_status.numpy()[0])
            self._status_read = True

    # ------------------------------------------------------------------ solve
    def _normalize_rhs(self, rhs_root, rhs_tail):
        B, T, n = self.num_tails, self.tail_length, self.block_size

        # tail rhs -> (B, T, n, q)
        def to_np(a):
            if isinstance(a, np.ndarray):
                return a
            if isinstance(a, wp.array):
                return a.numpy()
            raise TypeError(f"unsupported rhs type {type(a)}")

        rt = to_np(rhs_tail)
        squeeze = False
        if rt.ndim == 3:
            if rt.shape != (B, T, n):
                raise ValueError(
                    f"rhs_tail: expected {(B, T, n)} or {(B, T, n, '*')}, got {rt.shape}")
            rt = rt[..., None]
            squeeze = True
        elif rt.ndim == 4:
            if rt.shape[:3] != (B, T, n):
                raise ValueError(
                    f"rhs_tail: expected leading dims {(B, T, n)}, got {rt.shape}")
        else:
            raise ValueError(f"rhs_tail must be 3D or 4D, got ndim={rt.ndim}")
        q = rt.shape[3]

        rr = to_np(rhs_root)
        if rr.ndim == 1:
            rr = rr.reshape(1, n, 1)
        elif rr.ndim == 2:
            if rr.shape == (n, q):
                rr = rr.reshape(1, n, q)
            elif rr.shape == (n, 1) and q == 1:
                rr = rr.reshape(1, n, 1)
            else:
                raise ValueError(
                    f"rhs_root: expected {(n, q)}, got {rr.shape}")
        elif rr.ndim == 3:
            if rr.shape != (1, n, q):
                raise ValueError(
                    f"rhs_root: expected {(1, n, q)}, got {rr.shape}")
        else:
            raise ValueError(f"rhs_root must be 1D/2D/3D, got ndim={rr.ndim}")
        if rr.shape[2] != q:
            raise ValueError(
                f"rhs_root has {rr.shape[2]} columns but rhs_tail has {q}")
        return np.ascontiguousarray(rr), np.ascontiguousarray(rt), q, squeeze

    def _ensure_q_buffers(self, q):
        B, T, n = self.num_tails, self.tail_length, self.block_size
        dev, dt = self._device, self.dtype
        if q not in self._v:
            self._v[q] = wp.zeros((B, T, n, q), dtype=dt, device=dev)
            self._reduced_rhs[q] = wp.zeros((1, n, q), dtype=dt, device=dev)
            self._x_tail[q] = wp.zeros((B, T, n, q), dtype=dt, device=dev)
            self._tail_solve_launch[q] = create_cholesky_solve_launch(
                self._L, self._E, self._v[q], device=dev, dtype=dt)

    def _stage_rhs(self, q, rr, rt):
        """Copy right-hand sides into the per-``q`` device buffers.

        ``rt`` -> ``_v[q]`` (overwritten with the tail solution), ``rr`` ->
        ``_reduced_rhs[q]`` (overwritten with ``x_root``).  Separated from
        :meth:`_solve_numeric` so a benchmark can stage once and re-run the
        numeric solve.
        """
        self._stage(self._v[q], rt, "rhs_tail")
        self._stage(self._reduced_rhs[q], rr, "rhs_root")

    def _solve_numeric(self, q):
        """Run the rhs-dependent numeric solve (no staging, no host sync).

        Mutates ``_v[q]`` (rhs -> tail solution), ``_reduced_rhs[q]`` (r0 ->
        x_root) and ``_x_tail[q]``; reads the factor buffers ``_L``, ``_E``,
        ``_U``, ``_root_factor`` (untouched)."""
        B, T, n = self.num_tails, self.tail_length, self.block_size

        # 1. batched tail solve v_i = K_i^{-1} r_i
        self._tail_solve_launch[q]()

        # 2. reduced root rhs r0_hat = r0 - sum_i G_i^T v_i[interface]
        wp.launch_tiled(
            _k.create_build_reduced_root_rhs_kernel(n, q, self.dtype),
            dim=[1],
            inputs=[B, self.interface_pos, self._reduced_rhs[q], self._G,
                    self._v[q], self._reduced_rhs[q]],
            block_dim=self.root_block_dim,
            device=self._device,
        )

        # 3. root solve x0 = S0^{-1} r0_hat (overwrites _reduced_rhs in place)
        wp.launch_tiled(
            _k.create_root_solve_kernel(n, q, self.dtype),
            dim=[1],
            inputs=[self._root_factor, self._reduced_rhs[q]],
            block_dim=self.root_block_dim,
            device=self._device,
        )

        # 4. recover tails w_i = v_i - U_i x0
        wp.launch_tiled(
            _k.create_recover_tails_kernel(n, q, self.dtype),
            dim=[B, T],
            inputs=[self._U, self._v[q], self._reduced_rhs[q], self._x_tail[q]],
            block_dim=self.root_block_dim,
            device=self._device,
        )

    def solve(self, rhs_root, rhs_tail, copy_to_host=True):
        """Solve the tree system for the given right-hand side(s).

        Reuses the factorization from :meth:`factorize`; only rhs-dependent work
        is performed.  Returns ``(x_root, x_tail)``.

        With ``copy_to_host=True`` (default) numpy arrays are returned with
        shapes ``(n, q)`` / ``(B, T, n, q)`` (``q`` squeezed for a single vector
        rhs).  With ``copy_to_host=False`` the internal warp arrays
        ``(1, n, q)`` / ``(B, T, n, q)`` are returned (no device sync).
        """
        if not self._factored:
            raise RuntimeError("solve() called before factorize()")
        self._read_status()
        if not (self._tail_ok and self._root_ok):
            raise RuntimeError(
                "solve() called on a failed factorization; "
                "check solver.factor_status")

        rr, rt, q, squeeze = self._normalize_rhs(rhs_root, rhs_tail)
        self._ensure_q_buffers(q)
        self._stage_rhs(q, rr, rt)
        self._solve_numeric(q)

        x_root = self._reduced_rhs[q]      # (1, n, q), physical == tree order
        x_tail = self._x_tail[q]           # (B, T, n, q), physical tail order

        if not copy_to_host:
            return x_root, x_tail

        x_root_np = x_root.numpy()[0]      # (n, q)
        x_tail_np = x_tail.numpy()         # (B, T, n, q)
        if squeeze:
            x_root_np = x_root_np[..., 0]
            x_tail_np = x_tail_np[..., 0]
        return x_root_np, x_tail_np

    def factorize_and_solve(self, root_diag, tail_diag, tail_offdiag,
                            root_tail_coupling, rhs_root, rhs_tail,
                            copy_to_host=True, check=True):
        """Convenience one-shot: factorize then solve a single right-hand side."""
        self.factorize(root_diag, tail_diag, tail_offdiag, root_tail_coupling,
                       check=check)
        return self.solve(rhs_root, rhs_tail, copy_to_host=copy_to_host)

    # ------------------------------------------------------------ diagnostics
    @property
    def schur_complement(self):
        """The root Schur complement ``S0`` as a ``(n, n)`` numpy array."""
        if not self._factored:
            raise RuntimeError("schur_complement is only available after factorize()")
        return self._root_schur.numpy()[0]

    @property
    def tail_factor_status(self):
        """``True`` if every tail Cholesky factorization succeeded."""
        if self._factored:
            self._read_status()
        return self._tail_ok

    @property
    def root_factor_status(self):
        """``True`` if the root Schur complement Cholesky succeeded."""
        if self._factored:
            self._read_status()
        return self._root_ok

    @property
    def factor_status(self):
        """``True`` iff both tail and root factorizations succeeded."""
        if not self._factored:
            return None
        self._read_status()
        return bool(self._tail_ok and self._root_ok)

    def compute_residual(self, root_diag, tail_diag, tail_offdiag,
                         root_tail_coupling, rhs_root, rhs_tail,
                         x_root, x_tail):
        """Structured (block-wise) residual ``||K x - r||_inf`` for debugging.

        Computed with numpy directly from the structured blocks -- no dense
        global ``K`` is assembled.  Intended for tests/diagnostics only.
        """
        B, T, n = self.num_tails, self.tail_length, self.block_size

        def np4(a, shp):
            a = a.numpy() if isinstance(a, wp.array) else np.asarray(a)
            return a.reshape(shp)

        D0 = np4(root_diag, (n, n))  # accepts (n, n) or (1, n, n)
        Kd = np4(tail_diag, (B, T, n, n))
        Ke = np4(tail_offdiag, (B, T - 1, n, n)) if T > 1 else None
        G = np4(root_tail_coupling, (B, n, n))

        xr = np.asarray(x_root).reshape(n, -1)
        q = xr.shape[1]
        xt = np.asarray(x_tail).reshape(B, T, n, q)
        rr = np.asarray(rhs_root).reshape(n, q) if np.asarray(rhs_root).size == n * q \
            else np.asarray(rhs_root).reshape(1, n, q)[0]
        rt = np.asarray(rhs_tail).reshape(B, T, n, q)

        max_res = 0.0
        # root block row: sum_i G_i^T x_{i,0} + D0 x0 - r0
        root_res = D0 @ xr - rr
        for i in range(B):
            root_res += G[i].T @ xt[i, 0]
        max_res = max(max_res, np.abs(root_res).max())

        # tail block rows
        for i in range(B):
            for k in range(T):
                res = Kd[i, k] @ xt[i, k] - rt[i, k]
                if k == 0:
                    res += G[i] @ xr
                if T > 1 and k < T - 1:
                    res += Ke[i, k].T @ xt[i, k + 1]
                if T > 1 and k > 0:
                    res += Ke[i, k - 1] @ xt[i, k - 1]
                max_res = max(max_res, np.abs(res).max())
        return float(max_res)

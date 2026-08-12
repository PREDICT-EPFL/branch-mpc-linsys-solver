"""The structured GPU scenario-tree solver (Warp-only).

:class:`TreeSolver` implements one fixed algorithm.  With
``K_i = L_i L_i^T`` factored by SOCU's batched parallel cyclic reduction
and ``F`` denoting SOCU's forward-substitution operator
(``F^T F = K^{-1}``):

- factorize: ``G_i = F C_i^T`` (columnwise, forward only, fused with the
  factorization in one launch), ``S = R - sum_i G_i^T G_i`` (lower
  triangle only, deterministic pairwise cross-branch reduction),
  ``S = L_S L_S^T``;
- solve: ``u_i = F r_i``, ``s = q - sum_i G_i^T u_i``,
  ``L_S L_S^T y = s``, ``t_i = u_i - G_i y``, ``w_i = F^T t_i``.

Everything runs on the GPU, batched over all branches, without
assembling the global matrix and without any explicit inverse.  The
cross-branch Schur reduction uses a fixed pairwise order (no atomics),
adding no scheduling dependence beyond SOCU's own blocked-path
epsilon-level FMA-order caveat.  Alternatives that measured slower
(explicit inverse action,
sequential block-Thomas tails, per-branch dispatch, atomic reductions,
unfused factor+forward) were removed; their measurements are archived
with the benchmark results.

Usage::

    solver = TreeSolver(shape, device="cuda:0")
    solver.stage_matrix(matrix)          # explicit host -> device
    solver.factorize(check=True)         # all matrix-dependent work
    solution = solver.solve(rhs)         # owned device TreeVector

For allocation-free warm loops, bind buffers once and reuse::

    rhs_dev = solver.upload_rhs(rhs)
    prepared = solver.prepare_solve(rhs.nrhs, use_cuda_graph=True)
    solver.factorize()
    prepared.solve_into(rhs_dev)         # -> prepared.out, no allocation

Implementation notes:

- All GPU work runs through upstream SOCU launches and the custom Warp
  tile kernels of :mod:`src.kernels`; there is no CuPy dependency.
- The dense root Schur complement is factored and solved by SOCU itself
  (treated as a one-branch, one-stage chain with block size ``n_y``) for
  SOCU-aligned ``n_y``; small unaligned ``n_y`` falls back to a
  single-tile Warp Cholesky kernel.
- Matrix-side buffers are allocated once at construction; solve-side
  buffers and launches live on explicit :class:`PreparedSolve` objects
  (the convenience ``solve``/``solve_into`` keep a single lazily created
  one).  The timed factorize/solve paths perform no allocations.
"""

import enum
from dataclasses import dataclass, asdict

import numpy as np
import warp as wp
from socu.block_tridiag_solver import (
    create_cholesky_factor_launch,
    create_cholesky_solve_launch,
)

from src import workspace as _ws
from src.utils import as_device_array, copy_into
from src.kernels import BLOCK_DIM, TILE_M, num_separator_tiles
from src.kernels.diagnostics import (
    create_check_pivots_kernel,
    create_min_pivot_kernel,
)
from src.kernels.reduction import (
    create_finalize_rhs_kernel,
    create_finalize_schur_kernel,
    create_pair_reduce_kernel,
)
from src.kernels.root import (
    create_root_factor_kernel,
    create_root_solve_kernel,
)
from src.kernels.schur import (
    create_chunked_recover_kernel,
    create_chunked_schur_kernel,
    create_chunked_schur_rhs_kernel,
    num_row_chunks,
)
from src.problem import TreeMatrix, TreeShape, TreeVector
from src.runtime import require_cuda_device, wp_dtype
from src.socu_adapter import SocuTailEngine, is_block_size_aligned

_TILE_ROOT_LIMIT = 64  # single-tile root fallback size limit


class _State(enum.Enum):
    EMPTY = "empty"
    STAGED = "staged"
    FACTORED = "factored"
    CLOSED = "closed"


@dataclass(frozen=True)
class SolverStats:
    """Compact runtime report of one solver instance.

    ``socu`` nests the upstream SOCU provenance and launch settings.
    Benchmark records store :meth:`to_dict`.
    """

    device: str
    precision: str
    root_via_socu: bool
    tile_m: int
    workspace_bytes: int
    socu: dict

    def to_dict(self) -> dict:
        """Plain-dict form for benchmark records."""
        return asdict(self)


class PreparedSolve:
    """Buffers, bound launches, and (optionally) one CUDA graph for
    solves of one right-hand-side width into one output.

    Created by :meth:`TreeSolver.prepare_solve`; owns exactly one set of
    launches, one output binding (:attr:`out`), and at most one captured
    graph, so its memory and lifetime are explicit.  After construction,
    :meth:`solve_into` performs no allocations, launch construction, or
    kernel compilation.
    """

    def __init__(self, solver: "TreeSolver", nrhs: int,
                 out: TreeVector | None, use_cuda_graph: bool):
        solver._require_open()
        nrhs = int(nrhs)
        if nrhs < 1:
            raise ValueError(f"nrhs must be >= 1, got {nrhs}")
        self._solver = solver
        self.nrhs = nrhs
        self.use_cuda_graph = bool(use_cuda_graph)
        B, T, n_b, n_y = solver.shape.dims()
        dt, dev = solver.dtype, solver.device

        if out is None:
            out = solver.create_solution(nrhs)
        else:
            solver._check_vector(out, "out")
            as_device_array(out.branch, dt, dev, "out.branch")
            as_device_array(out.separator, dt, dev, "out.separator")
            if out.nrhs != nrhs:
                raise ValueError(f"out has {out.nrhs} columns, expected {nrhs}")
        #: The bound output vector every :meth:`solve_into` writes to.
        self.out = out

        self._rhs = wp.zeros((B, T, n_b, nrhs), dtype=dt, device=dev)
        self._rhs3 = self._rhs.reshape((B, T * n_b, nrhs))
        self._s4 = wp.zeros((1, 1, n_y, nrhs), dtype=dt, device=dev)
        self._s = self._s4.reshape((n_y, nrhs))
        self._contrib = wp.zeros((B, n_y, nrhs), dtype=dt, device=dev)
        self._out3 = out.branch.reshape((B, T * n_b, nrhs))

        engine = solver._engine
        self._forward = engine.build_forward_launch(self._rhs)
        self._backward = engine.build_backward_launch(out.branch)
        if solver._root_via_socu:
            self._root_solve = create_cholesky_solve_launch(
                solver._bufs.S_factor4, solver._bufs.root_E, self._s4,
                dtype=dt, device=dev)
        else:
            kernel = create_root_solve_kernel(n_y, nrhs, dt)
            s_factor, s_view = solver._bufs.S_factor, self._s

            def tile_root_solve():
                wp.launch_tiled(kernel, dim=[1], inputs=[s_factor, s_view],
                                block_dim=BLOCK_DIM, device=dev)
            self._root_solve = tile_root_solve
        self._graph = None
        _ws.probe_rhs_kernels(solver._bufs, self._rhs3, self._contrib,
                              self._s, self._out3, solver.shape, dt, dev)

    def solve_into(self, rhs: TreeVector, timer=None) -> TreeVector:
        """Solve for ``rhs`` into the bound output and return it.

        ``rhs`` is a :class:`TreeVector` of host or device arrays with
        this object's RHS width.  The recovery and backward substitution
        write straight into ``out.branch``; only the small
        ``(n_y, nrhs)`` separator part is copied from the root-solve
        buffer.  With ``use_cuda_graph=True`` the numeric phases are
        captured once and replayed.

        Phases marked on ``timer``: ``rhs_stage``, ``tail_forward_rhs``,
        ``schur_rhs``, ``root_forward_backward``, ``recovery_update``,
        ``tail_backward_rhs``, ``solution_store`` (``graph_replay``
        replaces the middle phases under graphs).
        """
        solver = self._solver
        solver._require_factored()
        solver._check_vector(rhs, "rhs")
        if rhs.nrhs != self.nrhs:
            raise ValueError(f"rhs has {rhs.nrhs} columns, expected {self.nrhs}")
        copy_into(self._rhs, rhs.branch, "rhs.branch")
        copy_into(self._s, rhs.separator, "rhs.separator")
        if timer: timer.mark("rhs_stage")
        if self.use_cuda_graph:
            if self._graph is None:
                self._numeric(None)  # warm compile + real compute
                with wp.ScopedCapture(device=solver.device) as cap:
                    self._numeric(None)
                self._graph = cap.graph
            else:
                wp.capture_launch(self._graph)
            if timer: timer.mark("graph_replay")
        else:
            self._numeric(timer)
        wp.copy(self.out.separator, self._s)
        if timer: timer.mark("solution_store")
        return self.out

    def _numeric(self, timer):
        solver = self._solver
        B, T, n_b, n_y = solver.shape.dims()
        dt = solver.dtype
        bufs = solver._bufs
        # u = F r (forward substitution only)
        self._forward()
        if timer: timer.mark("tail_forward_rhs")
        # s = q - sum_i G_i^T u_i (deterministic pairwise reduction)
        wp.launch_tiled(
            create_chunked_schur_rhs_kernel(self.nrhs, dt),
            dim=[B, solver._mt],
            inputs=[num_row_chunks(T * n_b), bufs.G3, self._rhs3,
                    self._contrib],
            block_dim=BLOCK_DIM, device=solver.device)
        solver._pair_reduce(self._contrib, B, n_y, self.nrhs)
        wp.launch(create_finalize_rhs_kernel(dt), dim=[n_y, self.nrhs],
                  inputs=[self._contrib, self._s], device=solver.device)
        if timer: timer.mark("schur_rhs")
        # L_S L_S^T y = s
        self._root_solve()
        if timer: timer.mark("root_forward_backward")
        # t = u - G y, written directly to the bound output
        wp.launch_tiled(create_chunked_recover_kernel(self.nrhs, dt),
                        dim=[B, num_row_chunks(T * n_b)],
                        inputs=[solver._mt, bufs.G3, self._rhs3, self._s,
                                self._out3],
                        block_dim=BLOCK_DIM, device=solver.device)
        if timer: timer.mark("recovery_update")
        # w = F^T t, in place in the bound output
        self._backward()
        if timer: timer.mark("tail_backward_rhs")

    def device_bytes(self) -> int:
        """Device bytes held by this prepared solve's buffers (excluding
        the bound output, which the caller owns)."""
        return sum(buf.capacity for buf in (self._rhs, self._s4,
                                            self._contrib))


class TreeSolver:
    """Structured GPU solver for one-level scenario-tree SPD systems.

    Parameters
    ----------
    shape : TreeShape
        Problem dimensions and precision (shared with the matrix and
        vector containers, so dimensions are stated exactly once).
    device : str
        CUDA device, e.g. ``"cuda:0"``.

    Construction allocates all matrix-side device buffers, builds the
    SOCU engine, and runs cached capability probes (Warp tile kernels
    whose shapes exceed device shared memory fail to launch with only a
    warning; the probes convert that into an immediate error).  See the
    module docstring for the lifecycle.
    """

    def __init__(self, shape: TreeShape, device: str = "cuda:0"):
        if not isinstance(shape, TreeShape):
            raise TypeError("shape must be a TreeShape")
        self._shape = shape
        self._device = require_cuda_device(device)
        self._wp_dtype = wp_dtype(shape.precision)

        B, T, n_b, n_y = shape.dims()
        self._mt = num_separator_tiles(n_y)

        # root path: SOCU dense-chain factorization for aligned n_y, else
        # a single-tile Warp Cholesky for small unaligned n_y
        self._root_via_socu = is_block_size_aligned(n_y, self._wp_dtype)
        if not self._root_via_socu and n_y > _TILE_ROOT_LIMIT:
            raise ValueError(
                f"separator_dim {n_y} is neither SOCU-aligned nor small "
                f"enough (<= {_TILE_ROOT_LIMIT}) for the single-tile root "
                f"fallback")

        self._engine = SocuTailEngine(B, T, n_b, self._wp_dtype,
                                      self._device)
        self._bufs = _ws.MatrixBuffers(shape, self._wp_dtype, self._device)

        self._combined = None
        self._root_factor_launch = None
        self._use_cuda_graph = False
        self._factor_graph = None
        self._prepared = None   # the single lazily created simple-path solve
        self._state = _State.EMPTY

        _ws.probe_matrix_kernels(self._bufs, shape, self._wp_dtype,
                                 self._device, self._root_via_socu)

    # ----------------------------------------------------------- public info
    @property
    def shape(self) -> TreeShape:
        """The problem dimensions this solver was built for."""
        return self._shape

    @property
    def device(self):
        """The Warp CUDA device this solver runs on."""
        return self._device

    @property
    def dtype(self):
        """The Warp scalar dtype (``wp.float64`` or ``wp.float32``)."""
        return self._wp_dtype

    # ------------------------------------------------------------- lifecycle
    def prepare(self, use_cuda_graph: bool = False):
        """Build and compile the bound launches now (idempotent).

        Called lazily by :meth:`factorize`; call it explicitly to move
        Warp kernel compilation out of latency-sensitive regions, or to
        enable CUDA-graph capture/replay of the factorization (and, by
        default, of solves prepared afterwards) -- the one measured warm
        optimization (4-48% at latency-bound sizes).
        """
        self._require_open()
        self._use_cuda_graph = bool(use_cuda_graph)
        self._prepare_launches()
        return self

    def _prepare_launches(self):
        if self._combined is not None:
            return
        self._combined = self._engine.build_factor_forward_launch(
            self._bufs.G)
        if self._root_via_socu:
            self._root_factor_launch = create_cholesky_factor_launch(
                self._bufs.S_factor4, self._bufs.root_E,
                dtype=self._wp_dtype, device=self._device)

    def stage_matrix(self, matrix: TreeMatrix):
        """Copy the matrix block data to the device (explicit transfer,
        outside any timed region).

        ``matrix`` arrays may be NumPy host arrays or device arrays of
        the solver's dtype.  Invalidates the factored state until the
        next :meth:`factorize`.
        """
        self._require_open()
        if not isinstance(matrix, TreeMatrix):
            raise TypeError("stage_matrix takes a TreeMatrix")
        if matrix.shape != self._shape:
            raise ValueError(f"matrix shape {matrix.shape} does not match "
                             f"solver shape {self._shape}")
        self._engine.stage(matrix.D, matrix.E)
        copy_into(self._bufs.C_T, matrix.C_T, "C_T")
        copy_into(self._bufs.R, matrix.R, "R")
        self._state = _State.STAGED
        return self

    def factorize(self, check: bool = False, timer=None):
        """Run all matrix-dependent work (reused by every later solve).

        Phases (marked on ``timer`` when given): ``refresh`` (restore
        factor buffers from pristine data), ``tail_factor_forward_c``
        (one fused batched SOCU launch), ``schur_branch_syrk`` and
        ``schur_branch_reduction`` (``S = R - sum G^T G``, lower
        triangle), ``root_potrf``; ``graph_replay`` replaces them all
        after :meth:`prepare` with ``use_cuda_graph=True``.

        With ``check=True`` the factor pivots are scanned and read back
        (synchronizes), raising ``RuntimeError`` on breakdown.  Health
        scans are not launched otherwise, keeping the warm path free of
        diagnostic work; call :meth:`check_factorization` for on-demand
        validation.  Repeated calls refresh and refactor in place.
        """
        self._require_open()
        if self._state is _State.EMPTY:
            raise RuntimeError("factorize() called before stage_matrix()")
        self._prepare_launches()
        if self._use_cuda_graph:
            if self._factor_graph is None:
                self._factorize_numeric(None)  # warm compile + real compute
                with wp.ScopedCapture(device=self._device) as cap:
                    self._factorize_numeric(None)
                self._factor_graph = cap.graph
            else:
                wp.capture_launch(self._factor_graph)
            if timer: timer.mark("graph_replay")
        else:
            self._factorize_numeric(timer)
        self._state = _State.FACTORED
        if check:
            self.check_factorization()
        return self

    def _require_open(self):
        if self._state is _State.CLOSED:
            raise RuntimeError("this TreeSolver has been closed")

    def _require_factored(self):
        self._require_open()
        if self._state is not _State.FACTORED:
            raise RuntimeError("solve requires a factorization; call "
                               "factorize() after staging the matrix")

    # ------------------------------------------------------- numeric phases
    def _factorize_numeric(self, timer=None):
        B, T, n_b, n_y = self._shape.dims()
        dt = self._wp_dtype
        bufs = self._bufs
        self._engine.refresh()
        if timer: timer.mark("refresh")

        # fused: factor every K_i in place and G = F C^T in one launch
        wp.copy(bufs.G, bufs.C_T)
        self._combined()
        if timer: timer.mark("tail_factor_forward_c")

        # S = R - sum G^T G, lower triangle only, over the contiguous
        # (B, T*n_b, n_y) view; deterministic pairwise reduction
        wp.launch_tiled(
            create_chunked_schur_kernel(dt),
            dim=[B, self._mt, self._mt],
            inputs=[num_row_chunks(T * n_b), bufs.G3, bufs.G3, bufs.contrib],
            block_dim=BLOCK_DIM, device=self._device)
        if timer: timer.mark("schur_branch_syrk")
        self._pair_reduce(bufs.contrib, B, n_y, n_y)
        wp.launch(create_finalize_schur_kernel(dt), dim=[n_y, n_y],
                  inputs=[bufs.R, bufs.contrib, bufs.S],
                  device=self._device)
        if timer: timer.mark("schur_branch_reduction")

        wp.copy(bufs.S_factor, bufs.S)
        if self._root_via_socu:
            self._root_factor_launch()
        else:
            wp.launch_tiled(create_root_factor_kernel(n_y, dt), dim=[1],
                            inputs=[bufs.S_factor], block_dim=BLOCK_DIM,
                            device=self._device)
        if timer: timer.mark("root_potrf")

    def _pair_reduce(self, buf, count, r, c):
        """Fixed pairwise reduction tree into ``buf[0]`` (deterministic:
        the summation order is independent of thread scheduling)."""
        kernel = create_pair_reduce_kernel(self._wp_dtype)
        stride = 1
        while stride < count:
            pairs = (count + 2 * stride - 1) // (2 * stride)
            wp.launch(kernel, dim=[pairs, r, c],
                      inputs=[stride, count, buf], device=self._device)
            stride *= 2

    # ----------------------------------------------------------------- solve
    def prepare_solve(self, nrhs: int, out: TreeVector | None = None,
                      use_cuda_graph: bool | None = None) -> PreparedSolve:
        """Create a :class:`PreparedSolve` for ``nrhs`` right-hand-side
        columns.

        ``out`` binds caller-owned output buffers (see
        :meth:`create_solution`); ``None`` allocates fresh ones, exposed
        as ``prepared.out``.  ``use_cuda_graph=None`` inherits the
        setting from :meth:`prepare`.  The returned object owns its
        buffers and launches; drop it to release them.
        """
        self._require_open()
        self._prepare_launches()
        if use_cuda_graph is None:
            use_cuda_graph = self._use_cuda_graph
        return PreparedSolve(self, nrhs, out, use_cuda_graph)

    def solve(self, rhs: TreeVector, timer=None) -> TreeVector:
        """Solve for ``rhs`` and return a solution the caller owns.

        Convenience path: the returned device :class:`TreeVector` is
        freshly allocated on every call.  Internally one
        :class:`PreparedSolve` is kept and reused while the RHS width
        stays the same; use :meth:`prepare_solve` directly for explicit
        control in warm loops.
        """
        prepared = self._simple_prepared(rhs)
        prepared.solve_into(rhs, timer=timer)
        return TreeVector(self._shape, wp.clone(prepared.out.branch),
                          wp.clone(prepared.out.separator))

    def solve_into(self, rhs: TreeVector, out: TreeVector,
                   timer=None) -> TreeVector:
        """Solve for ``rhs`` into caller-owned device buffers.

        Convenience wrapper over :meth:`prepare_solve`: the internal
        prepared solve is rebuilt when ``out`` or the RHS width changes,
        so warm loops that reuse the same ``out`` pay no per-call
        allocation.  Returns ``out``.
        """
        prepared = self._simple_prepared(rhs, out)
        return prepared.solve_into(rhs, timer=timer)

    def _simple_prepared(self, rhs, out=None) -> PreparedSolve:
        self._require_factored()
        self._check_vector(rhs, "rhs")
        p = self._prepared
        if (p is None or p.nrhs != rhs.nrhs
                or (out is not None and p.out is not out)):
            self._prepared = p = self.prepare_solve(rhs.nrhs, out=out)
        return p

    # ------------------------------------------------------------- staging
    def upload_rhs(self, rhs: TreeVector) -> TreeVector:
        """Copy a right-hand side to new device arrays (staging helper for
        callers that refresh device-to-device in warm loops).  Device
        inputs are validated (dtype and device) and passed through;
        NumPy inputs are transferred."""
        self._require_open()
        self._check_vector(rhs, "rhs")
        return TreeVector(
            self._shape,
            as_device_array(rhs.branch, self._wp_dtype, self._device,
                            "rhs.branch"),
            as_device_array(rhs.separator, self._wp_dtype, self._device,
                            "rhs.separator"))

    def create_solution(self, nrhs: int) -> TreeVector:
        """Allocate a zeroed device :class:`TreeVector` suitable as the
        ``out`` binding of :meth:`prepare_solve`."""
        self._require_open()
        if int(nrhs) < 1:
            raise ValueError(f"nrhs must be >= 1, got {nrhs}")
        B, T, n_b, n_y = self._shape.dims()
        return TreeVector(
            self._shape,
            wp.zeros((B, T, n_b, nrhs), dtype=self._wp_dtype,
                     device=self._device),
            wp.zeros((n_y, nrhs), dtype=self._wp_dtype, device=self._device))

    def _check_vector(self, vec, name):
        if not isinstance(vec, TreeVector):
            raise TypeError(f"{name} must be a TreeVector (see "
                            f"tree_vector_from_arrays for raw arrays)")
        if vec.shape != self._shape:
            raise ValueError(f"{name} shape {vec.shape} does not match "
                             f"solver shape {self._shape}")

    # ------------------------------------------------------------ statuses
    def check_factorization(self):
        """Scan the factor pivots (launches the health-check kernels),
        read the flags back (synchronizes), and raise on breakdown
        (non-SPD input or numerical indefiniteness).  Not part of the
        timed warm path; run it once per staged matrix outside timed
        regions."""
        if self._state is not _State.FACTORED:
            raise RuntimeError("check_factorization() called before factorize()")
        B, T, n_b, n_y = self._shape.dims()
        bufs = self._bufs
        kernel = create_check_pivots_kernel(self._wp_dtype)
        bufs.tail_status.fill_(1)
        wp.launch(kernel, dim=[B * T],
                  inputs=[n_b,
                          self._engine.diag_factor.reshape((B * T, n_b, n_b)),
                          bufs.tail_status],
                  device=self._device)
        bufs.root_status.fill_(1)
        wp.launch(kernel, dim=[1],
                  inputs=[n_y, bufs.S_factor.reshape((1, n_y, n_y)),
                          bufs.root_status],
                  device=self._device)
        wp.synchronize_device(self._device)
        if not bool(bufs.tail_status.numpy()[0]):
            raise RuntimeError("SOCU branch Cholesky produced a non-positive "
                               "or NaN pivot (input not SPD?)")
        if not bool(bufs.root_status.numpy()[0]):
            raise RuntimeError("root Schur complement Cholesky failed")
        return True

    def compute_min_pivot(self) -> float:
        """Smallest Cholesky pivot across branch and root factors.
        Launches reduction kernels and synchronizes; diagnostics only,
        never on the timed warm path."""
        if self._state is not _State.FACTORED:
            raise RuntimeError("compute_min_pivot() called before factorize()")
        B, T, n_b, n_y = self._shape.dims()
        bufs = self._bufs
        big = 1e300 if self._shape.precision == "float64" else 1e30
        bufs.min_pivot.fill_(big)
        kernel = create_min_pivot_kernel(self._wp_dtype)
        wp.launch(kernel, dim=[B * T],
                  inputs=[n_b,
                          self._engine.diag_factor.reshape((B * T, n_b, n_b)),
                          bufs.min_pivot],
                  device=self._device)
        wp.launch(kernel, dim=[1],
                  inputs=[n_y, bufs.S_factor.reshape((1, n_y, n_y)),
                          bufs.min_pivot],
                  device=self._device)
        wp.synchronize_device(self._device)
        return float(bufs.min_pivot.numpy()[0])

    def schur_complement(self) -> np.ndarray:
        """Host copy of the separator Schur complement
        ``S = R - sum_i G_i^T G_i`` of the current factorization
        (synchronizes; diagnostics and validation only)."""
        if self._state is not _State.FACTORED:
            raise RuntimeError("schur_complement() called before factorize()")
        return self._bufs.S.numpy()

    # ---------------------------------------------------------------- misc
    def synchronize(self):
        """Block until all enqueued GPU work on this device finished."""
        wp.synchronize_device(self._device)

    def workspace_bytes(self) -> int:
        """Device bytes held by the solver's persistent buffers plus the
        internal simple-path prepared solve, if one exists.  Explicitly
        created :class:`PreparedSolve` objects report their own
        :meth:`PreparedSolve.device_bytes`."""
        total = self._engine.workspace_bytes() + self._bufs.device_bytes()
        if self._prepared is not None:
            total += self._prepared.device_bytes()
        return total

    def stats(self) -> SolverStats:
        """Compact runtime report (see :class:`SolverStats`)."""
        return SolverStats(
            device=str(self._device),
            precision=self._shape.precision,
            root_via_socu=self._root_via_socu,
            tile_m=TILE_M,
            workspace_bytes=self.workspace_bytes(),
            socu=self._engine.stats(),
        )

    def close(self):
        """Release all device buffers and launches held by this solver.
        The solver must not be used afterwards."""
        self._prepared = None
        self._factor_graph = None
        self._combined = None
        self._root_factor_launch = None
        self._bufs = None
        self._engine = None
        self._state = _State.CLOSED

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

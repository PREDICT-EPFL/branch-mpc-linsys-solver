"""Boundary-node and root kernels (everything outside the SOCU prefix).

Fused per plan section 6.5: one batched kernel finishes the final-node
condensation, factors it, and transforms the root coupling; one batched
kernel does the final-node RHS update and triangular solve; one batched
kernel recovers the final node and prepares the path-sparse prefix
correction.  The root has a single-launch fused accumulate+factor (and
accumulate+solve) tile kernel for small roots, with the deterministic
scalar-reduction + tile-factor pair kept as the measured fallback for
larger roots.  Every kernel touches exactly ``n_b`` boundary rows per
tail; nothing here scales with the horizon.
"""

from functools import lru_cache

import warp as wp

from src.endpoint_tree.kernels import make_module

#: roots up to this size use the fused single-launch tile kernels --
#: but only for small batches: the fused kernel reduces over the tails
#: sequentially in one thread block, so it wins on launch latency only
#: while B is small (measured: at B=512 it is ~50x slower than the
#: two-launch scalar-reduction fallback)
FUSED_ROOT_MAX = 32
FUSED_ROOT_MAX_TAILS = 16


@lru_cache(maxsize=None)
def _sub_func(dtype):
    @wp.func
    def sub2(a: dtype, b: dtype):
        return a - b

    return sub2


@lru_cache(maxsize=None)
def create_boundary_factor_kernel(n: int, n_r: int, Lp: int,
                                  dtype=wp.float64):
    """Fused final-node factor step, one thread block per tail:
    ``Dbar = D0 - sum_p H_p^T H_p`` over the ``Lp`` path blocks,
    ``J = chol(Dbar)`` (stored), ``M = J^{-1} G`` (stored in place over
    the coupling buffer)."""
    module = make_module(f"boundary_factor_{n}_{n_r}_{Lp}")
    sub2 = _sub_func(dtype)

    @wp.kernel(module=module)
    def boundary_factor_kernel(D0: wp.array3d(dtype=dtype),  # type: ignore
                               H: wp.array4d(dtype=dtype),   # type: ignore
                               J: wp.array3d(dtype=dtype),   # type: ignore
                               M: wp.array3d(dtype=dtype)):  # type: ignore
        b = wp.tid()
        Dbar = wp.tile_load(D0[b], shape=(n, n))
        for p in range(Lp):
            Hp = wp.tile_load(H[b, p], shape=(n, n))
            HpT = wp.tile_transpose(Hp)
            prod = wp.tile_zeros(shape=(n, n), dtype=dtype)
            wp.tile_matmul(HpT, Hp, prod)
            Dbar = wp.tile_map(sub2, Dbar, prod)
        wp.tile_cholesky_inplace(Dbar)
        wp.tile_store(J[b], Dbar)
        Gb = wp.tile_load(M[b], shape=(n, n_r))
        wp.tile_lower_solve_inplace(Dbar, Gb)
        wp.tile_store(M[b], Gb)

    return boundary_factor_kernel


# --------------------------------------------------------------- root
@lru_cache(maxsize=None)
def create_fused_root_factor_kernel(n: int, n_r: int, dtype=wp.float64):
    """Single-launch small-root kernel: read every ``M_b``, accumulate
    ``S = R - sum_b M_b^T M_b``, and factor ``S`` in place."""
    module = make_module(f"fused_root_factor_{n}_{n_r}")
    sub2 = _sub_func(dtype)

    @wp.kernel(module=module)
    def fused_root_factor_kernel(B: int,
                                 R: wp.array2d(dtype=dtype),   # type: ignore
                                 M: wp.array3d(dtype=dtype),   # type: ignore
                                 S: wp.array2d(dtype=dtype)):  # type: ignore
        _ = wp.tid()
        St = wp.tile_load(R, shape=(n_r, n_r))
        for b in range(B):
            Mb = wp.tile_load(M[b], shape=(n, n_r))
            MbT = wp.tile_transpose(Mb)
            prod = wp.tile_zeros(shape=(n_r, n_r), dtype=dtype)
            wp.tile_matmul(MbT, Mb, prod)
            St = wp.tile_map(sub2, St, prod)
        wp.tile_cholesky_inplace(St)
        wp.tile_store(S, St)

    return fused_root_factor_kernel


@lru_cache(maxsize=None)
def create_fused_root_solve_kernel(n: int, n_r: int, dtype=wp.float64):
    """Single-launch small-root solve: ``qbar = q - sum_b M_b^T v0_b``
    then ``y = (L_R L_R^T)^{-1} qbar``, all in one kernel."""
    module = make_module(f"fused_root_solve_{n}_{n_r}")
    sub2 = _sub_func(dtype)

    @wp.kernel(module=module)
    def fused_root_solve_kernel(B: int,
                                q: wp.array2d(dtype=dtype),    # type: ignore
                                M: wp.array3d(dtype=dtype),    # type: ignore
                                v0: wp.array2d(dtype=dtype),   # type: ignore
                                S: wp.array2d(dtype=dtype),    # type: ignore
                                y: wp.array2d(dtype=dtype)):   # type: ignore
        _ = wp.tid()
        acc = wp.tile_load(q, shape=(n_r, 1))
        for b in range(B):
            Mb = wp.tile_load(M[b], shape=(n, n_r))
            MbT = wp.tile_transpose(Mb)
            vb = wp.tile_load(v0, shape=(1, n), offset=(b, 0))
            vbc = wp.tile_transpose(vb)
            prod = wp.tile_zeros(shape=(n_r, 1), dtype=dtype)
            wp.tile_matmul(MbT, vbc, prod)
            acc = wp.tile_map(sub2, acc, prod)
        Lt = wp.tile_load(S, shape=(n_r, n_r))
        wp.tile_lower_solve_inplace(Lt, acc)
        LT = wp.tile_transpose(Lt)
        wp.tile_upper_solve_inplace(LT, acc)
        wp.tile_store(y, acc)

    return fused_root_solve_kernel


@lru_cache(maxsize=None)
def create_root_schur_kernel(dtype=wp.float64):
    """Fallback (large batches/roots): batch-parallel atomic reduction
    ``S[i, j] -= sum_k M[b, k, i] M[b, k, j]`` launched with
    ``dim=[B, n_r, n_r]`` into ``S`` pre-initialized with ``R``.
    Accumulation order is scheduling dependent (like SOCU's own
    substitutions); solution tolerances cover it."""
    module = make_module("root_schur")

    @wp.kernel(module=module)
    def root_schur_kernel(M: wp.array3d(dtype=dtype),   # type: ignore
                          S: wp.array2d(dtype=dtype)):  # type: ignore
        b, i, j = wp.tid()
        acc = dtype(0.0)
        for k in range(M.shape[1]):
            acc += M[b, k, i] * M[b, k, j]
        wp.atomic_add(S, i, j, -acc)

    return root_schur_kernel


@lru_cache(maxsize=None)
def create_root_rhs_kernel(dtype=wp.float64):
    """Fallback (large batches/roots): batch-parallel atomic reduction
    ``qbar[i] -= sum_k M[b, k, i] v0[b, k]`` launched with
    ``dim=[B, n_r]`` into ``qbar`` pre-initialized with ``q``."""
    module = make_module("root_rhs")

    @wp.kernel(module=module)
    def root_rhs_kernel(M: wp.array3d(dtype=dtype),      # type: ignore
                        v0: wp.array2d(dtype=dtype),     # type: ignore
                        qbar: wp.array2d(dtype=dtype)):  # type: ignore
        b, i = wp.tid()
        acc = dtype(0.0)
        for k in range(M.shape[1]):
            acc += M[b, k, i] * v0[b, k]
        wp.atomic_add(qbar, i, 0, -acc)

    return root_rhs_kernel


# -------------------------------------------------------- solve phase
@lru_cache(maxsize=None)
def create_final_node_rhs_kernel(n: int, Lp: int, dtype=wp.float64):
    """Fused final-node RHS update and triangular solve, one thread
    block per tail: ``v0_b = J_b^{-1} (r0_b - sum_p H_p^T v[b, slot_p])``
    reading only the ``Lp`` ancestor-path slots of the prefix solve
    buffer ``v``."""
    module = make_module(f"final_node_rhs_{n}_{Lp}")
    sub2 = _sub_func(dtype)

    @wp.kernel(module=module)
    def final_node_rhs_kernel(slots: wp.array(dtype=wp.int32),  # type: ignore
                              r0: wp.array2d(dtype=dtype),      # type: ignore
                              H: wp.array4d(dtype=dtype),       # type: ignore
                              v: wp.array4d(dtype=dtype),       # type: ignore
                              J: wp.array3d(dtype=dtype),       # type: ignore
                              v0: wp.array2d(dtype=dtype)):     # type: ignore
        b = wp.tid()
        r = wp.tile_load(r0, shape=(1, n), offset=(b, 0))
        acc = wp.tile_transpose(r)
        for p in range(Lp):
            Hp = wp.tile_load(H[b, p], shape=(n, n))
            HpT = wp.tile_transpose(Hp)
            vp = wp.tile_load(v[b, slots[p]], shape=(n, 1))
            prod = wp.tile_zeros(shape=(n, 1), dtype=dtype)
            wp.tile_matmul(HpT, vp, prod)
            acc = wp.tile_map(sub2, acc, prod)
        Jb = wp.tile_load(J[b], shape=(n, n))
        wp.tile_lower_solve_inplace(Jb, acc)
        wp.tile_store(v0, wp.tile_transpose(acc), offset=(b, 0))

    return final_node_rhs_kernel


@lru_cache(maxsize=None)
def create_final_node_recover_kernel(n: int, n_r: int, Lp: int,
                                     dtype=wp.float64):
    """Fused final-node recovery and prefix correction, one thread
    block per tail: ``w0_b = J_b^{-T} (v0_b - M_b y)`` (stored), then
    ``v[b, slot_p] -= H_p w0_b`` on the ancestor-path slots only."""
    module = make_module(f"final_node_recover_{n}_{n_r}_{Lp}")
    sub2 = _sub_func(dtype)

    @wp.kernel(module=module)
    def final_node_recover_kernel(slots: wp.array(dtype=wp.int32),  # type: ignore
                                  M: wp.array3d(dtype=dtype),   # type: ignore
                                  y: wp.array2d(dtype=dtype),   # type: ignore
                                  J: wp.array3d(dtype=dtype),   # type: ignore
                                  H: wp.array4d(dtype=dtype),   # type: ignore
                                  v0: wp.array2d(dtype=dtype),  # type: ignore
                                  w0: wp.array2d(dtype=dtype),  # type: ignore
                                  v: wp.array4d(dtype=dtype)):  # type: ignore
        b = wp.tid()
        row = wp.tile_load(v0, shape=(1, n), offset=(b, 0))
        acc = wp.tile_transpose(row)
        Mb = wp.tile_load(M[b], shape=(n, n_r))
        yt = wp.tile_load(y, shape=(n_r, 1))
        prod = wp.tile_zeros(shape=(n, 1), dtype=dtype)
        wp.tile_matmul(Mb, yt, prod)
        acc = wp.tile_map(sub2, acc, prod)
        Jb = wp.tile_load(J[b], shape=(n, n))
        JbT = wp.tile_transpose(Jb)
        wp.tile_upper_solve_inplace(JbT, acc)
        wp.tile_store(w0, wp.tile_transpose(acc), offset=(b, 0))
        for p in range(Lp):
            Hp = wp.tile_load(H[b, p], shape=(n, n))
            corr = wp.tile_zeros(shape=(n, 1), dtype=dtype)
            wp.tile_matmul(Hp, acc, corr)
            vp = wp.tile_load(v[b, slots[p]], shape=(n, 1))
            vp = wp.tile_map(sub2, vp, corr)
            wp.tile_store(v[b, slots[p]], vp)

    return final_node_recover_kernel

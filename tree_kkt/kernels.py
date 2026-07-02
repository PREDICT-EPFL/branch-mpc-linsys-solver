"""Warp kernels for the one-level tree (batched SOCU tails) solver.

All performance-critical work introduced by the tree solver runs in these
kernels; nothing is brought back to the host on the normal GPU path.

Notation used in the docstrings:

* ``B``  - number of tails (the SOCU batch dimension)
* ``T``  - tail length (blocks per tail)
* ``n``  - block size
* ``q``  - number of right-hand sides
* ``G[i]``            coupling block of tail ``i`` (root <-> physical node 0)
* ``interface_pos``   storage position of physical node 0 (== 0 for socu)

Kernel factories are cached by their static shape parameters so that each
concrete ``(n, dtype)`` / ``(n, q, dtype)`` combination compiles exactly once.
Tile shapes must be compile-time constants (closed over via ``wp.static``);
``num_tails`` and ``interface_pos`` are passed as runtime kernel arguments.
"""

from functools import lru_cache

import warp as wp


@lru_cache(maxsize=None)
def create_build_coupling_rhs_kernel(n: int, dtype=wp.float64):
    """Kernel that writes each ``G[i]`` into ``out[i, interface_pos]``.

    ``out`` (shape ``(B, T, n, n)``) must be zeroed by the caller first; this
    kernel only stamps the single non-zero block of each implicit ``C_i``.
    """
    module = wp.Module('tree_build_coupling_rhs_kernel', None)
    module.options['enable_backward'] = False

    @wp.kernel(module=module)
    def build_coupling_rhs_kernel(interface_pos: int,
                                  G: wp.array3d(dtype=dtype),   # type: ignore
                                  out: wp.array4d(dtype=dtype)):  # type: ignore
        i = wp.tid()
        Gi = wp.tile_load(G[i], shape=(n, n))
        wp.tile_store(out[i, interface_pos], Gi)

    return build_coupling_rhs_kernel


@lru_cache(maxsize=None)
def create_build_root_schur_kernel(n: int, dtype=wp.float64):
    """Kernel that forms ``S0 = D0 - sum_i G_i^T U_i[interface_pos]``.

    Launched with a single tile (``dim=[1]``); it loops over the (small) number
    of tails internally, avoiding cross-tail atomics.  ``D0`` and ``out`` have
    shape ``(1, n, n)``; ``U`` has shape ``(B, T, n, n)``.
    """
    module = wp.Module('tree_build_root_schur_kernel', None)
    module.options['enable_backward'] = False

    @wp.kernel(module=module)
    def build_root_schur_kernel(num_tails: int,
                                interface_pos: int,
                                D0: wp.array3d(dtype=dtype),  # type: ignore
                                G: wp.array3d(dtype=dtype),   # type: ignore
                                U: wp.array4d(dtype=dtype),    # type: ignore
                                out: wp.array3d(dtype=dtype)):  # type: ignore
        _ = wp.tid()
        S = wp.tile_load(D0[0], shape=(n, n))
        for i in range(num_tails):
            Gi = wp.tile_load(G[i], shape=(n, n))
            Ui = wp.tile_load(U[i, interface_pos], shape=(n, n))
            GiT = wp.tile_transpose(Gi)
            # S += -1 * G_i^T @ U_i   (accumulate: out = alpha*a@b + beta*out)
            wp.tile_matmul(GiT, Ui, S, alpha=-1.0)
        wp.tile_store(out[0], S)

    return build_root_schur_kernel


@lru_cache(maxsize=None)
def create_build_reduced_root_rhs_kernel(n: int, n_rhs: int, dtype=wp.float64):
    """Kernel that forms ``r0_hat = r0 - sum_i G_i^T v_i[interface_pos]``.

    ``r0`` and ``out`` have shape ``(1, n, q)``; ``v`` has shape ``(B, T, n, q)``.
    """
    module = wp.Module('tree_build_reduced_root_rhs_kernel', None)
    module.options['enable_backward'] = False

    @wp.kernel(module=module)
    def build_reduced_root_rhs_kernel(num_tails: int,
                                      interface_pos: int,
                                      r0: wp.array3d(dtype=dtype),  # type: ignore
                                      G: wp.array3d(dtype=dtype),   # type: ignore
                                      v: wp.array4d(dtype=dtype),    # type: ignore
                                      out: wp.array3d(dtype=dtype)):  # type: ignore
        _ = wp.tid()
        r = wp.tile_load(r0[0], shape=(n, n_rhs))
        for i in range(num_tails):
            Gi = wp.tile_load(G[i], shape=(n, n))
            vi = wp.tile_load(v[i, interface_pos], shape=(n, n_rhs))
            GiT = wp.tile_transpose(Gi)
            wp.tile_matmul(GiT, vi, r, alpha=-1.0)
        wp.tile_store(out[0], r)

    return build_reduced_root_rhs_kernel


@lru_cache(maxsize=None)
def create_root_factor_kernel(n: int, dtype=wp.float64):
    """In-place dense Cholesky ``S0 = L0 L0^T`` of the ``n x n`` root Schur block.

    Operates on ``S`` with shape ``(1, n, n)``; the lower factor overwrites it.
    """
    module = wp.Module('tree_root_factor_kernel', None)
    module.options['enable_backward'] = False

    @wp.kernel(module=module)
    def root_factor_kernel(S: wp.array3d(dtype=dtype)):  # type: ignore
        _ = wp.tid()
        L0 = wp.tile_load(S[0], shape=(n, n))
        wp.tile_cholesky_inplace(L0)
        wp.tile_store(S[0], L0)

    return root_factor_kernel


@lru_cache(maxsize=None)
def create_root_solve_kernel(n: int, n_rhs: int, dtype=wp.float64):
    """Solve ``L0 L0^T x = rhs`` in place given the stored lower factor ``L0``.

    ``L0`` has shape ``(1, n, n)`` (the factor produced by
    :func:`create_root_factor_kernel`); ``rhs`` has shape ``(1, n, q)`` and is
    overwritten with the solution.
    """
    module = wp.Module('tree_root_solve_kernel', None)
    module.options['enable_backward'] = False

    @wp.kernel(module=module)
    def root_solve_kernel(L0: wp.array3d(dtype=dtype),   # type: ignore
                          rhs: wp.array3d(dtype=dtype)):  # type: ignore
        _ = wp.tid()
        L = wp.tile_load(L0[0], shape=(n, n))
        y = wp.tile_load(rhs[0], shape=(n, n_rhs))
        wp.tile_lower_solve_inplace(L, y)   # forward:  L y1 = rhs
        LT = wp.tile_transpose(L)
        wp.tile_upper_solve_inplace(LT, y)  # backward: L^T x = y1
        wp.tile_store(rhs[0], y)

    return root_solve_kernel


@lru_cache(maxsize=None)
def create_recover_tails_kernel(n: int, n_rhs: int, dtype=wp.float64):
    """Kernel that recovers ``w_i = v_i - U_i x_root`` for every tail block.

    Launched over ``dim=[B, T]``.  ``U`` has shape ``(B, T, n, n)``, ``v`` has
    shape ``(B, T, n, q)``, ``x_root`` has shape ``(1, n, q)``.  The result is
    written to ``out`` (shape ``(B, T, n, q)``) so ``v`` can be preserved.
    """
    module = wp.Module('tree_recover_tails_kernel', None)
    module.options['enable_backward'] = False

    @wp.kernel(module=module)
    def recover_tails_kernel(U: wp.array4d(dtype=dtype),       # type: ignore
                             v: wp.array4d(dtype=dtype),       # type: ignore
                             x_root: wp.array3d(dtype=dtype),  # type: ignore
                             out: wp.array4d(dtype=dtype)):     # type: ignore
        i, k = wp.tid()
        Uik = wp.tile_load(U[i, k], shape=(n, n))
        xr = wp.tile_load(x_root[0], shape=(n, n_rhs))
        wik = wp.tile_load(v[i, k], shape=(n, n_rhs))
        # w = v - U @ x_root
        wp.tile_matmul(Uik, xr, wik, alpha=-1.0)
        wp.tile_store(out[i, k], wik)

    return recover_tails_kernel


@lru_cache(maxsize=None)
def create_check_spd_diag_kernel(n: int, dtype=wp.float64):
    """Kernel that flags a failed Cholesky by inspecting the factor diagonal.

    For a successful Cholesky every diagonal entry of the lower factor is real,
    finite and strictly positive.  A non-SPD input produces ``NaN`` (from the
    square root of a non-positive pivot) or a non-positive diagonal.  This
    kernel sets ``status[0] = 0`` if any inspected block fails the test.

    Launched over ``dim=[num_blocks]`` where ``L`` is viewed as
    ``(num_blocks, n, n)``.  ``status`` must be initialised to ``1``.
    """
    module = wp.Module('tree_check_spd_diag_kernel', None)
    module.options['enable_backward'] = False

    @wp.kernel(module=module)
    def check_spd_diag_kernel(L: wp.array3d(dtype=dtype),        # type: ignore
                              status: wp.array(dtype=wp.int32)):  # type: ignore
        b = wp.tid()
        for d in range(n):
            val = L[b, d, d]
            # NaN fails (val != val); non-positive pivots fail too.
            if val != val or val <= dtype(0.0):
                status[0] = wp.int32(0)

    return check_spd_diag_kernel

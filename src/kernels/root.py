"""Single-tile dense root Cholesky/solve fallback.

Used only for small separator sizes not covered by SOCU's padding rules;
SOCU-aligned sizes factor the root through SOCU itself (treated as a
one-branch, one-stage chain).
"""

from functools import lru_cache

import warp as wp

from src.kernels import make_module


@lru_cache(maxsize=None)
def create_root_factor_kernel(n_y: int, dtype=wp.float64):
    """In-place dense Cholesky of the ``(n_y, n_y)`` root Schur
    complement."""
    module = make_module("root_factor")

    @wp.kernel(module=module)
    def root_factor_kernel(S: wp.array2d(dtype=dtype)):  # type: ignore
        _ = wp.tid()
        L = wp.tile_load(S, shape=(n_y, n_y))
        wp.tile_cholesky_inplace(L)
        wp.tile_store(S, L)

    return root_factor_kernel


@lru_cache(maxsize=None)
def create_root_solve_kernel(n_y: int, nrhs: int, dtype=wp.float64):
    """Solve ``L L^T y = s`` in place given the stored lower factor."""
    module = make_module("root_solve")

    @wp.kernel(module=module)
    def root_solve_kernel(L: wp.array2d(dtype=dtype),    # type: ignore
                          s: wp.array2d(dtype=dtype)):   # type: ignore
        _ = wp.tid()
        Lt = wp.tile_load(L, shape=(n_y, n_y))
        y = wp.tile_load(s, shape=(n_y, nrhs))
        wp.tile_lower_solve_inplace(Lt, y)
        LT = wp.tile_transpose(Lt)
        wp.tile_upper_solve_inplace(LT, y)
        wp.tile_store(s, y)

    return root_solve_kernel

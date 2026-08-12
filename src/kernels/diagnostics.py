"""Factor health checks (pivot scans; diagnostics only, never on the
timed warm path)."""

from functools import lru_cache

import warp as wp

from src.kernels import make_module


@lru_cache(maxsize=None)
def create_check_pivots_kernel(dtype=wp.float64):
    """Flag non-finite or non-positive Cholesky pivots.

    ``L`` is viewed as ``(num_blocks, n, n)`` with ``n`` a runtime
    argument; ``status`` (int32, length 1, initialized to 1) is cleared to
    0 if any inspected diagonal entry fails."""
    module = make_module("check_pivots")

    @wp.kernel(module=module)
    def check_pivots_kernel(n: int,
                            L: wp.array3d(dtype=dtype),         # type: ignore
                            status: wp.array(dtype=wp.int32)):  # type: ignore
        i = wp.tid()
        for d in range(n):
            val = L[i, d, d]
            if val != val or val <= dtype(0.0):
                status[0] = wp.int32(0)

    return check_pivots_kernel


@lru_cache(maxsize=None)
def create_min_pivot_kernel(dtype=wp.float64):
    """Atomic-min reduction of the factor diagonals into ``out[0]``
    (initialize ``out[0]`` to a large value first).  Diagnostics only."""
    module = make_module("min_pivot")

    @wp.kernel(module=module)
    def min_pivot_kernel(n: int,
                         L: wp.array3d(dtype=dtype),      # type: ignore
                         out: wp.array(dtype=dtype)):     # type: ignore
        i = wp.tid()
        for d in range(n):
            wp.atomic_min(out, 0, L[i, d, d])

    return min_pivot_kernel

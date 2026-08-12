"""Cross-branch reductions and Schur finalization (scalar kernels).

Per-branch contributions are reduced with a fixed pairwise tree whose
result is independent of thread scheduling, then finalized against the
root matrix; the Schur complement is therefore bitwise-reproducible run
to run.
"""

from functools import lru_cache

import warp as wp

from src.kernels import TILE_M, make_module


@lru_cache(maxsize=None)
def create_pair_reduce_kernel(dtype=wp.float64):
    """One level of the fixed pairwise reduction tree over ``(B, r, c)``
    contributions: ``buf[2*s*p] += buf[2*s*p + s]`` elementwise.

    Launched with ``dim=[num_pairs, r, c]``; repeated launches with stride
    1, 2, 4, ... implement a deterministic binary reduction whose result
    is independent of thread scheduling.
    """
    module = make_module("pair_reduce")

    @wp.kernel(module=module)
    def pair_reduce_kernel(stride: int,
                           count: int,
                           buf: wp.array3d(dtype=dtype)):  # type: ignore
        p, i, j = wp.tid()
        dst = 2 * stride * p
        src = dst + stride
        if src < count:
            buf[dst, i, j] = buf[dst, i, j] + buf[src, i, j]

    return pair_reduce_kernel


@lru_cache(maxsize=None)
def create_finalize_schur_kernel(dtype=wp.float64):
    """Form ``S = R - M`` elementwise from the reduced contribution
    ``M = buf[0]``, of which only the lower tile triangle is valid; the
    upper triangle of ``S`` is mirrored from the valid region (an entry is
    valid iff its column tile index does not exceed its row tile index).
    Launched with ``dim=[n_y, n_y]``."""
    module = make_module("finalize_schur")

    @wp.kernel(module=module)
    def finalize_schur_kernel(R: wp.array2d(dtype=dtype),    # type: ignore
                              M: wp.array3d(dtype=dtype),    # type: ignore
                              S: wp.array2d(dtype=dtype)):   # type: ignore
        i, j = wp.tid()
        if j // TILE_M <= i // TILE_M:
            S[i, j] = R[i, j] - M[0, i, j]
        else:
            S[i, j] = R[j, i] - M[0, j, i]

    return finalize_schur_kernel


@lru_cache(maxsize=None)
def create_finalize_rhs_kernel(dtype=wp.float64):
    """Form ``s = s - h`` elementwise from the reduced RHS contribution
    ``h = buf[0]`` (``s`` holds the staged root RHS ``q``).
    ``dim=[n_y, nrhs]``."""
    module = make_module("finalize_rhs")

    @wp.kernel(module=module)
    def finalize_rhs_kernel(h: wp.array3d(dtype=dtype),    # type: ignore
                            s: wp.array2d(dtype=dtype)):   # type: ignore
        i, j = wp.tid()
        s[i, j] = s[i, j] - h[0, i, j]

    return finalize_rhs_kernel

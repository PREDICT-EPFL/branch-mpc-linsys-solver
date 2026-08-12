"""Per-branch Schur products and the tail recovery update (tiled kernels).

The Schur kernels compute per-branch blocks of ``sum_t U[i,t]^T V[i,t]``,
where ``U``/``V`` are stage-rows-by-separator-columns arrays: in factor
space both operands are the transformed coupling ``G``; in the
inverse-action ablation they are ``C_T`` and ``X = K^{-1} C^T``.

The chunked kernels view each branch as one contiguous
``(T * n_b, n_y)`` matrix and iterate ``ROW_CHUNK``-row slabs -- few,
large tile operations (profiling: a per-stage inner loop is ILP-bound;
chunking runs the SYRK ~1.2x and the RHS/recovery products several
times faster).  Rows beyond the array are handled by Warp's
bounds-checked loads (zero fill), so the row count need not be a
multiple of the chunk size.
"""

from functools import lru_cache

import warp as wp

from src.kernels import TILE_M, make_module

#: Rows per chunk of the chunked Schur kernels (compile-time constant).
#: 32 measured best among the launchable sizes on RTX 5090; 64-row
#: chunks exceed Warp tile launch limits for FP64.
ROW_CHUNK = 32


def num_row_chunks(num_rows: int) -> int:
    """Number of ``ROW_CHUNK``-row slabs covering ``num_rows`` rows."""
    return (num_rows + ROW_CHUNK - 1) // ROW_CHUNK


@lru_cache(maxsize=None)
def create_chunked_schur_kernel(dtype=wp.float64):
    """Tile ``(i, j)`` of ``M_b = U[b]^T V[b]``, with
    ``U``/``V`` viewed as ``(B, T * n_b, n_y)`` and the row dimension
    processed in ``ROW_CHUNK`` slabs.

    Launched with ``dim=[B, MT, MT]``; ``n_chunks`` is
    ``num_row_chunks(T * n_b)``; ``out`` has shape ``(B, n_y, n_y)``.
    Only the lower tile triangle is computed.
    """
    module = make_module("chunked_schur")

    @wp.kernel(module=module)
    def chunked_schur_kernel(n_chunks: int,
                             U: wp.array3d(dtype=dtype),     # type: ignore
                             V: wp.array3d(dtype=dtype),     # type: ignore
                             out: wp.array3d(dtype=dtype)):  # type: ignore
        b, i, j = wp.tid()
        if j > i:
            return
        acc = wp.tile_zeros(shape=(TILE_M, TILE_M), dtype=dtype)
        for c in range(n_chunks):
            Uc = wp.tile_load(U[b], shape=(ROW_CHUNK, TILE_M),
                              offset=(ROW_CHUNK * c, TILE_M * i))
            Vc = wp.tile_load(V[b], shape=(ROW_CHUNK, TILE_M),
                              offset=(ROW_CHUNK * c, TILE_M * j))
            UcT = wp.tile_transpose(Uc)
            wp.tile_matmul(UcT, Vc, acc)
        wp.tile_store(out[b], acc, offset=(TILE_M * i, TILE_M * j))

    return chunked_schur_kernel


@lru_cache(maxsize=None)
def create_chunked_schur_rhs_kernel(nrhs: int, dtype=wp.float64):
    """Row-tile ``i`` of ``h_b = U[b]^T v[b]``,
    with ``U`` viewed as ``(B, T * n_b, n_y)`` and ``v`` as
    ``(B, T * n_b, nrhs)``.  Launched with ``dim=[B, MT]``."""
    module = make_module("chunked_schur_rhs")

    @wp.kernel(module=module)
    def chunked_schur_rhs_kernel(n_chunks: int,
                                 U: wp.array3d(dtype=dtype),     # type: ignore
                                 v: wp.array3d(dtype=dtype),     # type: ignore
                                 out: wp.array3d(dtype=dtype)):  # type: ignore
        b, i = wp.tid()
        acc = wp.tile_zeros(shape=(TILE_M, nrhs), dtype=dtype)
        for c in range(n_chunks):
            Uc = wp.tile_load(U[b], shape=(ROW_CHUNK, TILE_M),
                              offset=(ROW_CHUNK * c, TILE_M * i))
            vc = wp.tile_load(v[b], shape=(ROW_CHUNK, nrhs),
                              offset=(ROW_CHUNK * c, 0))
            UcT = wp.tile_transpose(Uc)
            wp.tile_matmul(UcT, vc, acc)
        wp.tile_store(out[b], acc, offset=(TILE_M * i, 0))

    return chunked_schur_rhs_kernel


@lru_cache(maxsize=None)
def create_chunked_recover_kernel(nrhs: int, dtype=wp.float64):
    """Recovery ``out[b, rows] = v[b, rows] - G[b, rows] @ y`` for one
    ``ROW_CHUNK``-row slab per block, with ``G`` viewed as
    ``(B, T * n_b, n_y)`` and ``v``/``out`` as ``(B, T * n_b, nrhs)``.

    Launched with ``dim=[B, num_row_chunks(T * n_b)]``;
    ``mt = num_separator_tiles(n_y)`` is a runtime argument.
    """
    module = make_module("chunked_recover")

    @wp.kernel(module=module)
    def chunked_recover_kernel(mt: int,
                               G: wp.array3d(dtype=dtype),     # type: ignore
                               v: wp.array3d(dtype=dtype),     # type: ignore
                               y: wp.array2d(dtype=dtype),     # type: ignore
                               out: wp.array3d(dtype=dtype)):  # type: ignore
        b, c = wp.tid()
        w = wp.tile_load(v[b], shape=(ROW_CHUNK, nrhs),
                         offset=(ROW_CHUNK * c, 0))
        for i in range(mt):
            Gc = wp.tile_load(G[b], shape=(ROW_CHUNK, TILE_M),
                              offset=(ROW_CHUNK * c, TILE_M * i))
            yc = wp.tile_load(y, shape=(TILE_M, nrhs),
                              offset=(TILE_M * i, 0))
            wp.tile_matmul(Gc, yc, w, alpha=-1.0)
        wp.tile_store(out[b], w, offset=(ROW_CHUNK * c, 0))

    return chunked_recover_kernel

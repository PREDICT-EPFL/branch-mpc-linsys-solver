"""Root updates and the tail correction of the permuted factor.

Kernel families over the root coupling ``M_i^T`` (stored as
``(B, T * n_b, n_r)`` with rows in physical stage order):

- the tiled chunked kernels iterate ``ROW_CHUNK``-row slabs of large
  tile operations; they are the default for root dimensions of at least
  ``TILE_M`` (profiling: a per-stage inner loop is ILP-bound; chunking
  runs the diagonal update ~1.2x and the RHS/correction products several
  times faster).  Rows beyond the array are handled by Warp's
  bounds-checked loads (zero fill), so the row count need not be a
  multiple of the chunk size;
- the single-RHS large-root kernels use an eight-way partitioned GEMV
  for the root RHS and one scalar output thread per tail row for the
  correction. They avoid degenerate ``TILE_M x 1`` tile matmuls while
  preserving the canonical coupling layout. The partition count is
  fixed from measurements over ``n_r in {16, 32, 64, 128}``;
- the scalar small-root kernels compute deterministic partial dot
  products over ``(tail, row_chunk, root_row, root_col)`` and reduce
  them per tail in a fixed order; they avoid loading mostly
  out-of-bounds ``TILE_M`` tiles when the root dimension is small
  (prototype-validated 2.7x on the profiled ``n_r = 2`` bottleneck).

Cross-tail accumulation is shared by both families: one atomic
kernel subtracts every per-tail contribution into the initialized
root target.  Floating-point summation order is scheduling dependent;
solution and residual tolerances cover it (plan 4 measured the atomic
path no slower than the removed fixed-order pairwise tree at every
representative point, so no fallback is kept).
"""

from functools import lru_cache

import warp as wp

from src.general_arrow.kernels import TILE_M, make_module

#: Rows per chunk of the chunked root-update kernels (compile-time constant).
#: 32 measured best among the launchable sizes on RTX 5090; 64-row
#: chunks exceed Warp tile launch limits for FP64.
ROW_CHUNK = 32


def num_row_chunks(num_rows: int) -> int:
    """Number of ``ROW_CHUNK``-row slabs covering ``num_rows`` rows."""
    return (num_rows + ROW_CHUNK - 1) // ROW_CHUNK


@lru_cache(maxsize=None)
def create_chunked_root_update_kernel(dtype=wp.float64):
    """Tile ``(i, j)`` of ``M_b = U[b]^T V[b]``, with
    ``U``/``V`` viewed as ``(B, T * n_b, n_r)`` and the row dimension
    processed in ``ROW_CHUNK`` slabs.

    Launched with ``dim=[B, MT, MT]``; ``n_chunks`` is
    ``num_row_chunks(T * n_b)``; ``out`` has shape ``(B, n_r, n_r)``.
    Only the lower tile triangle is computed.
    """
    module = make_module("chunked_root_update")

    @wp.kernel(module=module)
    def chunked_root_update_kernel(n_chunks: int,
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

    return chunked_root_update_kernel


@lru_cache(maxsize=None)
def create_chunked_root_rhs_kernel(nrhs: int, dtype=wp.float64):
    """Row-tile ``i`` of ``h_b = U[b]^T v[b]``,
    with ``U`` viewed as ``(B, T * n_b, n_r)`` and ``v`` as
    ``(B, T * n_b, nrhs)``.  Launched with ``dim=[B, MT]``."""
    module = make_module("chunked_root_rhs")

    @wp.kernel(module=module)
    def chunked_root_rhs_kernel(n_chunks: int,
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

    return chunked_root_rhs_kernel


@lru_cache(maxsize=None)
def create_root_rhs_vector_partial_kernel(partitions: int, n_r: int,
                                          dtype=wp.float64):
    """Parallel partial GEMV for one RHS.

    Each ``(tail, partition, root row)`` thread reduces one contiguous
    tail-row segment into ``partial``. A second kernel combines the small
    fixed number of partitions into the existing per-tail contribution.
    This exposes more parallelism than one long dot product per root row
    while retaining the canonical coupling layout.
    """
    module = make_module(f"root_rhs_vector_partial_{partitions}_{n_r}")

    @wp.kernel(module=module)
    def root_rhs_vector_partial_kernel(
            segment: int,
            num_rows: int,
            M_T: wp.array3d(dtype=dtype),       # type: ignore
            v: wp.array3d(dtype=dtype),         # type: ignore
            partial: wp.array3d(dtype=dtype)):  # type: ignore
        b, p, j = wp.tid()
        r0 = p * segment
        r1 = wp.min(r0 + segment, num_rows)
        acc = dtype(0.0)
        for r in range(r0, r1):
            acc += M_T[b, r, j] * v[b, r, 0]
        partial[b, p, j] = acc

    return root_rhs_vector_partial_kernel


@lru_cache(maxsize=None)
def create_root_rhs_vector_reduce_kernel(partitions: int, n_r: int,
                                         dtype=wp.float64):
    """Reduce single-RHS row partitions into ``contrib[b, j, 0]``."""
    module = make_module(f"root_rhs_vector_reduce_{partitions}_{n_r}")

    @wp.kernel(module=module)
    def root_rhs_vector_reduce_kernel(
            partial: wp.array3d(dtype=dtype),   # type: ignore
            contrib: wp.array3d(dtype=dtype)):  # type: ignore
        b, j = wp.tid()
        acc = dtype(0.0)
        for p in range(partitions):
            acc += partial[b, p, j]
        contrib[b, j, 0] = acc

    return root_rhs_vector_reduce_kernel


#: Eight row partitions saturate the single-RHS GEMV on the target grid;
#: sixteen does not improve n_r <= 64 and regresses n_r = 128.
ROOT_RHS_VECTOR_PARTITIONS = 8


@lru_cache(maxsize=None)
def create_chunked_recover_kernel(nrhs: int, dtype=wp.float64):
    """Recovery ``out[b, rows] = v[b, rows] - G[b, rows] @ y`` for one
    ``ROW_CHUNK``-row slab per block, with ``G`` viewed as
    ``(B, T * n_b, n_r)`` and ``v``/``out`` as ``(B, T * n_b, nrhs)``.

    Launched with ``dim=[B, num_row_chunks(T * n_b)]``;
    ``mt = num_root_tiles(n_r)`` is a runtime argument.
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


@lru_cache(maxsize=None)
def create_tail_root_vector_correction_kernel(n_r: int, dtype=wp.float64):
    """Single-RHS tail correction ``out = v - M_T @ y``.

    One thread writes one tail row directly into the buffer consumed by
    SOCU's backward substitution. The private specialization removes
    the degenerate ``ROW_CHUNK x 1`` tile matmul used by the multi-RHS
    path. ``n_r`` remains in the builder cache key for shape tuning.

    Launched with ``dim=[B, num_rows]``.
    """
    module = make_module(f"tail_root_vector_correction_{n_r}")

    @wp.kernel(module=module)
    def tail_root_vector_correction_kernel(
            root_dim: int,
            M_T: wp.array3d(dtype=dtype),       # type: ignore
            v: wp.array3d(dtype=dtype),         # type: ignore
            y: wp.array2d(dtype=dtype),         # type: ignore
            out: wp.array3d(dtype=dtype)):      # type: ignore
        b, r = wp.tid()
        acc = v[b, r, 0]
        for j in range(root_dim):
            acc -= M_T[b, r, j] * y[j, 0]
        out[b, r, 0] = acc

    return tail_root_vector_correction_kernel


# ---------------------------------------------------------------------------
# Cross-tail accumulation (shared by both kernel families)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def create_atomic_root_update_kernel(dtype=wp.float64):
    """Atomically subtract every per-tail root-update contribution into
    the root factor input: ``root_diag[i, j] -= contrib[b, i, j]`` over all
    tails ``b``.  ``S`` must be initialized (from ``R``) before this
    kernel runs (``root_diag`` holds ``R``); initialization and
    accumulation are separate launches,
    so there is no init/update race.  The tiled family fills only the
    lower tile triangle of ``contrib``; entries in the upper triangle
    read the mirrored element (for small roots both index the same
    single tile, which is fully populated).  Cross-tail summation
    order is scheduling dependent (atomic floating-point adds).
    Launched with ``dim=[B, n_r, n_r]``."""
    module = make_module("atomic_root_update")

    @wp.kernel(module=module)
    def atomic_root_update(contrib: wp.array3d(dtype=dtype),      # type: ignore
                           root_diag: wp.array2d(dtype=dtype)):  # type: ignore
        b, i, j = wp.tid()
        if j // TILE_M <= i // TILE_M:
            v = contrib[b, i, j]
        else:
            v = contrib[b, j, i]
        wp.atomic_add(root_diag, i, j, -v)

    return atomic_root_update


@lru_cache(maxsize=None)
def create_atomic_root_rhs_kernel(dtype=wp.float64):
    """Atomically subtract every per-tail RHS contribution into the
    root RHS: ``s[i, q] -= contrib[b, i, q]`` over all tails ``b``
    (``s`` already holds the root RHS ``q``, staged before any solve
    kernel runs).  Launched with ``dim=[B, n_r, nrhs]``."""
    module = make_module("atomic_root_rhs")

    @wp.kernel(module=module)
    def atomic_root_rhs(contrib: wp.array3d(dtype=dtype),  # type: ignore
                        s: wp.array2d(dtype=dtype)):       # type: ignore
        b, i, q = wp.tid()
        wp.atomic_add(s, i, q, -contrib[b, i, q])

    return atomic_root_rhs


# ---------------------------------------------------------------------------
# Scalar small-root kernels (root dimension below TILE_M)
# ---------------------------------------------------------------------------

#: Row partitions of the scalar partial-product kernels (fixed, so the
#: partial sums -- and therefore the reduced result -- are deterministic).
SMALL_ROOT_CHUNKS = 64

#: Root dimensions below TILE_M use the scalar small-root kernels (the
#: tiled kernels would load mostly out-of-bounds TILE_M-wide tiles).
#: Private test/benchmark hook: setting this to False forces the tiled
#: path for ablation.  Not a public option.
_SMALL_ROOT_ENABLED = True


def use_small_root(n_r: int) -> bool:
    """True when the scalar small-root kernels handle this root
    dimension."""
    return _SMALL_ROOT_ENABLED and n_r < TILE_M


def small_root_segment(num_rows: int) -> int:
    """Rows per partition of the scalar partial kernels."""
    return (num_rows + SMALL_ROOT_CHUNKS - 1) // SMALL_ROOT_CHUNKS


@lru_cache(maxsize=None)
def create_small_root_update_partial_kernel(dtype=wp.float64):
    """Scalar partial products of the root diagonal update:
    ``partial[b, c, i, j] = sum_r M_T[b, r, i] * M_T[b, r, j]`` over the
    rows of partition ``c``.  The full square is computed (the symmetric
    pair runs the same sum in the same order, so the result is exactly
    symmetric).  Launched with ``dim=[B, SMALL_ROOT_CHUNKS, n_r, n_r]``.
    """
    module = make_module("small_root_update")

    @wp.kernel(module=module)
    def small_root_update_partial(seg: int,
                                  num_rows: int,
                                  M_T: wp.array3d(dtype=dtype),      # type: ignore
                                  partial: wp.array4d(dtype=dtype)):  # type: ignore
        b, c, i, j = wp.tid()
        acc = dtype(0.0)
        r0 = c * seg
        r1 = wp.min(r0 + seg, num_rows)
        for r in range(r0, r1):
            acc += M_T[b, r, i] * M_T[b, r, j]
        partial[b, c, i, j] = acc

    return small_root_update_partial


@lru_cache(maxsize=None)
def create_small_root_rhs_partial_kernel(dtype=wp.float64):
    """Scalar partial products of the root RHS update:
    ``partial[b, c, j, q] = sum_r M_T[b, r, j] * v[b, r, q]`` over the
    rows of partition ``c``.  Launched with
    ``dim=[B, SMALL_ROOT_CHUNKS, n_r, nrhs]``."""
    module = make_module("small_root_rhs")

    @wp.kernel(module=module)
    def small_root_rhs_partial(seg: int,
                               num_rows: int,
                               M_T: wp.array3d(dtype=dtype),      # type: ignore
                               v: wp.array3d(dtype=dtype),        # type: ignore
                               partial: wp.array4d(dtype=dtype)):  # type: ignore
        b, c, j, q = wp.tid()
        acc = dtype(0.0)
        r0 = c * seg
        r1 = wp.min(r0 + seg, num_rows)
        for r in range(r0, r1):
            acc += M_T[b, r, j] * v[b, r, q]
        partial[b, c, j, q] = acc

    return small_root_rhs_partial


@lru_cache(maxsize=None)
def create_small_root_reduce_kernel(dtype=wp.float64):
    """Fixed-order reduction of the scalar partials into the per-tail
    contribution: ``contrib[b, i, j] = sum_c partial[b, c, i, j]``.
    Launched with ``dim=[B, r, c]``; the cross-tail reduction is the
    shared deterministic pairwise tree."""
    module = make_module("small_root_reduce")

    @wp.kernel(module=module)
    def small_root_reduce(partial: wp.array4d(dtype=dtype),     # type: ignore
                          contrib: wp.array3d(dtype=dtype)):    # type: ignore
        b, i, j = wp.tid()
        acc = dtype(0.0)
        for c in range(SMALL_ROOT_CHUNKS):
            acc += partial[b, c, i, j]
        contrib[b, i, j] = acc

    return small_root_reduce


@lru_cache(maxsize=None)
def create_small_root_correction_kernel(nrhs: int, dtype=wp.float64):
    """Scalar tail correction ``out[b, r, q] = v[b, r, q] -
    sum_j M_T[b, r, j] * y[j, q]`` (one thread per output element; the
    inner loop over the small root dimension is tiny).  Launched with
    ``dim=[B, T * n_b, nrhs_dim]`` where ``nrhs_dim = nrhs``."""
    module = make_module("small_root_correction")

    @wp.kernel(module=module)
    def small_root_correction(n_r: int,
                              M_T: wp.array3d(dtype=dtype),   # type: ignore
                              v: wp.array3d(dtype=dtype),     # type: ignore
                              y: wp.array2d(dtype=dtype),     # type: ignore
                              out: wp.array3d(dtype=dtype)):  # type: ignore
        b, r, q = wp.tid()
        acc = v[b, r, q]
        for j in range(n_r):
            acc -= M_T[b, r, j] * y[j, q]
        out[b, r, q] = acc

    return small_root_correction


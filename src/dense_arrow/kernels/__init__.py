"""Custom Warp kernels for the scenario-tree layer.

These are the only custom GPU kernels in the project; all tail
factorization/solve work is done by the upstream SOCU package (see
:mod:`src.socu`), which also factors the root
complement for SOCU-aligned root sizes.  Submodules:

- :mod:`src.kernels.coupling` -- root diagonal/RHS updates, the tail
  correction, and the deterministic cross-tail reductions (tiled
  chunked kernels plus scalar small-root variants);
- :mod:`src.kernels.root` -- single-tile dense root Cholesky/solve for
  small root dimensions.

All stage loops take runtime bounds (``t0``, ``t1``) so they are not
unrolled at compile time (unrolling grows shared memory with the number of
stages) and so coupling patterns that touch only some stages skip the zero
stages.  Tile loads/stores are bounds-checked by Warp, so ``n_r`` need not
be a multiple of the tile width.  Kernels are cached per compile-time
shape ``(n_b, nrhs, dtype)``.
"""

import os

import warp as wp

#: Tile width over the root dimension (compile-time constant).
#: Tuned on RTX 5090: 16 beats 8/32/64 at every measured point (up to
#: 43% on the root-update phases at small roots, ~10% on warm totals at
#: the default point); the environment variable TREE_SOCU_TILE_M
#: overrides it per process (kernels specialize on the value at import).
TILE_M = int(os.environ.get("TREE_SOCU_TILE_M", 16))

#: Threads per block for the tiled kernel launches (compile-time
#: constant; override with TREE_SOCU_BLOCK_DIM).
BLOCK_DIM = int(os.environ.get("TREE_SOCU_BLOCK_DIM", 128))


def num_root_tiles(n_r: int) -> int:
    """Number of ``TILE_M``-wide tiles covering a root of size
    ``n_r``."""
    return (n_r + TILE_M - 1) // TILE_M


def make_module(name):
    """A dedicated Warp module per kernel family (no backward pass)."""
    module = wp.Module(f"tree_socu_{name}", None)
    module.options["enable_backward"] = False
    return module

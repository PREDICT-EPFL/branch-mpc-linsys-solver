"""Custom Warp kernels for the scenario-tree layer.

These are the only custom GPU kernels in the project; all branch
factorization/solve work is done by the upstream SOCU package (see
:mod:`src.socu_adapter`), which also factors the root Schur
complement for SOCU-aligned separator sizes.  Submodules:

- :mod:`src.kernels.schur` -- per-branch Schur matrix/RHS products
  and the tail recovery update, tiled over the separator dimension;
- :mod:`src.kernels.reduction` -- deterministic pairwise and fused
  atomic cross-branch reductions plus finalization;
- :mod:`src.kernels.root` -- single-tile dense root Cholesky/solve
  fallback for small SOCU-unaligned separator sizes;
- :mod:`src.kernels.diagnostics` -- factor pivot checks and the
  minimum-pivot reduction.

All stage loops take runtime bounds (``t0``, ``t1``) so they are not
unrolled at compile time (unrolling grows shared memory with the number of
stages) and so coupling patterns that touch only some stages skip the zero
stages.  Tile loads/stores are bounds-checked by Warp, so ``n_y`` need not
be a multiple of the tile width.  Kernels are cached per compile-time
shape ``(n_b, nrhs, dtype)``.
"""

import os

import warp as wp

#: Tile width over the separator dimension (compile-time constant).
#: Tuned on RTX 5090: 16 beats 8/32/64 at every measured point (up to
#: 43% on the Schur phases at small separators, ~10% on warm totals at
#: the default point); the environment variable TREE_SOCU_TILE_M
#: overrides it per process (kernels specialize on the value at import).
TILE_M = int(os.environ.get("TREE_SOCU_TILE_M", 16))

#: Threads per block for the tiled kernel launches (compile-time
#: constant; override with TREE_SOCU_BLOCK_DIM).
BLOCK_DIM = int(os.environ.get("TREE_SOCU_BLOCK_DIM", 128))


def num_separator_tiles(n_y: int) -> int:
    """Number of ``TILE_M``-wide tiles covering a separator of size
    ``n_y``."""
    return (n_y + TILE_M - 1) // TILE_M


def make_module(name):
    """A dedicated Warp module per kernel family (no backward pass)."""
    module = wp.Module(f"tree_socu_{name}", None)
    module.options["enable_backward"] = False
    return module

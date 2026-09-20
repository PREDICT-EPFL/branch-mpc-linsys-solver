"""Warp kernels of the endpoint-tree solver.

Submodules:

- :mod:`endpoint_tree.kernels.tail` -- leaf-to-root block-tridiagonal
  Cholesky factorization and the forward/backward substitutions, one
  thread block per scenario, tiles specialized on the logical block
  size (no padding);
- :mod:`endpoint_tree.kernels.boundary` -- the boundary coupling
  transform ``M = L_last^{-1} G``, the root Schur update, the root RHS
  update, and the last-block correction.  All of them process exactly
  ``n_b`` boundary rows per tail, never ``T * n_b``.

Stage loops take runtime bounds so nothing unrolls with the horizon.
"""

import os

import warp as wp

#: Threads per block for the tiled per-scenario kernels.
BLOCK_DIM = int(os.environ.get("ENDPOINT_TREE_BLOCK_DIM", 64))


def make_module(name):
    """A dedicated Warp module per kernel family (no backward pass)."""
    module = wp.Module(f"endpoint_tree_{name}", None)
    module.options["enable_backward"] = False
    return module

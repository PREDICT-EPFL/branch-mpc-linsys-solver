"""Specialized GPU direct solver for scenario trees whose scenarios
share only the initial decision.

Every scenario tail is a block-tridiagonal chain stored and eliminated
in leaf-to-root order, and only the final (root-facing) block of each
tail couples to the shared root.  The coupling is stored as one
``(B, n_b, n_r)`` block per tail, so all root-related storage and work
is independent of the horizon; the general every-stage-coupling solver
lives in :mod:`src` and is unchanged.

Public API::

    EndpointTreeShape    problem dimensions (leaf-to-root contract)
    EndpointTreeMatrix   D, E, G_T, R block storage + conversions
    EndpointTreeVector   one structured vector
    EndpointTreeSolver   update / factorize / solve on the GPU
"""

from src.endpoint_tree.problem import (
    EndpointTreeMatrix,
    EndpointTreeShape,
    EndpointTreeVector,
)
from src.endpoint_tree.solver import EndpointTreeSolver

__all__ = [
    "EndpointTreeShape",
    "EndpointTreeMatrix",
    "EndpointTreeVector",
    "EndpointTreeSolver",
]

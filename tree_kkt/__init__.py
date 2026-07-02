"""One-level tree (single-branch) KKT solver built on batched SOCU tails.

See :class:`tree_kkt.one_level_tree.OneLevelTreeCholesky`.
"""

from tree_kkt.one_level_tree import OneLevelTreeCholesky
from tree_kkt.permutation import (
    permuted_block_position,
    interface_position,
    inverse_permutation,
    verify_identity_ordering,
)

__all__ = [
    "OneLevelTreeCholesky",
    "permuted_block_position",
    "interface_position",
    "inverse_permutation",
    "verify_identity_ordering",
]

"""Dense permutation/factor reconstruction helpers (test-only).

The solver factors the globally permuted matrix

    Phi_hat = Pi Phi Pi^T = L_hat L_hat^T,

where ``Pi = blkdiag(Pi_0, ..., Pi_{B-1}, I)`` and each ``Pi_i`` is the
recursive odd-even (cyclic-reduction) stage permutation used by SOCU.
The complete lower factor is stored structurally, never assembled:

    L_hat = [[L_hat_0                         0  ]
             [        ...                     ...]
             [               L_hat_{B-1}      0  ]
             [M_0    ...     M_{B-1}          L_R]]

with ``L_hat_i`` the permuted block-tridiagonal tail factor (SOCU
storage), ``M_i^T = L_hat_i^{-1} C_hat_i^T`` the root coupling columns,
and ``L_R L_R^T = R - sum_i M_i M_i^T`` the root diagonal block (a block
Cholesky diagonal update).

This module holds the permutation metadata, the persistent factor
container, and host-side validation utilities that reconstruct dense
factors from the structured storage (synchronizing; small cases only --
the solver itself never assembles anything dense).
"""

from dataclasses import dataclass

import numpy as np

def stage_elimination_levels(num_stages: int) -> np.ndarray:
    """Cyclic-reduction elimination level of every stage.

    Stage ``i`` is eliminated at level ``l`` where ``l`` is the number of
    trailing one-bits of ``i`` (level-``l`` pivots are the stages with
    ``i = 2^l - 1 (mod 2^(l+1))``).
    """
    T = int(num_stages)
    levels = np.zeros(T, dtype=np.int64)
    for i in range(T):
        v, l = i, 0
        while v & 1:
            v >>= 1
            l += 1
        levels[i] = l
    return levels


def tail_permutation(num_stages: int) -> np.ndarray:
    """Odd-even elimination order of one tail: ``perm[k]`` is the
    physical stage eliminated ``k``-th, i.e. placed at permuted block
    position ``k`` (rows of ``Pi_i`` select stages in this order)."""
    levels = stage_elimination_levels(num_stages)
    return np.lexsort((np.arange(num_stages), levels)).astype(np.int64)


@dataclass(frozen=True)
class TailPermutation:
    """Permutation metadata of one tail (all tails share it).

    ``order[k]`` is the physical stage at permuted position ``k``;
    ``position[i]`` is the permuted position of physical stage ``i``.
    """

    num_stages: int
    order: np.ndarray
    position: np.ndarray

    @classmethod
    def create(cls, num_stages: int) -> "TailPermutation":
        order = tail_permutation(num_stages)
        position = np.empty_like(order)
        position[order] = np.arange(len(order))
        return cls(num_stages=int(num_stages), order=order,
                   position=position)

    def row_order(self, block_dim: int) -> np.ndarray:
        """Row-level permutation: index array ``p`` such that
        ``x_permuted = x[p]`` for a tail vector of ``num_stages`` blocks
        of ``block_dim`` rows."""
        offsets = self.order[:, None] * block_dim + np.arange(block_dim)
        return offsets.reshape(-1)


# ---------------------------------------------------------------------------
# SOCU factor-storage layout (see socu.block_tridiag_solver)
# ---------------------------------------------------------------------------

def socu_level_segments(num_stages: int):
    """Off-diagonal storage segments per cyclic-reduction level:
    a list of ``(stride, offset, length)``.  Segment entry ``k`` at
    stride ``s`` couples the active stages ``a_k = s - 1 + s k`` and
    ``a_{k+1}``; after factorization it holds the corresponding
    off-diagonal block of the permuted factor (transposed for odd ``k``,
    see :func:`extract_tail_factor_dense`)."""
    T = int(num_stages)
    segments = []
    offset = 0
    stride = 1
    while stride <= T:
        length = max(T // stride - 1, 0)
        segments.append((stride, offset, length))
        offset += length
        stride *= 2
    return segments


# ---------------------------------------------------------------------------
# Host-side validation utilities (synchronize; small cases only)
# ---------------------------------------------------------------------------

def extract_tail_factor_dense(diag_factor: np.ndarray,
                              offdiag_factor: np.ndarray,
                              perm: TailPermutation) -> np.ndarray:
    """Assemble the dense permuted tail factor ``L_hat_i`` of one tail
    from host copies of SOCU's factor storage.

    ``diag_factor`` is ``(T, n_b, n_b)`` (the buffer SOCU overwrote with
    the diagonal factor blocks) and ``offdiag_factor``
    ``(n_off, n_b, n_b)``.  Validation only.
    """
    T = perm.num_stages
    n_b = diag_factor.shape[1]
    pos = perm.position
    L = np.zeros((T * n_b, T * n_b), dtype=np.float64)

    def block(bi, bj):
        return L[bi * n_b:(bi + 1) * n_b, bj * n_b:(bj + 1) * n_b]

    for i in range(T):
        block(pos[i], pos[i])[:] = np.tril(diag_factor[i])
    for stride, offset, length in socu_level_segments(T):
        for k in range(length):
            a_k = stride - 1 + stride * k
            a_next = a_k + stride
            if a_next >= T:
                continue
            stored = np.asarray(offdiag_factor[offset + k], dtype=np.float64)
            if k % 2 == 0:
                # pivot a_k: stored = E L^-T = factor block (a_next, a_k)
                block(pos[a_next], pos[a_k])[:] = stored
            else:
                # pivot a_next: stored = L^-1 E = transposed factor
                # block (a_k, a_next)
                block(pos[a_k], pos[a_next])[:] = stored.T
    return L


def assemble_dense_system(matrix) -> np.ndarray:
    """Dense symmetric ``Phi`` of a :class:`~src.problem.TreeMatrix`
    (tail blocks first, root last).  Validation only."""
    B, T, n_b, n_r = matrix.shape.dims()
    D = np.asarray(matrix.D, dtype=np.float64)
    E = np.asarray(matrix.E, dtype=np.float64)
    C_T = np.asarray(matrix.C_T, dtype=np.float64)
    R = np.asarray(matrix.R, dtype=np.float64)
    n = B * T * n_b + n_r
    Phi = np.zeros((n, n))
    for b in range(B):
        base = b * T * n_b
        for t in range(T):
            r = base + t * n_b
            Phi[r:r + n_b, r:r + n_b] = D[b, t]
            if t + 1 < T:
                Phi[r + n_b:r + 2 * n_b, r:r + n_b] = E[b, t]
                Phi[r:r + n_b, r + n_b:r + 2 * n_b] = E[b, t].T
            Phi[r:r + n_b, -n_r:] = C_T[b, t]
            Phi[-n_r:, r:r + n_b] = C_T[b, t].T
    Phi[-n_r:, -n_r:] = R
    return Phi


def global_permutation(shape) -> np.ndarray:
    """Row index array of the global permutation ``Pi`` (identity on the
    root): ``x_hat = x[p]``.  Validation only."""
    B, T, n_b, n_r = shape.dims()
    perm = TailPermutation.create(T)
    rows = perm.row_order(n_b)
    parts = [b * T * n_b + rows for b in range(B)]
    parts.append(B * T * n_b + np.arange(n_r))
    return np.concatenate(parts)


def reference_permuted_factor(matrix) -> np.ndarray:
    """CPU reference: the dense lower Cholesky factor ``L_hat`` of the
    permuted system, built structurally (per-tail permuted tail
    factors, root coupling columns, root diagonal update).  Validation
    only."""
    import scipy.linalg as sla

    B, T, n_b, n_r = matrix.shape.dims()
    perm = TailPermutation.create(T)
    rows = perm.row_order(n_b)
    D = np.asarray(matrix.D, dtype=np.float64)
    E = np.asarray(matrix.E, dtype=np.float64)
    C_T = np.asarray(matrix.C_T, dtype=np.float64)
    R = np.asarray(matrix.R, dtype=np.float64)

    n = B * T * n_b + n_r
    L = np.zeros((n, n))
    root_update = np.zeros((n_r, n_r))
    for b in range(B):
        K = np.zeros((T * n_b, T * n_b))
        for t in range(T):
            r = t * n_b
            K[r:r + n_b, r:r + n_b] = D[b, t]
            if t + 1 < T:
                K[r + n_b:r + 2 * n_b, r:r + n_b] = E[b, t]
                K[r:r + n_b, r + n_b:r + 2 * n_b] = E[b, t].T
        K_hat = K[np.ix_(rows, rows)]
        L_hat = np.linalg.cholesky(K_hat)
        C_hat_T = C_T[b].reshape(T * n_b, n_r)[rows]      # C_hat_i^T
        M_T = sla.solve_triangular(L_hat, C_hat_T, lower=True)
        base = b * T * n_b
        L[base:base + T * n_b, base:base + T * n_b] = L_hat
        L[-n_r:, base:base + T * n_b] = M_T.T             # M_i rows
        root_update += M_T.T @ M_T
    L[-n_r:, -n_r:] = np.linalg.cholesky(R - root_update)
    return L

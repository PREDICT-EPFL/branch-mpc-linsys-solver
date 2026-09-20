"""NumPy reference of the leaf-to-root endpoint factorization/solve.

Correctness anchor only, not a performance baseline: implements exactly
the block recursion the GPU solver runs (section 6 of the plan), in
FP64, so the GPU implementation can be validated block by block.

Factorization (per scenario, storage order ``k = 0..T-1``)::

    L_0 = chol(D_0)
    F_k = E_k L_k^{-T};  L_{k+1} = chol(D_{k+1} - F_k F_k^T)
    M_i = L_{T-1}^{-1} G_i
    S   = R - sum_i M_i^T M_i;   L_R = chol(S)

Solve (one vector)::

    forward tails -> v;  qbar = q - sum_i M_i^T v_i[T-1]
    y = (L_R L_R^T)^{-1} qbar
    v_i[T-1] -= M_i y;   backward tails -> w
"""

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla

from src.endpoint_tree.problem import EndpointTreeMatrix, EndpointTreeVector


@dataclass(frozen=True, slots=True)
class EndpointTreeFactor:
    """Host factor: tail Cholesky blocks, transformed boundary coupling,
    and the root factor."""

    matrix: EndpointTreeMatrix
    L: np.ndarray       # (B, T, n_b, n_b), lower tail factors
    F: np.ndarray       # (B, T-1, n_b, n_b), subdiagonal factor blocks
    M: np.ndarray       # (B, n_b, n_r), L_last^{-1} G
    L_R: np.ndarray     # (n_r, n_r), lower root factor


def factorize_reference(matrix: EndpointTreeMatrix) -> EndpointTreeFactor:
    """Leaf-to-root block Cholesky of the endpoint system (FP64)."""
    B, T, n_b, n_r = matrix.shape.dims()
    D = np.asarray(matrix.D, dtype=np.float64)
    E = np.asarray(matrix.E, dtype=np.float64)
    G = np.asarray(matrix.G_T, dtype=np.float64)
    R = np.asarray(matrix.R, dtype=np.float64)

    L = np.zeros((B, T, n_b, n_b))
    F = np.zeros((B, max(T - 1, 0), n_b, n_b))
    for b in range(B):
        L[b, 0] = np.linalg.cholesky(D[b, 0])
        for k in range(T - 1):
            # F_k = E_k L_k^{-T}  (triangular solve, never an inverse)
            F[b, k] = sla.solve_triangular(
                L[b, k], E[b, k].T, lower=True).T
            Dbar = D[b, k + 1] - F[b, k] @ F[b, k].T
            L[b, k + 1] = np.linalg.cholesky(Dbar)

    M = np.stack([sla.solve_triangular(L[b, T - 1], G[b], lower=True)
                  for b in range(B)])
    S = R - np.einsum("bki,bkj->ij", M, M)
    L_R = np.linalg.cholesky(S)
    return EndpointTreeFactor(matrix=matrix, L=L, F=F, M=M, L_R=L_R)


def solve_reference(factor: EndpointTreeFactor,
                    rhs: EndpointTreeVector) -> EndpointTreeVector:
    """One-vector solve with the reference factor (FP64)."""
    matrix = factor.matrix
    B, T, n_b, n_r = matrix.shape.dims()
    r = np.asarray(rhs.tail, dtype=np.float64).copy()
    q = np.asarray(rhs.root, dtype=np.float64).copy()
    L, F, M, L_R = factor.L, factor.F, factor.M, factor.L_R

    # forward substitution, leaf to root
    v = np.empty_like(r)
    for b in range(B):
        v[b, 0] = sla.solve_triangular(L[b, 0], r[b, 0], lower=True)
        for k in range(1, T):
            s = r[b, k] - F[b, k - 1] @ v[b, k - 1]
            v[b, k] = sla.solve_triangular(L[b, k], s, lower=True)

    # boundary-only root RHS update, root solve, boundary correction
    qbar = q - np.einsum("bki,bk->i", M, v[:, T - 1])
    y = sla.cho_solve((L_R, True), qbar)
    v[:, T - 1] -= M @ y

    # backward substitution, root to leaf
    w = np.empty_like(v)
    for b in range(B):
        w[b, T - 1] = sla.solve_triangular(L[b, T - 1], v[b, T - 1],
                                           lower=True, trans="T")
        for k in range(T - 2, -1, -1):
            s = v[b, k] - F[b, k].T @ w[b, k + 1]
            w[b, k] = sla.solve_triangular(L[b, k], s, lower=True,
                                           trans="T")

    dt = matrix.shape.np_dtype
    return EndpointTreeVector(matrix.shape, w.astype(dt),
                              y.astype(dt))

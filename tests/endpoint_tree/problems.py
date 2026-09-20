"""Reproducible SPD endpoint-coupled test/benchmark systems.

The generator constructs valid Cholesky factors first (random
well-conditioned tail factor blocks, random boundary coupling, and a
root built as ``R = S_target + sum_i M_i^T M_i`` with an SPD
``S_target``), so positive definiteness holds by construction and the
exact root Schur complement is known.  Everything is built in FP64;
block data are cast to the requested precision at the end while the
ground truth stays in FP64.
"""

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla

from src.endpoint_tree.problem import (
    EndpointTreeMatrix,
    EndpointTreeShape,
    EndpointTreeVector,
)


@dataclass(frozen=True, slots=True)
class GeneratedEndpointProblem:
    """One generated instance: the matrix, one RHS, the FP64 ground
    truth, and the exact root Schur complement."""

    shape: EndpointTreeShape
    matrix: EndpointTreeMatrix
    rhs: EndpointTreeVector
    exact_solution: EndpointTreeVector
    root_schur_target: np.ndarray


def _random_lower(rng, n, scale=0.3):
    L = np.eye(n) + scale * np.tril(rng.standard_normal((n, n)), k=-1)
    L[np.diag_indices(n)] += scale * rng.uniform(0.0, 1.0, n)
    return L


def generate_endpoint_problem(shape: EndpointTreeShape, seed=0,
                              coupling_scale=0.5
                              ) -> GeneratedEndpointProblem:
    """Generate one SPD endpoint-coupled system for ``shape``."""
    B, T, n_b, n_r = shape.dims()
    rng = np.random.default_rng(seed)

    L = np.stack([[_random_lower(rng, n_b) for _ in range(T)]
                  for _ in range(B)])
    # keep the subdiagonal factor spectral norm well below 1 for every
    # block size, so the chain conditioning stays bounded in T
    F = (0.4 / np.sqrt(n_b)) * rng.standard_normal(
        (B, max(T - 1, 0), n_b, n_b))
    D = np.einsum("btij,btkj->btik", L, L)
    if T > 1:
        D[:, 1:] += np.einsum("btij,btkj->btik", F, F)
        E = np.einsum("btij,btkj->btik", F, L[:, :-1])
    else:
        E = np.zeros((B, 0, n_b, n_b))

    G = coupling_scale * rng.standard_normal((B, n_b, n_r))
    G /= np.sqrt(B * n_b)
    M = np.stack([sla.solve_triangular(L[b, T - 1], G[b], lower=True)
                  for b in range(B)])
    A = rng.standard_normal((n_r, n_r)) / np.sqrt(n_r)
    S_target = A @ A.T + np.eye(n_r)
    R = S_target + np.einsum("bki,bkj->ij", M, M)
    R = 0.5 * (R + R.T)

    matrix = EndpointTreeMatrix(
        shape, D=D.astype(shape.np_dtype), E=E.astype(shape.np_dtype),
        G_T=G.astype(shape.np_dtype), R=R.astype(shape.np_dtype))
    w_true = rng.standard_normal((B, T, n_b))
    y_true = rng.standard_normal(n_r)
    exact = EndpointTreeVector(shape, w_true, y_true)
    fp64 = EndpointTreeMatrix(
        EndpointTreeShape(B, T, n_b, n_r, "float64"),
        D=D, E=E, G_T=G, R=R)
    rhs64 = fp64.matvec(EndpointTreeVector(fp64.shape, w_true, y_true))
    rhs = EndpointTreeVector(shape,
                             np.asarray(rhs64.tail,
                                        dtype=shape.np_dtype),
                             np.asarray(rhs64.root,
                                        dtype=shape.np_dtype))
    return GeneratedEndpointProblem(
        shape=shape, matrix=matrix, rhs=rhs, exact_solution=exact,
        root_schur_target=S_target)

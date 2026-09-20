"""CPU reference tests: leaf-to-root recursion versus dense/sparse
ground truth, with aggressively nonsymmetric random blocks."""

import numpy as np
import pytest
import scipy.sparse as sp

from src.endpoint_tree import EndpointTreeShape, EndpointTreeVector
from tests.endpoint_tree.problems import generate_endpoint_problem
from src.endpoint_tree.reference import factorize_reference, solve_reference

SHAPES = [EndpointTreeShape(1, 1, 3, 2), EndpointTreeShape(1, 6, 4, 3),
          EndpointTreeShape(5, 3, 4, 6), EndpointTreeShape(3, 8, 9, 9),
          EndpointTreeShape(4, 5, 6, 16)]


def _dense(matrix):
    A = matrix.to_csr_lower(dtype=np.float64)
    return (A + sp.tril(A, k=-1).T).toarray()


@pytest.mark.parametrize("shape", SHAPES)
def test_reference_solve_matches_dense(shape):
    p = generate_endpoint_problem(shape, seed=11)
    factor = factorize_reference(p.matrix)
    x = solve_reference(factor, p.rhs)
    K = _dense(p.matrix)
    x_dense = np.linalg.solve(K, p.rhs.flat())
    np.testing.assert_allclose(x.flat(), x_dense, rtol=1e-9, atol=1e-9)
    # and the generator's exact solution is recovered
    np.testing.assert_allclose(x.flat(), p.exact_solution.flat(),
                               rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("shape", SHAPES)
def test_root_schur_matches_generator_target(shape):
    p = generate_endpoint_problem(shape, seed=12)
    factor = factorize_reference(p.matrix)
    S = factor.L_R @ factor.L_R.T
    np.testing.assert_allclose(S, p.root_schur_target, rtol=1e-9,
                               atol=1e-9)


def test_reference_factor_blocks_reproduce_matrix():
    shape = EndpointTreeShape(2, 4, 3, 2)
    p = generate_endpoint_problem(shape, seed=13)
    f = factorize_reference(p.matrix)
    D = np.asarray(p.matrix.D, dtype=np.float64)
    E = np.asarray(p.matrix.E, dtype=np.float64)
    for b in range(2):
        np.testing.assert_allclose(f.L[b, 0] @ f.L[b, 0].T, D[b, 0],
                                   rtol=1e-10, atol=1e-12)
        for k in range(3):
            np.testing.assert_allclose(f.F[b, k] @ f.L[b, k].T, E[b, k],
                                       rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(
                f.L[b, k + 1] @ f.L[b, k + 1].T
                + f.F[b, k] @ f.F[b, k].T, D[b, k + 1],
                rtol=1e-10, atol=1e-12)


def test_reference_random_rhs_residual():
    shape = EndpointTreeShape(3, 7, 5, 4)
    p = generate_endpoint_problem(shape, seed=14)
    rng = np.random.default_rng(99)
    rhs = EndpointTreeVector(shape, rng.standard_normal((3, 7, 5)),
                             rng.standard_normal(4))
    x = solve_reference(factorize_reference(p.matrix), rhs)
    r = p.matrix.matvec(x).flat() - rhs.flat()
    assert np.linalg.norm(r) / np.linalg.norm(rhs.flat()) < 1e-12

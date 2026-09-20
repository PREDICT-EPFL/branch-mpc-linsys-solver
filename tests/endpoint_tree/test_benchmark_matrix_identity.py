"""The benchmark comparison must feed both solvers the identical
numerical matrix: the converted general matrix equals the endpoint
matrix entry for entry, and both solvers produce the same solution on
it."""

import numpy as np
import pytest
import scipy.sparse as sp

from src.endpoint_tree import EndpointTreeShape, EndpointTreeSolver
from tests.endpoint_tree.problems import generate_endpoint_problem

wp = pytest.importorskip("warp")
pytestmark = pytest.mark.gpu

SHAPE = EndpointTreeShape(4, 8, 6, 16)


def test_converted_matrix_is_numerically_identical():
    p = generate_endpoint_problem(SHAPE, seed=61)
    general = p.matrix.to_general_tree_matrix()
    A_end = p.matrix.to_csr_lower(dtype=np.float64)
    A_gen = general.to_csr_lower(dtype=np.float64)
    d = (A_end - A_gen)
    d.eliminate_zeros()
    assert d.nnz == 0


def test_both_solvers_agree_on_the_identical_matrix():
    p = generate_endpoint_problem(SHAPE, seed=62)
    s = EndpointTreeSolver(SHAPE)
    s.update(p.matrix)
    s.factorize()
    x = s.solve(p.rhs)
    from src.endpoint_tree._reuse import TreeVector
    from src.general_arrow.solver import Solver as GeneralSolver
    general = p.matrix.to_general_tree_matrix()
    gs = GeneralSolver(general.shape)
    gs.update(general)
    gs.factorize()
    gx = gs.solve(TreeVector(general.shape,
                             np.asarray(p.rhs.tail)[..., None].copy(),
                             np.asarray(p.rhs.root)[:, None].copy()))
    xf = np.concatenate([x.tail.numpy().reshape(-1), x.root.numpy()])
    gf = np.concatenate([gx.tail.numpy().reshape(-1),
                         gx.root.numpy().reshape(-1)])
    A = p.matrix.to_csr_lower(dtype=np.float64)
    K = A + sp.tril(A, k=-1).T
    b = p.rhs.flat()
    r_end = np.linalg.norm(K @ xf - b) / np.linalg.norm(b)
    r_gen = np.linalg.norm(K @ gf - b) / np.linalg.norm(b)
    assert r_end < 1e-12 and r_gen < 1e-12
    assert (np.linalg.norm(xf - gf) / np.linalg.norm(gf)) < 1e-11

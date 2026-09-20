"""Structure and API tests: shapes, ordering contract, round trips,
matvec, and conversions."""

import numpy as np
import pytest

from src.endpoint_tree import (
    EndpointTreeMatrix,
    EndpointTreeShape,
    EndpointTreeVector,
)
from tests.endpoint_tree.problems import generate_endpoint_problem

SHAPE = EndpointTreeShape(num_tails=3, num_stages=5, tail_block_dim=4,
                          root_dim=6)


@pytest.mark.parametrize("field,value", [
    ("num_tails", 0), ("num_stages", -1), ("tail_block_dim", 0),
    ("root_dim", 0)])
def test_shape_field_validation(field, value):
    kw = dict(num_tails=2, num_stages=2, tail_block_dim=2, root_dim=2)
    kw[field] = value
    with pytest.raises(ValueError):
        EndpointTreeShape(**kw)


def test_shape_precision_validation():
    with pytest.raises(ValueError):
        EndpointTreeShape(2, 2, 2, 2, precision="float16")


def test_storage_order_contract():
    assert SHAPE.storage_order == "leaf_to_root"
    assert SHAPE.tail_dimension == 3 * 5 * 4
    assert SHAPE.total_dimension == 3 * 5 * 4 + 6


def test_matrix_shape_validation():
    p = generate_endpoint_problem(SHAPE, seed=1)
    m = p.matrix
    with pytest.raises(ValueError):
        EndpointTreeMatrix(SHAPE, D=m.D[:, :-1], E=m.E, G_T=m.G_T, R=m.R)
    with pytest.raises(ValueError):
        EndpointTreeMatrix(SHAPE, D=m.D, E=m.E, G_T=m.G_T[:, :-1],
                           R=m.R)


def test_vector_flat_round_trip():
    p = generate_endpoint_problem(SHAPE, seed=2)
    v = p.rhs
    z = v.flat()
    assert z.shape == (SHAPE.total_dimension,)
    v2 = EndpointTreeVector.from_flat(SHAPE, z)
    np.testing.assert_array_equal(v.tail, v2.tail)
    np.testing.assert_array_equal(v.root, v2.root)
    with pytest.raises(ValueError):
        EndpointTreeVector.from_flat(SHAPE, z[:-1])


def test_matvec_matches_assembled_sparse():
    p = generate_endpoint_problem(SHAPE, seed=3)
    A = p.matrix.to_csr_lower(dtype=np.float64)
    import scipy.sparse as sp
    K = A + sp.tril(A, k=-1).T
    y_struct = p.matrix.matvec(p.exact_solution).flat()
    y_sparse = K @ p.exact_solution.flat()
    np.testing.assert_allclose(y_struct, y_sparse, rtol=1e-12, atol=1e-12)


def test_endpoint_general_endpoint_round_trip():
    p = generate_endpoint_problem(SHAPE, seed=4)
    general = p.matrix.to_general_tree_matrix()
    # the general coupling tensor holds G_T at the last stage only
    C_T = np.asarray(general.C_T)
    assert C_T.shape == (3, 5, 4, 6)
    assert np.all(C_T[:, :-1] == 0.0)
    back = EndpointTreeMatrix.from_general_tree_matrix(general)
    for name in ("D", "E", "G_T", "R"):
        np.testing.assert_array_equal(getattr(p.matrix, name),
                                      getattr(back, name))


def test_rejects_early_stage_coupling():
    p = generate_endpoint_problem(SHAPE, seed=5)
    general = p.matrix.to_general_tree_matrix()
    C_T = np.asarray(general.C_T).copy()
    C_T[0, 0, 0, 0] = 1e-3
    from src.endpoint_tree._reuse import TreeMatrix
    bad = TreeMatrix(general.shape, D=general.D, E=general.E, C_T=C_T,
                     R=general.R)
    with pytest.raises(ValueError):
        EndpointTreeMatrix.from_general_tree_matrix(bad)
    # atol permits genuinely negligible noise, never silent projection
    ok = EndpointTreeMatrix.from_general_tree_matrix(bad, atol=1e-2)
    assert np.all(ok.G_T == p.matrix.G_T)


@pytest.mark.parametrize("dims", [(1, 1, 2, 2), (1, 4, 3, 2),
                                  (4, 1, 2, 3), (2, 3, 9, 9)])
def test_edge_shapes_matvec_consistent(dims):
    shape = EndpointTreeShape(*dims)
    p = generate_endpoint_problem(shape, seed=6)
    A = p.matrix.to_csr_lower(dtype=np.float64)
    import scipy.sparse as sp
    K = A + sp.tril(A, k=-1).T
    np.testing.assert_allclose(
        p.matrix.matvec(p.exact_solution).flat(),
        K @ p.exact_solution.flat(), rtol=1e-12, atol=1e-12)

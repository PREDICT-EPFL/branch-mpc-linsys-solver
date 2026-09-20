"""API contract: lifecycle errors, type rejection, RHS preservation,
and the boundary-coupling storage guarantees."""

import numpy as np
import pytest

from src.endpoint_tree import (
    EndpointTreeMatrix,
    EndpointTreeShape,
    EndpointTreeSolver,
    EndpointTreeVector,
)
from tests.endpoint_tree.problems import generate_endpoint_problem

wp = pytest.importorskip("warp")
pytestmark = pytest.mark.gpu

SHAPE = EndpointTreeShape(3, 5, 4, 6)


def _ready_solver(seed=41):
    p = generate_endpoint_problem(SHAPE, seed=seed)
    s = EndpointTreeSolver(SHAPE)
    s.update(p.matrix)
    s.factorize()
    return s, p


def test_lifecycle_errors():
    s = EndpointTreeSolver(SHAPE)
    p = generate_endpoint_problem(SHAPE, seed=42)
    with pytest.raises(RuntimeError):
        s.factorize()
    s.update(p.matrix)
    with pytest.raises(RuntimeError):
        s.solve(p.rhs)
    s.factorize()
    s.solve(p.rhs)


def test_rejects_general_api_objects():
    s, p = _ready_solver()
    general = p.matrix.to_general_tree_matrix()
    with pytest.raises(TypeError):
        s.update(general)
    from src.endpoint_tree._reuse import TreeVector
    gv = TreeVector(general.shape,
                    np.asarray(p.rhs.tail)[..., None].copy(),
                    np.asarray(p.rhs.root)[:, None].copy())
    with pytest.raises(TypeError):
        s.solve(gv)


def test_shape_mismatch_rejected():
    s, _ = _ready_solver()
    other = EndpointTreeShape(3, 5, 4, 7)
    p2 = generate_endpoint_problem(other, seed=43)
    with pytest.raises(ValueError):
        s.update(p2.matrix)


def test_device_rhs_is_not_consumed():
    s, p = _ready_solver()
    dev = wp.get_device("cuda:0")
    rhs = EndpointTreeVector(
        SHAPE,
        wp.array(np.asarray(p.rhs.tail), dtype=wp.float64, device=dev),
        wp.array(np.asarray(p.rhs.root), dtype=wp.float64, device=dev))
    before_t = rhs.tail.numpy().copy()
    before_r = rhs.root.numpy().copy()
    out = EndpointTreeVector(
        SHAPE, wp.zeros((3, 5, 4), dtype=wp.float64, device=dev),
        wp.zeros((6,), dtype=wp.float64, device=dev))
    s.solve(rhs, out=out)
    np.testing.assert_array_equal(rhs.tail.numpy(), before_t)
    np.testing.assert_array_equal(rhs.root.numpy(), before_r)
    xf = np.concatenate([out.tail.numpy().reshape(-1),
                         out.root.numpy()])
    err = (np.linalg.norm(xf - p.exact_solution.flat())
           / np.linalg.norm(p.exact_solution.flat()))
    assert err < 1e-10


def test_boundary_coupling_storage_is_endpoint_only():
    s, _ = _ready_solver()
    B, T, n_b, n_r = SHAPE.dims()
    assert s.boundary_coupling_entries == B * n_b * n_r
    shapes = s.persistent_buffer_shapes
    full_tensor = B * T * n_b * n_r
    for name, shp in shapes.items():
        assert int(np.prod(shp)) < full_tensor, (
            f"buffer {name} with shape {shp} is as large as a "
            f"full-stage coupling tensor")
    assert shapes["G_values"] == (B, n_b, n_r)
    assert shapes["M_coupling"] == (B, n_b, n_r)


def test_out_none_returns_solver_owned_result():
    s, p = _ready_solver()
    x1 = s.solve(p.rhs)
    x2 = s.solve(p.rhs)
    assert x1.tail.ptr == x2.tail.ptr  # reused staging output

"""CUDA graph reuse and allocation-freedom of the warm paths."""

import numpy as np
import pytest

from src.endpoint_tree import (
    EndpointTreeShape,
    EndpointTreeSolver,
    EndpointTreeVector,
)
from tests.endpoint_tree.problems import generate_endpoint_problem

wp = pytest.importorskip("warp")
pytestmark = pytest.mark.gpu

SHAPE = EndpointTreeShape(4, 6, 5, 8)


def _device_vectors(p, dev):
    rhs = EndpointTreeVector(
        SHAPE,
        wp.array(np.asarray(p.rhs.tail), dtype=wp.float64, device=dev),
        wp.array(np.asarray(p.rhs.root), dtype=wp.float64, device=dev))
    out = EndpointTreeVector(
        SHAPE, wp.zeros((4, 6, 5), dtype=wp.float64, device=dev),
        wp.zeros((8,), dtype=wp.float64, device=dev))
    return rhs, out


def test_warm_factorize_and_solve_allocate_nothing():
    dev = wp.get_device("cuda:0")
    p = generate_endpoint_problem(SHAPE, seed=51)
    solver = EndpointTreeSolver(SHAPE)
    solver.update(p.matrix)
    solver.factorize()          # cold: capture
    rhs, out = _device_vectors(p, dev)
    solver.solve(rhs, out=out)  # cold: bind + capture
    solver.factorize()
    solver.solve(rhs, out=out)
    wp.synchronize_device(dev)
    before = wp.get_mempool_used_mem_current(dev)
    for _ in range(5):
        solver.factorize()
        solver.solve(rhs, out=out)
    wp.synchronize_device(dev)
    assert wp.get_mempool_used_mem_current(dev) == before


def test_rebinding_on_new_pointers():
    dev = wp.get_device("cuda:0")
    p = generate_endpoint_problem(SHAPE, seed=52)
    solver = EndpointTreeSolver(SHAPE)
    solver.update(p.matrix)
    solver.factorize()
    rhs1, out1 = _device_vectors(p, dev)
    rhs2, out2 = _device_vectors(p, dev)
    x1 = solver.solve(rhs1, out=out1)
    x2 = solver.solve(rhs2, out=out2)
    # SOCU's substitutions accumulate with atomics, so two
    # separate executions may differ in the last ulps
    np.testing.assert_allclose(x1.tail.numpy(), x2.tail.numpy(),
                               rtol=1e-12, atol=1e-13)
    assert x1.tail.ptr != x2.tail.ptr


def test_update_keeps_graphs_valid():
    dev = wp.get_device("cuda:0")
    solver = EndpointTreeSolver(SHAPE)
    p1 = generate_endpoint_problem(SHAPE, seed=53)
    solver.update(p1.matrix)
    solver.factorize()
    rhs, out = _device_vectors(p1, dev)
    solver.solve(rhs, out=out)
    # new values, same buffers: graph replay must give the new solution
    p2 = generate_endpoint_problem(SHAPE, seed=54)
    solver.update(p2.matrix)
    solver.factorize()
    wp.copy(rhs.tail, wp.array(np.asarray(p2.rhs.tail),
                               dtype=wp.float64, device=dev))
    wp.copy(rhs.root, wp.array(np.asarray(p2.rhs.root),
                               dtype=wp.float64, device=dev))
    solver.solve(rhs, out=out)
    xf = np.concatenate([out.tail.numpy().reshape(-1),
                         out.root.numpy()])
    err = (np.linalg.norm(xf - p2.exact_solution.flat())
           / np.linalg.norm(p2.exact_solution.flat()))
    assert err < 1e-10

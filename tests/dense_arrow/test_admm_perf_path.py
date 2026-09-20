"""Warm-path guarantees of the fused ADMM implementation: iteration
graphs match the eager path, warm iterations allocate nothing, device
inputs are accepted without staging, and over-relaxation follows the
NumPy reference."""

import numpy as np
import pytest

from admm import Settings, Solver
from baselines.admm_numpy import solve_qp_reference
from experiments.general_arrow.benchmarks.problems import generate_scenario_qp

wp = pytest.importorskip("warp")
pytestmark = pytest.mark.gpu

BASE = dict(rho=1.0, max_iter=120, eps_abs=1e-7, eps_rel=1e-7,
            check_every=7)  # 120 % 7 != 0 exercises the remainder graph


def _qp(seed=3):
    return generate_scenario_qp(num_tails=3, num_stages=6, nx=4, nu=2,
                                seed=seed)


def _solve(settings_extra, qp, warm=None, updates=None):
    P, q, A, l, u, meta = qp
    solver = Solver()
    solver.setup(P, q, A, l, u,
                 settings=Settings(rho_eq_scale=1.0, linear_solver="tree",
                                   **BASE, **settings_extra))
    try:
        if warm is not None:
            solver.warm_start(**warm)
        if updates is not None:
            solver.update(**updates)
        res = solver.solve()
        return (res.x.numpy(), res.z.numpy(), res.dual.numpy(),
                res.info.iterations, res.info.converged)
    finally:
        solver.close()


def test_iteration_graph_matches_eager_path():
    qp = _qp()
    graph = _solve({"iteration_graph": True}, qp)
    eager = _solve({"iteration_graph": False}, qp)
    assert graph[3] == eager[3] and graph[4] == eager[4]
    for g, e in zip(graph[:3], eager[:3]):
        np.testing.assert_array_equal(g, e)


def test_graph_survives_matrix_update():
    P, q, A, l, u, meta = _qp()
    solver = Solver()
    solver.setup(P, q, A, l, u,
                 settings=Settings(rho_eq_scale=1.0, linear_solver="tree",
                                   **BASE))
    try:
        solver.solve()  # builds the iteration graphs
        rng = np.random.default_rng(0)
        new_Px = np.asarray(P.data) * (1.0 + 0.1 * rng.uniform(size=P.nnz))
        info = solver.update(Px=new_Px)
        assert info.factorized
        solver.warm_start(x=np.zeros(P.shape[0]), z=np.zeros(A.shape[0]),
                          dual=np.zeros(A.shape[0]))
        res = solver.solve()
        import scipy.sparse as sp
        P2 = sp.csc_matrix((new_Px, P.indices, P.indptr), shape=P.shape)
        ref = solve_qp_reference(P2, q, A, l, u, **BASE)
        assert res.info.iterations == ref.iterations
        np.testing.assert_allclose(res.x.numpy(), ref.x, atol=1e-9)
    finally:
        solver.close()


def test_warm_solve_allocates_nothing():
    P, q, A, l, u, meta = _qp()
    solver = Solver()
    solver.setup(P, q, A, l, u,
                 settings=Settings(rho_eq_scale=1.0, linear_solver="tree",
                                   **BASE))
    try:
        dev = wp.get_device("cuda:0")
        zeros = dict(x=np.zeros(P.shape[0]), z=np.zeros(A.shape[0]),
                     dual=np.zeros(A.shape[0]))
        solver.solve()  # cold: builds graphs, compiles, binds
        solver.warm_start(**zeros)
        solver.solve(copy_result=False)  # second call warms any stragglers
        solver.warm_start(**zeros)
        wp.synchronize_device(dev)
        before = wp.get_mempool_used_mem_current(dev)
        solver.solve(copy_result=False)
        wp.synchronize_device(dev)
        assert wp.get_mempool_used_mem_current(dev) == before

        # device-to-device vector and matrix-value updates: no growth
        q_dev = wp.array(q * 1.01, dtype=wp.float64, device=dev)
        px_dev = wp.array(np.asarray(P.data) * 1.02, dtype=wp.float64,
                          device=dev)
        wp.synchronize_device(dev)
        before = wp.get_mempool_used_mem_current(dev)
        solver.update(q=q_dev)
        solver.update(Px=px_dev)
        solver.solve(copy_result=False)  # graph replay after the update
        wp.synchronize_device(dev)
        assert wp.get_mempool_used_mem_current(dev) == before
    finally:
        solver.close()


def test_warm_start_accepts_device_arrays():
    P, q, A, l, u, meta = _qp()
    n, m = P.shape[0], A.shape[0]
    rng = np.random.default_rng(7)
    x0 = rng.standard_normal(n)
    z0 = rng.standard_normal(m)
    v0 = rng.standard_normal(m)
    dev = wp.get_device("cuda:0")
    host = _solve({}, (P, q, A, l, u, meta),
                  warm=dict(x=x0, z=z0, dual=v0))
    device = _solve({}, (P, q, A, l, u, meta),
                    warm=dict(x=wp.array(x0, dtype=wp.float64, device=dev),
                              z=wp.array(z0, dtype=wp.float64, device=dev),
                              dual=wp.array(v0, dtype=wp.float64,
                                            device=dev)))
    assert host[3] == device[3]
    for h, d in zip(host[:3], device[:3]):
        np.testing.assert_array_equal(h, d)


def test_warm_start_rejects_bad_device_arrays():
    P, q, A, l, u, meta = _qp()
    solver = Solver()
    solver.setup(P, q, A, l, u,
                 settings=Settings(rho_eq_scale=1.0, linear_solver="tree",
                                   **BASE))
    try:
        dev = wp.get_device("cuda:0")
        with pytest.raises(ValueError):
            solver.warm_start(x=wp.zeros(3, dtype=wp.float64, device=dev))
        with pytest.raises(TypeError):
            solver.warm_start(x=wp.zeros(P.shape[0], dtype=wp.float32,
                                         device=dev))
        with pytest.raises(TypeError):
            solver.update(l=wp.zeros(A.shape[0], dtype=wp.float64,
                                     device=dev))
    finally:
        solver.close()


def test_device_updates_match_host_updates():
    P, q, A, l, u, meta = _qp()
    rng = np.random.default_rng(1)
    new_Px = np.asarray(P.data) * (1.0 + 0.1 * rng.uniform(size=P.nnz))
    new_Ax = np.asarray(A.data) * (1.0 + 0.05 * rng.uniform(size=A.nnz))
    new_q = q + 0.1 * rng.standard_normal(len(q))
    dev = wp.get_device("cuda:0")
    host = _solve({}, (P, q, A, l, u, meta),
                  updates=dict(Px=new_Px, Ax=new_Ax, q=new_q))
    device = _solve(
        {}, (P, q, A, l, u, meta),
        updates=dict(Px=wp.array(new_Px, dtype=wp.float64, device=dev),
                     Ax=wp.array(new_Ax, dtype=wp.float64, device=dev),
                     q=wp.array(new_q, dtype=wp.float64, device=dev)))
    assert host[3] == device[3]
    for h, d in zip(host[:3], device[:3]):
        np.testing.assert_array_equal(h, d)


def test_over_relaxation_matches_reference():
    P, q, A, l, u, meta = _qp(seed=5)
    for alpha in (1.4, 1.7):
        gpu = _solve({"alpha": alpha}, (P, q, A, l, u, meta))
        ref = solve_qp_reference(P, q, A, l, u, alpha=alpha, **BASE)
        assert gpu[3] == ref.iterations and gpu[4] == ref.converged
        np.testing.assert_allclose(gpu[0], ref.x, atol=1e-9)
        np.testing.assert_allclose(gpu[1], ref.z, atol=1e-9)
        np.testing.assert_allclose(gpu[2], ref.v, atol=1e-9)


def test_non_copying_result_returns_solver_views():
    P, q, A, l, u, meta = _qp()
    solver = Solver()
    solver.setup(P, q, A, l, u,
                 settings=Settings(rho_eq_scale=1.0, linear_solver="tree",
                                   **BASE))
    try:
        zeros = dict(x=np.zeros(P.shape[0]), z=np.zeros(A.shape[0]),
                     dual=np.zeros(A.shape[0]))
        solver.warm_start(**zeros)
        owned = solver.solve()
        solver.warm_start(**zeros)
        view = solver.solve(copy_result=False)
        # identical run, but the view aliases solver state while the
        # default result is an owned snapshot
        np.testing.assert_array_equal(owned.x.numpy(), view.x.numpy())
        assert view.x.ptr != owned.x.ptr
        solver.warm_start(**zeros)
        again = solver.solve(copy_result=False)
        assert again.x.ptr == view.x.ptr
    finally:
        solver.close()


def test_instrumentation_preserves_state_and_counters():
    P, q, A, l, u, meta = _qp()
    solver = Solver()
    solver.setup(P, q, A, l, u,
                 settings=Settings(rho_eq_scale=1.0, linear_solver="tree",
                                   **BASE))
    try:
        zeros = dict(x=np.zeros(P.shape[0]), z=np.zeros(A.shape[0]),
                     dual=np.zeros(A.shape[0]))
        solver.warm_start(**zeros)
        res1 = solver.solve()
        count = solver.factorization_count
        # instrumentation between two identical runs must not change
        # the start state, the iterates, or the factorization count
        solver.warm_start(**zeros)
        solver.time_linear_system(repeats=5)
        solver.time_iteration_phases(iters=4, repeats=2, phase_repeats=3)
        assert solver.factorization_count == count
        res2 = solver.solve()
        assert res2.info.iterations == res1.info.iterations
        np.testing.assert_array_equal(res1.x.numpy(), res2.x.numpy())
    finally:
        solver.close()

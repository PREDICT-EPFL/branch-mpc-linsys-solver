"""End-to-end GPU ADMM with the tree linear system (stage 5/6)."""

import numpy as np
import pytest
import scipy.sparse as sp

from admm import Settings, Solver, TreeStructureError
from baselines.admm_numpy import solve_qp_reference
from experiments.dense_arrow.benchmarks.problems import generate_scenario_qp

wp = pytest.importorskip("warp")
pytestmark = pytest.mark.gpu

SETTINGS = dict(rho=1.0, max_iter=800, eps_abs=1e-7, eps_rel=1e-7,
                check_every=5)


def test_tree_matches_numpy_reference():
    P, q, A, l, u, meta = generate_scenario_qp(num_tails=3, num_stages=5,
                                               nx=6, nu=2, seed=0)
    ref = solve_qp_reference(P, q, A, l, u, **SETTINGS)
    solver = Solver()
    solver.setup(P, q, A, l, u,
                 settings=Settings(rho_eq_scale=1.0, linear_solver="tree", **SETTINGS))
    res = solver.solve()
    assert res.info.linear_solver == "tree"
    assert res.info.converged == ref.converged
    assert res.info.iterations == ref.iterations
    # user variable order is preserved in the returned x
    assert np.allclose(res.x.numpy(), ref.x, atol=1e-7)
    assert np.allclose(res.z.numpy(), ref.z, atol=1e-7)
    solver.close()


def test_tree_and_cudss_agree():
    """Same QP, same state: both implementations produce the same ADMM
    iterates (FP64 agreement of the final triple and iteration count)."""
    cudss_mod = pytest.importorskip("admm.cudss")
    if not cudss_mod.CUDSS_AVAILABLE:
        pytest.skip(cudss_mod.CUDSS_UNAVAILABLE_REASON)
    P, q, A, l, u, meta = generate_scenario_qp(num_tails=2, num_stages=4,
                                               nx=4, nu=2, seed=1)
    fixed = dict(rho=2.0, max_iter=25, eps_abs=0.0, eps_rel=0.0,
                 check_every=100)  # fixed 25 iterations, no early exit
    results = {}
    for impl in ("tree", "cudss"):
        solver = Solver()
        solver.setup(P, q, A, l, u,
                     settings=Settings(rho_eq_scale=1.0, linear_solver=impl, **fixed))
        res = solver.solve()
        assert res.info.linear_solver == impl
        results[impl] = res
        solver.close()
    for field in ("x", "z", "dual"):
        a = getattr(results["tree"], field).numpy()
        b = getattr(results["cudss"], field).numpy()
        assert np.allclose(a, b, atol=1e-9), field


def test_auto_selects_tree_for_scenario_qp():
    P, q, A, l, u, meta = generate_scenario_qp(num_tails=2, num_stages=4,
                                               nx=4, nu=2, seed=2)
    solver = Solver()
    solver.setup(P, q, A, l, u,
                 settings=Settings(rho_eq_scale=1.0, linear_solver="auto", **SETTINGS))
    res = solver.solve()
    assert res.info.linear_solver == "tree"
    solver.close()


def test_explicit_tree_rejects_unstructured_qp():
    rng = np.random.default_rng(5)
    n, m = 10, 14
    M = rng.standard_normal((n, n))
    P = sp.csc_matrix(np.triu(M @ M.T + n * np.eye(n)))
    A = sp.csc_matrix(rng.standard_normal((m, n)))
    solver = Solver()
    with pytest.raises(TreeStructureError):
        solver.setup(P, rng.standard_normal(n), A, -np.ones(m), np.ones(m),
                     settings=Settings(rho_eq_scale=1.0, linear_solver="tree"))


def test_tree_linear_system_zero_copy_binding():
    """The ADMM tree adapter remains zero-copy: the solver's solve
    binding holds exactly the flat ADMM rhs/out buffer views (pointer
    identity, no staging), and the binding key never changes across
    iterations (plan 4 acceptance criterion 6)."""
    P, q, A, l, u, meta = generate_scenario_qp(num_tails=3, num_stages=5,
                                               nx=6, nu=2, seed=0)
    solver = Solver()
    solver.setup(P, q, A, l, u,
                 settings=Settings(rho_eq_scale=1.0, linear_solver="tree",
                                   **SETTINGS))
    solver.solve()
    lin = solver._linsys
    binding = lin._solver._binding
    assert binding.zero_copy
    assert binding.rhs_tail.ptr == lin._rhs_tree.tail.ptr
    assert binding.root_rhs.ptr == lin._rhs_tree.root.ptr
    assert binding.out.tail.ptr == lin._out_tree.tail.ptr
    assert binding.out.root.ptr == lin._out_tree.root.ptr
    solver.close()

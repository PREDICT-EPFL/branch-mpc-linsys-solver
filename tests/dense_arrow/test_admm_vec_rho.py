"""Fixed per-constraint penalty vector (plan section 19): equality rows
carry rho * rho_eq_scale, all other rows rho.  Verifies assembly,
iterate-level agreement with the NumPy reference, row-scaling
equivalence, lifecycle rules for bound updates, and the convergence
regression that motivated the feature."""

import numpy as np
import pytest
import scipy.sparse as sp

from admm import Settings, Solver
from baselines.admm_numpy import solve_qp_reference
from experiments.general_arrow.benchmarks.problems import generate_scenario_qp

wp = pytest.importorskip("warp")
pytestmark = pytest.mark.gpu


def _qp(seed=0, **kw):
    kw.setdefault("num_tails", 3)
    kw.setdefault("num_stages", 5)
    kw.setdefault("nx", 6)
    kw.setdefault("nu", 2)
    P, q, A, l, u, meta = generate_scenario_qp(seed=seed, **kw)
    return P, q, A, l, u


def _rho_vec(l, u, rho, scale):
    eq = np.isfinite(l) & np.isfinite(u) & (l == u)
    return np.where(eq, rho * scale, rho)


def _setup(P, q, A, l, u, **settings_kw):
    solver = Solver()
    solver.setup(P, q, A, l, u, settings=Settings(**settings_kw))
    return solver


def test_scale_one_reproduces_scalar_matrix_and_iterates():
    P, q, A, l, u = _qp(seed=0)
    common = dict(rho=1.3, max_iter=30, eps_abs=0.0, eps_rel=0.0,
                  check_every=100)
    solver = _setup(P, q, A, l, u, rho_eq_scale=1.0, linear_solver="tree",
                    **common)
    res = solver.solve()
    ref = solve_qp_reference(P, q, A, l, u, **common)  # scalar rho path
    assert np.allclose(res.x.numpy(), ref.x, atol=1e-9)
    assert np.allclose(res.z.numpy(), ref.z, atol=1e-9)
    assert np.allclose(res.dual.numpy(), ref.v, atol=1e-9)
    assert res.info.rho_min == res.info.rho_max == 1.3
    # assembled K equals the scalar-rho normal matrix
    P_full = sp.triu(P) + sp.triu(P, k=1).T
    K_ref = sp.tril(P_full + 1.3 * (A.T @ A)).tocsr()
    K_ref.sort_indices()
    assert np.allclose(solver._K_values.numpy(), K_ref.data, atol=1e-12)
    solver.close()


def test_gpu_assembly_matches_dense_weighted_normal_matrix():
    P, q, A, l, u = _qp(seed=1)
    rho, scale = 0.7, 250.0
    solver = _setup(P, q, A, l, u, rho=rho, rho_eq_scale=scale,
                    linear_solver="tree", max_iter=1)
    rho_vec = _rho_vec(l, u, rho, scale)
    P_full = (sp.triu(P) + sp.triu(P, k=1).T).toarray()
    K_dense = P_full + (A.T @ sp.diags(rho_vec) @ A).toarray()
    plan = solver._plan
    K = sp.csr_matrix((solver._K_values.numpy(), plan.K_indices,
                       plan.K_indptr), shape=(plan.n, plan.n)).toarray()
    ref = np.tril(K_dense)
    assert np.allclose(K, ref, atol=1e-11 * max(1.0, np.abs(ref).max()))
    eq = np.isfinite(l) & np.isfinite(u) & (l == u)
    res = solver.solve()
    assert res.info.num_equality_constraints == int(eq.sum())
    assert res.info.rho_max == pytest.approx(rho * scale)
    assert res.info.rho_min == pytest.approx(rho)
    solver.close()


def test_vector_rho_iterates_match_numpy_reference():
    P, q, A, l, u = _qp(seed=2)
    rho, scale = 1.0, 1000.0
    common = dict(max_iter=40, eps_abs=0.0, eps_rel=0.0, check_every=100)
    solver = _setup(P, q, A, l, u, rho=rho, rho_eq_scale=scale,
                    linear_solver="tree", **common)
    res = solver.solve()
    rho_vec = _rho_vec(l, u, rho, scale)
    ref = solve_qp_reference(P, q, A, l, u, rho=rho_vec, **common)
    assert np.allclose(res.x.numpy(), ref.x, atol=1e-8)
    assert np.allclose(res.z.numpy(), ref.z, atol=1e-8)
    assert np.allclose(res.dual.numpy(), ref.v, atol=1e-8)
    solver.close()


def test_weighted_multiplier_stationarity():
    """lambda = rho_vec * dual satisfies the KKT stationarity residual
    Px + q + A' lambda ~ 0 at a converged solution."""
    P, q, A, l, u = _qp(seed=3)
    solver = _setup(P, q, A, l, u, rho=1.0, rho_eq_scale=1000.0,
                    linear_solver="tree", max_iter=2000, eps_abs=1e-9,
                    eps_rel=1e-9, check_every=10)
    res = solver.solve()
    assert res.info.converged
    x = res.x.numpy()
    lam = _rho_vec(l, u, 1.0, 1000.0) * res.dual.numpy()
    P_full = sp.triu(P) + sp.triu(P, k=1).T
    stat = P_full @ x + q + A.T @ lam
    assert np.max(np.abs(stat)) < 1e-6
    solver.close()


def test_tree_and_cudss_agree_with_vector_rho():
    cudss_mod = pytest.importorskip("admm.cudss")
    if not cudss_mod.CUDSS_AVAILABLE:
        pytest.skip(cudss_mod.CUDSS_UNAVAILABLE_REASON)
    P, q, A, l, u = _qp(seed=4, num_tails=2, num_stages=4, nx=4)
    fixed = dict(rho=1.0, rho_eq_scale=1000.0, max_iter=25, eps_abs=0.0,
                 eps_rel=0.0, check_every=100)
    results = {}
    for impl in ("tree", "cudss"):
        solver = _setup(P, q, A, l, u, linear_solver=impl, **fixed)
        results[impl] = solver.solve()
        solver.close()
    for field in ("x", "z", "dual"):
        a = getattr(results["tree"], field).numpy()
        b = getattr(results["cudss"], field).numpy()
        assert np.allclose(a, b, atol=1e-9), field


def test_equality_weighting_equals_row_scaling():
    """Weighting equality rows by rho_eq_scale = s is equivalent to
    scaling those rows of A/l/u by sqrt(s) under scalar rho: same x
    iterates."""
    P, q, A, l, u = _qp(seed=5)
    s = 100.0
    fixed = dict(max_iter=30, eps_abs=0.0, eps_rel=0.0, check_every=100)
    solver = _setup(P, q, A, l, u, rho=1.0, rho_eq_scale=s,
                    linear_solver="tree", **fixed)
    weighted = solver.solve()
    solver.close()
    eq = np.isfinite(l) & np.isfinite(u) & (l == u)
    d = np.where(eq, np.sqrt(s), 1.0)
    A_s = sp.diags(d) @ A
    ref = solve_qp_reference(P, q, sp.csc_matrix(A_s), d * l, d * u,
                             rho=1.0, **fixed)
    assert np.allclose(weighted.x.numpy(), ref.x, atol=1e-8)


def test_bounds_update_changing_equality_class_rejected():
    P, q, A, l, u = _qp(seed=6)
    solver = _setup(P, q, A, l, u, rho_eq_scale=1000.0,
                    linear_solver="tree", max_iter=50)
    eq = np.isfinite(l) & np.isfinite(u) & (l == u)
    assert eq.any() and (~eq).any()
    l_before = solver._l.numpy().copy()
    u_before = solver._u.numpy().copy()
    # opening an equality row into an interval changes its class
    i = int(np.argmax(eq))
    bad_u = u.copy()
    bad_u[i] = u[i] + 1.0
    with pytest.raises(ValueError, match="equality classification"):
        solver.update(u=bad_u)
    # rejected atomically: no device state was mutated
    assert np.array_equal(solver._l.numpy(), l_before)
    assert np.array_equal(solver._u.numpy(), u_before)
    solver.close()


def test_class_preserving_bounds_update_does_not_refactor():
    P, q, A, l, u = _qp(seed=7)
    solver = _setup(P, q, A, l, u, rho_eq_scale=1000.0,
                    linear_solver="tree", max_iter=800, eps_abs=1e-7,
                    eps_rel=1e-7, check_every=5)
    before = solver.factorization_count
    # shift every row's bounds by the same amount: equality rows stay
    # equalities, boxes stay boxes
    info = solver.update(l=l + 0.05, u=u + 0.05)
    assert not info.matrix_updated and not info.factorized
    assert solver.factorization_count == before
    res = solver.solve()
    rho_vec = _rho_vec(l, u, 1.0, 1000.0)
    ref = solve_qp_reference(P, q, A, l + 0.05, u + 0.05, rho=rho_vec,
                             max_iter=800, eps_abs=1e-7, eps_rel=1e-7,
                             check_every=5)
    assert np.allclose(res.x.numpy(), ref.x, atol=1e-7)
    solver.close()


def test_matrix_update_with_vector_rho_factorizes_once():
    P, q, A, l, u = _qp(seed=8)
    solver = _setup(P, q, A, l, u, rho_eq_scale=1000.0,
                    linear_solver="tree", max_iter=800, eps_abs=1e-7,
                    eps_rel=1e-7, check_every=5)
    before = solver.factorization_count
    rng = np.random.default_rng(0)
    new_Px = np.asarray(P.data) * (1.0 + 0.1 * rng.uniform(size=P.nnz))
    new_Ax = np.asarray(A.data) * (1.0 + 0.05 * rng.uniform(size=A.nnz))
    info = solver.update(Px=new_Px, Ax=new_Ax)
    assert info.matrix_updated and info.factorized
    assert solver.factorization_count == before + 1
    res = solver.solve()
    P2 = sp.csc_matrix((new_Px, P.indices, P.indptr), shape=P.shape)
    A2 = sp.csc_matrix((new_Ax, A.indices, A.indptr), shape=A.shape)
    rho_vec = _rho_vec(l, u, 1.0, 1000.0)
    ref = solve_qp_reference(P2, q, A2, l, u, rho=rho_vec, max_iter=800,
                             eps_abs=1e-7, eps_rel=1e-7, check_every=5)
    assert np.allclose(res.x.numpy(), ref.x, atol=1e-7)
    solver.close()


def test_invalid_penalty_settings_rejected():
    P, q, A, l, u = _qp(seed=9, num_tails=2, num_stages=4, nx=4)
    for bad in (dict(rho=-1.0), dict(rho=np.inf),
                dict(rho_eq_scale=0.0), dict(rho_eq_scale=np.inf),
                dict(rho=1e300, rho_eq_scale=1e300)):
        with pytest.raises(ValueError):
            Solver().setup(P, q, A, l, u, settings=Settings(**bad))


@pytest.mark.slow
def test_convergence_regression_large_scenario():
    """B=41, T=64: the equality-weighted penalty converges to the 1e-4
    stopping criteria within the iteration budget while scalar rho does
    not (the gap that motivated section 19).  No wall-clock dependence."""
    P, q, A, l, u, meta = generate_scenario_qp(num_tails=41,
                                               num_stages=64, nx=8, nu=2,
                                               seed=10)
    common = dict(rho=1.0, max_iter=2000, eps_abs=1e-4, eps_rel=1e-4,
                  check_every=25, linear_solver="tree")
    solver = _setup(P, q, A, l, u, rho_eq_scale=1000.0, **common)
    weighted = solver.solve()
    solver.close()
    solver = _setup(P, q, A, l, u, rho_eq_scale=1.0, **common)
    scalar = solver.solve()
    solver.close()
    assert weighted.info.converged
    assert weighted.info.iterations < scalar.info.iterations
    assert not scalar.info.converged

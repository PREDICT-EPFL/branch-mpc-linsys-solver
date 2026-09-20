"""End-to-end GPU ADMM with the cuDSS linear system (stage 4)."""

import numpy as np
import pytest
import scipy.sparse as sp

from baselines.admm_numpy import solve_qp_reference

wp = pytest.importorskip("warp")
pytestmark = pytest.mark.gpu

cudss_mod = pytest.importorskip("admm.cudss")
if not cudss_mod.CUDSS_AVAILABLE:
    pytest.skip(f"cuDSS unavailable: {cudss_mod.CUDSS_UNAVAILABLE_REASON}",
                allow_module_level=True)

from admm import Settings, Solver  # noqa: E402


def _random_qp(n=12, m=18, seed=0):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n)) / np.sqrt(n)
    P = sp.csc_matrix(np.triu(M @ M.T + 0.5 * np.eye(n)))
    A_d = rng.standard_normal((m, n)) / np.sqrt(n)
    A_d[np.abs(A_d) < 0.3] = 0.0
    A = sp.csc_matrix(A_d)
    q = rng.standard_normal(n)
    l = -rng.uniform(0.5, 1.5, m)
    u = rng.uniform(0.5, 1.5, m)
    l[:2] = u[:2] = 0.1              # equalities
    u[2] = np.inf                    # lower-only
    l[3] = -np.inf                   # upper-only
    l[4], u[4] = -np.inf, np.inf     # unbounded
    return P, q, A, l, u


SETTINGS = dict(rho=1.0, max_iter=600, eps_abs=1e-7, eps_rel=1e-7,
                check_every=5)


def test_cudss_matches_numpy_reference():
    P, q, A, l, u = _random_qp()
    ref = solve_qp_reference(P, q, A, l, u, **SETTINGS)
    solver = Solver()
    solver.setup(P, q, A, l, u,
                 settings=Settings(rho_eq_scale=1.0, linear_solver="cudss", **SETTINGS))
    res = solver.solve()
    assert res.info.linear_solver == "cudss"
    assert res.info.converged == ref.converged
    assert res.info.iterations == ref.iterations
    assert np.allclose(res.x.numpy(), ref.x, atol=1e-8)
    assert np.allclose(res.z.numpy(), ref.z, atol=1e-8)
    assert np.allclose(res.dual.numpy(), ref.v, atol=1e-8)
    solver.close()


def test_cudss_warm_start_matches_reference():
    P, q, A, l, u = _random_qp(seed=2)
    rng = np.random.default_rng(9)
    x0, z0, v0 = (rng.standard_normal(P.shape[0]),
                  rng.standard_normal(A.shape[0]),
                  rng.standard_normal(A.shape[0]))
    ref = solve_qp_reference(P, q, A, l, u, x0=x0, z0=z0, v0=v0, **SETTINGS)
    solver = Solver()
    solver.setup(P, q, A, l, u,
                 settings=Settings(rho_eq_scale=1.0, linear_solver="cudss", **SETTINGS))
    solver.warm_start(x=x0, z=z0, dual=v0)
    res = solver.solve()
    assert res.info.iterations == ref.iterations
    assert np.allclose(res.x.numpy(), ref.x, atol=1e-8)
    solver.close()


def test_auto_selects_cudss_for_unstructured_qp():
    P, q, A, l, u = _random_qp(seed=3)
    solver = Solver()
    solver.setup(P, q, A, l, u,
                 settings=Settings(rho_eq_scale=1.0, linear_solver="auto", **SETTINGS))
    res = solver.solve()
    assert res.info.linear_solver == "cudss"
    solver.close()


def test_lifecycle_errors():
    solver = Solver()
    with pytest.raises(RuntimeError, match="setup"):
        solver.solve()
    P, q, A, l, u = _random_qp(seed=4)
    solver.setup(P, q, A, l, u, settings=Settings(rho_eq_scale=1.0, linear_solver="cudss"))
    with pytest.raises(RuntimeError, match="once"):
        solver.setup(P, q, A, l, u)
    solver.close()
    with pytest.raises(RuntimeError, match="closed"):
        solver.solve()

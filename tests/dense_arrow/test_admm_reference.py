"""NumPy ADMM reference: convergence and KKT checks (CPU only)."""

import numpy as np
import pytest
import scipy.sparse as sp

from baselines.admm_numpy import solve_qp_reference


def _random_qp(n=12, m=18, seed=0, equalities=True):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n)) / np.sqrt(n)
    P_full = M @ M.T + 0.5 * np.eye(n)
    P = sp.csc_matrix(np.triu(P_full))
    A = sp.csc_matrix(rng.standard_normal((m, n)) / np.sqrt(n))
    q = rng.standard_normal(n)
    l = -rng.uniform(0.5, 1.5, m)
    u = rng.uniform(0.5, 1.5, m)
    if equalities:
        l[:3] = u[:3] = rng.standard_normal(3) * 0.1  # equality rows
        u[3] = np.inf                                  # lower-only
        l[4] = -np.inf                                 # upper-only
        l[5], u[5] = -np.inf, np.inf                   # unbounded
    return P, q, A, l, u


def test_reference_converges_and_satisfies_kkt():
    P, q, A, l, u = _random_qp()
    res = solve_qp_reference(P, q, A, l, u, rho=1.0, max_iter=3000,
                             eps_abs=1e-8, eps_rel=1e-8, check_every=5)
    assert res.converged
    ax = A @ res.x
    assert np.all(ax >= l - 1e-6) and np.all(ax <= u + 1e-6)
    # stationarity: P x + q + A' lambda = 0 with lambda = rho * v
    P_full = sp.triu(P) + sp.triu(P, k=1).T
    lam = 1.0 * res.v
    grad = P_full @ res.x + q + A.T @ lam
    assert np.max(np.abs(grad)) <= 1e-5


def test_reference_warm_start_reduces_iterations():
    P, q, A, l, u = _random_qp(seed=3)
    cold = solve_qp_reference(P, q, A, l, u, max_iter=5000, eps_abs=1e-7,
                              eps_rel=1e-7, check_every=1)
    warm = solve_qp_reference(P, q, A, l, u, max_iter=5000, eps_abs=1e-7,
                              eps_rel=1e-7, check_every=1,
                              x0=cold.x, z0=cold.z, v0=cold.v)
    assert warm.converged and warm.iterations <= cold.iterations


def test_reference_rejects_indefinite_normal_matrix():
    n = 4
    P = sp.csc_matrix(np.triu(-np.eye(n)))  # negative definite P
    A = sp.csc_matrix(np.zeros((1, n)))
    with pytest.raises(np.linalg.LinAlgError):
        solve_qp_reference(P, np.zeros(n), A, [-1.0], [1.0], rho=1.0)


def test_reference_scaled_dual_convention():
    """The returned dual is the scaled v: lambda = rho * v must satisfy
    stationarity for rho != 1 (catches mixing v with lambda)."""
    P, q, A, l, u = _random_qp(seed=5, equalities=False)
    rho = 7.5
    res = solve_qp_reference(P, q, A, l, u, rho=rho, max_iter=4000,
                             eps_abs=1e-9, eps_rel=1e-9, check_every=5)
    assert res.converged
    P_full = sp.triu(P) + sp.triu(P, k=1).T
    grad = P_full @ res.x + q + A.T @ (rho * res.v)
    assert np.max(np.abs(grad)) <= 1e-5

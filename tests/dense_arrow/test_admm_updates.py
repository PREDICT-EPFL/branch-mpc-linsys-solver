"""OSQP-style value updates: atomicity, factorization counting (GPU)."""

import numpy as np
import pytest

from admm import Settings, Solver
from baselines.admm_numpy import solve_qp_reference
from experiments.dense_arrow.benchmarks.problems import generate_scenario_qp

wp = pytest.importorskip("warp")
pytestmark = pytest.mark.gpu

SETTINGS = dict(rho=1.0, max_iter=800, eps_abs=1e-7, eps_rel=1e-7,
                check_every=5)


@pytest.fixture(params=["tree", "cudss"])
def qp_solver(request):
    if request.param == "cudss":
        cudss_mod = pytest.importorskip("admm.cudss")
        if not cudss_mod.CUDSS_AVAILABLE:
            pytest.skip(cudss_mod.CUDSS_UNAVAILABLE_REASON)
    P, q, A, l, u, meta = generate_scenario_qp(num_tails=2, num_stages=4,
                                               nx=4, nu=2, seed=3)
    solver = Solver()
    solver.setup(P, q, A, l, u,
                 settings=Settings(rho_eq_scale=1.0, linear_solver=request.param, **SETTINGS))
    yield solver, (P, q, A, l, u)
    solver.close()


def test_vector_update_does_not_factorize(qp_solver):
    solver, (P, q, A, l, u) = qp_solver
    before = solver.factorization_count
    info = solver.update(q=q * 2.0, l=l - 0.1, u=u + 0.1)
    assert not info.matrix_updated and not info.factorized
    assert solver.factorization_count == before
    res = solver.solve()
    ref = solve_qp_reference(P, q * 2.0, A, l - 0.1, u + 0.1, **SETTINGS)
    assert np.allclose(res.x.numpy(), ref.x, atol=1e-7)


def test_matrix_update_factorizes_exactly_once(qp_solver):
    solver, (P, q, A, l, u) = qp_solver
    before = solver.factorization_count
    rng = np.random.default_rng(0)
    new_Px = np.asarray(P.data) * (1.0 + 0.1 * rng.uniform(size=P.nnz))
    new_Ax = np.asarray(A.data) * (1.0 + 0.05 * rng.uniform(size=A.nnz))
    info = solver.update(Px=new_Px, Ax=new_Ax)
    assert info.matrix_updated and info.factorized
    assert solver.factorization_count == before + 1
    res = solver.solve()
    import scipy.sparse as sp
    P2 = sp.csc_matrix((new_Px, P.indices, P.indptr), shape=P.shape)
    A2 = sp.csc_matrix((new_Ax, A.indices, A.indptr), shape=A.shape)
    ref = solve_qp_reference(P2, q, A2, l, u, **SETTINGS)
    assert np.allclose(res.x.numpy(), ref.x, atol=1e-7)


def test_invalid_updates_rejected_before_mutation(qp_solver):
    solver, (P, q, A, l, u) = qp_solver
    q_before = solver._q.numpy().copy()
    with pytest.raises(ValueError, match="length"):
        solver.update(Px=np.zeros(P.nnz + 1))
    with pytest.raises(ValueError, match="length"):
        solver.update(Ax=np.zeros(A.nnz - 1))
    with pytest.raises(ValueError, match="l <= u"):
        solver.update(q=q * 0.5, l=u + 1.0, u=u)
    # nothing was applied (atomicity): q unchanged by the failed calls
    assert np.array_equal(solver._q.numpy(), q_before)


def test_solution_tracks_q_updates(qp_solver):
    solver, (P, q, A, l, u) = qp_solver
    first = solver.solve()
    solver.update(q=-q)
    second = solver.solve()
    ref = solve_qp_reference(P, -q, A, l, u,
                             x0=first.x.numpy(), z0=first.z.numpy(),
                             v0=first.dual.numpy(), **SETTINGS)
    assert np.allclose(second.x.numpy(), ref.x, atol=1e-7)


def test_indefinite_p_never_reports_convergence(qp_solver):
    """A finite but indefinite P breaks the Cholesky factor into NaNs;
    the NaN-safe residual reduction must report huge residuals and
    converged=False, never a silent false convergence (the float
    atomic_max reduction would otherwise drop NaN candidates and leave
    zero residuals)."""
    solver, (P, q, A, l, u) = qp_solver
    info = solver.update(Px=-np.abs(np.asarray(P.data)) - 1.0)
    assert info.factorized
    res = solver.solve()
    assert not res.info.converged
    assert (res.info.primal_residual >= 1e290
            or res.info.dual_residual >= 1e290
            or not np.isfinite(res.info.primal_residual))


def test_partial_vector_update_keeps_other_vectors(qp_solver):
    """Updating only q must not touch l/u (and vice versa): the host
    caches replace device readbacks, so a later partial update still
    validates and solves against the correct current values."""
    solver, (P, q, A, l, u) = qp_solver
    solver.update(q=q * 3.0)
    solver.update(l=l - 0.5)  # validated against the cached u
    res = solver.solve()
    ref = solve_qp_reference(P, q * 3.0, A, l - 0.5, u, **SETTINGS)
    assert res.info.converged == ref.converged
    assert np.allclose(res.x.numpy(), ref.x, atol=1e-7)

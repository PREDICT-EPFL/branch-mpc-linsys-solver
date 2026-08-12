"""CPU reference solvers and metrics (plan sections 7.2 and 14.2-14.3)."""

import numpy as np
import pytest

from benchmarks.problems import ProblemSpec, generate_problem
from src import validation
from baselines import scipy_reference as ref

SMALL = ProblemSpec(num_branches=3, horizon=5, block_size=4,
                    separator_dim=6, num_rhs=2, seed=7)


@pytest.mark.parametrize("solver", [ref.solve_sparse_direct,
                                    ref.solve_structured_cpu])
def test_reference_solvers_recover_x_true(solver):
    p = generate_problem(SMALL)
    xt, xr = solver(p)
    metrics = validation.compute_metrics(p, xt, xr)
    assert metrics["forward_error"] <= 1e-9
    assert metrics["scaled_residual"] <= 1e-10
    assert not metrics["nan_or_inf"]


def test_structured_cpu_schur_matches_target():
    p = generate_problem(SMALL)
    xt, xr, S = ref.solve_structured_cpu(p, return_schur=True)
    assert np.allclose(S, p.schur_target,
                       atol=1e-10 * np.linalg.norm(p.schur_target))


def test_structured_matches_sparse_reference():
    p = generate_problem(SMALL)
    xt_a, xr_a = ref.solve_structured_cpu(p)
    xt_b, xr_b = ref.solve_sparse_direct(p)
    assert np.allclose(xt_a, xt_b, atol=1e-10)
    assert np.allclose(xr_a, xr_b, atol=1e-10)


@pytest.mark.parametrize("spec", [
    ProblemSpec(num_branches=1, horizon=4, block_size=3, separator_dim=5, seed=1),
    ProblemSpec(num_branches=4, horizon=1, block_size=3, separator_dim=4, seed=2),
    ProblemSpec(num_branches=3, horizon=4, block_size=2, separator_dim=1, seed=3),
    ProblemSpec(num_branches=2, horizon=3, block_size=4, separator_dim=5,
                num_rhs=4, seed=4),
    ProblemSpec(num_branches=2, horizon=3, block_size=4, separator_dim=5,
                rho_tail=2.0, rho_sep=5.0, seed=5),
])
def test_structured_cpu_edge_cases(spec):
    p = generate_problem(spec)
    xt, xr = ref.solve_structured_cpu(p)
    assert validation.compute_metrics(p, xt, xr)["forward_error"] <= 1e-8


def test_ill_conditioned_spd_still_solves():
    spec = ProblemSpec(num_branches=2, horizon=5, block_size=4,
                       separator_dim=6, condition_target=1e6, seed=9)
    p = generate_problem(spec)
    xt, xr = ref.solve_structured_cpu(p)
    metrics = validation.compute_metrics(p, xt, xr)
    eps = np.finfo(np.float64).eps
    assert metrics["forward_error"] <= 100 * p.kappa_estimate * eps


def test_chain_cholesky_reconstructs():
    p = generate_problem(SMALL)
    D, E = p.matrix.D, p.matrix.E
    Ld, Lsub = validation.chain_cholesky(D, E)
    B, T, n_b, _ = D.shape
    # reconstruct block-tridiagonal blocks from the factors
    diag_rec = Ld @ np.swapaxes(Ld, -1, -2)
    diag_rec[:, 1:] += Lsub @ np.swapaxes(Lsub, -1, -2)
    sub_rec = np.zeros_like(Lsub)
    for t in range(T - 1):
        sub_rec[:, t] = Lsub[:, t] @ np.swapaxes(Ld[:, t], -1, -2)
    assert np.allclose(diag_rec, D, atol=1e-12)
    assert np.allclose(sub_rec, E, atol=1e-12)

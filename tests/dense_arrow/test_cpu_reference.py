"""CPU reference solvers and metrics (plan sections 7.2 and 14.2-14.3)."""

import numpy as np
import pytest

from experiments.general_arrow.benchmarks.problems import ProblemSpec, generate_problem
from baselines import reference as validation
from baselines import scipy_reference as ref

SMALL = ProblemSpec(num_tails=3, horizon=5, block_size=4,
                    root_dim=6, num_rhs=2, seed=7)


def test_reference_solver_recovers_x_true():
    p = generate_problem(SMALL)
    xt, xr = ref.solve_structured_cpu(p)
    metrics = validation.compute_metrics(p, xt, xr)
    assert metrics["forward_error"] <= 1e-9
    assert metrics["scaled_residual"] <= 1e-10
    assert not metrics["nan_or_inf"]


def test_structured_cpu_root_update_matches_target():
    p = generate_problem(SMALL)
    xt, xr, S = ref.solve_structured_cpu(p, return_root_update=True)
    assert np.allclose(S, p.root_target,
                       atol=1e-10 * np.linalg.norm(p.root_target))


def test_structured_matches_dense_reference():
    """Cross-check the structured block-Cholesky reference against a
    dense LAPACK solve of the assembled system: an independent method,
    affordable because these problems are tiny."""
    from src.general_arrow.problem import TreeVector
    p = generate_problem(SMALL)
    xt_a, xr_a = ref.solve_structured_cpu(p)
    L = p.matrix.to_csr_lower(dtype=np.float64).toarray()
    A = L + np.tril(L, k=-1).T          # symmetrize the stored triangle
    r = TreeVector(p.shape, p.rhs.tail.astype(np.float64),
                   p.rhs.root.astype(np.float64)).flat()
    z = np.linalg.solve(A, r)
    v = TreeVector.from_flat(p.shape, z)
    assert np.allclose(xt_a, v.tail, atol=1e-10)
    assert np.allclose(xr_a, v.root, atol=1e-10)


@pytest.mark.parametrize("spec", [
    ProblemSpec(num_tails=1, horizon=4, block_size=3, root_dim=5, seed=1),
    ProblemSpec(num_tails=4, horizon=1, block_size=3, root_dim=4, seed=2),
    ProblemSpec(num_tails=3, horizon=4, block_size=2, root_dim=1, seed=3),
    ProblemSpec(num_tails=2, horizon=3, block_size=4, root_dim=5,
                num_rhs=4, seed=4),
    ProblemSpec(num_tails=2, horizon=3, block_size=4, root_dim=5,
                rho_tail=2.0, rho_sep=5.0, seed=5),
])
def test_structured_cpu_edge_cases(spec):
    p = generate_problem(spec)
    xt, xr = ref.solve_structured_cpu(p)
    assert validation.compute_metrics(p, xt, xr)["forward_error"] <= 1e-8


def test_ill_conditioned_spd_still_solves():
    spec = ProblemSpec(num_tails=2, horizon=5, block_size=4,
                       root_dim=6, condition_target=1e6, seed=9)
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

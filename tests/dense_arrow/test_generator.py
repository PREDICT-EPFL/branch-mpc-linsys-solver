"""Generator invariants (plan section 5.3), CPU only."""

import numpy as np
import pytest

from tests.dense_arrow.problems import ProblemSpec, generate_problem
from src.dense_arrow.problem import TreeVector
from src.dense_arrow.problem import structural_matvec

SMALL = ProblemSpec(num_tails=3, horizon=5, block_size=4,
                    root_dim=6, num_rhs=2, seed=7)

EDGE_SPECS = [
    ProblemSpec(num_tails=1, horizon=4, block_size=3, root_dim=5, seed=1),
    ProblemSpec(num_tails=4, horizon=1, block_size=3, root_dim=4, seed=2),
    ProblemSpec(num_tails=3, horizon=4, block_size=2, root_dim=1, seed=3),
    ProblemSpec(num_tails=2, horizon=3, block_size=1, root_dim=2, seed=4),
]


def _assemble(problem):
    import scipy.sparse as sp
    lower = problem.matrix.to_csr_lower(dtype=np.float64)
    return (lower + sp.tril(lower, k=-1).T).toarray()


def test_shapes_and_layout():
    p = generate_problem(SMALL)
    B, T, n_b, n_r = 3, 5, 4, 6
    assert p.matrix.D.shape == (B, T, n_b, n_b)
    assert p.matrix.E.shape == (B, T - 1, n_b, n_b)
    assert p.matrix.C_T.shape == (B, T, n_b, n_r)
    assert p.matrix.R.shape == (n_r, n_r)
    assert p.rhs.tail.shape == (B, T, n_b, 2)
    assert p.rhs.root.shape == (n_r, 2)
    assert p.rhs.nrhs == 2
    assert p.spec.total_dimension == B * T * n_b + n_r
    assert p.shape == p.spec.shape


def test_symmetry_and_spd():
    p = generate_problem(SMALL)
    A = _assemble(p)
    assert np.allclose(A, A.T, atol=1e-14 * np.abs(A).max())
    np.linalg.cholesky(A)  # raises if not SPD


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_spd_across_seeds(seed):
    spec = ProblemSpec(num_tails=2, horizon=4, block_size=3,
                       root_dim=5, seed=seed)
    np.linalg.cholesky(_assemble(generate_problem(spec)))


def test_root_update_target_matches():
    p = generate_problem(SMALL)
    A = _assemble(p)
    off = p.spec.tail_dimension
    K = A[:off, :off]
    C = A[:off, off:]
    R = A[off:, off:]
    S_explicit = R - C.T @ np.linalg.solve(K, C)
    scale = np.linalg.norm(p.root_target)
    assert np.linalg.norm(S_explicit - p.root_target) <= 1e-10 * scale


def test_structural_matvec_matches_assembled():
    p = generate_problem(SMALL)
    A = _assemble(p)
    m = p.matrix
    rng = np.random.default_rng(0)
    xt = rng.standard_normal(p.rhs.tail.shape)
    xr = rng.standard_normal(p.rhs.root.shape)
    yt, yr = structural_matvec(m.D, m.E, m.C_T, m.R, xt, xr)
    ref = A @ TreeVector(p.shape, xt, xr).flat()
    assert np.allclose(TreeVector(p.shape, yt, yr).flat(), ref,
                       atol=1e-12 * np.abs(ref).max())


def test_rhs_equals_A_times_x_true():
    p = generate_problem(SMALL)
    A = _assemble(p)
    truth = p.exact_solution
    r = A @ truth.flat()
    assert np.allclose(p.rhs.flat(), r, atol=1e-12 * np.abs(r).max())


def test_same_seed_identical_different_seed_differs():
    a = generate_problem(SMALL)
    b = generate_problem(SMALL)
    assert np.array_equal(a.matrix.D, b.matrix.D)
    assert np.array_equal(a.matrix.C_T, b.matrix.C_T)
    assert np.array_equal(a.rhs.tail, b.rhs.tail)
    c = generate_problem(ProblemSpec(**{**SMALL.to_dict(), "seed": 8}))
    assert not np.array_equal(a.matrix.D, c.matrix.D)


def test_gamma_recorded_and_nontrivial():
    p = generate_problem(SMALL)
    assert 0.01 < p.gamma < 1.0


@pytest.mark.parametrize("pattern,band", [("dense", 8), ("root_only", 8),
                                          ("terminal_only", 8),
                                          ("banded_root", 2)])
def test_coupling_patterns_spd_and_sparsity(pattern, band):
    spec = ProblemSpec(num_tails=2, horizon=4, block_size=3,
                       root_dim=6, coupling_pattern=pattern,
                       band_width=band, seed=5)
    p = generate_problem(spec)
    np.linalg.cholesky(_assemble(p))
    nz = np.abs(p.matrix.C_T).sum(axis=(0, 2))  # (T, n_r) stage activity
    if pattern == "root_only":
        assert nz[1:].sum() == 0 and nz[0].sum() > 0
    elif pattern == "terminal_only":
        assert nz[:-1].sum() == 0 and nz[-1].sum() > 0
    elif pattern == "banded_root":
        assert (nz > 0).sum(axis=1).max() <= band


@pytest.mark.parametrize("spec", EDGE_SPECS)
def test_edge_dimensions(spec):
    p = generate_problem(spec)
    A = _assemble(p)
    np.linalg.cholesky(A)
    off = spec.tail_dimension
    S_explicit = A[off:, off:] - A[:off, off:].T @ np.linalg.solve(
        A[:off, :off], A[:off, off:])
    assert np.allclose(S_explicit, p.root_target,
                       atol=1e-9 * np.linalg.norm(p.root_target))


def test_condition_target_reaches_regime():
    for target in (1e2, 1e4):
        spec = ProblemSpec(num_tails=2, horizon=6, block_size=4,
                           root_dim=6, condition_target=target, seed=11)
        p = generate_problem(spec)
        assert p.kappa_method == "exact_dense"
        assert target / 30 <= p.kappa_estimate <= target * 30
        np.linalg.cholesky(_assemble(p))


def test_condition_estimate_matches_exact():
    spec = ProblemSpec(num_tails=2, horizon=6, block_size=4,
                       root_dim=6, condition_target=1e3, seed=12)
    exact = generate_problem(spec, exact_condition_dim=10**9)
    est = generate_problem(spec, exact_condition_dim=0)
    assert est.kappa_method == "power_iteration_estimate"
    assert 0.2 <= est.kappa_estimate / exact.kappa_estimate <= 5.0


def test_diagonal_dominant_mode():
    spec = ProblemSpec(num_tails=2, horizon=4, block_size=3,
                       root_dim=5, generator_mode="diagonal_dominant",
                       seed=6)
    p = generate_problem(spec)
    A = _assemble(p)
    np.linalg.cholesky(A)
    # strict scalar diagonal dominance of the tail rows by construction
    off = spec.tail_dimension
    tail = A[:off]
    assert np.all(np.diag(A)[:off] > np.abs(tail).sum(axis=1) - np.abs(np.diag(A)[:off]))


def test_csr_lower_assembly_matches_dense():
    p = generate_problem(SMALL)
    m = p.matrix
    A = _assemble(p)
    lower = m.to_csr_lower()
    assert np.allclose(lower.toarray(), np.tril(A), atol=1e-14 * np.abs(A).max())

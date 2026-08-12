"""GPU solver tests: SOCU integration and the full structured tree solve.

All tests are marked ``gpu`` and skip automatically without a CUDA device.
"""

import numpy as np
import pytest

from benchmarks.problems import ProblemSpec, generate_problem
from src import validation

wp = pytest.importorskip("warp")

pytestmark = pytest.mark.gpu

SMALL = ProblemSpec(num_branches=3, horizon=5, block_size=16,
                    separator_dim=8, seed=7)
# n_b=16 is SOCU-aligned for fp64; smaller n_b uses the fused path


def _solve_gpu(spec, num_rhs=None, precision=None, use_cuda_graph=False):
    from src.solver import TreeSolver
    spec_d = spec.to_dict()
    if num_rhs is not None:
        spec_d["num_rhs"] = num_rhs
    if precision is not None:
        spec_d["precision"] = precision
    spec = ProblemSpec(**spec_d)
    problem = generate_problem(spec)
    solver = TreeSolver(problem.shape)
    solver.prepare(use_cuda_graph=use_cuda_graph)
    solver.stage_matrix(problem.matrix)
    solver.factorize(check=True)
    solution = solver.solve(problem.rhs).numpy()
    return problem, solver, solution.branch, solution.separator


# ----------------------------------------------------------------- SOCU layer
def test_socu_batched_factor_solve_matches_numpy():
    """Upstream SOCU batched multi-RHS solve against the NumPy chain solve
    (plan 6.1 integration test, B > 1 and multiple RHS)."""
    from src.socu_adapter import SocuTailEngine
    spec = ProblemSpec(num_branches=4, horizon=6, block_size=16,
                       separator_dim=4, seed=3)
    p = generate_problem(spec)
    B, T, n_b = 4, 6, 16
    nrhs = 5
    rng = np.random.default_rng(0)
    rhs = rng.standard_normal((B, T, n_b, nrhs))

    from socu.block_tridiag_solver import create_cholesky_solve_launch
    engine = SocuTailEngine(B, T, n_b, wp.float64, "cuda:0")
    engine.stage(p.matrix.D, p.matrix.E)
    engine.factor()
    x = wp.array(rhs, dtype=wp.float64, device="cuda:0")
    create_cholesky_solve_launch(engine.diag_factor, engine.offdiag_factor,
                                 x, dtype=wp.float64, device="cuda:0")()
    wp.synchronize()

    Ld, Lsub = validation.chain_cholesky(p.matrix.D, p.matrix.E)
    ref = validation.chain_solve(Ld, Lsub, rhs)
    assert np.allclose(x.numpy(), ref, atol=1e-9)


# ------------------------------------------------------------- full tree solve
def test_solver_recovers_x_true_fp64():
    problem, solver, xt, xr = _solve_gpu(SMALL)
    metrics = validation.compute_metrics(problem, xt, xr)
    assert metrics["forward_error"] <= 1e-9
    assert metrics["scaled_residual"] <= 1e-10
    assert not metrics["nan_or_inf"]


def test_solver_schur_matches_target():
    problem, solver, xt, xr = _solve_gpu(SMALL)
    S = solver.schur_complement()
    assert np.allclose(S, problem.schur_target,
                       atol=1e-9 * np.linalg.norm(problem.schur_target))


def test_solver_fp32_thresholds():
    problem, solver, xt, xr = _solve_gpu(SMALL, precision="float32")
    metrics = validation.compute_metrics(problem, xt, xr)
    assert metrics["scaled_residual"] <= 5e-5
    assert metrics["forward_error"] <= 5e-4


def test_solver_multi_rhs():
    problem, solver, xt, xr = _solve_gpu(SMALL, num_rhs=4)
    metrics = validation.compute_metrics(problem, xt, xr)
    assert metrics["forward_error"] <= 1e-9


def test_repeated_factorization_stable():
    """factorize() refreshes in place, so repeated calls reproduce the
    same solution.  SOCU's blocked multi-stream path (selected when
    n_b >= its internal factor block size, e.g. n_b = 16 in FP64) is
    not run-to-run bitwise deterministic under concurrent GPU load
    (upstream FMA-order caveat), so compare at a tight scale-aware
    tolerance rather than bitwise."""
    from src.solver import TreeSolver
    problem = generate_problem(SMALL)
    solver = TreeSolver(problem.shape)
    solver.stage_matrix(problem.matrix)
    solver.factorize(check=True)
    s1 = solver.solve(problem.rhs).numpy()
    solver.factorize(check=True)
    s2 = solver.solve(problem.rhs).numpy()
    eps = np.finfo(np.float64).eps
    scale = max(float(np.abs(s1.branch).max()), 1.0)
    assert np.allclose(s1.branch, s2.branch, atol=100 * eps * scale)
    assert np.allclose(s1.separator, s2.separator, atol=100 * eps * scale)


@pytest.mark.parametrize("use_cuda_graph", [False, True])
def test_solver_graph_variants(use_cuda_graph):
    problem, solver, xt, xr = _solve_gpu(SMALL, use_cuda_graph=use_cuda_graph)
    metrics = validation.compute_metrics(problem, xt, xr)
    assert metrics["forward_error"] <= 1e-9, solver.stats()


def test_long_horizon_no_unroll_regression():
    """Regression: the stage loops must not unroll (unrolled kernels exceed
    shared memory for long horizons and silently fail to launch)."""
    spec = ProblemSpec(num_branches=2, horizon=128, block_size=16,
                       separator_dim=64, seed=6)
    problem, solver, xt, xr = _solve_gpu(spec)
    assert validation.compute_metrics(problem, xt, xr)["forward_error"] <= 1e-9


def test_large_separator_paths():
    """n_y > 64 exercises the tiled Schur kernels and the SOCU dense-root
    factorization."""
    for n_y in (128, 256):
        spec = ProblemSpec(num_branches=2, horizon=8, block_size=16,
                           separator_dim=n_y, seed=6)
        problem, solver, xt, xr = _solve_gpu(spec)
        assert solver.stats().root_via_socu
        assert validation.compute_metrics(problem, xt, xr)["forward_error"] <= 1e-9


def test_unaligned_separator_tile_root():
    """Small SOCU-unaligned n_y uses the single-tile root fallback."""
    spec = ProblemSpec(num_branches=2, horizon=4, block_size=16,
                       separator_dim=41, seed=6)
    problem, solver, xt, xr = _solve_gpu(spec)
    assert not solver.stats().root_via_socu
    assert validation.compute_metrics(problem, xt, xr)["forward_error"] <= 1e-9


def test_sparse_coupling_patterns_solve():
    """Sparse coupling patterns (structurally zero stages) solve
    correctly through the same fixed full-range kernels."""
    for pattern in ("root_only", "terminal_only"):
        spec = ProblemSpec(num_branches=2, horizon=6, block_size=16,
                           separator_dim=8, coupling_pattern=pattern, seed=6)
        problem, solver, xt, xr = _solve_gpu(spec)
        assert validation.compute_metrics(problem, xt, xr)["forward_error"] <= 1e-9


def test_schur_reproducible():
    """The fixed pairwise reduction removes scheduling dependence from
    the cross-branch sum; residual variation across independent solver
    instances is bounded by SOCU's blocked-path FMA-order caveat
    (epsilon-level), so compare at 100 eps scale."""
    prob, solver_a, *_ = _solve_gpu(SMALL)
    _, solver_b, *_ = _solve_gpu(SMALL)
    Sa, Sb = solver_a.schur_complement(), solver_b.schur_complement()
    eps = np.finfo(np.float64).eps
    assert np.allclose(Sa, Sb, atol=100 * eps * max(np.abs(Sa).max(), 1.0))


def test_no_warm_allocations():
    """Prepared warm factorize/solve calls must not grow the Warp
    allocator (allocator-stability requirement)."""
    from src.solver import TreeSolver
    problem = generate_problem(SMALL)
    solver = TreeSolver(problem.shape)
    solver.stage_matrix(problem.matrix)
    solver.factorize()
    rhs_dev = solver.upload_rhs(problem.rhs)
    prepared = solver.prepare_solve(problem.rhs.nrhs)
    for _ in range(3):  # warm-up: compiles kernels
        solver.factorize()
        prepared.solve_into(rhs_dev)
    solver.synchronize()
    before = wp.get_mempool_used_mem_current("cuda:0")
    for _ in range(5):
        solver.factorize()
        prepared.solve_into(rhs_dev)
    solver.synchronize()
    after = wp.get_mempool_used_mem_current("cuda:0")
    assert after == before, (before, after)


def test_prepared_solve_with_graph():
    from src.solver import TreeSolver
    problem = generate_problem(SMALL)
    solver = TreeSolver(problem.shape)
    solver.stage_matrix(problem.matrix)
    solver.factorize(check=True)
    rhs_dev = solver.upload_rhs(problem.rhs)
    prepared = solver.prepare_solve(problem.rhs.nrhs, use_cuda_graph=True)
    for _ in range(3):  # first call captures, later calls replay
        out = prepared.solve_into(rhs_dev)
    host = out.numpy()
    metrics = validation.compute_metrics(problem, host.branch,
                                         host.separator)
    assert metrics["forward_error"] <= 1e-9


@pytest.mark.parametrize("spec", [
    ProblemSpec(num_branches=1, horizon=5, block_size=16, separator_dim=8, seed=1),
    ProblemSpec(num_branches=4, horizon=1, block_size=16, separator_dim=8, seed=2),
    ProblemSpec(num_branches=3, horizon=4, block_size=16, separator_dim=1, seed=3),
    ProblemSpec(num_branches=2, horizon=3, block_size=16, separator_dim=8,
                rho_tail=2.0, rho_sep=5.0, seed=4),
    ProblemSpec(num_branches=2, horizon=8, block_size=32, separator_dim=96, seed=5),
])
def test_solver_edge_cases(spec):
    problem, solver, xt, xr = _solve_gpu(spec)
    assert validation.compute_metrics(problem, xt, xr)["forward_error"] <= 1e-8


def test_solver_matches_cpu_reference_bitwise_tolerance():
    from baselines import scipy_reference as ref
    problem, solver, xt, xr = _solve_gpu(SMALL)
    xt_c, xr_c = ref.solve_structured_cpu(problem)
    assert np.allclose(xt, xt_c, atol=1e-9)
    assert np.allclose(xr, xr_c.reshape(xr.shape), atol=1e-9)


def test_ill_conditioned_gpu():
    spec = ProblemSpec(num_branches=2, horizon=5, block_size=16,
                       separator_dim=8, condition_target=1e6, seed=9)
    problem, solver, xt, xr = _solve_gpu(spec)
    metrics = validation.compute_metrics(problem, xt, xr)
    eps = np.finfo(np.float64).eps
    assert metrics["forward_error"] <= 1000 * problem.kappa_estimate * eps


def test_non_spd_raises():
    from src.problem import TreeMatrix
    from src.solver import TreeSolver
    problem = generate_problem(SMALL)
    D = problem.matrix.D.copy()
    D[0, 0] -= 100.0 * np.eye(16)  # break positive definiteness
    broken = TreeMatrix(problem.shape, D=D, E=problem.matrix.E,
                        C_T=problem.matrix.C_T, R=problem.matrix.R)
    solver = TreeSolver(problem.shape)
    solver.stage_matrix(broken)
    with pytest.raises(RuntimeError):
        solver.factorize(check=True)


def test_stats_and_workspace():
    problem, solver, xt, xr = _solve_gpu(SMALL)
    stats = solver.stats()
    assert stats.socu["socu_commit"] and stats.socu["socu_version"]
    assert stats.socu["socu_levels"] >= 1
    assert stats.workspace_bytes > 0
    assert stats.precision == "float64"
    assert solver.compute_min_pivot() > 0

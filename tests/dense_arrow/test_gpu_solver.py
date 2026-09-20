"""GPU solver tests: SOCU integration and the full structured tree solve.

All tests are marked ``gpu`` and skip automatically without a CUDA device.
"""

import numpy as np
import pytest

from experiments.dense_arrow.benchmarks.problems import ProblemSpec, generate_problem
from baselines import reference as validation

wp = pytest.importorskip("warp")

pytestmark = pytest.mark.gpu

SMALL = ProblemSpec(num_tails=3, horizon=5, block_size=16,
                    root_dim=8, seed=7)
# n_b=16 is SOCU-aligned for fp64; smaller n_b uses the fused path


def _solve_gpu(spec, num_rhs=None, precision=None):
    from src.dense_arrow.solver import Solver
    spec_d = spec.to_dict()
    if num_rhs is not None:
        spec_d["num_rhs"] = num_rhs
    if precision is not None:
        spec_d["precision"] = precision
    spec = ProblemSpec(**spec_d)
    problem = generate_problem(spec)
    solver = Solver(problem.shape)
    solver.update(problem.matrix)
    solver.factorize()
    solution = solver.solve(problem.rhs).numpy()
    return problem, solver, solution.tail, solution.root


# ----------------------------------------------------------------- SOCU layer
def test_socu_batched_factor_solve_matches_numpy():
    """Upstream SOCU batched multi-RHS solve against the NumPy chain solve
    (plan 6.1 integration test, B > 1 and multiple RHS)."""
    from src.dense_arrow.socu import TailEngine as SocuTailEngine
    spec = ProblemSpec(num_tails=4, horizon=6, block_size=16,
                       root_dim=4, seed=3)
    p = generate_problem(spec)
    B, T, n_b = 4, 6, 16
    nrhs = 5
    rng = np.random.default_rng(0)
    rhs = rng.standard_normal((B, T, n_b, nrhs))

    from socu.block_tridiag_solver import create_cholesky_solve_launch
    from socu.block_tridiag_solver import create_cholesky_factor_launch
    engine = SocuTailEngine(B, T, n_b, wp.float64, "cuda:0")
    engine.stage(p.matrix.D, p.matrix.E)
    engine.refresh()
    create_cholesky_factor_launch(engine.diag_factor,
                                  engine.offdiag_factor,
                                  dtype=wp.float64, device="cuda:0")()
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


def test_solver_root_factor_matches_target():
    """The stored root factor reproduces the generator's updated-root
    target: tril(L_R) @ tril(L_R)^T == R - sum_i M_i M_i^T."""
    problem, solver, xt, xr = _solve_gpu(SMALL)
    L = np.tril(solver._root_diagonal.numpy())
    assert np.allclose(L @ L.T, problem.root_target,
                       atol=1e-9 * np.linalg.norm(problem.root_target))


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
    from src.dense_arrow.solver import Solver
    problem = generate_problem(SMALL)
    solver = Solver(problem.shape)
    solver.update(problem.matrix)
    solver.factorize()
    s1 = solver.solve(problem.rhs).numpy()
    solver.factorize()
    s2 = solver.solve(problem.rhs).numpy()
    eps = np.finfo(np.float64).eps
    scale = max(float(np.abs(s1.tail).max()), 1.0)
    assert np.allclose(s1.tail, s2.tail, atol=100 * eps * scale)
    assert np.allclose(s1.root, s2.root, atol=100 * eps * scale)


def test_solver_graph_replay_accuracy():
    problem, solver, xt, xr = _solve_gpu(SMALL)
    metrics = validation.compute_metrics(problem, xt, xr)
    assert metrics["forward_error"] <= 1e-9


def test_long_horizon_no_unroll_regression():
    """Regression: the stage loops must not unroll (unrolled kernels exceed
    shared memory for long horizons and silently fail to launch)."""
    spec = ProblemSpec(num_tails=2, horizon=128, block_size=16,
                       root_dim=64, seed=6)
    problem, solver, xt, xr = _solve_gpu(spec)
    assert validation.compute_metrics(problem, xt, xr)["forward_error"] <= 1e-9


def test_large_root_paths():
    """n_r > 64 exercises the tiled root-update kernels and the SOCU dense-root
    factorization."""
    for n_r in (128, 256):
        spec = ProblemSpec(num_tails=2, horizon=8, block_size=16,
                           root_dim=n_r, seed=6)
        problem, solver, xt, xr = _solve_gpu(spec)
        assert solver._root_via_socu
        assert validation.compute_metrics(problem, xt, xr)["forward_error"] <= 1e-9


def test_unaligned_large_root_rejected():
    """Root dimensions of at least TILE_M must be SOCU-aligned (there is
    no third fallback path)."""
    from src.dense_arrow.problem import TreeShape
    from src.dense_arrow.solver import Solver
    with pytest.raises(ValueError, match="aligned"):
        Solver(TreeShape(2, 4, 16, 41, "float64"))


def test_sparse_coupling_patterns_solve():
    """Sparse coupling patterns (structurally zero stages) solve
    correctly through the same fixed full-range kernels."""
    for pattern in ("root_only", "terminal_only"):
        spec = ProblemSpec(num_tails=2, horizon=6, block_size=16,
                           root_dim=8, coupling_pattern=pattern, seed=6)
        problem, solver, xt, xr = _solve_gpu(spec)
        assert validation.compute_metrics(problem, xt, xr)["forward_error"] <= 1e-9


def test_root_update_reproducible():
    """Run-to-run variation of the updated root across independent
    solver instances stays at epsilon scale (atomic cross-tail
    accumulation and SOCU's blocked-path FMA order are both
    scheduling-dependent, so bitwise equality is not claimed)."""
    prob, solver_a, *_ = _solve_gpu(SMALL)
    _, solver_b, *_ = _solve_gpu(SMALL)
    Sa = np.tril(solver_a._root_diagonal.numpy())
    Sb = np.tril(solver_b._root_diagonal.numpy())
    eps = np.finfo(np.float64).eps
    assert np.allclose(Sa, Sb, atol=100 * eps * max(np.abs(Sa).max(), 1.0))


def test_bound_solve_no_warm_allocations():
    """Warm factorize/solve with bound device buffers must not grow the
    Warp allocator."""
    from src.dense_arrow.problem import TreeVector
    from src.dense_arrow.solver import Solver
    problem = generate_problem(SMALL)
    solver = Solver(problem.shape)
    solver.update(problem.matrix)
    solver.factorize()
    rhs = TreeVector(problem.shape,
                     wp.array(problem.rhs.tail, dtype=wp.float64,
                              device="cuda:0"),
                     wp.array(problem.rhs.root, dtype=wp.float64,
                              device="cuda:0"))
    out = TreeVector(problem.shape,
                     wp.zeros(rhs.tail.shape, dtype=wp.float64,
                              device="cuda:0"),
                     wp.zeros(rhs.root.shape, dtype=wp.float64,
                              device="cuda:0"))
    for _ in range(3):  # warm-up: compiles kernels and captures graphs
        solver.factorize()
        solver.solve(rhs, out=out)
    wp.synchronize_device("cuda:0")
    before = wp.get_mempool_used_mem_current("cuda:0")
    for _ in range(5):
        solver.factorize()
        solver.solve(rhs, out=out)
    wp.synchronize_device("cuda:0")
    after = wp.get_mempool_used_mem_current("cuda:0")
    assert after == before, (before, after)


@pytest.mark.parametrize("spec", [
    ProblemSpec(num_tails=1, horizon=5, block_size=16, root_dim=8, seed=1),
    ProblemSpec(num_tails=4, horizon=1, block_size=16, root_dim=8, seed=2),
    ProblemSpec(num_tails=3, horizon=4, block_size=16, root_dim=1, seed=3),
    ProblemSpec(num_tails=2, horizon=3, block_size=16, root_dim=8,
                rho_tail=2.0, rho_sep=5.0, seed=4),
    ProblemSpec(num_tails=2, horizon=8, block_size=32, root_dim=96, seed=5),
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
    spec = ProblemSpec(num_tails=2, horizon=5, block_size=16,
                       root_dim=8, condition_target=1e6, seed=9)
    problem, solver, xt, xr = _solve_gpu(spec)
    metrics = validation.compute_metrics(problem, xt, xr)
    eps = np.finfo(np.float64).eps
    assert metrics["forward_error"] <= 1000 * problem.kappa_estimate * eps


def test_non_spd_produces_nan():
    """Non-SPD input breaks the Cholesky factor; the solution carries
    NaNs (there is no default-path pivot scan -- validation is the
    caller's choice, outside the warm path)."""
    from src.dense_arrow.problem import TreeMatrix
    from src.dense_arrow.solver import Solver
    problem = generate_problem(SMALL)
    D = problem.matrix.D.copy()
    D[0, 0] -= 100.0 * np.eye(16)  # break positive definiteness
    broken = TreeMatrix(problem.shape, D=D, E=problem.matrix.E,
                        C_T=problem.matrix.C_T, R=problem.matrix.R)
    solver = Solver(problem.shape)
    solver.update(broken)
    solver.factorize()
    x = solver.solve(problem.rhs).numpy()
    assert not np.isfinite(x.tail).all()

"""Small-root kernel path (plan 3, stages 4-5): scalar root updates,
reduction, and correction for root dimensions below TILE_M."""

import numpy as np
import pytest

from src.general_arrow.kernels import TILE_M
from tests.general_arrow.permuted_factor_reference import assemble_dense_system

wp = pytest.importorskip("warp")
pytestmark = pytest.mark.gpu


def _problem(B, T, n_b, n_r, nrhs=1, seed=0, precision="float64"):
    from experiments.general_arrow.benchmarks.problems import ProblemSpec, generate_problem
    spec = ProblemSpec(num_tails=B, horizon=T, block_size=n_b,
                       root_dim=n_r, num_rhs=nrhs,
                       precision=precision, seed=seed)
    return generate_problem(spec, estimate_condition=False)


def _dense_solve(problem):
    Phi = assemble_dense_system(problem.matrix)
    B, T, n_b, n_r = problem.matrix.shape.dims()
    r = np.asarray(problem.rhs.tail, dtype=np.float64)
    q = np.asarray(problem.rhs.root, dtype=np.float64)
    nrhs = q.shape[-1]
    rhs = np.concatenate([r.reshape(B * T * n_b, nrhs), q])
    x = np.linalg.solve(Phi, rhs)
    return x[:B * T * n_b].reshape(B, T, n_b, nrhs), x[B * T * n_b:]


@pytest.mark.parametrize("n_r", [1, 2, 4, 8, 15])
@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_small_root_path_matches_dense(n_r, precision):
    from src.general_arrow.solver import Solver
    problem = _problem(B=5, T=13, n_b=8, n_r=n_r, seed=n_r,
                       precision=precision)
    solver = Solver(problem.matrix.shape)
    assert solver._small_root  # the scalar path is active
    solver.update(problem.matrix)
    solver.factorize()
    x = solver.solve(problem.rhs).numpy()
    wb, yb = _dense_solve(problem)
    tol = 1e-9 if precision == "float64" else 5e-4
    scale = max(np.abs(wb).max(), np.abs(yb).max(), 1.0)
    assert np.abs(x.tail - wb).max() <= tol * scale
    assert np.abs(x.root - yb).max() <= tol * scale


@pytest.mark.parametrize("nrhs", [1, 3, 16])
def test_small_root_multi_rhs(nrhs):
    from src.general_arrow.solver import Solver
    problem = _problem(B=3, T=8, n_b=8, n_r=2, nrhs=nrhs, seed=7)
    solver = Solver(problem.matrix.shape)
    solver.update(problem.matrix)
    solver.factorize()
    x = solver.solve(problem.rhs).numpy()
    wb, yb = _dense_solve(problem)
    assert np.abs(x.tail - wb).max() <= 1e-9
    assert np.abs(x.root - yb).max() <= 1e-9


def test_small_and_tiled_paths_agree(monkeypatch):
    """The scalar small-root path and the tiled path compute the same
    factorization and solution (FP64 tolerance)."""
    import src.general_arrow.kernels.coupling as coupling
    from src.general_arrow.solver import Solver
    problem = _problem(B=4, T=16, n_b=8, n_r=8, seed=9)
    results = {}
    for label, enabled in (("small", True), ("tiled", False)):
        monkeypatch.setattr(coupling, "_SMALL_ROOT_ENABLED", enabled)
        solver = Solver(problem.matrix.shape)
        assert solver._small_root is enabled
        solver.update(problem.matrix)
        solver.factorize()
        results[label] = (
            solver.solve(problem.rhs).numpy(),
            np.tril(solver._root_diagonal.numpy()))
    xs, Ss = results["small"], results["tiled"]
    assert np.abs(xs[1] - Ss[1]).max() <= 1e-11 * np.abs(Ss[1]).max()
    assert np.abs(xs[0].tail - Ss[0].tail).max() <= 1e-11
    assert np.abs(xs[0].root - Ss[0].root).max() <= 1e-11


def test_small_root_deterministic_repeat():
    """The scalar partials and fixed-order reductions add nothing
    scheduling-dependent: repeated factorize/solve agrees to epsilon
    scale.  (SOCU's fused forward substitution uses atomic neighbor
    updates, so whole-solver bitwise determinism is not claimed --
    plan 9.4.)"""
    from src.general_arrow.solver import Solver
    problem = _problem(B=7, T=12, n_b=8, n_r=2, seed=11)
    solver = Solver(problem.matrix.shape)
    runs = []
    for _ in range(2):
        solver.update(problem.matrix)
        solver.factorize()
        x = solver.solve(problem.rhs).numpy()
        runs.append((x.tail.copy(), x.root.copy(),
                     np.tril(solver._root_diagonal.numpy())))
    tol = 100 * np.finfo(np.float64).eps
    for i in range(3):
        a, b = runs[0][i], runs[1][i]
        assert np.abs(a - b).max() <= tol * max(np.abs(b).max(), 1.0)


def test_dispatch_boundary():
    """n_r >= TILE_M uses the tiled path; below uses the scalar path."""
    from src.general_arrow.solver import Solver
    from src.general_arrow.problem import TreeShape
    hi = TreeShape(2, 4, 8, TILE_M, "float64")
    lo = TreeShape(2, 4, 8, TILE_M - 1, "float64")
    assert not Solver(hi)._small_root
    assert Solver(lo)._small_root
    # unaligned roots of at least TILE_M are rejected at construction
    with pytest.raises(ValueError, match="aligned"):
        Solver(TreeShape(2, 4, 8, TILE_M + 1, "float64"))

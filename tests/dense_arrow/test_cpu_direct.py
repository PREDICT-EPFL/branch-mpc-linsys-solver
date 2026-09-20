"""CPU sparse direct baselines (qdldl, CHOLMOD, PARDISO): correctness
on generated tree problems and the benchmark-runner record contract."""

import numpy as np
import pytest

from baselines import cpu_direct
from baselines import reference as validation
from experiments.dense_arrow.benchmarks.problems import ProblemSpec, generate_problem
from src.dense_arrow.problem import TreeVector

SMALL = ProblemSpec(num_tails=3, horizon=5, block_size=4,
                    root_dim=6, num_rhs=2, seed=7)


def _solver_classes():
    classes = []
    if cpu_direct.QDLDL_AVAILABLE:
        classes.append(lambda lower: cpu_direct.QdldlDirect(lower))
    if cpu_direct.CHOLMOD_AVAILABLE:
        classes.append(lambda lower: cpu_direct.CholmodDirect(lower))
    if cpu_direct.PARDISO_AVAILABLE:
        classes.append(lambda lower: cpu_direct.PardisoDirect(lower))
    return classes


@pytest.mark.parametrize("make", _solver_classes())
def test_direct_solvers_recover_x_true(make):
    p = generate_problem(SMALL)
    solver = make(p.matrix.to_csr_lower(dtype=np.float64))
    solver.factorize()  # refactorization keeps the factorization valid
    x = solver.solve(p.rhs.flat())
    z = TreeVector.from_flat(SMALL.shape, x)
    metrics = validation.compute_metrics(p, z.tail, z.root)
    assert metrics["forward_error"] <= 1e-9


@pytest.mark.parametrize("make", _solver_classes())
def test_direct_solvers_single_vector_rhs(make):
    spec = ProblemSpec(num_tails=2, horizon=4, block_size=4,
                       root_dim=6, num_rhs=1, seed=3)
    p = generate_problem(spec)
    solver = make(p.matrix.to_csr_lower(dtype=np.float64))
    x = solver.solve(p.rhs.flat()[:, 0])
    z = TreeVector.from_flat(spec.shape, x[:, None])
    metrics = validation.compute_metrics(p, z.tail, z.root)
    assert metrics["forward_error"] <= 1e-9


def test_runner_records_have_benchmark_contract():
    from experiments.dense_arrow.benchmarks.runners import RUNNERS
    p = generate_problem(SMALL)
    rules = {"warmups": 1, "min_reps": 3, "slow_reps": 2,
             "min_seconds": 0.0, "max_reps": 5}
    for kind in ("qdldl", "cholmod"):
        rec = RUNNERS[kind](p, {"kind": kind}, rules, "cuda:0")
        if rec["status"] != "ok":  # optional package not installed
            assert kind in ("qdldl", "cholmod")
            continue
        for key in ("warm_factor", "warm_solve", "warm_total",
                    "warm_total_ms", "cold_first_factor_s",
                    "cold_first_solve_s", "rhs_relative_residual"):
            assert key in rec, (kind, key)
        assert rec["rhs_relative_residual"] < 1e-10

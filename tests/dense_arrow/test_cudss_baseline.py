"""cuDSS baseline adapter tests (GPU; skipped without CUDA/cuDSS)."""

import pytest

from experiments.dense_arrow.benchmarks.problems import ProblemSpec, generate_problem
from baselines import reference as validation
from src.dense_arrow.problem import TreeVector


def _lower_and_rhs(problem):
    m = problem.matrix
    lower = m.to_csr_lower()
    return lower, problem.rhs.flat()

pytestmark = pytest.mark.gpu

cudss_mod = pytest.importorskip("baselines.cudss")

SMALL = ProblemSpec(num_tails=3, horizon=5, block_size=8,
                    root_dim=6, seed=7)


def _run_cudss(spec, ordering="default", num_rhs=None):
    import warp as wp
    from baselines.cudss import CudssCholesky
    d = spec.to_dict()
    if num_rhs:
        d["num_rhs"] = num_rhs
    spec = ProblemSpec(**d)
    s = generate_problem(spec)
    lower, rhs = _lower_and_rhs(s)
    rhs = rhs[:, 0] if spec.num_rhs == 1 else rhs
    solver = CudssCholesky(lower, rhs, precision=spec.precision,
                           ordering=ordering)
    solver.plan()
    solver.factorize()
    solver.solve()
    wp.synchronize_device("cuda:0")
    z = solver.solution()
    solver.free()
    v = TreeVector.from_flat(spec.shape, z)
    return s, v.tail, v.root


def test_cudss_available_or_reason():
    from baselines.cudss import CUDSS_AVAILABLE, CUDSS_UNAVAILABLE_REASON
    assert CUDSS_AVAILABLE or CUDSS_UNAVAILABLE_REASON


def test_cudss_solves_small_system():
    if not cudss_mod.CUDSS_AVAILABLE:
        pytest.skip(cudss_mod.CUDSS_UNAVAILABLE_REASON)
    s, xt, xr = _run_cudss(SMALL)
    metrics = validation.compute_metrics(s, xt, xr)
    assert metrics["forward_error"] <= 1e-9
    assert metrics["scaled_residual"] <= 1e-10


@pytest.mark.parametrize("ordering", ["default", "amd", "nested_dissection"])
def test_cudss_ordering_alternatives(ordering):
    if not cudss_mod.CUDSS_AVAILABLE:
        pytest.skip(cudss_mod.CUDSS_UNAVAILABLE_REASON)
    from nvmath.bindings.cudss import cuDSSError
    try:
        s, xt, xr = _run_cudss(SMALL, ordering=ordering)
    except cuDSSError as exc:
        if "NOT_SUPPORTED" in str(exc):
            pytest.skip(f"ordering {ordering} unsupported by cuDSS "
                        f"{cudss_mod.cudss_version()}")
        raise
    assert validation.compute_metrics(s, xt, xr)["forward_error"] <= 1e-9


def test_cudss_multi_rhs():
    if not cudss_mod.CUDSS_AVAILABLE:
        pytest.skip(cudss_mod.CUDSS_UNAVAILABLE_REASON)
    s, xt, xr = _run_cudss(SMALL, num_rhs=4)
    assert validation.compute_metrics(s, xt, xr)["forward_error"] <= 1e-9


def test_cudss_refactorize_reuses_analysis():
    if not cudss_mod.CUDSS_AVAILABLE:
        pytest.skip(cudss_mod.CUDSS_UNAVAILABLE_REASON)
    import warp as wp
    from baselines.cudss import CudssCholesky
    s = generate_problem(SMALL)
    lower, rhs = _lower_and_rhs(s)
    solver = CudssCholesky(lower, rhs[:, 0])
    solver.plan()
    for _ in range(3):  # repeated numeric factorization, one analysis
        solver.factorize()
        solver.solve()
    wp.synchronize_device("cuda:0")
    z = solver.solution()
    solver.free()
    v = TreeVector.from_flat(s.shape, z)
    assert validation.compute_metrics(s, v.tail,
                                      v.root)["forward_error"] <= 1e-9


def test_cudss_metadata():
    if not cudss_mod.CUDSS_AVAILABLE:
        pytest.skip(cudss_mod.CUDSS_UNAVAILABLE_REASON)
    from baselines.cudss import CudssCholesky
    s = generate_problem(SMALL)
    lower, rhs = _lower_and_rhs(s)
    solver = CudssCholesky(lower, rhs[:, 0])
    md = solver.metadata()
    assert md["matrix_type"] == "SPD" and md["matrix_view"] == "lower"
    assert md["cudss_version"] not in ("unavailable",)
    assert set(solver.setup_seconds) >= {"transfer", "create"}
    solver.plan()
    assert "plan" in solver.setup_seconds
    solver.free()

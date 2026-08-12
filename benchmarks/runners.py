"""Timed method runners: the structured tree solver, cuDSS, and the CPU
references, each measured on identical generated instances.

Timing scopes (see plan section 8):

- ``cold_*``   : construction, JIT, staging, analysis, first
  factorization and first solve, wall-clock with synchronization, phases
  separated;
- ``warm_*``   : CUDA-event kernel times of repeated factorize/solve
  after warmups, with staging/refresh/store overhead reported separately
  and included only in ``warm_total_ms``;
- accuracy metrics are computed once per case outside all timed regions.
"""

import time

import numpy as np

from src import validation
from src.problem import TreeVector
from benchmarks.timing import summarize

#: Phase marks that are repeated-overhead, not solver work (reported
#: separately as ``warm_overhead_ms``; included in the paired factor and
#: solve intervals, which time the complete calls).
_OVERHEAD_PHASES = {"refresh", "rhs_stage", "solution_store"}


class _PairedTimer:
    """Three CUDA events timing one factorize+solve iteration, so the
    reported factor, solve, and total come from the same iteration and
    are exactly additive (total = factor + solve by construction)."""

    def __init__(self, device):
        import warp as wp
        dev = wp.get_device(device)
        self._wp = wp
        self._e = [wp.Event(dev, enable_timing=True) for _ in range(3)]

    def measure(self, factor_fn, solve_fn):
        wp = self._wp
        e0, e1, e2 = self._e
        wp.record_event(e0)
        factor_fn()
        wp.record_event(e1)
        solve_fn()
        wp.record_event(e2)
        wp.synchronize_event(e2)
        f = float(wp.get_event_elapsed_time(e0, e1, synchronize=False))
        s = float(wp.get_event_elapsed_time(e1, e2, synchronize=False))
        return f, s


def _paired_loop(paired_measure, rules):
    """Adaptive repetition loop over paired iterations: at least
    ``min_reps`` and ``min_seconds`` of measured time, capped at
    ``max_reps``.  Returns (factor, solve, total) sample lists (ms) taken
    from the same iterations."""
    min_reps = int(rules.get("min_reps", 30))
    slow_reps = int(rules.get("slow_reps", 10))
    slow_ms = float(rules.get("slow_threshold_ms", 200.0))
    min_seconds = float(rules.get("min_seconds", 1.0))
    max_reps = int(rules.get("max_reps", 500))
    factor, solve, total = [], [], []
    elapsed = 0.0
    while len(total) < max_reps:
        f, s = paired_measure()
        factor.append(f)
        solve.append(s)
        total.append(f + s)
        elapsed += (f + s) / 1e3
        target = slow_reps if np.median(total) >= slow_ms else min_reps
        if len(total) >= target and elapsed >= min_seconds:
            break
    return factor, solve, total


def run_tree_method(problem, method_def, rules, device):
    """Benchmark the structured tree solver on one generated problem."""
    from src.solver import TreeSolver
    from benchmarks.timing import PhaseTimer

    spec = problem.spec
    options = method_def.get("options", {})
    use_cuda_graph = bool(options.get("use_cuda_graph", False))
    rec = {"status": "ok"}

    # ---- cold path: construction + staging + preparation + first
    # factor/solve
    t0 = time.perf_counter()
    solver = TreeSolver(spec.shape, device=device)
    rec["cold_alloc_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    solver.stage_matrix(problem.matrix)
    solver.synchronize()
    rec["cold_transfer_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    solver.prepare(use_cuda_graph=use_cuda_graph)
    rec["cold_analyze_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    solver.factorize()
    solver.synchronize()
    rec["cold_first_factor_s"] = time.perf_counter() - t0

    # pre-stage device rhs and bind output buffers so warm solves
    # refresh device-to-device and allocate nothing
    rhs_dev = solver.upload_rhs(problem.rhs)
    prepared = solver.prepare_solve(problem.rhs.nrhs)
    t0 = time.perf_counter()
    prepared.solve_into(rhs_dev)
    solver.synchronize()
    rec["cold_first_solve_s"] = time.perf_counter() - t0

    # ---- accuracy (outside timed regions)
    solution = prepared.out.numpy()
    rec.update(validation.compute_metrics(problem, solution.branch,
                                          solution.separator))
    solver.factorize(check=True)  # raises on breakdown
    rec["min_pivot"] = solver.compute_min_pivot()

    # ---- warm timing: paired factor/solve intervals from the same
    # iteration (exactly additive), plus a few phase-instrumented
    # iterations for the diagnostic breakdown
    warmups = int(rules.get("warmups", 10))
    for _ in range(warmups):
        solver.factorize()
        prepared.solve_into(rhs_dev)
    solver.synchronize()

    paired = _PairedTimer(device)
    factor_samples, solve_samples, total_samples = _paired_loop(
        lambda: paired.measure(
            solver.factorize,
            lambda: prepared.solve_into(rhs_dev)),
        rules)

    timer = PhaseTimer(device)
    phase_totals, phase_counts = {}, {}
    for _ in range(int(rules.get("phase_reps", 10))):
        timer.begin()
        solver.factorize(timer=timer)
        prepared.solve_into(rhs_dev, timer=timer)
        for k, v in timer.collect().items():
            phase_totals[k] = phase_totals.get(k, 0.0) + v
            phase_counts[k] = phase_counts.get(k, 0) + 1

    rec["warm_factor"] = summarize(factor_samples)
    rec["warm_solve"] = summarize(solve_samples)
    rec["warm_total"] = summarize(total_samples)
    rec["raw_factor_ms"] = factor_samples
    rec["raw_solve_ms"] = solve_samples
    rec["raw_total_ms"] = total_samples
    phase_means = {k: phase_totals[k] / phase_counts[k] for k in phase_totals}
    rec["phase_ms"] = phase_means
    rec["warm_total_ms"] = rec["warm_total"]["median"]
    rec["warm_overhead_ms"] = sum(phase_means.get(k, 0.0)
                                  for k in _OVERHEAD_PHASES)
    rec["measurement"] = "paired_events"
    rec["jit_s"] = max(
        rec["cold_first_factor_s"] + rec["cold_first_solve_s"]
        - (rec["warm_factor"]["median"] + rec["warm_solve"]["median"]) / 1e3,
        0.0)
    # persistent workspace is the dominant, exactly-known device footprint
    rec["peak_device_bytes"] = solver.workspace_bytes()
    rec["workspace_bytes"] = solver.workspace_bytes()
    rec["solver_stats"] = solver.stats().to_dict()

    solver.close()
    return rec


def run_cudss_method(problem, method_def, rules, device):
    """Benchmark cuDSS on the assembled lower triangle of the same
    system."""
    from baselines import cudss as cudss_mod


    rec = {"status": "ok"}
    if not cudss_mod.CUDSS_AVAILABLE:
        return {"status": "unavailable",
                "skip_reason": cudss_mod.CUDSS_UNAVAILABLE_REASON}
    import warp as wp

    spec = problem.spec
    matrix = problem.matrix
    t0 = time.perf_counter()
    lower = matrix.to_csr_lower(dtype=spec.np_dtype)
    rec["host_assembly_s"] = time.perf_counter() - t0
    rec["nnz_lower"] = int(lower.nnz)

    rhs = problem.rhs.flat()
    rhs = rhs[:, 0] if spec.num_rhs == 1 else rhs
    solver = cudss_mod.CudssCholesky(
        lower, rhs, precision=spec.precision,
        ordering=method_def.get("ordering", "default"),
        ir_steps=int(method_def.get("ir_steps", 0)), device=device)
    stream_ptr = wp.get_device(device).stream.cuda_stream
    solver.plan(stream_ptr)
    rec["cold_transfer_s"] = solver.setup_seconds["transfer"]
    rec["cold_alloc_s"] = solver.setup_seconds["create"]
    rec["cold_analyze_s"] = solver.setup_seconds["plan"]

    t0 = time.perf_counter()
    solver.factorize(stream_ptr)
    wp.synchronize_device(device)
    rec["cold_first_factor_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    solver.solve(stream_ptr)
    wp.synchronize_device(device)
    rec["cold_first_solve_s"] = time.perf_counter() - t0

    # accuracy outside timed regions
    z = TreeVector.from_flat(spec.shape, solver.solution())
    rec.update(validation.compute_metrics(problem, z.branch, z.separator))

    warmups = int(rules.get("warmups", 10))
    for _ in range(warmups):
        solver.factorize(stream_ptr)
        solver.solve(stream_ptr)
    wp.synchronize_device(device)

    paired = _PairedTimer(device)
    factor_samples, solve_samples, total_samples = _paired_loop(
        lambda: paired.measure(lambda: solver.factorize(stream_ptr),
                               lambda: solver.solve(stream_ptr)),
        rules)

    rec["warm_factor"] = summarize(factor_samples)
    rec["warm_solve"] = summarize(solve_samples)
    rec["warm_total"] = summarize(total_samples)
    rec["raw_factor_ms"] = factor_samples
    rec["raw_solve_ms"] = solve_samples
    rec["raw_total_ms"] = total_samples
    rec["warm_total_ms"] = rec["warm_total"]["median"]
    rec["warm_overhead_ms"] = 0.0
    rec["jit_s"] = 0.0
    rec["measurement"] = "paired_events"
    perm_bytes, peak_bytes = solver.memory_estimates()
    rec["cudss_memory_estimates"] = [perm_bytes, peak_bytes]
    buffer_bytes = (solver.values.capacity + solver.rhs.capacity * 2
                    + lower.indptr.nbytes + lower.indices.nbytes)
    rec["peak_device_bytes"] = int((peak_bytes or 0) + buffer_bytes)
    rec["solver_stats"] = solver.metadata()
    solver.free()
    return rec


def run_cpu_method(problem, method_def, rules, device):
    """CPU reference (structured NumPy solve); correctness anchor with
    coarse wall-clock timing, small/medium systems only."""
    from baselines import scipy_reference as ref
    spec = problem.spec
    max_dim = int(method_def.get("max_dimension", 50000))
    if spec.total_dimension > max_dim:
        return {"status": "skipped",
                "skip_reason": f"dimension {spec.total_dimension} exceeds "
                               f"CPU reference limit {max_dim}"}
    variant = method_def.get("variant", "structured")
    fn = {"structured": ref.solve_structured_cpu,
          "sparse": ref.solve_sparse_direct}[variant]
    rec = {"status": "ok"}
    t0 = time.perf_counter()
    xt, xr = fn(problem)
    rec["cpu_solve_s"] = time.perf_counter() - t0
    rec.update(validation.compute_metrics(problem, xt, xr))
    rec["warm_factor"] = {}
    rec["warm_solve"] = {}
    rec["warm_total_ms"] = rec["cpu_solve_s"] * 1e3
    return rec


RUNNERS = {"tree": run_tree_method, "cudss": run_cudss_method,
           "cpu": run_cpu_method}

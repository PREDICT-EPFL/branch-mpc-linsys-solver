"""Isolated linear-solver campaign: tree vs cuDSS (plan 3, section 10).

Runs the named campaigns A-H (FP64) and the FP32 subset with three
seeds, measuring warm numerical factorize() and warm solve(rhs, out=...)
in SEPARATE timed loops under the fairness rules of section 10.1
(identical values to both backends, symbolic analysis outside timed
regions, graphs on for the tree headline, batched CUDA events, no
per-repetition synchronization).  One raw JSON per
(campaign, point, seed) under results/plan3_campaign/raw (resumable:
existing records are skipped).

The sweeps, seeds, and measurement rules are configured in
experiments/plan3.yaml (--config to override).

Usage::

    python scripts/plan3_campaign.py [--smoke] [--campaigns A B ...]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# measurement rules; overwritten from the YAML config in main()
SEEDS = [0, 1, 2]
WARMUPS = 20
MIN_CALLS = 100
MIN_DEVICE_MS = 100.0
MAX_CALLS = 2000
BATCH = 100

_PARAM_KEYS = ("B", "N", "n_b", "n_r", "nrhs", "precision")


def _pt(B, T, n_b, n_r, nrhs=1, precision="float64"):
    return dict(B=B, T=T, n_b=n_b, n_r=n_r, nrhs=nrhs, precision=precision)


def load_config(path):
    """Parse the campaign YAML: returns (campaigns dict, seeds,
    measurement dict).  Campaign entries expand `grid` products and
    `points` lists over layered defaults; the horizon key is ``N`` in
    the config and ``T`` internally."""
    import itertools
    import yaml
    cfg = yaml.safe_load(Path(path).read_text())
    base = dict(cfg.get("defaults", {}))
    campaigns = {}
    for name, spec in cfg["campaigns"].items():
        defaults = {**base, **(spec.get("defaults") or {})}
        points = []
        grid = spec.get("grid")
        grids = grid if isinstance(grid, list) else ([grid] if grid else [])
        for g in grids:
            keys = list(g)
            for combo in itertools.product(*(g[k] for k in keys)):
                points.append({**defaults, **dict(zip(keys, combo))})
        for extra in spec.get("points") or []:
            points.append({**defaults, **extra})
        expanded = []
        for pt in points:
            unknown = set(pt) - set(_PARAM_KEYS)
            if unknown:
                raise ValueError(f"campaign {name}: unknown parameter(s) "
                                 f"{sorted(unknown)}")
            expanded.append(_pt(pt["B"], pt["N"], pt["n_b"], pt["n_r"],
                                nrhs=pt.get("nrhs", 1),
                                precision=pt.get("precision", "float64")))
        campaigns[str(name)] = expanded
    return campaigns, list(cfg.get("seeds", SEEDS)), \
        dict(cfg.get("measurement", {}))


class _EventTimer:
    """Per-call CUDA-event timing without per-repetition host sync:
    events are recorded around every call and read after one batch-level
    synchronization."""

    def __init__(self, device, batch):
        import warp as wp
        self._wp = wp
        self._dev = wp.get_device(device)
        self._events = [wp.Event(self._dev, enable_timing=True)
                        for _ in range(2 * batch)]

    def time_batch(self, fn, n):
        wp = self._wp
        ev = self._events
        for i in range(n):
            wp.record_event(ev[2 * i])
            fn()
            wp.record_event(ev[2 * i + 1])
        wp.synchronize_event(ev[2 * n - 1])
        return [float(wp.get_event_elapsed_time(ev[2 * i], ev[2 * i + 1],
                                                synchronize=False))
                for i in range(n)]


def _timed_loop(timer, fn):
    samples = []
    while (len(samples) < MIN_CALLS or sum(samples) < MIN_DEVICE_MS) \
            and len(samples) < MAX_CALLS:
        samples.extend(timer.time_batch(fn, BATCH))
    return samples


def _summ(samples):
    a = np.asarray(samples)
    return {"median_ms": float(np.median(a)),
            "p10_ms": float(np.percentile(a, 10)),
            "p90_ms": float(np.percentile(a, 90)),
            "mean_ms": float(a.mean()),
            "repetition_count": int(len(a))}


def run_point(campaign, point, seed, device="cuda:0"):
    import warp as wp
    from baselines import cudss as cudss_mod
    from baselines import reference
    from experiments.general_arrow.benchmarks.problems import ProblemSpec, generate_problem
    from src.general_arrow._utils import wp_dtype
    from src.general_arrow.problem import TreeVector
    from src.general_arrow.solver import Solver

    spec = ProblemSpec(num_tails=point["B"], horizon=point["T"],
                       block_size=point["n_b"], root_dim=point["n_r"],
                       num_rhs=point["nrhs"], precision=point["precision"],
                       seed=seed)
    problem = generate_problem(spec, estimate_condition=False)
    lower = problem.matrix.to_csr_lower(dtype=spec.np_dtype)
    dt = wp_dtype(spec.precision)
    B, T, n_b, n_r = spec.shape.dims()
    nrhs = spec.num_rhs
    base = {**point, "seed": seed, "campaign": campaign,
            "total_dimension": spec.shape.total_dimension,
            "nnz": int(lower.nnz), "warmup_count": WARMUPS,
            "graph_enabled": True, "cudss_ordering": "default"}
    timer = _EventTimer(device, BATCH)
    out_rec = {"point": base, "backends": {}}

    # ------------------------------------------------------------- tree
    solver = Solver(spec.shape, device=device)
    solver.update(problem.matrix)          # outside the timed region
    solver.factorize()                     # compile + capture
    rhs = TreeVector(
        spec.shape,
        wp.array(np.ascontiguousarray(problem.rhs.tail), dtype=dt,
                 device=device),
        wp.array(np.ascontiguousarray(problem.rhs.root), dtype=dt,
                 device=device))
    out = TreeVector(spec.shape,
                     wp.zeros((B, T, n_b, nrhs), dtype=dt, device=device),
                     wp.zeros((n_r, nrhs), dtype=dt, device=device))
    solver.solve(rhs, out=out)             # bind + capture
    for _ in range(WARMUPS):
        solver.factorize()
    wp.synchronize_device(device)
    tree_factor = _timed_loop(timer, solver.factorize)
    for _ in range(WARMUPS):
        solver.solve(rhs, out=out)
    wp.synchronize_device(device)
    tree_solve = _timed_loop(timer, lambda: solver.solve(rhs, out=out))
    # residual outside timed regions, with a fresh RHS (solves consumed it)
    wp.copy(rhs.tail, wp.array(np.ascontiguousarray(problem.rhs.tail),
                                 dtype=dt, device=device))
    wp.copy(rhs.root,
            wp.array(np.ascontiguousarray(problem.rhs.root), dtype=dt,
                     device=device))
    solver.solve(rhs, out=out)
    x_tree = out.numpy()
    tree_metrics = reference.compute_metrics(problem, x_tree.tail,
                                             x_tree.root)
    out_rec["backends"]["tree"] = {
        "factor": _summ(tree_factor), "solve": _summ(tree_solve),
        "relative_residual": tree_metrics["rhs_relative_residual"],
        "status": "ok" if not tree_metrics["nan_or_inf"] else "nan"}

    # ------------------------------------------------------------ cuDSS
    if cudss_mod.CUDSS_AVAILABLE:
        flat = problem.rhs.flat()
        flat = flat[:, 0] if nrhs == 1 else flat
        cd = cudss_mod.CudssCholesky(lower, flat, precision=spec.precision,
                                     ordering="default", ir_steps=0,
                                     device=device)
        stream_ptr = wp.get_device(device).stream.cuda_stream
        cd.plan(stream_ptr)                 # symbolic, outside timing
        cd.factorize(stream_ptr)
        cd.solve(stream_ptr)
        wp.synchronize_device(device)
        for _ in range(WARMUPS):
            cd.factorize(stream_ptr)
        wp.synchronize_device(device)
        cd_factor = _timed_loop(timer, lambda: cd.factorize(stream_ptr))
        for _ in range(WARMUPS):
            cd.solve(stream_ptr)
        wp.synchronize_device(device)
        cd_solve = _timed_loop(timer, lambda: cd.solve(stream_ptr))
        z = TreeVector.from_flat(spec.shape, cd.solution())
        cd_metrics = reference.compute_metrics(problem, z.tail,
                                               z.root)
        scale = max(float(np.abs(z.tail).max()), 1.0)
        disagreement = max(
            float(np.abs(x_tree.tail - z.tail).max()),
            float(np.abs(x_tree.root
                         - z.root.reshape(x_tree.root.shape)
                         ).max())) / scale
        out_rec["backends"]["cudss"] = {
            "factor": _summ(cd_factor), "solve": _summ(cd_solve),
            "relative_residual": cd_metrics["rhs_relative_residual"],
            "status": "ok" if not cd_metrics["nan_or_inf"] else "nan"}
        out_rec["tree_cudss_disagreement"] = disagreement
        cd.free()
    else:
        out_rec["backends"]["cudss"] = {
            "status": "unavailable",
            "skip_reason": cudss_mod.CUDSS_UNAVAILABLE_REASON}
    return out_rec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--campaigns", nargs="+", default=None)
    p.add_argument("--config", default="experiments/plan3.yaml")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", default="results/plan3_campaign/raw")
    args = p.parse_args()

    global SEEDS, WARMUPS, MIN_CALLS, MIN_DEVICE_MS, MAX_CALLS, BATCH
    all_c, SEEDS, meas = load_config(args.config)
    WARMUPS = int(meas.get("warmups", WARMUPS))
    MIN_CALLS = int(meas.get("min_calls", MIN_CALLS))
    MIN_DEVICE_MS = float(meas.get("min_device_ms", MIN_DEVICE_MS))
    MAX_CALLS = int(meas.get("max_calls", MAX_CALLS))
    BATCH = int(meas.get("batch", BATCH))

    import warp as wp
    wp.init()
    from experiments.general_arrow.benchmarks import metadata
    md = metadata.collect_metadata(args.device)

    if args.smoke:
        todo = {"smoke": [_pt(11, 16, 8, 2), _pt(41, 64, 8, 2)]}
        seeds = [0]
    else:
        names = args.campaigns or list(all_c)
        todo = {k: all_c[k] for k in names}
        seeds = SEEDS
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_done = n_skip = 0
    for campaign, points in todo.items():
        for point in points:
            for seed in seeds:
                key = ("{campaign}_B{B}_T{T}_n{n_b}_m{n_r}_q{nrhs}_"
                       "{precision}_s{seed}").format(campaign=campaign,
                                                     seed=seed, **point)
                path = out_dir / f"{key}.json"
                if path.exists():
                    n_skip += 1
                    continue
                t0 = time.perf_counter()
                rec = run_point(campaign, point, seed, args.device)
                rec["meta"] = {k: md.get(k) for k in
                               ("git_commit", "git_dirty", "warp",
                                "socu_version", "socu_commit",
                                "cudss_version", "gpu_name")}
                rec["wall_s"] = time.perf_counter() - t0
                path.write_text(json.dumps(rec, indent=1, default=float))
                tr = rec["backends"]["tree"]
                cu = rec["backends"].get("cudss", {})
                spd_f = (cu.get("factor", {}).get("median_ms", float("nan"))
                         / tr["factor"]["median_ms"])
                spd_s = (cu.get("solve", {}).get("median_ms", float("nan"))
                         / tr["solve"]["median_ms"])
                print(f"{key:44s} tree f={tr['factor']['median_ms']:8.4f} "
                      f"s={tr['solve']['median_ms']:8.4f}  "
                      f"speedup f={spd_f:5.2f} s={spd_s:5.2f}")
                n_done += 1
    print(f"done: {n_done} new, {n_skip} skipped (resume)")


if __name__ == "__main__":
    main()

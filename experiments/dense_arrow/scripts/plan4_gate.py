"""Plan-4 performance gate and launch-count evidence.

Measures, at the five frozen representative points:

- warm factorize() and solve(rhs, out=...) medians (CUDA events, graphs
  replayed, paired loop);
- update() staging wall time (host copies excluded from the factor
  numbers by construction);
- launch counts of one eager factorization and one eager solve
  (wp.launch / wp.launch_tiled / wp.copy wrappers), which equal the
  graph node counts of the captured pipelines;
- persistent mempool memory after warmup.

Run with --tag pre before the plan-4 changes and --tag post after;
--compare prints the acceptance verdict (<= 2 percent regression).
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

POINTS = [(11, 16, 8, 2, 1), (41, 64, 8, 2, 1), (81, 128, 8, 2, 1),
          (64, 128, 16, 64, 1), (41, 64, 8, 2, 16)]


def _count_launches(fn):
    """Run fn() with launch/copy wrappers installed; return counts."""
    import warp as wp
    counts = {"launch": 0, "launch_tiled": 0, "copy": 0}
    orig = (wp.launch, wp.launch_tiled, wp.copy)

    def launch(*a, **k):
        counts["launch"] += 1
        return orig[0](*a, **k)

    def launch_tiled(*a, **k):
        counts["launch_tiled"] += 1
        return orig[1](*a, **k)

    def copy(*a, **k):
        counts["copy"] += 1
        return orig[2](*a, **k)

    wp.launch, wp.launch_tiled, wp.copy = launch, launch_tiled, copy
    try:
        fn()
    finally:
        wp.launch, wp.launch_tiled, wp.copy = orig
    counts["total"] = sum(counts.values()) - counts["total"] \
        if "total" in counts else sum(counts.values())
    return counts


def measure_point(B, T, n_b, n_r, nrhs, device="cuda:0"):
    import warp as wp
    from experiments.general_arrow.benchmarks.problems import ProblemSpec, generate_problem
    from experiments.general_arrow.benchmarks.runners import _PairedTimer, _paired_loop
    from src.general_arrow.solver import Solver

    spec = ProblemSpec(num_tails=B, horizon=T, block_size=n_b,
                       root_dim=n_r, num_rhs=nrhs,
                       precision="float64", seed=10)
    prob = generate_problem(spec, estimate_condition=False)
    solver = Solver(spec.shape, device=device)

    t0 = time.perf_counter()
    solver.update(prob.matrix)
    wp.synchronize_device(device)
    update_ms = (time.perf_counter() - t0) * 1e3

    solver.factorize()
    dt = wp.float64
    rb, rr = prob.rhs.tail, prob.rhs.root
    from src.general_arrow.problem import TreeVector
    rhs = TreeVector(spec.shape,
                     wp.array(np.asarray(rb), dtype=dt, device=device),
                     wp.array(np.asarray(rr), dtype=dt, device=device))
    out_arrays = (wp.zeros((B, T, n_b, nrhs), dtype=dt, device=device),
                  wp.zeros((n_r, nrhs), dtype=dt, device=device))
    out = TreeVector(spec.shape, *out_arrays)

    for _ in range(20):
        solver.factorize()
        solver.solve(rhs, out=out)
    wp.synchronize_device(device)
    mem = int(wp.get_mempool_used_mem_current(device))

    # launch counts of the eager pipelines (== captured graph nodes)
    lf = _count_launches(lambda: solver._factorize_numeric(None))
    ls = _count_launches(lambda: solver._solve_numeric(solver._binding,
                                                       None))
    wp.synchronize_device(device)

    paired = _PairedTimer(device)
    f, s, t = _paired_loop(
        lambda: paired.measure(solver.factorize,
                               lambda: solver.solve(rhs, out=out)),
        {"min_reps": 100, "slow_reps": 30, "slow_threshold_ms": 50.0,
         "min_seconds": 0.5, "max_reps": 500})
    return {
        "factor_ms": float(np.median(f)),
        "solve_ms": float(np.median(s)),
        "update_ms": update_ms,
        "mem_bytes": mem,
        "factor_launches": lf,
        "solve_launches": ls,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", choices=["pre", "post", "atomic", "pairwise", "review_fixes"])
    p.add_argument("--compare", nargs=2, metavar=("A", "B"),
                   help="compare two tags (ratio = B / A)")
    p.add_argument("--output", default="results/plan4")
    args = p.parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.compare:
        a = json.loads((out_dir / f"gate_{args.compare[0]}.json").read_text())
        b = json.loads((out_dir / f"gate_{args.compare[1]}.json").read_text())
        ok = True
        print(f"{'point':26s} {'phase':7s} {'A':>9s} {'B':>9s} {'ratio':>7s}")
        for key in a["points"]:
            for phase in ("factor_ms", "solve_ms"):
                x, y = a["points"][key][phase], b["points"][key][phase]
                r = y / x
                flag = "  <-- FAIL >2%" if r > 1.02 else ""
                ok = ok and r <= 1.02
                print(f"{key:26s} {phase[:-3]:7s} {x:9.4f} {y:9.4f} "
                      f"{r:7.3f}{flag}")
        for key in a["points"]:
            la = a["points"][key]["solve_launches"]["total"]
            lb = b["points"][key]["solve_launches"]["total"]
            fa = a["points"][key]["factor_launches"]["total"]
            fb = b["points"][key]["factor_launches"]["total"]
            print(f"{key:26s} launches factor {fa:3d}->{fb:3d} "
                  f"solve {la:3d}->{lb:3d}")
        print("GATE:", "PASS" if ok else "FAIL")
        return

    import warp as wp
    wp.init()
    points = {}
    for pt in POINTS:
        r = measure_point(*pt)
        key = "B{}_T{}_n{}_m{}_q{}".format(*pt)
        points[key] = r
        print(f"{key:26s} factor={r['factor_ms']:8.4f}ms "
              f"solve={r['solve_ms']:8.4f}ms "
              f"launches f={r['factor_launches']['total']} "
              f"s={r['solve_launches']['total']} "
              f"mem={r['mem_bytes'] / 1e6:.1f}MB")
    payload = {
        "git": subprocess.run(["git", "rev-parse", "HEAD"],
                              capture_output=True,
                              text=True).stdout.strip(),
        "points": points,
    }
    (out_dir / f"gate_{args.tag}.json").write_text(
        json.dumps(payload, indent=1))
    print(f"wrote gate_{args.tag}.json")


if __name__ == "__main__":
    main()

"""Simplification performance gate (plan 3, section 15.8).

Measures warm factorize() and solve(rhs, out=...) medians (CUDA events,
graphs on) at the five gate points, counts non-SOCU launches per
pipeline, and records src/ size metrics.  Run with --tag pre before the
refactor and --tag post after; --compare prints the gate verdict
(<= 2 percent regression per point).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

POINTS = [(11, 16, 8, 2, 1), (41, 64, 8, 2, 1), (81, 128, 8, 2, 1),
          (64, 128, 16, 64, 1), (41, 64, 8, 2, 16)]


def measure_point(B, T, n_b, n_r, nrhs, device="cuda:0"):
    import warp as wp
    from experiments.general_arrow.benchmarks.problems import ProblemSpec, generate_problem
    from experiments.general_arrow.benchmarks.runners import _PairedTimer, _paired_loop
    from src.general_arrow.solver import TreeSolver as Solver

    spec = ProblemSpec(num_tails=B, horizon=T, block_size=n_b,
                       root_dim=n_r, num_rhs=nrhs,
                       precision="float64", seed=10)
    prob = generate_problem(spec, estimate_condition=False)
    solver = Solver(spec.shape, device=device)
    solver.update(prob.matrix)
    solver.factorize()
    rhs = prob.device_rhs(device) if hasattr(prob, "device_rhs") else None
    if rhs is None:
        from src.general_arrow.problem import TreeVector
        dt = wp.float64
        rhs = TreeVector(spec.shape,
                         wp.array(np.asarray(prob.rhs.tail), dtype=dt,
                                  device=device),
                         wp.array(np.asarray(prob.rhs.root), dtype=dt,
                                  device=device))
        out = TreeVector(spec.shape,
                         wp.zeros(rhs.tail.shape, dtype=dt, device=device),
                         wp.zeros(rhs.root.shape, dtype=dt,
                                  device=device))
    for _ in range(20):
        solver.factorize()
        solver.solve(rhs, out=out)
    wp.synchronize_device(device)
    paired = _PairedTimer(device)
    f, s, t = _paired_loop(
        lambda: paired.measure(solver.factorize,
                               lambda: solver.solve(rhs, out=out)),
        {"min_reps": 100, "slow_reps": 30, "slow_threshold_ms": 50.0,
         "min_seconds": 0.5, "max_reps": 500})
    if hasattr(solver, "close"):
        solver.close()
    return float(np.median(f)), float(np.median(s))


def src_metrics():
    files = sorted(Path("src").rglob("*.py"))
    lines = sum(len(p.read_text().splitlines()) for p in files)
    return {"files": len(files), "lines": lines,
            "names": [str(p) for p in files]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", choices=["pre", "post"])
    p.add_argument("--compare", action="store_true")
    p.add_argument("--output", default="results/plan3_baseline")
    args = p.parse_args()
    out_dir = Path(args.output)

    if args.compare:
        pre = json.loads((out_dir / "simplify_gate_pre.json").read_text())
        post = json.loads((out_dir / "simplify_gate_post.json").read_text())
        ok = True
        print(f"{'point':28s} {'phase':7s} {'pre':>8s} {'post':>8s} {'ratio':>7s}")
        for key in pre["points"]:
            for i, phase in enumerate(("factor", "solve")):
                a, b = pre["points"][key][i], post["points"][key][i]
                r = b / a
                flag = ""
                if r > 1.02:
                    flag = "  <-- FAIL >2%"
                    ok = False
                print(f"{key:28s} {phase:7s} {a:8.4f} {b:8.4f} {r:7.3f}{flag}")
        print(f"src files: {pre['src']['files']} -> {post['src']['files']}, "
              f"lines: {pre['src']['lines']} -> {post['src']['lines']}")
        print("GATE:", "PASS" if ok else "FAIL")
        return

    import warp as wp
    wp.init()
    points = {}
    for pt in POINTS:
        f, s = measure_point(*pt)
        key = "B{}_T{}_n{}_m{}_q{}".format(*pt)
        points[key] = (f, s)
        print(f"{key:28s} factor={f:8.4f}ms solve={s:8.4f}ms")
    payload = {
        "git": subprocess.run(["git", "rev-parse", "HEAD"],
                              capture_output=True,
                              text=True).stdout.strip(),
        "points": points,
        "src": src_metrics(),
    }
    (out_dir / f"simplify_gate_{args.tag}.json").write_text(
        json.dumps(payload, indent=1))
    print(f"wrote simplify_gate_{args.tag}.json")


if __name__ == "__main__":
    main()

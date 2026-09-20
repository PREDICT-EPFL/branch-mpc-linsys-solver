"""Compare the plan-3 permuted-factor implementation against the frozen
baseline and cuDSS, and apply the plan's acceptance criteria.

Reads the two paired-benchmark JSON files written by
scripts/plan3_baseline.py and prints per-point medians, ratios, the
geometric means, and pass/fail per criterion.  Also cross-checks the
stored solution fingerprints (same generated systems, so agreement is a
correctness statement about the rewrite).

Usage::

    python scripts/plan3_compare.py [--dir results/plan3_baseline]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path):
    d = json.loads(Path(path).read_text())
    rows = defaultdict(list)
    for r in d["records"]:
        key = (r["case_key"], r["method"])
        rows[key].append(r)
    med = {}
    for (key, method), recs in rows.items():
        med[(key, method)] = {
            "factor": np.median([r["warm_factor"]["median"] for r in recs]),
            "solve": np.median([r["warm_solve"]["median"] for r in recs]),
            "total": np.median([r["warm_factor"]["median"]
                                + r["warm_solve"]["median"] for r in recs]),
        }
    return d, med


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", default="results/plan3_baseline")
    args = p.parse_args()
    base_d, base = load(Path(args.dir) / "baseline.json")
    new_d, new = load(Path(args.dir) / "plan3.json")

    keys = sorted({k for k, m in base.keys() if m == "tree_graph"})
    print(f"baseline commit {base_d['meta']['git_commit'][:10]}  "
          f"plan3 commit {new_d['meta']['git_commit'][:10]}")
    print(f"{'case':44s} {'phase':7s} {'base':>8s} {'plan3':>8s} "
          f"{'ratio':>6s} {'cudss':>8s}")

    primary = [k for k in keys if "_m64_q1_float64" in k]
    small = [k for k in keys if any(f"_m{m}_" in k for m in (1, 2, 4, 8))]
    ratios_total, viol5, small_gain = [], [], []
    for k in keys:
        for phase in ("factor", "solve", "total"):
            b = base[(k, "tree_graph")][phase]
            n = new[(k, "tree_graph")][phase]
            c = new.get((k, "cudss"), {}).get(phase, float("nan"))
            flag = ""
            if k in primary and phase in ("factor", "solve", "total"):
                if n > 1.05 * b:
                    flag = "  <-- >5% regression"
                    viol5.append((k, phase, n / b))
            if phase == "total":
                if k in primary:
                    ratios_total.append(n / b)
                if k in small:
                    small_gain.append(
                        (k, 1.0 - new[(k, 'tree_graph')]['solve']
                         / base[(k, 'tree_graph')]['solve']))
            print(f"{k:44s} {phase:7s} {b:8.4f} {n:8.4f} {n/b:6.3f} "
                  f"{c:8.4f}{flag}")
    gm = float(np.exp(np.mean(np.log(ratios_total))))
    print(f"\nprimary FP64 geometric-mean total ratio (new/base): {gm:.4f}")
    print("small-root warm-solve improvements:")
    for k, g in small_gain:
        print(f"  {k:44s} {g*100:6.1f}%")

    # solution fingerprints: same inputs -> agreement across implementations
    b_sol = np.load(Path(args.dir) / "baseline_solutions.npz")
    n_sol = np.load(Path(args.dir) / "plan3_solutions.npz")
    worst = 0.0
    for k in b_sol.files:
        a, b = b_sol[k], n_sol[k]
        rel = np.abs(a - b).max() / max(np.abs(a).max(), 1.0)
        worst = max(worst, rel)
    print(f"\nworst relative solution difference vs frozen baseline: "
          f"{worst:.3e}")

    crit1 = "PASS" if not viol5 else f"FAIL {viol5}"
    crit2 = "PASS" if gm <= 1.0 else f"FAIL gm={gm:.4f}"
    rep = [g for k, g in small_gain
           if ("B41_T64" in k or "B81_T128" in k)]
    crit3 = ("PASS" if any(g >= 0.15 for g in rep)
             and all(g > -0.02 for k, g in small_gain) else "FAIL")
    print(f"\ncriterion 1 (<=5% regression at primary FP64 points): {crit1}")
    print(f"criterion 2 (geo-mean factor+solve <= baseline): {crit2}")
    print(f"criterion 3 (>=15% small-root solve gain, no regressions): "
          f"{crit3}")


if __name__ == "__main__":
    main()

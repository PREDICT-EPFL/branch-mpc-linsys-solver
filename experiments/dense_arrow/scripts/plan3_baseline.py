"""Plan-3 paired baseline: freeze/compare evidence for the permuted-factor
rewrite.

Runs the tree solver (with and without CUDA graphs) and cuDSS on the
plan-3 point grid with identical generated matrices and right-hand
sides, using the campaign's paired CUDA-event measurement, and writes
one JSON per tag plus solution fingerprints for later agreement tests.

Reproduce the frozen baseline::

    python scripts/plan3_baseline.py --tag baseline

After the rewrite, run with ``--tag plan3`` and compare the two JSON
files (scripts/plan3_compare.py).
"""

import argparse
import hashlib
import sys
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _git(*args):
    out = subprocess.run(["git", *args], capture_output=True, text=True)
    return out.stdout.strip()


# (B, T, n_b, n_r) points; nrhs=1 unless listed in MULTI_RHS
PRIMARY_FP64 = [(11, 32, 8, 64), (21, 64, 8, 64), (41, 96, 8, 64),
                (81, 128, 8, 64)]
SMALL_ROOT = [(41, 64, 8, 1), (41, 64, 8, 2), (41, 64, 8, 4),
              (41, 64, 8, 8), (81, 128, 8, 2)]
MULTI_RHS = [(41, 96, 8, 64, q) for q in (4, 16, 64)]
SEEDS = [10, 11, 12]
RULES = {"warmups": 10, "min_reps": 30, "slow_reps": 10,
         "slow_threshold_ms": 200.0, "min_seconds": 0.5, "max_reps": 200,
         "phase_reps": 5}


def _cases():
    for B, T, n, m in PRIMARY_FP64:
        yield dict(num_tails=B, horizon=T, block_size=n,
                   root_dim=m, num_rhs=1, precision="float64")
    for B, T, n, m in PRIMARY_FP64:
        yield dict(num_tails=B, horizon=T, block_size=n,
                   root_dim=m, num_rhs=1, precision="float32")
    for B, T, n, m in SMALL_ROOT:
        yield dict(num_tails=B, horizon=T, block_size=n,
                   root_dim=m, num_rhs=1, precision="float64")
    for B, T, n, m, q in MULTI_RHS:
        yield dict(num_tails=B, horizon=T, block_size=n,
                   root_dim=m, num_rhs=q, precision="float64")


def _fingerprint(problem, device):
    """Solve once with the tree solver and return solution digest data
    (fixed inputs, so the digest is comparable across implementations up
    to FP tolerance; store norms plus a low-precision hash)."""
    from src.dense_arrow.solver import Solver
    solver = Solver(problem.spec.shape, device=device)
    solver.update(problem.matrix)
    solver.factorize()
    sol = solver.solve(problem.rhs).numpy()
    flat = np.concatenate([sol.tail.ravel(), sol.root.ravel()])
    rounded = np.round(flat.astype(np.float64), 8)
    return {
        "norm": float(np.linalg.norm(flat)),
        "hash8": hashlib.sha256(rounded.tobytes()).hexdigest()[:16],
        "solution": flat,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", required=True, help="baseline or plan3")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", default="results/plan3_baseline")
    args = p.parse_args()

    import warp as wp
    wp.init()
    from experiments.dense_arrow.benchmarks.problems import ProblemSpec, generate_problem
    from experiments.dense_arrow.benchmarks.runners import run_tree_method, run_cudss_method

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "tag": args.tag,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "warp_version": wp.__version__,
        "device_name": wp.get_device(args.device).name,
        "platform": platform.platform(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rules": RULES,
        "seeds": SEEDS,
    }
    records = []
    solutions = {}
    for case in _cases():
        for seed in SEEDS:
            spec = ProblemSpec(seed=seed, **case)
            problem = generate_problem(spec, estimate_condition=False)
            key = (f"B{spec.num_tails}_T{spec.horizon}"
                   f"_n{spec.block_size}_m{spec.root_dim}"
                   f"_q{spec.num_rhs}_{spec.precision}_s{seed}")
            # graphs are the only production mode now; keep the
            # historical "tree_graph" label so records stay comparable
            for method, runner, mdef in (
                    ("tree_graph", run_tree_method, {"kind": "tree"}),
                    ("cudss", run_cudss_method,
                     {"kind": "cudss", "ordering": "default"})):
                rec = runner(problem, mdef, RULES, args.device)
                rec.update(case_key=key, method=method, seed=seed, **case)
                rec.pop("raw_factor_ms", None)
                rec.pop("raw_solve_ms", None)
                rec.pop("raw_total_ms", None)
                records.append(rec)
                print(f"{key:44s} {method:10s} "
                      f"factor={rec['warm_factor'].get('median', float('nan')):8.4f}ms "
                      f"solve={rec['warm_solve'].get('median', float('nan')):8.4f}ms")
            if seed == SEEDS[0]:
                fp = _fingerprint(problem, args.device)
                solutions[key] = fp.pop("solution")
                meta.setdefault("fingerprints", {})[key] = fp

    payload = {"meta": meta, "records": records}
    json_path = out_dir / f"{args.tag}.json"
    json_path.write_text(json.dumps(payload, indent=1, default=float))
    np.savez_compressed(out_dir / f"{args.tag}_solutions.npz", **solutions)
    print(f"\nwrote {json_path} ({len(records)} records) and solution "
          f"fingerprints")


if __name__ == "__main__":
    main()

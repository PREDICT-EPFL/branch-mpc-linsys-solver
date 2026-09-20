"""Plan-admm-2 measurement campaign for the optimized ADMM warm path.

Runs three experiment groups and writes one JSON file per group under
``results/admm2/``:

1. ``matrix.json``: the required benchmark matrix around the headline
   case (B=41, T=64, nx=8, nu=2, FP64, tree) -- a tail-count sweep, a
   horizon sweep, and a block-size sweep, each case reporting the cold
   first solve, warm fixed-iteration medians (eager and graph), phase
   breakdown, launch counts, allocation delta, and solution quality for
   both the tree and cuDSS linear systems;
2. ``check_every.json``: fixed 400-iteration warm runs of the headline
   case at check_every 10, 25, and 400, separating cost per check from
   amortized cost per iteration;
3. ``alpha.json``: iterations and warm time to solution of the headline
   case under over-relaxation alpha in {1.0, 1.4, 1.6, 1.8}.

Usage::

    python scripts/admm2_campaign.py [--output results/admm2]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from admm import Settings, Solver  # noqa: E402
from experiments.dense_arrow.benchmarks.admm import run_case, _sync  # noqa: E402
from experiments.dense_arrow.benchmarks.problems import generate_scenario_qp  # noqa: E402

HEADLINE = dict(num_tails=41, num_stages=64, nx=8, nu=2)

MATRIX = [
    # tail-count sweep at the headline horizon/block
    dict(B=11, T=64, nx=8, nu=2), dict(B=21, T=64, nx=8, nu=2),
    dict(B=41, T=64, nx=8, nu=2), dict(B=81, T=64, nx=8, nu=2),
    # horizon sweep at the headline tail count
    dict(B=41, T=16, nx=8, nu=2), dict(B=41, T=128, nx=8, nu=2),
    # block-size sweep (nx + nu in {8, 10, 16})
    dict(B=41, T=64, nx=6, nu=2), dict(B=41, T=64, nx=12, nu=4),
]


def run_matrix(out_dir):
    records = []
    for pt in MATRIX:
        for impl in ("tree", "cudss"):
            try:
                rec, solver, _ = run_case(pt["B"], pt["T"], pt["nx"],
                                          pt["nu"], impl, max_iter=400,
                                          fixed_iters=200)
                solver.close()
            except Exception as exc:  # noqa: BLE001 (cuDSS not critical)
                rec = {"impl": impl, **pt, "status": f"failed: {exc}"}
            records.append(rec)
            ph = rec.get("phases")
            if ph:
                g = ph["graph_us_per_iter"]
                gtxt = f" graph={g:.1f}us" if g is not None else ""
                print(f"B={pt['B']:3d} T={pt['T']:4d} nb={pt['nx']+pt['nu']} "
                      f"{impl:6s}: warm_iter={ph['total_us_per_iter']:.1f}us"
                      f"{gtxt} solve={rec['linear_solve_ms']*1e3:.1f}us "
                      f"iters={rec['iterations']}")
            else:
                print(f"B={pt['B']:3d} {impl}: {rec.get('status')}")
    (out_dir / "matrix.json").write_text(
        json.dumps(records, indent=1, default=str))
    print(f"wrote {out_dir / 'matrix.json'}")


def _fixed_solver(P, q, A, l, u, max_iter, check_every):
    # eps = 0 never converges, so every run performs exactly max_iter
    # iterations regardless of the check period
    s = Solver()
    s.setup(P, q, A, l, u,
            settings=Settings(rho=1.0, rho_eq_scale=1000.0,
                              max_iter=max_iter, eps_abs=0.0, eps_rel=0.0,
                              check_every=check_every,
                              linear_solver="tree"))
    return s


def run_check_every(out_dir, max_iter=400, samples=20):
    P, q, A, l, u, _ = generate_scenario_qp(**HEADLINE, seed=10)
    n, m = P.shape[0], A.shape[0]
    zeros = dict(x=np.zeros(n), z=np.zeros(m), dual=np.zeros(m))
    records = []
    for check_every in (10, 25, max_iter):
        solver = _fixed_solver(P, q, A, l, u, max_iter, check_every)
        solver.solve()  # cold: graphs + compile
        wall = []
        for _ in range(samples):
            solver.warm_start(**zeros)
            _sync()
            t0 = time.perf_counter()
            solver.solve(copy_result=False)
            _sync()
            wall.append(time.perf_counter() - t0)
        ph = solver.time_iteration_phases(iters=200, repeats=10)
        solver.close()
        med = float(np.median(wall))
        checks = max_iter // check_every
        rec = {
            "check_every": check_every,
            "max_iter": max_iter,
            "checks": checks,
            "warm_wall_s": med,
            "us_per_iter": med * 1e6 / max_iter,
            "check_total_ms": ph["check_total_ms"],
            "check_gpu_ms": ph["check_gpu_ms"],
            "uncheckd_us_per_iter": ph["graph_us_per_iter"]
            or ph["total_us_per_iter"],
        }
        records.append(rec)
        print(f"check_every={check_every:4d}: {rec['us_per_iter']:.1f} "
              f"us/iter over {max_iter} iters "
              f"(check={ph['check_total_ms']*1e3:.1f}us x {checks})")
    (out_dir / "check_every.json").write_text(
        json.dumps(records, indent=1))
    print(f"wrote {out_dir / 'check_every.json'}")


def run_alpha(out_dir, samples=10):
    P, q, A, l, u, _ = generate_scenario_qp(**HEADLINE, seed=10)
    n, m = P.shape[0], A.shape[0]
    zeros = dict(x=np.zeros(n), z=np.zeros(m), dual=np.zeros(m))
    records = []
    for alpha in (1.0, 1.4, 1.6, 1.8):
        solver = Solver()
        solver.setup(P, q, A, l, u,
                     settings=Settings(rho=1.0, rho_eq_scale=1000.0,
                                       max_iter=2000, eps_abs=1e-4,
                                       eps_rel=1e-4, check_every=10,
                                       alpha=alpha, linear_solver="tree"))
        res = solver.solve()  # cold
        wall = []
        for _ in range(samples):
            solver.warm_start(**zeros)
            _sync()
            t0 = time.perf_counter()
            res = solver.solve(copy_result=False)
            _sync()
            wall.append(time.perf_counter() - t0)
        rec = {
            "alpha": alpha,
            "iterations": res.info.iterations,
            "converged": res.info.converged,
            "warm_time_to_solution_ms": float(np.median(wall)) * 1e3,
            "primal_residual": res.info.primal_residual,
            "dual_residual": res.info.dual_residual,
        }
        solver.close()
        records.append(rec)
        print(f"alpha={alpha:.1f}: iters={rec['iterations']:4d} "
              f"time={rec['warm_time_to_solution_ms']:.2f}ms "
              f"conv={rec['converged']}")
    (out_dir / "alpha.json").write_text(json.dumps(records, indent=1))
    print(f"wrote {out_dir / 'alpha.json'}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="results/admm2")
    p.add_argument("--skip", nargs="*", default=[],
                   choices=["matrix", "check_every", "alpha"])
    args = p.parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    if "matrix" not in args.skip:
        run_matrix(out_dir)
    if "check_every" not in args.skip:
        run_check_every(out_dir)
    if "alpha" not in args.skip:
        run_alpha(out_dir)


if __name__ == "__main__":
    main()

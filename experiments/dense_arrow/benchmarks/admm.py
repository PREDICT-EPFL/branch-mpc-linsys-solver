"""Unified end-to-end ADMM benchmark: tree vs cuDSS linear systems.

Runs the identical scenario QP (same CSC input, rho, ADMM kernels,
initial state, stopping rule, FP64) once per explicit linear-system
choice and reports cold and warm scopes separately:

1. application-facing times: symbolic/setup work, the cold first solve
   (includes kernel compilation and CUDA graph construction; never
   divided by the iteration count), and the warm convergence-based time
   to solution from a restored initial state;
2. warm fixed-iteration throughput and the per-phase breakdown from
   ``Solver.time_iteration_phases`` (eager and captured-graph paths),
   plus host launch counts and the device-allocation delta of a warm
   solve;
3. numerical-kernel times: factorization and the prepared per-iteration
   linear solve (non-mutating instrumentation);
4. with ``--mpc N``, an MPC-style sequence of N steps, each performing a
   full matrix-value ``update()`` (asynchronous; timed to completion)
   followed by a warm ``solve()``.

Usage::

    python benchmarks/admm.py [--tails 11 21 41] [--stages 16]
                              [--mpc 20] [--json out]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from admm import Settings, Solver  # noqa: E402
from admm.problem import analyze_tree_structure, build_plan  # noqa: E402
from experiments.dense_arrow.benchmarks.problems import generate_scenario_qp  # noqa: E402


def _sync():
    import warp as wp
    wp.synchronize()


def _settings(impl, rho=1.0, rho_eq_scale=1000.0, max_iter=1000,
              check_every=10, alpha=1.0, iteration_graph=True):
    return Settings(rho=rho, rho_eq_scale=rho_eq_scale, max_iter=max_iter,
                    eps_abs=1e-4, eps_rel=1e-4, check_every=check_every,
                    alpha=alpha, iteration_graph=iteration_graph,
                    linear_solver=impl)


class _LaunchCounter:
    """Count the host-side enqueue calls (kernel launches, graph
    replays, copies) issued inside the ``with`` block."""

    def __init__(self):
        import warp as wp
        self._wp = wp
        self.count = 0

    def __enter__(self):
        wp = self._wp
        self._saved = (wp.launch, wp.capture_launch, wp.copy)

        def wrap(fn):
            def inner(*a, **k):
                self.count += 1
                return fn(*a, **k)
            return inner

        wp.launch, wp.capture_launch, wp.copy = map(wrap, self._saved)
        return self

    def __exit__(self, *exc):
        (self._wp.launch, self._wp.capture_launch,
         self._wp.copy) = self._saved
        return False


def run_case(B, T, nx, nu, impl, seed=10, repeats=50, max_iter=1000,
             check_every=10, rho=1.0, rho_eq_scale=1000.0, alpha=1.0,
             iteration_graph=True, fixed_iters=200, throughput_repeats=20):
    import warp as wp
    P, q, A, l, u, meta = generate_scenario_qp(B, T, nx, nu, seed=seed)
    n, m = P.shape[0], A.shape[0]
    rec = {"impl": impl, "B": B, "T": T, "nx": nx, "nu": nu,
           "n": n, "m": m, "alpha": alpha,
           "iteration_graph": iteration_graph}
    zeros = dict(x=np.zeros(n), z=np.zeros(m), dual=np.zeros(m))

    t0 = time.perf_counter()
    plan = build_plan(P, A)
    rec["symbolic_plan_s"] = time.perf_counter() - t0
    rec["nnz_K"] = plan.nnz_K
    rec["nnz_P"] = plan.nnz_P
    rec["nnz_A"] = plan.nnz_A
    if impl == "tree":
        t0 = time.perf_counter()
        analyze_tree_structure(plan)
        rec["tree_analysis_s"] = time.perf_counter() - t0

    solver = Solver()
    t0 = time.perf_counter()
    solver.setup(P, q, A, l, u,
                 settings=_settings(impl, rho=rho, max_iter=max_iter,
                                    check_every=check_every,
                                    rho_eq_scale=rho_eq_scale,
                                    alpha=alpha,
                                    iteration_graph=iteration_graph))
    rec["setup_s"] = time.perf_counter() - t0  # includes factorization

    # cold scope: includes kernel compilation and graph construction;
    # never divided by the iteration count
    t0 = time.perf_counter()
    res = solver.solve()
    _sync()
    rec["cold_first_solve_s"] = time.perf_counter() - t0
    rec["iterations"] = res.info.iterations
    rec["converged"] = res.info.converged
    rec["primal_residual"] = res.info.primal_residual
    rec["dual_residual"] = res.info.dual_residual
    rec["rho_min"] = res.info.rho_min
    rec["rho_max"] = res.info.rho_max
    rec["num_equality_constraints"] = res.info.num_equality_constraints
    # objective and true constraint violation of the returned x
    import scipy.sparse as sp
    x = res.x.numpy()
    P_full = sp.triu(P) + sp.triu(P, k=1).T
    rec["objective"] = float(0.5 * x @ (P_full @ x) + q @ x)
    ax = A @ x
    rec["max_violation"] = float(max(
        np.max(np.maximum(l - ax, 0.0)), np.max(np.maximum(ax - u, 0.0))))

    # warm convergence-based time to solution (identical start state),
    # with launch counting and the device-allocation delta
    solver.warm_start(**zeros)
    solver.solve(copy_result=False)  # warm every remaining path once
    dev = wp.get_device("cuda:0")
    warm = []
    for _ in range(5):
        solver.warm_start(**zeros)
        _sync()
        before_bytes = wp.get_mempool_used_mem_current(dev)
        t0 = time.perf_counter()
        r = solver.solve(copy_result=False)
        _sync()
        warm.append(time.perf_counter() - t0)
        rec["warm_alloc_delta_bytes"] = int(
            wp.get_mempool_used_mem_current(dev) - before_bytes)
    rec["warm_solve_s"] = float(np.median(warm))
    rec["warm_iterations"] = r.info.iterations
    solver.warm_start(**zeros)
    _sync()
    with _LaunchCounter() as lc:
        r = solver.solve(copy_result=False)
    rec["warm_solve_host_calls"] = lc.count
    rec["warm_solve_host_calls_per_iter"] = lc.count / max(
        r.info.iterations, 1)

    # warm fixed-iteration throughput and phase breakdown (tree and
    # cuDSS both support this instrumentation; graph_us_per_iter is
    # None when the linear system cannot be captured)
    rec["phases"] = solver.time_iteration_phases(
        iters=fixed_iters, repeats=throughput_repeats)

    # repeated solve after a q/l/u update (warm fixed-matrix pattern)
    rng = np.random.default_rng(seed + 1)
    t0 = time.perf_counter()
    solver.update(q=q + 0.01 * rng.standard_normal(len(q)))
    res2 = solver.solve()
    _sync()
    rec["vector_update_solve_s"] = time.perf_counter() - t0
    rec["vector_update_iterations"] = res2.info.iterations

    # isolated numerical-kernel scope (non-mutating instrumentation)
    fact_ms, solve_ms = solver.time_linear_system(repeats)
    rec["factorize_ms"] = fact_ms
    rec["linear_solve_ms"] = solve_ms
    rec["linsys_stats"] = solver.linear_system_stats()
    return rec, solver, (P, q, A, l, u)


def run_osqp(P, q, A, l, u, max_iter=1000, rho=1.0, check_every=10):
    """OSQP (CPU) on the identical QP with settings matched as closely
    as its algorithm allows: fixed rho, no adaptive rho, no scaling, no
    relaxation (alpha=1), minimal sigma regularization, no polishing,
    the same max_iter cap and termination-check period, and the same
    eps_abs/eps_rel.  OSQP still differs algorithmically (sigma-
    regularized KKT solve on the CPU), so iterates need not match ours;
    the comparison is total time and time per iteration at the same
    iteration cap."""
    import osqp
    # OSQP always applies its internal ~1e3 equality-rho boost, so this
    # matched configuration corresponds to our rho_eq_scale=1000 setting
    # (rho_is_vec: true on both sides).
    settings = dict(rho=rho, adaptive_rho=False, sigma=1e-12, alpha=1.0,
                    scaling=0, max_iter=max_iter, eps_abs=1e-4,
                    eps_rel=1e-4, check_termination=check_every,
                    verbose=False)
    m = osqp.OSQP()
    t0 = time.perf_counter()
    try:
        m.setup(P=P, q=q, A=A, l=l, u=u, polishing=False, **settings)
    except TypeError:  # older setting name
        m.setup(P=P, q=q, A=A, l=l, u=u, polish=False, **settings)
    setup_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    res = m.solve()
    wall_solve_s = time.perf_counter() - t0
    info = res.info
    return {
        "impl": "osqp",
        "osqp_version": osqp.__version__,
        "iterations": int(info.iter),
        "status": str(info.status),
        "setup_s": setup_s,
        "first_solve_s": wall_solve_s,
        "osqp_solve_time_s": float(info.solve_time),
        "per_iter_ms": wall_solve_s * 1e3 / max(int(info.iter), 1),
        "primal_residual": float(info.prim_res if hasattr(info, "prim_res")
                                 else info.pri_res),
        "dual_residual": float(info.dual_res if hasattr(info, "dual_res")
                               else info.dua_res),
        "objective": float(info.obj_val),
        "rho_is_vec": True,  # OSQP's internal equality-rho boost
    }


def mpc_sequence(solver, P, q, A, l, u, steps, seed=100):
    """MPC-style loop (18.3): matrix-value update + warm solve per step,
    identical numerical data for every implementation."""
    rng = np.random.default_rng(seed)
    step_s, update_s, iters = [], [], []
    P0, A0 = np.asarray(P.data), np.asarray(A.data)
    for k in range(steps):
        scale_p = 1.0 + 0.02 * rng.uniform(-1, 1, size=len(P0))
        scale_a = 1.0 + 0.01 * rng.uniform(-1, 1, size=len(A0))
        t0 = time.perf_counter()
        solver.update(Px=P0 * scale_p, Ax=A0 * scale_a,
                      q=q + 0.02 * rng.standard_normal(len(q)))
        _sync()  # update() is asynchronous; time it to completion here
        t1 = time.perf_counter()
        res = solver.solve()
        _sync()
        step_s.append(time.perf_counter() - t0)
        update_s.append(t1 - t0)
        iters.append(res.info.iterations)
    return {
        "steps": steps,
        "step_median_s": float(np.median(step_s)),
        "iterations_median": int(np.median(iters)),
        "update_median_ms": float(np.median(update_s)) * 1e3,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tails", type=int, nargs="+", default=[11, 21, 41])
    p.add_argument("--stages", type=int, default=16)
    p.add_argument("--nx", type=int, default=6)
    p.add_argument("--nu", type=int, default=2)
    p.add_argument("--mpc", type=int, default=0,
                   help="run an MPC-style matrix-update sequence of N steps")
    p.add_argument("--osqp", action="store_true",
                   help="also run OSQP (CPU) on the identical QP")
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--check-every", type=int, default=10)
    p.add_argument("--rho", type=float, default=1.0,
                   help="base penalty rho_base")
    p.add_argument("--rho-eq-scale", type=float, default=1000.0,
                   help="equality-row penalty scale (1 = plain scalar rho)")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="ADMM over-relaxation parameter (1 = plain ADMM)")
    p.add_argument("--no-iteration-graph", action="store_true",
                   help="enqueue iterations eagerly instead of replaying "
                        "captured blocks")
    p.add_argument("--fixed-iters", type=int, default=200,
                   help="iterations per warm fixed-iteration sample")
    p.add_argument("--json", default=None)
    args = p.parse_args()

    records = []
    for B in args.tails:
        for impl in ("tree", "cudss"):
            rec, solver, qp = run_case(
                B, args.stages, args.nx, args.nu, impl,
                max_iter=args.max_iter, check_every=args.check_every,
                rho=args.rho, rho_eq_scale=args.rho_eq_scale,
                alpha=args.alpha,
                iteration_graph=not args.no_iteration_graph,
                fixed_iters=args.fixed_iters)
            ph = rec["phases"]
            g = ph["graph_us_per_iter"]
            graph_part = f"graph_iter={g:.1f}us " if g is not None else ""
            print(f"B={B:4d} {impl:6s}: n={rec['n']:6d} "
                  f"conv={str(rec['converged'])[0]} "
                  f"iters={rec['iterations']:4d} "
                  f"warm={rec['warm_solve_s']*1e3:.2f}ms "
                  f"warm_iter={ph['total_us_per_iter']:.1f}us "
                  + graph_part +
                  f"cold={rec['cold_first_solve_s']*1e3:.0f}ms "
                  f"r_p={rec['primal_residual']:.2e} "
                  f"r_d={rec['dual_residual']:.2e} "
                  f"obj={rec['objective']:.3f} "
                  f"viol={rec['max_violation']:.1e}")
            print(f"    linsys: factor={rec['factorize_ms']*1e3:.1f}us "
                  f"solve={rec['linear_solve_ms']*1e3:.1f}us | phases "
                  f"rhs={ph['rhs_ms']*1e3:.1f}us "
                  f"proj={ph['project_ms']*1e3:.1f}us "
                  f"check_gpu={ph['check_gpu_ms']*1e3:.1f}us "
                  f"check_total={ph['check_total_ms']*1e3:.1f}us | "
                  f"host_calls/iter="
                  f"{rec['warm_solve_host_calls_per_iter']:.2f} "
                  f"alloc_delta={rec['warm_alloc_delta_bytes']}")
            if args.mpc:
                rec["mpc"] = mpc_sequence(solver, *qp, steps=args.mpc)
                mpc = rec["mpc"]
                print(f"    mpc/step={mpc['step_median_s']*1e3:.1f}ms "
                      f"(update={mpc['update_median_ms']:.3f} ms, "
                      f"iters={mpc['iterations_median']})")
            solver.close()
            records.append(rec)
        if args.osqp:
            P, q, A, l, u, meta = generate_scenario_qp(
                B, args.stages, args.nx, args.nu, seed=10)
            rec = run_osqp(P, q, A, l, u, max_iter=args.max_iter,
                           rho=args.rho, check_every=args.check_every)
            rec.update({"B": B, "T": args.stages, "n": P.shape[0]})
            records.append(rec)
            print(f"B={B:4d} osqp  : n={rec['n']:6d} "
                  f"iters={rec['iterations']:4d} "
                  f"per_iter={rec['per_iter_ms']:.3f}ms "
                  f"total={rec['first_solve_s']*1e3:.1f}ms "
                  f"r_p={rec['primal_residual']:.2e} "
                  f"r_d={rec['dual_residual']:.2e} "
                  f"obj={rec['objective']:.3f} [{rec['status']}]")
    if args.json:
        Path(args.json).write_text(json.dumps(records, indent=1, default=str))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()

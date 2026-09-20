"""Phase breakdown at the representative points (plan 3, section 10.4
item 6): per-stage CUDA-event times of the eager numerical pipelines at
(41,64,8,2,1) and the legacy reference point (64,128,16,64,1).

Uses the solver's private stage timers (the production path replays
graphs, where stages are invisible); run only while the GPU is otherwise
idle.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

POINTS = [(50, 128, 8, 8, 1), (50, 128, 16, 16, 1), (50, 128, 32, 32, 1)]


def main():
    import warp as wp
    wp.init()
    import numpy as np
    from experiments.general_arrow.benchmarks.problems import ProblemSpec, generate_problem
    from experiments.general_arrow.benchmarks.timing import PhaseTimer
    from src.general_arrow.problem import TreeVector
    from src.general_arrow.solver import Solver
    from src.general_arrow._utils import wp_dtype

    out = {}
    for B, T, n_b, n_r, nrhs in POINTS:
        spec = ProblemSpec(num_tails=B, horizon=T, block_size=n_b,
                           root_dim=n_r, num_rhs=nrhs,
                           precision="float64", seed=0)
        prob = generate_problem(spec, estimate_condition=False)
        solver = Solver(spec.shape)
        solver.update(prob.matrix)
        solver.factorize()
        dt = wp_dtype("float64")
        rhs = TreeVector(spec.shape,
                         wp.array(np.ascontiguousarray(prob.rhs.tail),
                                  dtype=dt, device="cuda:0"),
                         wp.array(np.ascontiguousarray(prob.rhs.root),
                                  dtype=dt, device="cuda:0"))
        o = TreeVector(spec.shape,
                       wp.zeros((B, T, n_b, nrhs), dtype=dt,
                                device="cuda:0"),
                       wp.zeros((n_r, nrhs), dtype=dt, device="cuda:0"))
        solver.solve(rhs, out=o)
        for _ in range(10):
            solver._factorize_numeric(None)
            solver._solve_numeric(solver._binding, None)
        wp.synchronize_device("cuda:0")
        timer = PhaseTimer("cuda:0")
        tot, cnt = {}, 30
        for _ in range(cnt):
            timer.begin()
            solver._factorize_numeric(timer)
            solver._solve_numeric(solver._binding, timer)
            for k, v in timer.collect().items():
                tot[k] = tot.get(k, 0.0) + v / cnt
        key = f"B{B}_T{T}_n{n_b}_m{n_r}_q{nrhs}"
        out[key] = {k: round(v * 1000, 2) for k, v in tot.items()}  # us
        print(f"{key}:")
        for k, v in out[key].items():
            print(f"  {k:24s} {v:8.1f} us")
    path = Path("results/plan3_campaign/phase_breakdown.json")
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

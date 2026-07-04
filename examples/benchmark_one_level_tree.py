"""Phase 8 benchmark for the one-level tree solver.

Measures the separable stages of factorize/solve across a small sweep of
problem sizes.  Warp is warmed up before timing and the GPU is synchronized
around each measured region.

Note: this reports *observed* wall-clock scaling on one machine.  It does NOT
by itself prove the O(log T) critical-path bound -- that requires sufficient
parallel hardware and careful isolation of the tail-solve depth.

Run with::

    python examples/benchmark_one_level_tree.py
    python examples/benchmark_one_level_tree.py --quick
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import warp as wp

# silence Warp module generate/load + init-banner logging
wp.config.log_level = logging.WARNING

# allow running as `python examples/benchmark_one_level_tree.py` from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tree_kkt import OneLevelTreeCholesky


def build_spd_one_level_tree(B, T, n, q, seed=0):
    rng = np.random.default_rng(seed)
    Kd = np.zeros((B, T, n, n))
    Ke = np.zeros((B, max(T - 1, 0), n, n))
    G = np.zeros((B, n, n))
    for i in range(B):
        Dl = np.zeros((T, n, n)); El = np.zeros((max(T - 1, 0), n, n))
        for k in range(T):
            Dl[k] = np.tril(rng.standard_normal((n, n))) + (n + 2) * np.eye(n)
            if k < T - 1:
                El[k] = 0.3 * rng.standard_normal((n, n))
        for k in range(T):
            Kd[i, k] = Dl[k] @ Dl[k].T
            if k > 0:
                Kd[i, k] += El[k - 1] @ El[k - 1].T
            if k < T - 1:
                Ke[i, k] = El[k] @ Dl[k].T
        G[i] = 0.2 * rng.standard_normal((n, n))
    # Diagonally dominant root: SPD and cheap to build (benchmark only needs a
    # well-posed system, not the exact-S0 construction used in the tests).
    D0 = float(n) * np.eye(n)
    for i in range(B):
        D0 = D0 + G[i].T @ G[i]
    r_root = rng.standard_normal((n, q))
    r_tail = rng.standard_normal((B, T, n, q))
    return D0, Kd, Ke, G, r_root, r_tail


def _sync():
    wp.synchronize()


def time_region(fn, iters):
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def bench_case(B, T, n, q, dtype, device, iters=20):
    D0, Kd, Ke, G, r_root, r_tail = build_spd_one_level_tree(B, T, n, q)

    # --- setup (construction: allocation + launch recording) ---
    def setup():
        return OneLevelTreeCholesky(B, T, n, dtype=dtype, device=device)
    t_setup = time_region(setup, max(iters // 4, 1))
    solver = setup()

    # stage inputs once (kept out of the timed sections below)
    solver.factorize(D0, Kd, Ke, G)     # warm up + compile
    solver.solve(r_root, r_tail)

    # --- factorize (all matrix-dependent work) ---
    t_factor = time_region(
        lambda: solver.factorize(D0, Kd, Ke, G), iters)

    # --- solve (all rhs-dependent work) ---
    t_solve = time_region(
        lambda: solver.solve(r_root, r_tail, copy_to_host=False), iters)

    return dict(t_setup=t_setup, t_factor=t_factor, t_solve=t_solve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    wp.init()
    device = args.device

    if args.quick:
        block_sizes = [8, 16]
        tail_lengths = [15, 63]
        num_tails_list = [4, 16]
        dtypes = [wp.float64]
        q = 1
    else:
        block_sizes = [4, 8, 16, 32]
        tail_lengths = [7, 15, 31, 63, 127, 255]
        num_tails_list = [2, 4, 8, 16, 32]
        dtypes = [wp.float32, wp.float64]
        q = 1

    hdr = f"{'dtype':>8} {'n':>4} {'T':>5} {'B':>4} | " \
          f"{'setup[ms]':>10} {'factor[ms]':>11} {'solve[ms]':>10}"
    print(hdr)
    print("-" * len(hdr))
    for dtype in dtypes:
        dname = "f32" if dtype == wp.float32 else "f64"
        for n in block_sizes:
            for T in tail_lengths:
                for B in num_tails_list:
                    r = bench_case(B, T, n, q, dtype, device)
                    print(f"{dname:>8} {n:>4} {T:>5} {B:>4} | "
                          f"{r['t_setup']:>10.3f} {r['t_factor']:>11.3f} "
                          f"{r['t_solve']:>10.3f}")


if __name__ == "__main__":
    main()

"""Endpoint-solver benchmark on TRUE full-attitude quadrotor matrices.

For every (B, T) grid point this script assembles an immediate-
branching scenario-MPC system for a 12-state quadrotor -- position,
velocity, Euler attitude, and body rates, with the 4 rotor thrusts as
inputs -- in the LEAF-TO-ROOT variable ordering:

    tail i = [ (u_{T-1}^i, x_T^i) | ... | (u_1^i, x_2^i) | x_1^i ]
    root   = (x_0, u_0)                             (n_r = 16)

The initial state x_0 is kept as an optimization variable (pinned by
identity initial-condition rows), so the shared root is the complete
first decision node and n_r equals the stage block size n_b = 16.
Scenarios sample the vehicle mass and a constant wind disturbance, so
every scenario has its own RK4-linearized dynamics jacobians.  Unlike
the point-mass drone, the attitude dynamics couple ALL states (the
rotation matrix mixes translation with attitude, the Euler kinematics
mix attitude with body rates, and RK4 composes these couplings), so
the stage blocks are genuinely dense and the sparse baselines cannot
exploit intra-block structure.

Constraint rows: dynamics (t = 0 couples the root to x_1 only), 3-D
obstacle avoidance on the positions, terminal rows on x_T, thrust
bounds, and the root initial-condition/bound rows.  The benchmarked
matrix is K = diag(P) + A^T A.  Baselines receive the exact sparse K:

- cuDSS: SPD Cholesky on the exact lower CSR;
- CHOLMOD: supernodal sparse Cholesky (SuiteSparse, via scikit-sparse).
  NOT timed: it serves only as the untimed reference solution that the
  three timed solvers are checked against, so it never shares the CPU
  timing rotation with PARDISO.
- PARDISO: Intel MKL PARDISO (called directly through ctypes), the
  multicore CPU reference that parallelizes over independent
  elimination subtrees.  MKL's own defaults -- parallel nested-
  dissection ordering, classic factorization, parallel triangular
  solve -- with iterative refinement disabled (MKL's default of two
  refinement steps triples the solve work at no accuracy benefit
  here); thread count from MKL_NUM_THREADS; numeric-only refactor
  timed.  Skipped when libmkl_rt is not installed.

Timing protocol, identical for every solver and phase.  Clocks cannot
be locked without root, so fairness against frequency and thermal
drift comes from the protocol itself:

- the grid is visited in a seeded random order, so slow drift over the
  campaign is not correlated with problem size;
- before the first timed point both devices are loaded concurrently
  for --burn-in seconds so they reach their sustained operating point
  instead of the cold, boosted state; every later point re-warms both
  devices for --settle seconds with its own solvers (at least
  --warmups passes, which also covers JIT and cache warm-up);
- the two GPU solvers are timed INTERLEAVED in blocks, one phase at a
  time: the factorization pair takes turns in blocks of
  --interleave-block repetitions until each has run --reps times, then
  the solve pair does the same.  The order within the pair alternates
  every block (endpoint, cuDSS | cuDSS, endpoint | ...), so neither
  solver is systematically first in its block; --reverse-pairs starts
  with the opposite order.  Both sides of every speedup therefore see
  the same device temperature and clock, and each block still runs with
  its own factor hot in cache after the first call.  PARDISO is the
  only timed CPU solver, so it is timed alone; before each of its two
  timed groups the SAME phase runs for a fixed --cpu-settle seconds, so
  the CPU enters the measurement at the steady state of the workload
  being measured (a fixed interval, never a wait for a frequency
  reading, which would select favorable hardware states);
- GPU temperature, SM clock and power, and CPU package temperature and
  mean core frequency are sampled right before and after every timed
  block and stored in the CSV, so the thermal state of each measurement
  is on record.

The CSV holds the median and the 25th/75th percentiles of the timed
set.  Every individual sample is also appended, after each point's
measurements and outside every timed region, to the raw sidecar
``<output>_raw.csv`` with its block index and position in the block, so
the alternating schedule and any latency drift are auditable.  GPU
phases are timed with CUDA events, CPU phases with the wall clock
around the bare library call.

The endpoint solver receives the logically identical block form; the
non-uniform last block (12-wide x_1) is padded to the uniform 16-wide
storage block with identity dummies.  Solutions are cross-checked
between all solvers per point.

Usage (from this directory, socu env)::

    python quadrotor_endpoint_benchmark.py [--tails 20 40 ... 200]
    python quadrotor_endpoint_benchmark.py --replot results/quadrotor_endpoint_heatmap.csv
                                           [--horizons 20 30 ... 120]

Writes results/quadrotor_endpoint_heatmap.csv and the speedup heat
maps figures/quadrotor_endpoint_speedup_{cudss,cholmod}.pdf.
"""

import argparse
import csv
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

REG = 1e-8
N_X, N_U = 12, 4
BLK = N_X + N_U   # uniform storage block (16)
N_R = N_X + N_U   # root block (x_0, u_0): n_r = n_b

# quadrotor model constants
T_FINAL = 3.0
MASS = 1.0
J_INERTIA = np.array([0.010, 0.010, 0.018])
ARM = 0.17
C_TAU = 0.016
GRAVITY = 9.81
DRAG_V = 0.30
DRAG_W = 0.05
X_INIT = np.zeros(N_X)
X_FINAL = np.array([2.0, 1.0, 1.0] + [0.0] * 9)
OBS_POSITIONS = np.array([[0.7, 0.3, 0.4],
                          [1.3, 0.7, 0.7],
                          [1.0, 0.2, 1.1]])
OBS_RADII = np.array([0.30, 0.35, 0.25])


def quad_step(dt):
    """RK4 step of the full-attitude quadrotor as a jax function.

    State x = (p, v, (phi, theta, psi), omega); input u holds the four
    rotor thrusts.  The scenario parameters are the vehicle mass and a
    constant wind acceleration.
    """
    import jax.numpy as jnp

    Jv = jnp.asarray(J_INERTIA)

    def f(x, u, m, wind):
        v, ang, om = x[3:6], x[6:9], x[9:12]
        cph, sph = jnp.cos(ang[0]), jnp.sin(ang[0])
        cth, sth = jnp.cos(ang[1]), jnp.sin(ang[1])
        cps, sps = jnp.cos(ang[2]), jnp.sin(ang[2])
        # third column of the ZYX rotation matrix (thrust direction)
        r3 = jnp.array([cps * sth * cph + sps * sph,
                        sps * sth * cph - cps * sph,
                        cth * cph])
        thrust = jnp.sum(u)
        a = (r3 * thrust - DRAG_V * v) / m + wind \
            - jnp.array([0.0, 0.0, GRAVITY])
        # Euler-angle kinematics
        tth = sth / cth
        W = jnp.array([[1.0, sph * tth, cph * tth],
                       [0.0, cph, -sph],
                       [0.0, sph / cth, cph / cth]])
        ang_dot = W @ om
        tau = jnp.array([ARM * (u[1] - u[3]),
                         ARM * (u[2] - u[0]),
                         C_TAU * (u[0] - u[1] + u[2] - u[3])])
        om_dot = (tau - jnp.cross(om, Jv * om)) / Jv - DRAG_W * om
        return jnp.concatenate([v, a, ang_dot, om_dot])

    def step(x, u, m, wind):
        k1 = f(x, u, m, wind)
        k2 = f(x + 0.5 * dt * k1, u, m, wind)
        k3 = f(x + 0.5 * dt * k2, u, m, wind)
        k4 = f(x + dt * k3, u, m, wind)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return step


def build_true_problem(B, T, seed=0):
    """Assemble the exact scenario-MPC system in leaf-to-root ordering.

    Returns ``(K_true, b_true, blocks)`` where ``K_true`` is the exact
    sparse SPD matrix, ``b_true`` a random RHS, and ``blocks`` the
    padded (D, E, G_T, R) plus the index map needed to pad/strip
    vectors.
    """
    import jax
    import jax.numpy as jnp
    from jax import jacfwd, vmap

    jax.config.update("jax_enable_x64", True)
    dt = T_FINAL / T
    step = quad_step(dt)

    rng = np.random.default_rng(seed)
    masses = MASS * (1.0 + 0.10 * rng.standard_normal(B))
    winds = 0.25 * rng.standard_normal((B, 3))

    # gently wobbling reference controls (nonzero attitude and rates,
    # so the linearization point carries the full coupling)
    hover = MASS * GRAVITY / 4.0
    tt = np.arange(T)[:, None]
    u_ref = hover * (1.0 + 0.06 * np.sin(
        2.0 * np.pi * tt / max(T, 1) + np.array([0, 1.5, 3.0, 4.5])))
    u_ref = jnp.asarray(u_ref)

    def rollout(m, w):
        def g(x, u):
            xn = step(x, u, m, w)
            return xn, xn
        _, xs = jax.lax.scan(g, jnp.asarray(X_INIT), u_ref)
        return jnp.concatenate([jnp.asarray(X_INIT)[None], xs], axis=0)

    xs_all = np.asarray(vmap(rollout)(jnp.asarray(masses),
                                      jnp.asarray(winds)))

    def step_i(x, u, m, w):
        return step(x, u, m, w)

    m_rep = jnp.repeat(jnp.asarray(masses)[:, None], T, axis=1)
    w_rep = jnp.repeat(jnp.asarray(winds)[:, None], T, axis=1)
    u_rep = jnp.repeat(u_ref[None], B, axis=0)
    Jx = np.asarray(vmap(vmap(jacfwd(step_i, argnums=0)))(
        jnp.asarray(xs_all[:, :-1]), u_rep, m_rep, w_rep))
    Ju = np.asarray(vmap(vmap(jacfwd(step_i, argnums=1)))(
        jnp.asarray(xs_all[:, :-1]), u_rep, m_rep, w_rep))

    # ---- leaf-to-root column layout (no dummy variables) -----------
    tail_dim = BLK * (T - 1) + N_X
    n_true = B * tail_dim + N_R
    root0 = B * tail_dim          # root = (x_0 | u_0)
    root_u0 = root0 + N_X
    n_obs = len(OBS_RADII)

    def ucol(i, t):          # u_t^i, t = 1..T-1 (u_0 is the root)
        return i * tail_dim + (T - 1 - t) * BLK

    def xcol(i, t):          # x_t^i, t = 1..T
        if t == 1:
            return i * tail_dim + (T - 1) * BLK
        return ucol(i, t - 1) + N_U

    # vectorized dynamics rows: for each (i, t) the 12 rows carry the
    # identity on x_{t+1}, -Jx on x_t (root x_0 at t = 0), and -Ju on
    # u_t (root u_0 at t = 0)
    xc_next = np.empty((B, T), dtype=np.int64)
    xc_prev = np.empty((B, T), dtype=np.int64)
    uc = np.empty((B, T), dtype=np.int64)
    for i in range(B):
        for t in range(T):
            xc_next[i, t] = xcol(i, t + 1)
            xc_prev[i, t] = root0 if t == 0 else xcol(i, t)
            uc[i, t] = root_u0 if t == 0 else ucol(i, t)
    row0 = (np.arange(B * T) * N_X).reshape(B, T)

    # row equilibration of the dynamics rows: the body-rate rows
    # carry 1/J gains that would otherwise dominate A^T A
    rnrm = np.sqrt(1.0 + (Jx ** 2).sum(axis=3)
                   + (Ju ** 2).sum(axis=3))       # (B, T, N_X)

    r_eye = (row0[:, :, None] + np.arange(N_X)).ravel()
    c_eye = (xc_next[:, :, None] + np.arange(N_X)).ravel()
    v_eye = (1.0 / rnrm).ravel()

    rr = np.broadcast_to(row0[:, :, None, None]
                         + np.arange(N_X)[None, None, :, None],
                         (B, T, N_X, N_X)).ravel()
    cc = np.broadcast_to(xc_prev[:, :, None, None]
                         + np.arange(N_X)[None, None, None, :],
                         (B, T, N_X, N_X)).ravel()
    vv = (-Jx / rnrm[:, :, :, None]).ravel()

    ru = np.broadcast_to(row0[:, :, None, None]
                         + np.arange(N_X)[None, None, :, None],
                         (B, T, N_X, N_U)).ravel()
    cu = np.broadcast_to(uc[:, :, None, None]
                         + np.arange(N_U)[None, None, None, :],
                         (B, T, N_X, N_U)).ravel()
    vu = (-Ju / rnrm[:, :, :, None]).ravel()

    rows = [r_eye, rr, ru]
    cols = [c_eye, cc, cu]
    vals = [v_eye, vv, vu]
    row = B * T * N_X

    # 3-D obstacle rows, terminal rows, and bound rows (small; loops)
    r2, c2, v2 = [], [], []

    def add(r, c, v):
        if v != 0.0:
            r2.append(r)
            c2.append(c)
            v2.append(float(v))

    for i in range(B):
        for t in range(1, T + 1):
            p = xs_all[i, t, :3]
            for o in range(n_obs):
                d = p - OBS_POSITIONS[o]
                nrm = max(np.linalg.norm(d), 1e-6)
                g = -d / nrm
                for c in range(3):
                    add(row, xcol(i, t) + c, g[c])
                row += 1
        for r in range(N_X):
            add(row, xcol(i, T) + r, 1.0)
            row += 1
        for t in range(1, T):
            for c in range(N_U):
                add(row, ucol(i, t) + c, 1.0)
                row += 1
    # initial condition x_0 = x_init and bounds on u_0
    for c in range(N_R):
        add(row, root0 + c, 1.0)
        row += 1
    m = row
    rows.append(np.asarray(r2))
    cols.append(np.asarray(c2))
    vals.append(np.asarray(v2))
    A = sp.coo_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(m, n_true)).tocsc()
    A.sort_indices()

    P_diag = np.full(n_true, REG)
    P_diag[root_u0:root_u0 + N_U] += 2.0 * dt
    for i in range(B):
        for t in range(1, T):
            P_diag[ucol(i, t):ucol(i, t) + N_U] += 2.0 * dt / B
    K = (sp.diags(P_diag) + A.T @ A).tocsr()
    K.eliminate_zeros()
    K.sort_indices()

    b_true = np.random.default_rng(seed + 1).standard_normal(n_true)

    # ---- padded uniform block extraction for the endpoint solver ---
    D = np.zeros((B, T, BLK, BLK))
    E = np.zeros((B, T - 1, BLK, BLK))
    G_T = np.zeros((B, BLK, N_R))
    for i in range(B):
        r0 = i * tail_dim
        band = K[r0:r0 + tail_dim, r0:r0 + tail_dim].toarray()
        for k in range(T - 1):
            D[i, k] = band[k * BLK:(k + 1) * BLK, k * BLK:(k + 1) * BLK]
        last = band[(T - 1) * BLK:, (T - 1) * BLK:]
        D[i, T - 1, :N_X, :N_X] = last
        for j in range(N_X, BLK):
            D[i, T - 1, j, j] = 1.0
        for k in range(T - 2):
            E[i, k] = band[(k + 1) * BLK:(k + 2) * BLK,
                           k * BLK:(k + 1) * BLK]
        E[i, T - 2, :N_X, :] = band[(T - 1) * BLK:,
                                    (T - 2) * BLK:(T - 1) * BLK]
        G_T[i, :N_X, :] = K[r0 + (T - 1) * BLK:r0 + tail_dim,
                            root0:].toarray()
    R = K[root0:, root0:].toarray()
    blocks = {"D": D, "E": E, "G_T": G_T, "R": R,
              "tail_dim": tail_dim, "root0": root0}
    return K, b_true, blocks


def pad_rhs(b_true, B, T, blocks):
    tail_dim, root0 = blocks["tail_dim"], blocks["root0"]
    tail = np.zeros((B, T, BLK))
    for i in range(B):
        seg = b_true[i * tail_dim:(i + 1) * tail_dim]
        tail[i, :T - 1] = seg[:(T - 1) * BLK].reshape(T - 1, BLK)
        tail[i, T - 1, :N_X] = seg[(T - 1) * BLK:]
    return tail, b_true[root0:].copy()


def strip_solution(xt, xr, B, T, blocks):
    tail_dim = blocks["tail_dim"]
    z = np.empty(B * tail_dim + N_R)
    for i in range(B):
        z[i * tail_dim:i * tail_dim + (T - 1) * BLK] = \
            xt[i, :T - 1].reshape(-1)
        z[i * tail_dim + (T - 1) * BLK:(i + 1) * tail_dim] = \
            xt[i, T - 1, :N_X]
    z[B * tail_dim:] = xr
    return z


def _load_records(path):
    """Read a results CSV back into the record dicts the plots use."""
    ints = {"B", "T", "n", "nnz", "nnz_factor"}
    records = []
    for row in csv.DictReader(open(path)):
        rec = {}
        for k, v in row.items():
            if v == "":
                continue
            if k in ints:
                rec[k] = int(float(v))
            elif k.endswith("_ms") or k == "residual":
                rec[k] = float(v)
            else:
                rec[k] = v
        records.append(rec)
    return records


def _plot_heatmaps(records, args):
    """Draw one factorization/solve speedup heat map per baseline
    present in ``records`` (cuDSS, CHOLMOD, PARDISO) into figures/."""
    # -------------------------------------------------- heat maps
    import matplotlib.pyplot as plt
    plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                         "font.size": 13, "figure.dpi": 150,
                         "savefig.bbox": "tight"})

    def cell(B, T, solver, key):
        for r in records:
            if r["B"] == B and r["T"] == T and r["solver"] == solver:
                return r[key]
        return np.nan

    os.makedirs("figures", exist_ok=True)
    nT, nB = len(args.horizons), len(args.tails)
    fig_w = max(4.6, 0.52 * nT + 2.0)
    fig_h = max(4.8, 2.0 * (0.36 * nB + 0.66))
    ann_fs = 10 if nT <= 6 else 9
    baselines_present = [sv for sv in ("cudss", "cholmod", "pardiso")
                         if any(r["solver"] == sv for r in records)]
    for base_sv in baselines_present:
        fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h))
        for ax, key, phase in (
                (axes[0], "factor_ms", "Factorization"),
                (axes[1], "solve_ms", "Triangular solve")):
            G = np.array([[cell(B, T, base_sv, key)
                           / cell(B, T, "endpoint", key)
                           for T in args.horizons]
                          for B in args.tails])
            vmin, vmax = float(np.nanmin(G)), float(np.nanmax(G))
            im = ax.imshow(G, cmap="viridis", vmin=vmin, vmax=vmax,
                           aspect="auto", origin="lower")
            for i in range(len(args.tails)):
                for j in range(len(args.horizons)):
                    v = G[i, j]
                    frac = (v - vmin) / max(vmax - vmin, 1e-9)
                    ax.text(j, i, rf"{v:.1f}", ha="center",
                            va="center", fontsize=ann_fs,
                            color="black" if frac > 0.55
                            else "white")
            ax.set_xticks(range(len(args.horizons)))
            if ax is axes[1]:
                ax.set_xticklabels(args.horizons, fontsize=13)
                ax.set_xlabel(r"horizon $N$", fontsize=18)
            else:
                ax.set_xticklabels([])
            ax.set_yticks(range(len(args.tails)))
            ax.set_yticklabels(args.tails, fontsize=13)
            ax.set_ylabel(r"scenarios $M$", fontsize=18)
            name = {"cudss": "cuDSS", "cholmod": "CHOLMOD",
                    "pardiso": "PARDISO"}[base_sv]
            ax.set_title(phase, fontsize=18)
            cbar = fig.colorbar(im, ax=ax, pad=0.02)
            cbar.ax.tick_params(labelsize=11)
        fig.subplots_adjust(hspace=0.14)
        out_pdf = f"figures/quadrotor_endpoint_speedup_{base_sv}.pdf"
        fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print("saved", out_pdf)



def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tails", type=int, nargs="+",
                    default=list(range(20, 201, 20)))
    ap.add_argument("--horizons", type=int, nargs="+",
                    default=list(range(20, 121, 10)))
    ap.add_argument("--reps", type=int, default=50,
                    help="timed executions per phase, every solver alike")
    ap.add_argument("--warmups", type=int, default=10,
                    help="untimed executions before the timed ones, "
                         "every solver alike")
    ap.add_argument("--burn-in", type=float, default=60.0,
                    help="seconds of concurrent GPU+CPU load before the "
                         "first timed point (thermal steady state)")
    ap.add_argument("--settle", type=float, default=1.0,
                    help="seconds of concurrent GPU+CPU load before each "
                         "point's timed block (re-warm after problem "
                         "construction)")
    ap.add_argument("--interleave-block", type=int, default=10,
                    help="paired solvers take turns in blocks of this "
                         "many repetitions; 0 times each solver's "
                         "repetitions back to back")
    ap.add_argument("--cpu-settle", type=float, default=2.0,
                    help="seconds the PARDISO phase about to be timed runs "
                         "before its timed group (fixed, identical at "
                         "every point)")
    ap.add_argument("--reverse-pairs", action="store_true",
                    help="start every interleaved pair with the second "
                         "solver (cuDSS before endpoint) instead of the "
                         "first; the order still alternates per block")
    ap.add_argument("--points", nargs="+", metavar="M,N",
                    help="benchmark exactly these (scenarios, horizon) "
                         "points instead of the --tails x --horizons grid")
    ap.add_argument("--shuffle-seed", type=int, default=0,
                    help="seed for the random grid order; a negative "
                         "value keeps the nested horizon/tails order")
    ap.add_argument("--output",
                    default="results/quadrotor_endpoint_heatmap.csv")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--replot", metavar="CSV",
                    help="draw the heat maps from an existing results CSV "
                         "and exit without benchmarking")
    args = ap.parse_args()
    if args.replot:
        records = _load_records(args.replot)
        args.tails = sorted({r["B"] for r in records})
        args.horizons = sorted({r["T"] for r in records})
        _plot_heatmaps(records, args)
        return

    import warp as wp
    from baselines.cudss import CudssCholesky
    from baselines.cpu_direct import (PARDISO_AVAILABLE, CholmodDirect,
                                      PardisoDirect)
    from src.endpoint_tree import (EndpointTreeMatrix, EndpointTreeShape,
                               EndpointTreeSolver, EndpointTreeVector)
    from src.endpoint_tree.kernels.extract import (extract_blocks,
                                               init_padding)
    wp.init()
    dev = wp.get_device("cuda:0")
    st = dev.stream.cuda_stream
    e0 = wp.Event(dev, enable_timing=True)
    e1 = wp.Event(dev, enable_timing=True)

    def quartiles(ts):
        """Median with the 25th and 75th percentile around it.

        The interquartile range is the dispersion reported alongside
        every timing.  Run-to-run latency is right-skewed -- a stray
        interrupt only ever makes one sample slower -- so a standard
        deviation would be dominated by the tail rather than describing
        the typical spread, while the quartiles bracket the middle half
        of the samples and match the median already being reported.
        """
        p25, med, p75 = np.percentile(ts, [25.0, 50.0, 75.0])
        return float(med), float(p25), float(p75)

    # One protocol for every solver and phase: ``--warmups`` untimed
    # executions, then ``--reps`` timed ones, quartiles over the whole
    # timed set.  GPU phases are timed with CUDA events on the device
    # (host launch overhead excluded, matching the numerical-latency
    # interpretation of the plots); CPU phases with the wall clock
    # around a call that contains nothing but the library routine.

    def timed_gpu(fn):
        for _ in range(args.warmups):
            fn()
        wp.synchronize_device(dev)
        ts = []
        for _ in range(args.reps):
            wp.record_event(e0)
            fn()
            wp.record_event(e1)
            wp.synchronize_event(e1)
            ts.append(wp.get_event_elapsed_time(e0, e1,
                                                synchronize=False))
        return quartiles(ts)

    def timed_cpu(fn):
        for _ in range(args.warmups):
            fn()
        ts = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1e3)
        return quartiles(ts)

    def timing(factor, solve):
        """Flatten one factorize/solve quartile pair into CSV columns."""
        (f_med, f_lo, f_hi), (s_med, s_lo, s_hi) = factor, solve
        return {"factor_ms": f_med, "factor_p25": f_lo, "factor_p75": f_hi,
                "solve_ms": s_med, "solve_p25": s_lo, "solve_p75": s_hi}

    def settle(gpu_fns, cpu_fns, seconds):
        """Load both devices concurrently for ``seconds`` (untimed).

        GPU launches are asynchronous, so each pass enqueues the GPU
        work first and runs the CPU work while it executes.  Holding
        both devices at their sustained operating point before the
        timed block means no solver is measured on a cold, boosted
        chip.  At least ``--warmups`` passes always run, which also
        covers JIT compilation and cache warm-up.
        """
        t_end = time.perf_counter() + seconds
        passes = 0
        while passes < args.warmups or time.perf_counter() < t_end:
            for fn in gpu_fns.values():
                fn()
            for fn in cpu_fns.values():
                fn()
            wp.synchronize_device(dev)
            passes += 1
        return passes

    def rotation(fns):
        """The interleaving schedule of one timed group: the callables
        take turns in blocks of ``--interleave-block`` repetitions until
        each has run ``--reps`` times.

        Blocks rather than single calls: alternating two solvers call
        by call evicts each one's factor from the last-level cache
        between its own calls, so the timing would include re-reading
        the factor from DRAM -- a cache artifact, not the repeated-solve
        workload being measured.  A block of ten calls is still far
        shorter than any thermal time constant, so both sides of a
        speedup still see the same device temperature and clock, while
        only the first call of each block pays the cold-cache price
        (one sample in ten; the median is unaffected).

        The order of the callables inside a block alternates from one
        block to the next (AB, BA, AB, ...), so no callable is always
        the first to run after the other has warmed the device; the
        starting order follows ``--reverse-pairs``.
        """
        block = args.interleave_block if args.interleave_block > 0 \
            else args.reps
        names = list(fns)
        if args.reverse_pairs:
            names.reverse()
        order, done, i = [], 0, 0
        while done < args.reps:
            k = min(block, args.reps - done)
            seq = names if i % 2 == 0 else names[::-1]
            order.extend((name, k, i, pos) for pos, name in enumerate(seq))
            done += k
            i += 1
        return order

    def timed_gpu_group(fns, raw, phase):
        """Blocked-interleaved CUDA-event timing of several GPU
        callables (see :func:`rotation`).  Every sample is appended to
        ``raw`` with its schedule position."""
        samples = {k: [] for k in fns}
        for name, k, block_index, pos in rotation(fns):
            fn = fns[name]
            for _ in range(k):
                wp.record_event(e0)
                fn()
                wp.record_event(e1)
                wp.synchronize_event(e1)
                ms = wp.get_event_elapsed_time(e0, e1, synchronize=False)
                raw.append((phase, name, block_index, len(samples[name]),
                            pos, ms))
                samples[name].append(ms)
        return {k: quartiles(v) for k, v in samples.items()}

    def timed_cpu_group(fns, raw, phase):
        """Blocked-interleaved wall-clock timing of several CPU
        callables (see :func:`rotation`), samples appended to ``raw``."""
        samples = {k: [] for k in fns}
        for name, k, block_index, pos in rotation(fns):
            fn = fns[name]
            for _ in range(k):
                t0 = time.perf_counter()
                fn()
                ms = (time.perf_counter() - t0) * 1e3
                raw.append((phase, name, block_index, len(samples[name]),
                            pos, ms))
                samples[name].append(ms)
        return {k: quartiles(v) for k, v in samples.items()}

    def settle_cpu(fn, seconds):
        """Run one CPU phase back to back for a fixed interval (at least
        --warmups passes) right before that phase is timed."""
        t_end = time.perf_counter() + seconds
        passes = 0
        while passes < args.warmups or time.perf_counter() < t_end:
            fn()
            passes += 1
        return passes

    raw_path = os.path.splitext(args.output)[0] + "_raw.csv"
    raw_fields = ["campaign", "order_index", "M", "N", "phase", "solver",
                  "block_index", "sample_index", "order_in_block",
                  "elapsed_ms"]
    campaign = (f"seed{args.shuffle_seed}-"
                f"{'BA' if args.reverse_pairs else 'AB'}")

    # ---- thermal telemetry (outside every timed region)
    def _coretemp_package():
        for h in Path("/sys/class/hwmon").glob("hwmon*"):
            try:
                if (h / "name").read_text().strip() != "coretemp":
                    continue
                for lab in h.glob("temp*_label"):
                    if lab.read_text().strip().startswith("Package"):
                        raw = lab.with_name(lab.name.replace(
                            "_label", "_input")).read_text()
                        return float(raw) / 1000.0
            except OSError:
                continue
        return float("nan")

    def telemetry():
        """GPU temperature/SM clock/power and CPU package temperature
        and mean current frequency of the bound cores."""
        out = {"gpu_temp_c": float("nan"), "gpu_sm_mhz": float("nan"),
               "gpu_power_w": float("nan"), "cpu_temp_c": float("nan"),
               "cpu_mhz": float("nan")}
        try:
            q = subprocess.run(
                ["nvidia-smi", "--id=0",
                 "--query-gpu=temperature.gpu,clocks.sm,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5).stdout
            t, c, w = (float(v) for v in q.strip().split(","))
            out.update(gpu_temp_c=t, gpu_sm_mhz=c, gpu_power_w=w)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        out["cpu_temp_c"] = _coretemp_package()
        freqs = []
        for c in sorted(os.sched_getaffinity(0)):
            try:
                freqs.append(float(Path(
                    f"/sys/devices/system/cpu/cpu{c}/cpufreq/"
                    f"scaling_cur_freq").read_text()) / 1000.0)
            except OSError:
                pass
        if freqs:
            out["cpu_mhz"] = float(np.mean(freqs))
        return out

    def thermal(begin, end):
        """Flatten the before/after telemetry of one timed block."""
        rec = {f"{k}_begin": v for k, v in begin.items()}
        rec.update({f"{k}_end": v for k, v in end.items()})
        return rec

    if args.points:
        grid = []
        for pt in args.points:
            B, T = (int(v) for v in pt.split(","))
            grid.append((T, B))
    else:
        grid = [(T, B) for T in args.horizons for B in args.tails]
    if args.shuffle_seed >= 0:
        random.Random(args.shuffle_seed).shuffle(grid)
    records = []
    for idx, (T, B) in enumerate(grid):
        K, b_true, blocks = build_true_problem(B, T)
        n = K.shape[0]
        burn = args.burn_in if idx == 0 else 0.0
        base = {"B": B, "T": T, "n": n, "nnz": int(K.nnz),
                # provenance of the CPU-baseline protocol
                "cpu_affinity": ",".join(map(str, sorted(
                    os.sched_getaffinity(0)))),
                "cpu_threads": os.environ.get("MKL_NUM_THREADS", ""),
                "omp_wait_policy": os.environ.get("OMP_WAIT_POLICY", ""),
                "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS",
                                                   ""),
                "reps": args.reps, "warmups": args.warmups,
                # thermal protocol provenance
                "order_index": idx, "settle_s": args.settle,
                "burn_in_s": burn,
                "interleave_block": args.interleave_block,
                "cpu_settle_s": args.cpu_settle, "campaign": campaign,
                "pair_order": ("BA" if args.reverse_pairs else "AB")
                              + "-alternating"}

        # ---- block extraction from the device-resident CSR
        lower = sp.tril(K).tocsr()
        lower.sort_indices()
        d_vals = wp.array(lower.data, dtype=wp.float64,
                          device=dev)
        d_cols = wp.array(lower.indices.astype(np.int32),
                          dtype=wp.int32, device=dev)
        d_offs = wp.array(lower.indptr.astype(np.int32),
                          dtype=wp.int32, device=dev)
        D_x = wp.zeros((B, T, BLK, BLK), dtype=wp.float64,
                       device=dev)
        E_x = wp.zeros((B, T - 1, BLK, BLK), dtype=wp.float64,
                       device=dev)
        G_x = wp.zeros((B, BLK, N_R), dtype=wp.float64,
                       device=dev)
        R_x = wp.zeros((N_R, N_R), dtype=wp.float64, device=dev)
        init_padding(D_x, N_X)
        td = blocks["tail_dim"]

        def do_extract():
            extract_blocks(d_vals, d_cols, d_offs, n, td, T,
                           BLK, N_X, D_x, E_x, G_x, R_x,
                           device=dev)

        do_extract()
        wp.synchronize_device(dev)
        f_x = timed_gpu(do_extract)[0]

        # ---- build every solver BEFORE anything is timed
        shape = EndpointTreeShape(B, T, BLK, N_R)
        matrix = EndpointTreeMatrix(
            shape, D=blocks["D"], E=blocks["E"],
            G_T=blocks["G_T"], R=blocks["R"])
        solver = EndpointTreeSolver(shape)
        solver.update(matrix)
        solver.factorize()
        tail_h, root_h = pad_rhs(b_true, B, T, blocks)
        rhs = EndpointTreeVector(
            shape, wp.array(tail_h, dtype=wp.float64, device=dev),
            wp.array(root_h, dtype=wp.float64, device=dev))
        out = EndpointTreeVector(
            shape, wp.zeros((B, T, BLK), dtype=wp.float64,
                            device=dev),
            wp.zeros((N_R,), dtype=wp.float64, device=dev))
        solver.solve(rhs, out=out)

        s = CudssCholesky(lower, b_true, precision="float64")
        s.plan(st)
        s.factorize(st)
        s.solve(st)

        ch = CholmodDirect(lower)
        pd = PardisoDirect(lower) if PARDISO_AVAILABLE else None
        wp.synchronize_device(dev)

        # the two GPU solvers form one interleaved pair per phase; the
        # CPU side times PARDISO alone (CHOLMOD is the untimed reference)
        gpu_factor = {"endpoint": solver.factorize,
                      "cudss": lambda: s.factorize(st)}
        gpu_solve = {"endpoint": lambda: solver.solve(rhs, out=out),
                     "cudss": lambda: s.solve(st)}
        gpu_fns = {"endpoint_factor": gpu_factor["endpoint"],
                   "cudss_factor": gpu_factor["cudss"],
                   "endpoint_solve": gpu_solve["endpoint"],
                   "cudss_solve": gpu_solve["cudss"]}
        cpu_fns = {}
        if pd is not None:
            cpu_fns = {"pardiso_factor": pd.factorize,
                       "pardiso_solve": lambda: pd.solve(b_true)}

        # ---- steady state, then interleaved timing per device.  Both
        # devices are re-warmed right before EACH timed block: while
        # one device is being timed the other idles and its clock
        # drops, so without this the second block would start cold.
        raw = []
        passes = settle(gpu_fns, cpu_fns, burn or args.settle)
        tel0 = telemetry()
        gf = timed_gpu_group(gpu_factor, raw, "factor")
        gs = timed_gpu_group(gpu_solve, raw, "solve")
        # CPU: a fixed phase-specific settle before each timed group
        c = {}
        if cpu_fns:
            settle_cpu(cpu_fns["pardiso_factor"], args.cpu_settle)
            tel1 = telemetry()
            c.update(timed_cpu_group(
                {"pardiso": cpu_fns["pardiso_factor"]}, raw, "factor"))
            c["pardiso_factor"] = c.pop("pardiso")
            settle_cpu(cpu_fns["pardiso_solve"], args.cpu_settle)
            tel1b = telemetry()
            c.update(timed_cpu_group(
                {"pardiso": cpu_fns["pardiso_solve"]}, raw, "solve"))
            c["pardiso_solve"] = c.pop("pardiso")
            tel2 = telemetry()
            # the factor-group start snapshot is the one that showed
            # slow clock recovery in earlier runs; keep it as "begin"
            # and record the solve-group start separately
            tel1["cpu_mhz_solve"] = tel1b["cpu_mhz"]
        else:
            tel1 = tel2 = telemetry()
        f_e, s_e = gf["endpoint"], gs["endpoint"]
        f_c, s_c = gf["cudss"], gs["cudss"]

        # ---- solutions and residuals (untimed)
        solver.solve(rhs, out=out)
        s.solve(st)
        wp.synchronize_device(dev)
        x_e = strip_solution(out.tail.numpy(), out.root.numpy(),
                             B, T, blocks)
        x_c = np.asarray(s.solution()).ravel()
        # untimed reference solution (one factorization at construction)
        x_h = np.asarray(ch.solve(b_true))

        def rel_res(x):
            return float(np.linalg.norm(K @ x - b_true)
                         / np.linalg.norm(b_true))

        gpu_therm = thermal(tel0, {k: v for k, v in tel1.items()
                                   if k != "cpu_mhz_solve"})
        cpu_therm = thermal(tel1, tel2)   # adds cpu_mhz_solve_begin
        records.append({**base, "solver": "endpoint",
                        **timing(f_e, s_e), **gpu_therm,
                        "extract_ms": f_x, "settle_passes": passes,
                        "residual": rel_res(x_e)})
        records.append({**base, "solver": "cudss",
                        **timing(f_c, s_c), **gpu_therm,
                        "solver_version": s.metadata()["cudss_version"],
                        "solver_config": f"ordering={s.ordering}",
                        "residual": rel_res(x_c)})
        s.free()

        agree_p, pardiso_txt = 0.0, ""
        if pd is not None:
            f_p, s_p = c["pardiso_factor"], c["pardiso_solve"]
            x_p = np.asarray(pd.solve(b_true)).reshape(-1)
            pd.free()
            records.append({**base, "solver": "pardiso",
                            **timing(f_p, s_p), **cpu_therm,
                            "solver_version": pd.mkl_version,
                            "solver_config": " ".join(
                                f"iparm({k})={v}" for k, v
                                in sorted(pd.iparm_used.items())),
                            "nnz_factor": pd.nnz_factor,
                            "residual": rel_res(x_p)})
            agree_p = np.linalg.norm(x_p - x_h) / np.linalg.norm(x_h)
            pardiso_txt = (f" | pardiso {f_p[0]:8.3f}/{s_p[0]:6.3f}")

        agree = max(
            np.linalg.norm(x_e - x_h) / np.linalg.norm(x_h),
            np.linalg.norm(x_c - x_h) / np.linalg.norm(x_h),
            agree_p)
        assert agree < 1e-9, f"solution mismatch {agree:.2e}"
        print(f"[{idx + 1:3d}/{len(grid)}] B={B:3d} T={T:3d} "
              f"n={n:6d} nnz={K.nnz:8d}: "
              f"extract {f_x:6.3f} | "
              f"endpoint {f_e[0]:6.3f}/{s_e[0]:6.3f} | cudss "
              f"{f_c[0]:7.3f}/{s_c[0]:6.3f}{pardiso_txt} | "
              f"agree {agree:.1e} | "
              f"gpu {tel0['gpu_temp_c']:.0f}C {tel0['gpu_sm_mhz']:.0f}MHz"
              f" | cpu {tel1['cpu_temp_c']:.0f}C "
              f"{tel1['cpu_mhz']:.0f}MHz",
              flush=True)
        # raw samples of this point (written only after all its timing)
        new_file = not os.path.exists(raw_path) or idx == 0
        with open(raw_path, "w" if new_file else "a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(raw_fields)
            for phase, name, bi, si, pos, ms in raw:
                w.writerow([campaign, idx, B, T, phase, name, bi, si, pos,
                            f"{ms:.6f}"])
        del solver, rhs, out, gpu_fns, cpu_fns, raw

    os.makedirs("results", exist_ok=True)
    fields = sorted({k for r in records for k in r})
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)
    print(f"wrote {args.output}")
    if args.no_plots:
        return

    _plot_heatmaps(records, args)


if __name__ == "__main__":
    main()

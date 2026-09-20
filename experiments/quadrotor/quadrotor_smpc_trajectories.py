"""Open-loop scenario plan of the branch-MPC quadrotor, in 3-D.

Plans the immediate-branching open-loop maneuver the endpoint
benchmark's matrices come from: the first decision is shared by all
scenarios, every scenario carries its own inputs from stage 1 on, and
each scenario samples its own mass and a bounded random
disturbance sequence acting in the x-y plane.  Planning uses
the quadrotor's differential flatness: the trajectory is optimized on
the flat outputs (positions with acceleration inputs and per-scenario
wind) by sequential convex programming with hard cylindrical obstacle
constraints per scenario, and the full attitude and rotor thrust are
recovered from the planned accelerations for rendering.

The figure shows the plan in the x-y plane: the scenario trajectory
bundle, the circular cross sections of the z-aligned cylindrical
obstacles, and quadrotor body glyphs (arms and rotors oriented by the
recovered attitude) along one representative scenario.

Usage (from this directory, socu env)::

    python quadrotor_smpc_trajectories.py [--samples 20]
                                          [--horizon 30]
                                          [--scp-iters 10]

Writes figures/quadrotor_smpc_trajectories.pdf.
"""

import argparse
import os

import numpy as np
import scipy.sparse as sp

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import osqp
from jax import jacfwd, vmap

jax.config.update("jax_enable_x64", True)

from quadrotor_endpoint_benchmark import (
    DRAG_V, GRAVITY, MASS, T_FINAL, X_FINAL, X_INIT)

# z-aligned cylindrical obstacles placed ON the straight chord from
# x_0 to x_g, so that the avoidance constraints are active and the
# planned corridors visibly separate (the benchmark's timing is
# insensitive to these values); only the (x, y) centers matter
# the cylinders sit on the chord but are biased to its lower-right
# side, so the short way around is unambiguous and the planned
# corridor is essentially single modal
OBS_POSITIONS = np.array([[0.69, 0.18],
                          [1.27, 0.47],
                          [1.65, 0.73]])
OBS_RADII = np.array([0.28, 0.34, 0.26])

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--samples", type=int, default=20)
parser.add_argument("--horizon", type=int, default=30)
parser.add_argument("--scp-iters", type=int, default=10)
args = parser.parse_args()

B, T = args.samples, args.horizon
dt = T_FINAL / T
MARGIN = 0.05
N_A = 3                     # flat-output input: world acceleration

rng = np.random.default_rng(0)
masses = np.asarray(MASS * (1.0 + 0.10 * rng.standard_normal(B)))
# bounded random disturbance SEQUENCES in the x-y plane: scenario i
# is one sampled realization w_t^i with |w| <= W_DIST per axis
W_DIST = 0.5
winds = W_DIST * rng.uniform(-1.0, 1.0, (B, T, 3))
winds[:, :, 2] = 0.0
winds_j = jnp.asarray(winds)
# per-scenario obstacle inflation (perception uncertainty): each
# scenario must avoid its own sampled obstacle sizes, so the planned
# corridors genuinely differ across the branch
radii_all = np.asarray(OBS_RADII)[None] * (
    1.0 + 0.10 * rng.uniform(0, 1, (B, len(OBS_RADII))))
radii_j = jnp.asarray(radii_all)

p0 = jnp.asarray(X_INIT[:3])
v0 = jnp.asarray(X_INIT[3:6])


def rollout(a_all, w_seq):
    """Positions and velocities under accelerations a and the sampled
    disturbance sequence w_seq (one row per stage)."""
    def f(carry, aw):
        p, v = carry
        a, w = aw
        vn = v + dt * (a + w)
        pn = p + dt * v + 0.5 * dt * dt * (a + w)
        return (pn, vn), (pn, vn)
    _, (ps, vs) = jax.lax.scan(f, (p0, v0), (a_all, w_seq))
    ps = jnp.concatenate([p0[None], ps], axis=0)
    vs = jnp.concatenate([v0[None], vs], axis=0)
    return ps, vs


def cons_one(a0, Ai, w, radii):
    """Stacked constraints for one scenario (bounds defined below)."""
    a_all = jnp.concatenate([a0[None], Ai], axis=0)
    ps, vs = rollout(a_all, w)
    # cylinders along z: only the horizontal distance matters
    d = ps[1:, None, :2] - jnp.asarray(OBS_POSITIONS)[None]
    dist = jnp.sqrt(jnp.sum(d ** 2, axis=2) + 1e-12)
    c_obs = (radii[None] + MARGIN - dist).ravel()
    c_term = jnp.concatenate([ps[-1] - jnp.asarray(X_FINAL[:3]),
                              vs[-1]])
    return jnp.concatenate([c_obs, c_term])


n_obs = len(OBS_RADII)
n_con = T * n_obs + 6
lo = np.concatenate([np.full(T * n_obs, -np.inf), np.zeros(6)])
hi = np.concatenate([np.zeros(T * n_obs), np.zeros(6)])

cons_val = jax.jit(vmap(cons_one, in_axes=(None, 0, 0, 0)))
cons_grad = jax.jit(vmap(jacfwd(cons_one, argnums=(0, 1)),
                         in_axes=(None, 0, 0, 0)))

n_ui = (T - 1) * N_A
n_z = N_A + B * n_ui
A_MAX = 2.0 * GRAVITY
z_lo = np.full(n_z, -A_MAX)
z_hi = np.full(n_z, A_MAX)

P_cost = 2.0 * dt * np.ones(n_z)
P_cost[N_A:] /= B
PROX = 1e-2
W_SLACK = 1e4           # L1 penalty on obstacle slacks
n_s = B * T * n_obs     # one slack per obstacle row

# warm start on the upper route: a quadratic Bezier through a
# waypoint above the obstacle chain puts every scenario in the same
# homotopy class before the avoidance constraints activate
WAY = np.array([0.75, 1.05, 0.55])
ss = (np.arange(T + 1) / T)[:, None]
p_ref = ((1 - ss) ** 2 * X_INIT[:3] + 2 * ss * (1 - ss) * WAY
         + ss ** 2 * X_FINAL[:3])
a_ref = np.zeros((T, 3))
a_ref[0] = 2.0 * (p_ref[1] - p_ref[0]) / dt ** 2 * 0.5
a_ref[1:] = np.diff(p_ref, 2, axis=0) / dt ** 2
z = np.zeros(n_z)
z[:N_A] = np.clip(a_ref[0], -A_MAX, A_MAX)
for i in range(B):
    z[N_A + i * n_ui:N_A + (i + 1) * n_ui] = np.clip(
        a_ref[1:] - winds[i, 1:], -A_MAX, A_MAX).ravel()
for it in range(args.scp_iters):
    a0 = jnp.asarray(z[:N_A])
    Ai = jnp.asarray(z[N_A:].reshape(B, T - 1, N_A))
    # obstacle-size homotopy: grow the cylinders over the first
    # iterations so the corridor is routed before the wall closes
    scale = min(1.0, 0.35 + 0.65 * it / 8.0)
    rad_it = radii_j * scale
    vals = np.asarray(cons_val(a0, Ai, winds_j, rad_it))
    g0, gU = cons_grad(a0, Ai, winds_j, rad_it)
    g0 = np.asarray(g0)
    gU = np.asarray(gU).reshape(B, n_con, n_ui)
    blocks = []
    for i in range(B):
        row = [sp.csr_matrix(g0[i])] + \
              [sp.csr_matrix((n_con, n_ui))] * B
        row[1 + i] = sp.csr_matrix(gU[i])
        blocks.append(sp.hstack(row, format="csr"))
    A_lin = sp.vstack(blocks, format="csr")
    shift = (A_lin @ z) - vals.ravel()
    # L1 slacks on the obstacle rows keep every subproblem feasible;
    # the terminal rows stay hard
    obs_mask = np.zeros(B * n_con, dtype=bool)
    for i in range(B):
        obs_mask[i * n_con:i * n_con + T * n_obs] = True
    S_obs = sp.csr_matrix(
        (-np.ones(n_s), (np.where(obs_mask)[0], np.arange(n_s))),
        shape=(B * n_con, n_s))
    A = sp.vstack([
        sp.hstack([A_lin, S_obs], format="csr"),
        sp.hstack([sp.eye(n_z, format="csr"),
                   sp.csr_matrix((n_z, n_s))], format="csr"),
        sp.hstack([sp.csr_matrix((n_s, n_z)),
                   sp.eye(n_s, format="csr")], format="csr"),
    ], format="csc")
    low = np.concatenate([np.tile(lo, B) + shift, z_lo,
                          np.zeros(n_s)])
    up = np.concatenate([np.tile(hi, B) + shift, z_hi,
                         np.full(n_s, np.inf)])
    Pm = sp.diags(np.concatenate(
        [P_cost + PROX, np.zeros(n_s)])).tocsc()
    q = np.concatenate([-PROX * z, W_SLACK * np.ones(n_s)])
    prob = osqp.OSQP()
    prob.setup(Pm, q, A, low, up, eps_abs=1e-7, eps_rel=1e-7,
               max_iter=40000, polishing=True, verbose=False)
    res = prob.solve()
    if res.info.status not in ("solved", "solved inaccurate"):
        print(f"iter {it}: OSQP status {res.info.status}")
    z_new = res.x[:n_z]
    s_max = float(res.x[n_z:].max())
    err = np.linalg.norm(z_new - z) / max(np.linalg.norm(z_new), 1e-9)
    z = z_new
    print(f"SCP iter {it:2d}: relative change {err:.2e}, "
          f"max slack {s_max:.2e}", flush=True)
    if err < 1e-7 and s_max < 1e-8:
        break

a0 = jnp.asarray(z[:N_A])
Ai = jnp.asarray(z[N_A:].reshape(B, T - 1, N_A))
vals = np.asarray(cons_val(a0, Ai, winds_j, radii_j))
viol = np.maximum(vals - hi, 0) + np.maximum(lo - vals, 0)
print(f"worst constraint violation over all scenarios: "
      f"{viol.max():.3e}")

# rollouts and flatness recovery of attitude and thrust
a_full = np.asarray(jnp.concatenate(
    [jnp.repeat(a0[None, None], B, axis=0), Ai], axis=1))
ps_vs = [np.asarray(x) for x in vmap(rollout, in_axes=(0, 0))(
    jnp.asarray(a_full), winds_j)]
pos, vel = ps_vs                                     # (B, T+1, 3)


def recover_attitude(a_cmd, v, m, w):
    """Rotation matrix and thrust from the flatness map: the body z
    axis aligns with the required rotor-thrust vector."""
    tvec = m * (a_cmd + np.array([0, 0, GRAVITY]) - w) + DRAG_V * v
    f = np.linalg.norm(tvec)
    zb = tvec / max(f, 1e-9)
    xc = np.array([1.0, 0.0, 0.0])                   # zero yaw
    yb = np.cross(zb, xc)
    yb /= max(np.linalg.norm(yb), 1e-9)
    xb = np.cross(yb, zb)
    return np.stack([xb, yb, zb], axis=1), f


# ------------------------------------------------------------- figure
plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                     "figure.dpi": 150, "savefig.bbox": "tight"})
fig, ax = plt.subplots(figsize=(7.2, 4.0))

# cylindrical obstacles: circular cross sections in the x-y plane
from matplotlib.patches import Circle

for o, (opos, rad) in enumerate(zip(OBS_POSITIONS, OBS_RADII)):
    ax.add_patch(Circle(opos, radius=rad, color="#d9534f",
                        alpha=0.32, zorder=1))
    # each scenario's inflated radius (incl. the safety margin): the
    # boundaries the plans actually graze
    for i in range(B):
        ax.add_patch(Circle(opos, radius=radii_all[i, o] + MARGIN,
                            facecolor="none", edgecolor="#d9534f",
                            linewidth=0.6, alpha=0.30, zorder=1))

# scenario trajectory bundle (top view)
for i in range(B):
    ax.plot(pos[i, :, 0], pos[i, :, 1], color="#157DEC", alpha=0.35,
            lw=1.1, zorder=3)

# quadrotor glyphs (top view) along one representative scenario
GLYPH_L = 0.11
arms = np.array([[GLYPH_L, 0, 0], [-GLYPH_L, 0, 0],
                 [0, GLYPH_L, 0], [0, -GLYPH_L, 0]])
th = np.linspace(0, 2 * np.pi, 24)
rotor = 0.04 * np.stack([np.cos(th), np.sin(th),
                         np.zeros_like(th)], axis=1)
for k in (0, T - 1):  # glyphs only at start and goal
    Rw, _ = recover_attitude(a_full[0, k], vel[0, k], masses[0],
                             winds[0, k])
    c = pos[0, k]
    for a in (0, 1), (2, 3):
        seg = np.stack([c + Rw @ arms[a[0]], c + Rw @ arms[a[1]]])
        ax.plot(seg[:, 0], seg[:, 1], color="#17212B", lw=1.6,
                zorder=5)
    for j in range(4):
        ring = c + (Rw @ (arms[j][:, None] + rotor.T)).T
        ax.plot(ring[:, 0], ring[:, 1], color="#17212B", lw=0.8,
                alpha=0.85, zorder=5)

ax.scatter(X_INIT[0], X_INIT[1], color="k", s=32, zorder=6)
ax.scatter(X_FINAL[0], X_FINAL[1], color="r", s=32, zorder=6)
ax.text(X_INIT[0] - 0.05, X_INIT[1] - 0.16, r"$x_0$", fontsize=15,
        ha="center", va="top")
ax.text(X_FINAL[0] + 0.02, X_FINAL[1] - 0.16, r"$x_g$", fontsize=15,
        ha="center", va="top", color="r")

ax.set_xlabel(r"$p_x$", fontsize=14)
ax.set_ylabel(r"$p_y$", fontsize=14, rotation=0, labelpad=8)
ax.set_aspect("equal")
ax.tick_params(labelsize=9)
for side in ax.spines.values():
    side.set_linewidth(0.6)

os.makedirs("figures", exist_ok=True)
out = "figures/quadrotor_smpc_trajectories.pdf"
fig.savefig(out, bbox_inches="tight", dpi=300)
print("saved", out)

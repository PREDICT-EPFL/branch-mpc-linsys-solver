"""Closed-loop branch MPC for the quadrotor, run in receding horizon.

At every control step the immediate-branching scenario problem is
re-solved from the current state: all B scenarios (each a sampled
bounded x-y disturbance sequence, plus its own perceived obstacle
size) share the first acceleration, each scenario plans its own
recourse afterwards, and hard cylindrical obstacle constraints hold
per scenario.  The shared first action is applied to the true plant,
which experiences a fresh bounded random disturbance at every step
that the controller never knows, and the horizon recedes.  Planning uses the flat-output model of
quadrotor_smpc_trajectories.py; each re-solve is warm-started from the
previous plan.

The figure shows the executed closed-loop trajectory in the x-y plane
together with the predicted scenario fans at a few snapshots, the
nominal obstacles, and each scenario's inflated constraint boundary.

Usage (from this directory, socu env)::

    python quadrotor_smpc_closedloop.py [--samples 20] [--horizon 30]
                                        [--sim-steps 45]

Writes figures/quadrotor_smpc_closedloop.pdf.
"""

import argparse
import os

import numpy as np
import scipy.sparse as sp

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import jax
import jax.numpy as jnp
import osqp
from jax import jacfwd, vmap

jax.config.update("jax_enable_x64", True)

from quadrotor_endpoint_benchmark import (
    GRAVITY, MASS, T_FINAL, X_FINAL, X_INIT)

OBS_POSITIONS = np.array([[0.69, 0.18],
                          [1.27, 0.47],
                          [1.65, 0.73]])
OBS_RADII = np.array([0.28, 0.34, 0.26])

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--samples", type=int, default=20)
parser.add_argument("--horizon", type=int, default=30)
parser.add_argument("--sim-steps", type=int, default=45)
args = parser.parse_args()

B, T = args.samples, args.horizon
dt = T_FINAL / T
MARGIN = 0.05
N_A = 3
n_obs = len(OBS_RADII)

rng = np.random.default_rng(0)
# bounded random disturbance SEQUENCES in the x-y plane: scenario i
# is one sampled realization w_t^i with |w| <= W_DIST per axis
W_DIST = 0.5
winds = W_DIST * rng.uniform(-1.0, 1.0, (B, T, 3))
winds[:, :, 2] = 0.0
winds_j = jnp.asarray(winds)
radii_all = np.asarray(OBS_RADII)[None] * (
    1.0 + 0.10 * rng.uniform(0, 1, (B, n_obs)))
radii_j = jnp.asarray(radii_all)
# the true plant disturbance: a fresh bounded random draw at every
# simulation step (never known to the controller)
rng_true = np.random.default_rng(123)


def rollout(a_all, w_seq, p0, v0):
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


def cons_one(a0, Ai, w, radii, p0, v0):
    a_all = jnp.concatenate([a0[None], Ai], axis=0)
    ps, vs = rollout(a_all, w, p0, v0)
    d = ps[1:, None, :2] - jnp.asarray(OBS_POSITIONS)[None]
    dist = jnp.sqrt(jnp.sum(d ** 2, axis=2) + 1e-12)
    c_obs = (radii[None] + MARGIN - dist).ravel()
    c_term = jnp.concatenate([ps[-1] - jnp.asarray(X_FINAL[:3]),
                              vs[-1]])
    return jnp.concatenate([c_obs, c_term])


n_con = T * n_obs + 6
lo = np.concatenate([np.full(T * n_obs, -np.inf), np.zeros(6)])
hi = np.concatenate([np.zeros(T * n_obs), np.zeros(6)])

cons_val = jax.jit(vmap(cons_one,
                        in_axes=(None, 0, 0, 0, None, None)))
cons_grad = jax.jit(vmap(jacfwd(cons_one, argnums=(0, 1)),
                         in_axes=(None, 0, 0, 0, None, None)))

n_ui = (T - 1) * N_A
n_z = N_A + B * n_ui
A_MAX = 2.0 * GRAVITY
z_lo = np.full(n_z, -A_MAX)
z_hi = np.full(n_z, A_MAX)
P_cost = 2.0 * dt * np.ones(n_z)
P_cost[N_A:] /= B
PROX = 1e-2
W_SLACK = 1e4
n_s = B * T * n_obs

obs_mask = np.zeros(B * n_con, dtype=bool)
for i in range(B):
    obs_mask[i * n_con:i * n_con + T * n_obs] = True
S_obs = sp.csr_matrix(
    (-np.ones(n_s), (np.where(obs_mask)[0], np.arange(n_s))),
    shape=(B * n_con, n_s))


def solve_mpc(p0, v0, z, iters, homotopy):
    """SCP solve of the branch problem from (p0, v0), warm start z."""
    p0j, v0j = jnp.asarray(p0), jnp.asarray(v0)
    for it in range(iters):
        scale = min(1.0, 0.35 + 0.65 * it / 8.0) if homotopy else 1.0
        rad_it = radii_j * scale
        a0 = jnp.asarray(z[:N_A])
        Ai = jnp.asarray(z[N_A:].reshape(B, T - 1, N_A))
        vals = np.asarray(cons_val(a0, Ai, winds_j, rad_it, p0j, v0j))
        g0, gU = cons_grad(a0, Ai, winds_j, rad_it, p0j, v0j)
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
        z = res.x[:n_z]
    return z


# initial warm start: Bezier through a waypoint above the chain
WAY = np.array([0.75, 1.05, 0.55])
ss = (np.arange(T + 1) / T)[:, None]
p_ref = ((1 - ss) ** 2 * X_INIT[:3] + 2 * ss * (1 - ss) * WAY
         + ss ** 2 * X_FINAL[:3])
a_ref = np.zeros((T, 3))
a_ref[0] = (p_ref[1] - p_ref[0]) / dt ** 2
a_ref[1:] = np.diff(p_ref, 2, axis=0) / dt ** 2
z = np.zeros(n_z)
z[:N_A] = np.clip(a_ref[0], -A_MAX, A_MAX)
for i in range(B):
    z[N_A + i * n_ui:N_A + (i + 1) * n_ui] = np.clip(
        a_ref[1:] - winds[i, 1:], -A_MAX, A_MAX).ravel()

# ------------------------------------------------- receding horizon
p, v = X_INIT[:3].copy(), X_INIT[3:6].copy()
executed = [p.copy()]
snapshots = {}
SNAP_AT = (0, 12, 24)
for step in range(args.sim_steps):
    z = solve_mpc(p, v, z, iters=30 if step == 0 else 6,
                  homotopy=(step == 0))
    if step in SNAP_AT:
        a0 = jnp.asarray(z[:N_A])
        Ai = jnp.asarray(z[N_A:].reshape(B, T - 1, N_A))
        bundle = np.asarray(vmap(
            lambda U, w: rollout(
                jnp.concatenate([a0[None], U], axis=0), w,
                jnp.asarray(p), jnp.asarray(v))[0])(Ai, winds_j))
        snapshots[step] = bundle
    a_apply = z[:N_A]
    w_step = np.zeros(3)
    w_step[:2] = W_DIST * rng_true.uniform(-1.0, 1.0, 2)
    v_new = v + dt * (a_apply + w_step)
    p = p + dt * v + 0.5 * dt * dt * (a_apply + w_step)
    v = v_new
    executed.append(p.copy())
    err = np.linalg.norm(p - X_FINAL[:3])
    print(f"step {step:2d}: |p - x_g| = {err:.3f}", flush=True)
    if err < 0.03 and np.linalg.norm(v) < 0.15:
        break
executed = np.asarray(executed)

# distance of the executed path to every nominal obstacle
dmin = min(np.linalg.norm(executed[:, :2] - OBS_POSITIONS[o],
                          axis=1).min() - OBS_RADII[o]
           for o in range(n_obs))
print(f"closed-loop clearance to nominal obstacles: {dmin:.3f}")

# ------------------------------------------------------------- figure
plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                     "figure.dpi": 150, "savefig.bbox": "tight"})
fig, ax = plt.subplots(figsize=(7.2, 4.0))
for o, (opos, rad) in enumerate(zip(OBS_POSITIONS, OBS_RADII)):
    ax.add_patch(Circle(opos, radius=rad, color="#d9534f",
                        alpha=0.32, zorder=1))
    for i in range(B):
        ax.add_patch(Circle(opos, radius=radii_all[i, o] + MARGIN,
                            facecolor="none", edgecolor="#d9534f",
                            linewidth=0.6, alpha=0.30, zorder=1))
for step, bundle in snapshots.items():
    for i in range(B):
        ax.plot(bundle[i, :, 0], bundle[i, :, 1], color="#157DEC",
                alpha=0.18, lw=0.9, zorder=2)
    ax.scatter(bundle[0, 0, 0], bundle[0, 0, 1], color="#0b4f9e",
               s=16, zorder=5)
ax.plot(executed[:, 0], executed[:, 1], color="#0b4f9e", lw=2.4,
        zorder=4, label="closed loop")
ax.scatter(X_INIT[0], X_INIT[1], color="k", s=32, zorder=6)
ax.scatter(X_FINAL[0], X_FINAL[1], color="r", s=32, zorder=6)
ax.text(X_INIT[0] - 0.05, X_INIT[1] - 0.14, r"$x_0$", fontsize=15,
        ha="center", va="top")
ax.text(X_FINAL[0] + 0.02, X_FINAL[1] - 0.14, r"$x_g$", fontsize=15,
        ha="center", va="top", color="r")
ax.set_xlabel(r"$p_x$", fontsize=14)
ax.set_ylabel(r"$p_y$", fontsize=14, rotation=0, labelpad=8)
ax.set_aspect("equal")
ax.tick_params(labelsize=9)
for side in ax.spines.values():
    side.set_linewidth(0.6)
os.makedirs("figures", exist_ok=True)
out = "figures/quadrotor_smpc_closedloop.pdf"
fig.savefig(out, bbox_inches="tight", dpi=300)
print("saved", out)

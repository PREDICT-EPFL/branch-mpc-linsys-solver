"""Open-loop branch-MPC plan for the quadrotor via CasADi and IPOPT.

Concept illustration on the FULL 12-state model (position, velocity,
ZYX Euler attitude, body rates; rotor thrusts as inputs), solved as a
single multiple-shooting NLP with IPOPT -- no SCP, no flat-output
approximation.  The immediate-branching structure is explicit: the
first input u_0 is one shared variable, x_0 is the known initial
state, and every scenario carries its own states and inputs from stage
1 on.  Scenario i samples a bounded random disturbance sequence acting
in the x-y plane and its own perceived (inflated) obstacle radii;
obstacles are z-aligned cylinders, enforced as hard constraints for
every scenario and stage.

The figure shows the planned scenario bundle in the x-y plane with the
nominal cylinders, each scenario's inflated constraint circle, and
quadrotor glyphs at start and goal oriented by the SOLVED attitude.

Usage (from this directory, socu env)::

    python quadrotor_nlp_trajectories.py [--samples 20] [--horizon 30]

Writes figures/quadrotor_nlp_trajectories.pdf.
"""

import argparse
import os

import casadi as ca
import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from quadrotor_endpoint_benchmark import (
    ARM, C_TAU, DRAG_V, DRAG_W, GRAVITY, J_INERTIA, MASS, N_U, N_X,
    T_FINAL, X_FINAL, X_INIT)

OBS_POSITIONS = np.array([[0.69, 0.18],
                          [1.27, 0.47],
                          [1.65, 0.73]])
OBS_RADII = np.array([0.28, 0.34, 0.26])
MARGIN = 0.05

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--samples", type=int, default=20)
parser.add_argument("--horizon", type=int, default=30)
args = parser.parse_args()

B, T = args.samples, args.horizon
dt = T_FINAL / T
n_obs = len(OBS_RADII)
hover = MASS * GRAVITY / 4.0

rng = np.random.default_rng(0)
masses = MASS * (1.0 + 0.10 * rng.standard_normal(B))
W_DIST = 0.5
winds = W_DIST * rng.uniform(-1.0, 1.0, (B, T, 3))
winds[:, :, 2] = 0.0
radii_all = OBS_RADII[None] * (1.0 + 0.10 * rng.uniform(0, 1, (B, n_obs)))


def dynamics(x, u, m, w):
    """Continuous-time full-attitude quadrotor dynamics (CasADi)."""
    v, ang, om = x[3:6], x[6:9], x[9:12]
    cph, sph = ca.cos(ang[0]), ca.sin(ang[0])
    cth, sth = ca.cos(ang[1]), ca.sin(ang[1])
    cps, sps = ca.cos(ang[2]), ca.sin(ang[2])
    r3 = ca.vertcat(cps * sth * cph + sps * sph,
                    sps * sth * cph - cps * sph,
                    cth * cph)
    thrust = u[0] + u[1] + u[2] + u[3]
    a = (r3 * thrust - DRAG_V * v) / m + w \
        - ca.vertcat(0.0, 0.0, GRAVITY)
    tth = sth / cth
    ang_dot = ca.vertcat(om[0] + sph * tth * om[1] + cph * tth * om[2],
                         cph * om[1] - sph * om[2],
                         sph / cth * om[1] + cph / cth * om[2])
    tau = ca.vertcat(ARM * (u[1] - u[3]),
                     ARM * (u[2] - u[0]),
                     C_TAU * (u[0] - u[1] + u[2] - u[3]))
    Jv = ca.DM(J_INERTIA)
    om_dot = (tau - ca.cross(om, Jv * om)) / Jv - DRAG_W * om
    return ca.vertcat(v, a, ang_dot, om_dot)


def rk4(x, u, m, w):
    k1 = dynamics(x, u, m, w)
    k2 = dynamics(x + 0.5 * dt * k1, u, m, w)
    k3 = dynamics(x + 0.5 * dt * k2, u, m, w)
    k4 = dynamics(x + dt * k3, u, m, w)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def build_nlp(x0_par):
    """Multiple-shooting branch NLP; returns solver pieces."""
    u0 = ca.MX.sym("u0", N_U)
    w_list, w0, lbw, ubw = [u0], [hover * np.ones(N_U)], \
        [np.zeros(N_U)], [4.0 * hover * np.ones(N_U)]
    g_list, lbg, ubg = [], [], []
    cost = ca.MX(0)

    # a gently arcing position guess above the obstacle chain
    WAY = np.array([0.75, 1.05, 0.55])
    ss = (np.arange(T + 1) / T)[:, None]
    p_guess = ((1 - ss) ** 2 * X_INIT[:3]
               + 2 * ss * (1 - ss) * WAY + ss ** 2 * X_FINAL[:3])

    X_all = []
    for i in range(B):
        Xi = ca.MX.sym(f"X{i}", N_X, T)          # stages 1..T
        Ui = ca.MX.sym(f"U{i}", N_U, T - 1)      # inputs 1..T-1
        X_all.append(Xi)
        w_list += [ca.vec(Xi), ca.vec(Ui)]
        xg = np.zeros((N_X, T))
        xg[:3] = p_guess[1:].T
        w0 += [xg.reshape(-1, order="F"),
               hover * np.ones(N_U * (T - 1))]
        lbx = np.full((N_X, T), -np.inf)
        ubx = np.full((N_X, T), np.inf)
        lbx[6:8, :] = -1.2                       # Euler safety bounds
        ubx[6:8, :] = 1.2
        lbw += [lbx.reshape(-1, order="F"),
                np.zeros(N_U * (T - 1))]
        ubw += [ubx.reshape(-1, order="F"),
                4.0 * hover * np.ones(N_U * (T - 1))]

        # multiple-shooting dynamics; t = 0 uses the SHARED u_0
        xprev = x0_par
        for t in range(T):
            u_t = u0 if t == 0 else Ui[:, t - 1]
            g_list.append(Xi[:, t] - rk4(xprev, u_t, masses[i],
                                         ca.DM(winds[i, t])))
            lbg.append(np.zeros(N_X))
            ubg.append(np.zeros(N_X))
            xprev = Xi[:, t]
        # cylindrical obstacles, scenario-inflated radii
        for t in range(T):
            for o in range(n_obs):
                dsq = ((Xi[0, t] - OBS_POSITIONS[o, 0]) ** 2
                       + (Xi[1, t] - OBS_POSITIONS[o, 1]) ** 2)
                g_list.append(dsq)
                lbg.append(np.array(
                    [(radii_all[i, o] + MARGIN) ** 2]))
                ubg.append(np.array([np.inf]))
        # terminal: reach the goal position at rest
        g_list.append(Xi[:6, T - 1] - ca.DM(X_FINAL[:6]))
        lbg.append(np.zeros(6))
        ubg.append(np.zeros(6))
        # control energy around hover, plus velocity and body-rate
        # regularization (short, calm paths)
        cost = cost + dt * ca.sumsqr(Ui - hover) / B \
            + 0.20 * dt * ca.sumsqr(Xi[3:6, :]) / B \
            + 1e-2 * ca.sumsqr(Xi[9:12, :]) / B
    cost = cost + dt * ca.sumsqr(u0 - hover)

    nlp = {"x": ca.vertcat(*w_list), "f": cost,
           "g": ca.vertcat(*g_list), "p": x0_par}
    return (nlp, np.concatenate([np.atleast_1d(v).ravel()
                                 for v in w0]),
            np.concatenate(lbw), np.concatenate(ubw),
            np.concatenate(lbg), np.concatenate(ubg))


x0_par = ca.MX.sym("x0", N_X)
nlp, w0, lbw, ubw, lbg, ubg = build_nlp(x0_par)
solver = ca.nlpsol("solver", "ipopt", nlp,
                   {"ipopt.print_level": 0, "print_time": 0,
                    "ipopt.max_iter": 800})
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=X_INIT)
stats = solver.stats()
print("IPOPT status:", stats["return_status"],
      "iters:", stats["iter_count"])

wv = np.asarray(sol["x"]).ravel()
u0_sol = wv[:N_U]
per = N_X * T + N_U * (T - 1)
xs = np.empty((B, T + 1, N_X))
for i in range(B):
    seg = wv[N_U + i * per:N_U + i * per + N_X * T]
    xs[i, 0] = X_INIT
    xs[i, 1:] = seg.reshape(N_X, T, order="F").T

# feasibility check on the true nonlinear obstacle constraints
d = np.linalg.norm(xs[:, 1:, None, :2]
                   - OBS_POSITIONS[None, None], axis=3)
viol = np.maximum(radii_all[:, None, :] + MARGIN - d, 0).max()
print(f"worst obstacle violation: {viol:.2e}")

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
for i in range(B):
    ax.plot(xs[i, :, 0], xs[i, :, 1], color="#157DEC", alpha=0.35,
            lw=1.1, zorder=3)


def rot_zyx(ang):
    cph, sph = np.cos(ang[0]), np.sin(ang[0])
    cth, sth = np.cos(ang[1]), np.sin(ang[1])
    cps, sps = np.cos(ang[2]), np.sin(ang[2])
    return np.array([
        [cps * cth, cps * sth * sph - sps * cph,
         cps * sth * cph + sps * sph],
        [sps * cth, sps * sth * sph + cps * cph,
         sps * sth * cph - cps * sph],
        [-sth, cth * sph, cth * cph]])


GLYPH_L = 0.11
arms = np.array([[GLYPH_L, 0, 0], [-GLYPH_L, 0, 0],
                 [0, GLYPH_L, 0], [0, -GLYPH_L, 0]])
th = np.linspace(0, 2 * np.pi, 24)
rotor = 0.04 * np.stack([np.cos(th), np.sin(th),
                         np.zeros_like(th)], axis=1)
for k in (0, T):
    Rw = rot_zyx(xs[0, k, 6:9])
    c = xs[0, k, :3]
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
out = "figures/quadrotor_nlp_trajectories.pdf"
fig.savefig(out, bbox_inches="tight", dpi=300)
print("saved", out)

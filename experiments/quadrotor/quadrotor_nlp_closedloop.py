"""Closed-loop branch MPC for the quadrotor via CasADi and IPOPT.

Concept illustration on the FULL 12-state model, run in receding
horizon: at every control step the immediate-branching multiple-
shooting NLP of quadrotor_nlp_trajectories.py is re-solved by IPOPT
from the current state (the initial state enters as a parameter, the
first input u_0 is shared by all scenarios, every scenario carries its
own recourse and its own sampled disturbance sequence and inflated
obstacle radii).  The shared u_0 is applied to the true plant, which
experiences a fresh bounded random x-y disturbance at every step, and
the horizon recedes.  Warm starts reuse the previous solution.

The figure shows the executed closed-loop trajectory in the x-y plane
with predicted scenario fans at a few snapshots.

Usage (from this directory, socu env)::

    python quadrotor_nlp_closedloop.py [--samples 20] [--horizon 30]
                                       [--sim-steps 70]

Writes figures/quadrotor_nlp_closedloop.pdf.
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
parser.add_argument("--sim-steps", type=int, default=70)
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
rng_true = np.random.default_rng(123)


def dynamics(x, u, m, w):
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


# one-step plant function (numeric evaluation of the same model)
x_s = ca.MX.sym("x", N_X)
u_s = ca.MX.sym("u", N_U)
w_s = ca.MX.sym("w", 3)
F_plant = ca.Function("F", [x_s, u_s, w_s],
                      [rk4(x_s, u_s, MASS, w_s)])

# ------------------------------------------------ parametric NLP
x0_par = ca.MX.sym("x0", N_X)
u0 = ca.MX.sym("u0", N_U)
w_list = [u0]
lbw = [np.zeros(N_U)]
ubw = [4.0 * hover * np.ones(N_U)]
g_list, lbg, ubg = [], [], []
cost = ca.MX(0)
for i in range(B):
    Xi = ca.MX.sym(f"X{i}", N_X, T)
    Ui = ca.MX.sym(f"U{i}", N_U, T - 1)
    w_list += [ca.vec(Xi), ca.vec(Ui)]
    lbx = np.full((N_X, T), -np.inf)
    ubx = np.full((N_X, T), np.inf)
    lbx[6:8, :] = -1.2
    ubx[6:8, :] = 1.2
    lbw += [lbx.reshape(-1, order="F"), np.zeros(N_U * (T - 1))]
    ubw += [ubx.reshape(-1, order="F"),
            4.0 * hover * np.ones(N_U * (T - 1))]
    xprev = x0_par
    for t in range(T):
        u_t = u0 if t == 0 else Ui[:, t - 1]
        g_list.append(Xi[:, t] - rk4(xprev, u_t, masses[i],
                                     ca.DM(winds[i, t])))
        lbg.append(np.zeros(N_X))
        ubg.append(np.zeros(N_X))
        xprev = Xi[:, t]
    for t in range(T):
        for o in range(n_obs):
            dsq = ((Xi[0, t] - OBS_POSITIONS[o, 0]) ** 2
                   + (Xi[1, t] - OBS_POSITIONS[o, 1]) ** 2)
            g_list.append(dsq)
            lbg.append(np.array([(radii_all[i, o] + MARGIN) ** 2]))
            ubg.append(np.array([np.inf]))
    g_list.append(Xi[:6, T - 1] - ca.DM(X_FINAL[:6]))
    lbg.append(np.zeros(6))
    ubg.append(np.zeros(6))
    cost = cost + dt * ca.sumsqr(Ui - hover) / B \
        + 0.05 * dt * ca.sumsqr(Xi[3:6, :]) / B \
        + 0.10 * dt * ca.sumsqr(Xi[:3, :]
                                - ca.DM(X_FINAL[:3])) / B \
        + 1e-2 * ca.sumsqr(Xi[9:12, :]) / B
# move suppression: keep the shared first action close to the one of
# the previous MPC step to avoid flipping between local plans
u0_prev = ca.MX.sym("u0_prev", N_U)
cost = cost + dt * ca.sumsqr(u0 - hover) \
    + 0.05 * ca.sumsqr(u0 - u0_prev)

nlp = {"x": ca.vertcat(*w_list), "f": cost,
       "g": ca.vertcat(*g_list),
       "p": ca.vertcat(x0_par, u0_prev)}
solver = ca.nlpsol("solver", "ipopt", nlp,
                   {"ipopt.print_level": 0, "print_time": 0,
                    "ipopt.max_iter": 500,
                    "ipopt.warm_start_init_point": "yes"})
lbw = np.concatenate(lbw)
ubw = np.concatenate(ubw)
lbg = np.concatenate(lbg)
ubg = np.concatenate(ubg)

# initial guess: arcing positions, hover thrusts
WAY = np.array([0.75, 1.05, 0.55])
ss = (np.arange(T + 1) / T)[:, None]
p_guess = ((1 - ss) ** 2 * X_INIT[:3]
           + 2 * ss * (1 - ss) * WAY + ss ** 2 * X_FINAL[:3])
w0 = [hover * np.ones(N_U)]
for i in range(B):
    xg = np.zeros((N_X, T))
    xg[:3] = p_guess[1:].T
    w0 += [xg.reshape(-1, order="F"), hover * np.ones(N_U * (T - 1))]
w_guess = np.concatenate(w0)

per = N_X * T + N_U * (T - 1)

# ------------------------------------------------ receding horizon
x = X_INIT.copy()
executed = [x[:3].copy()]
snapshots = {}
SNAP_AT = (0, 15, 30)
u0_last = hover * np.ones(N_U)
for step in range(args.sim_steps):
    sol = solver(x0=w_guess, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg,
                 p=np.concatenate([x, u0_last]))
    status = solver.stats()["return_status"]
    if status not in ("Solve_Succeeded",
                      "Solved_To_Acceptable_Level"):
        print(f"step {step}: IPOPT {status}")
    w_guess = np.asarray(sol["x"]).ravel()
    u0_sol = w_guess[:N_U]
    u0_last = u0_sol
    if step in SNAP_AT:
        bundle = np.empty((B, T + 1, 3))
        for i in range(B):
            seg = w_guess[N_U + i * per:N_U + i * per + N_X * T]
            bundle[i, 0] = x[:3]
            bundle[i, 1:] = seg.reshape(N_X, T, order="F")[:3].T
        snapshots[step] = bundle
    w_step = np.zeros(3)
    w_step[:2] = W_DIST * rng_true.uniform(-1.0, 1.0, 2)
    x = np.asarray(F_plant(x, u0_sol, w_step)).ravel()
    executed.append(x[:3].copy())
    err = np.linalg.norm(x[:3] - X_FINAL[:3])
    print(f"step {step:2d}: |p - x_g| = {err:.3f}", flush=True)
    if err < 0.03 and np.linalg.norm(x[3:6]) < 0.15:
        break
executed = np.asarray(executed)

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
        zorder=4)
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
out = "figures/quadrotor_nlp_closedloop.pdf"
fig.savefig(out, bbox_inches="tight", dpi=300)
print("saved", out)

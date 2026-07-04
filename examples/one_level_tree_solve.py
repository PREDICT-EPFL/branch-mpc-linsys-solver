"""Runnable example: solve a one-level tree KKT system with OneLevelTreeCholesky.

A one-level tree couples a shared root block ``x0`` to ``B`` independent tails
of equal length ``T`` (all blocks of size ``n``)::

    x0 -> x_{1,1} -> x_{1,2} -> ... -> x_{1,T}
    x0 -> x_{2,1} -> ... -> x_{2,T}
    ...

Each tail is a block-tridiagonal system solved by SOCU; the root is eliminated
via a Schur complement.  Run with::

    python examples/one_level_tree_solve.py
"""

import logging
import os
import sys

import numpy as np
import warp as wp

# silence Warp module generate/load + init-banner logging
wp.config.log_level = logging.WARNING

# allow running as `python examples/one_level_tree_solve.py` from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tree_kkt import OneLevelTreeCholesky


def build_spd_one_level_tree(B, T, n, q, seed=0):
    """Random SPD one-level tree; returns structured blocks + dense reference."""
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

    # D0 = alpha I + sum_i C_i^T K_i^{-1} C_i  => root Schur complement = alpha I
    D0 = np.eye(n)
    for i in range(B):
        Ki = np.zeros((T * n, T * n))
        for k in range(T):
            Ki[k*n:(k+1)*n, k*n:(k+1)*n] = Kd[i, k]
            if k < T - 1:
                Ki[(k+1)*n:(k+2)*n, k*n:(k+1)*n] = Ke[i, k]
                Ki[k*n:(k+1)*n, (k+1)*n:(k+2)*n] = Ke[i, k].T
        Ci = np.zeros((T * n, n)); Ci[:n] = G[i]
        D0 += Ci.T @ np.linalg.solve(Ki, Ci)

    r_root = rng.standard_normal((n, q))
    r_tail = rng.standard_normal((B, T, n, q))
    return D0, Kd, Ke, G, r_root, r_tail


def main():
    wp.init()
    B, T, n, q = 4, 31, 8, 2
    device, dtype = "cuda:0", wp.float64

    print(f"One-level tree: B={B} tails, T={T} nodes/tail, n={n}, num_rhs={q}")
    D0, Kd, Ke, G, r_root, r_tail = build_spd_one_level_tree(B, T, n, q, seed=0)

    solver = OneLevelTreeCholesky(B, T, n, dtype=dtype, device=device)
    print(f"interface_pos = {solver.interface_pos}  (physical first tail node)")

    # Separate factorize / solve (reusable factorization).
    solver.factorize(D0, Kd, Ke, G)
    print(f"tail_factor_status = {solver.tail_factor_status}")
    print(f"root_factor_status = {solver.root_factor_status}")
    print(f"root Schur complement S0 ~= I:  ||S0 - I||_inf = "
          f"{np.abs(solver.schur_complement - np.eye(n)).max():.2e}")

    x_root, x_tail = solver.solve(r_root, r_tail)
    print(f"x_root shape {x_root.shape}, x_tail shape {x_tail.shape}")

    # Structured residual (no dense global matrix on the production path;
    # compute_residual is a debug helper that works block-wise).
    res = solver.compute_residual(D0, Kd, Ke, G, r_root, r_tail, x_root, x_tail)
    print(f"structured residual ||K x - r||_inf = {res:.3e}")

    # Reuse the factorization for a fresh right-hand side.
    rng = np.random.default_rng(123)
    r_root2 = rng.standard_normal((n, q))
    r_tail2 = rng.standard_normal((B, T, n, q))
    x_root2, x_tail2 = solver.solve(r_root2, r_tail2)
    res2 = solver.compute_residual(D0, Kd, Ke, G, r_root2, r_tail2, x_root2, x_tail2)
    print(f"reused factorization, second residual = {res2:.3e}")

    assert res < 1e-8 and res2 < 1e-8
    print("OK")


if __name__ == "__main__":
    main()

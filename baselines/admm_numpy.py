"""NumPy reference of the scaled ADMM used by the unified QP solver.

Correctness reference only: implements exactly the equations, projection,
residuals, and stopping rule of :mod:`admm` on the host with dense/SciPy
linear algebra, so the GPU implementations can be validated iterate by
iterate.

The scaled formulation with a fixed per-constraint penalty
``R = diag(rho_vec)`` (``rho`` may be one scalar or a length-m vector;
the dual variable is ``v = R^-1 lambda``, so the unscaled multiplier is
``lambda = rho_vec * v``)::

    b   = -q + A.T @ (rho_vec * (z - v))
    x   = solve(P + A.T @ diag(rho_vec) @ A, b)
    a   = A @ x
    z+  = clip(a + v, l, u)
    v+  = v + a - z+

Residuals and tolerances (infinity norms), checked every ``check_every``
iterations::

    r_p = a - z+
    r_d = A.T @ (rho_vec * (z+ - z))
    eps_p = eps_abs + eps_rel * max(||a||, ||z+||)
    eps_d = eps_abs + eps_rel * ||A.T @ (rho_vec * v+)||
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class ReferenceResult:
    """Solution triple and convergence report of the reference solver."""

    x: np.ndarray
    z: np.ndarray
    v: np.ndarray           # scaled dual; multiplier lambda = rho_vec * v
    iterations: int
    converged: bool
    primal_residual: float
    dual_residual: float


def solve_qp_reference(P, q, A, l, u, rho=1.0, max_iter=200, eps_abs=1e-5,
                       eps_rel=1e-4, check_every=10, alpha=1.0, x0=None,
                       z0=None, v0=None, record_iterates=False):
    """Solve ``min 0.5 x'Px + q'x  s.t.  l <= Ax <= u`` with scaled ADMM.

    ``P`` is the upper triangle of the symmetric cost (CSC), ``A`` the
    constraint matrix (CSC); ``rho`` is one scalar or a length-m
    per-constraint vector.  ``K = P + A' diag(rho) A`` must be SPD; a
    non-positive pivot raises ``numpy.linalg.LinAlgError``.  ``alpha``
    is the over-relaxation parameter (0 < alpha < 2): the projection
    and dual update see ``a_hat = alpha (A x) + (1 - alpha) z`` instead
    of ``A x``, while the residuals keep using the true ``A x``; the
    default 1.0 is plain ADMM.  With ``record_iterates=True`` the
    per-iteration ``(x, z, v)`` triples are returned as well (for
    iterate-level GPU comparisons).
    """
    P = sp.csc_matrix(P)
    A = sp.csc_matrix(A)
    n, m = P.shape[0], A.shape[0]
    rho_vec = np.broadcast_to(np.asarray(rho, dtype=np.float64),
                              (m,)).copy()
    P_full = sp.triu(P) + sp.triu(P, k=1).T
    K = (P_full + A.T @ sp.diags(rho_vec) @ A).tocsc()
    # dense Cholesky on the small reference systems proves SPD-ness
    K_dense = K.toarray()
    np.linalg.cholesky(K_dense)
    solve = spla.factorized(K)

    q = np.asarray(q, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=np.float64)
    z = np.zeros(m) if z0 is None else np.array(z0, dtype=np.float64)
    v = np.zeros(m) if v0 is None else np.array(v0, dtype=np.float64)

    iterates = []
    converged = False
    r_p = r_d = float("inf")
    it = 0
    for it in range(1, max_iter + 1):
        b = -q + A.T @ (rho_vec * (z - v))
        x = solve(b)
        a = A @ x
        a_hat = a if alpha == 1.0 else alpha * a + (1.0 - alpha) * z
        z_old = z
        z = np.clip(a_hat + v, l, u)
        v = v + a_hat - z
        if record_iterates:
            iterates.append((x.copy(), z.copy(), v.copy()))
        if it % check_every == 0 or it == max_iter:
            r_p = float(np.max(np.abs(a - z))) if m else 0.0
            r_d = (float(np.max(np.abs(A.T @ (rho_vec * (z - z_old)))))
                   if m else 0.0)
            eps_p = eps_abs + eps_rel * max(
                float(np.max(np.abs(a))) if m else 0.0,
                float(np.max(np.abs(z))) if m else 0.0)
            eps_d = eps_abs + eps_rel * (
                float(np.max(np.abs(A.T @ (rho_vec * v)))) if m else 0.0)
            if r_p <= eps_p and r_d <= eps_d:
                converged = True
                break

    result = ReferenceResult(x=x, z=z, v=v, iterations=it,
                             converged=converged, primal_residual=r_p,
                             dual_residual=r_d)
    if record_iterates:
        return result, iterates
    return result

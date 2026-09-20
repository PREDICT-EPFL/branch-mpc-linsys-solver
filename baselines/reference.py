"""CPU reference solves, structured assembly, and error metrics.

Everything here runs on the CPU in NumPy and exists to validate the GPU
solvers and the generator; the timed structured solver never calls these
routines.  The system matrix is never assembled densely anywhere in the
project -- sparse assembly for the cuDSS baseline lives on
:meth:`src.problem.TreeMatrix.to_csr_lower`.
"""

import numpy as np

from src.general_arrow.problem import structural_matvec


# --------------------------------------------------------------------------
# CPU block-chain Cholesky (batched over tails, sequential over stages)
# --------------------------------------------------------------------------
def chain_cholesky(D, E):
    """Block-tridiagonal Cholesky ``K_i = L_i L_i^T`` for every tail.

    Parameters are the ``(B, T, n_b, n_b)`` diagonal and
    ``(B, T-1, n_b, n_b)`` sub-diagonal blocks.  Returns
    ``(L_diag, L_sub)`` where ``L_diag`` holds lower-triangular Cholesky
    blocks and ``L_sub`` the sub-diagonal factor blocks, both batched over
    tails.  Raises ``LinAlgError`` if any pivot block is not positive
    definite.
    """
    B, T, n_b, _ = D.shape
    L_diag = np.empty_like(D)
    L_sub = np.empty_like(E)
    M = D[:, 0]
    L_diag[:, 0] = np.linalg.cholesky(M)
    for t in range(1, T):
        # L_sub[t-1] = E[t-1] @ L_diag[t-1]^-T, via
        # L_sub^T = L_diag^-1 @ E^T (batched solve with the lower factor)
        L_sub[:, t - 1] = np.swapaxes(
            np.linalg.solve(L_diag[:, t - 1],
                            np.swapaxes(E[:, t - 1], -1, -2)),
            -1, -2)
        M = D[:, t] - L_sub[:, t - 1] @ np.swapaxes(L_sub[:, t - 1], -1, -2)
        L_diag[:, t] = np.linalg.cholesky(M)
    return L_diag, L_sub


def chain_forward(L_diag, L_sub, rhs):
    """Solve the block-bidiagonal system ``L u = rhs`` (forward
    substitution) for every tail.  ``rhs`` has shape
    ``(B, T, n_b, nrhs)``."""
    B, T, n_b, nrhs = rhs.shape
    u = np.empty_like(rhs)
    u[:, 0] = np.linalg.solve(L_diag[:, 0], rhs[:, 0])
    for t in range(1, T):
        s = rhs[:, t] - L_sub[:, t - 1] @ u[:, t - 1]
        u[:, t] = np.linalg.solve(L_diag[:, t], s)
    return u


def chain_backward(L_diag, L_sub, u):
    """Solve ``L^T x = u`` (backward substitution) for every tail."""
    B, T, n_b, nrhs = u.shape
    x = np.empty_like(u)
    x[:, T - 1] = np.linalg.solve(np.swapaxes(L_diag[:, T - 1], -1, -2),
                                  u[:, T - 1])
    for t in range(T - 2, -1, -1):
        s = u[:, t] - np.swapaxes(L_sub[:, t], -1, -2) @ x[:, t + 1]
        x[:, t] = np.linalg.solve(np.swapaxes(L_diag[:, t], -1, -2), s)
    return x


def chain_solve(L_diag, L_sub, rhs):
    """Solve ``K_i x = rhs`` for every tail given the chain factors."""
    return chain_backward(L_diag, L_sub, chain_forward(L_diag, L_sub, rhs))


def structured_solve_cpu(D, E, C_T, R, r, q, return_root_update=False):
    """Reference CPU implementation of the structured block-Cholesky
    solve.

    Implements exactly the algorithm of the GPU solver (tail Cholesky,
    ``X = K^{-1} C^T`` columnwise, root update ``S = R - sum C X``,
    root solve, recovery) with plain NumPy.  Intended for validation and
    small cases.

    Returns ``(w, y)`` and, with ``return_root_update=True``, also the
    complement ``S``.
    """
    L_diag, L_sub = chain_cholesky(D, E)
    X = chain_solve(L_diag, L_sub, C_T)
    u = chain_solve(L_diag, L_sub, r)
    S = R - np.einsum("btim,btil->ml", C_T, X)
    s = q - np.einsum("btim,btiq->mq", C_T, u)
    S = 0.5 * (S + S.T)
    Ls = np.linalg.cholesky(S)
    y = np.linalg.solve(Ls.T, np.linalg.solve(Ls, s))
    w = u - np.einsum("btim,mq->btiq", X, y)
    if return_root_update:
        return w, y, S
    return w, y


# --------------------------------------------------------------------------
# Norm and error metrics
# --------------------------------------------------------------------------
def estimate_two_norm(D, E, C_T, R, iters=30, seed=0):
    """Power-iteration estimate of ``||A||_2`` using the structural
    matvec.

    Returns an estimate (documented as such); for symmetric ``A`` the
    power iteration converges to the spectral norm from a random start.
    """
    B, T, n_b, _ = D.shape
    n_r = R.shape[0]
    rng = np.random.default_rng(seed)
    xt = rng.standard_normal((B, T, n_b, 1))
    xr = rng.standard_normal((n_r, 1))
    lam = 1.0
    for _ in range(iters):
        yt, yr = structural_matvec(D, E, C_T, R, xt, xr)
        lam = float(np.sqrt(np.sum(yt * yt) + np.sum(yr * yr)))
        if lam == 0.0:
            return 0.0
        xt, xr = yt / lam, yr / lam
    return lam


def compute_metrics(problem, x_tail, x_root, a_norm=None):
    """Accuracy metrics of a candidate solution, computed in FP64.

    ``problem`` is a :class:`~src.problem.GeneratedProblem`;
    ``x_tail``/``x_root`` are the solution parts as arrays (host or
    anything :func:`numpy.asarray` accepts).  Returns a dict with
    ``scaled_residual``, ``rhs_relative_residual``, ``forward_error``,
    ``max_componentwise_backward_error``, and ``nan_or_inf`` (True if the
    solution contains non-finite values).  ``a_norm`` may pass a
    precomputed ``||A||_2`` (exact or estimated); otherwise a
    power-iteration estimate is used.
    """
    matrix = problem.matrix
    d64 = matrix.D.astype(np.float64)
    e64 = matrix.E.astype(np.float64)
    c64 = matrix.C_T.astype(np.float64)
    r64 = matrix.R.astype(np.float64)
    rhs_b = problem.rhs.tail.astype(np.float64)
    rhs_s = problem.rhs.root.astype(np.float64)
    xt = np.asarray(x_tail, dtype=np.float64).reshape(rhs_b.shape)
    xr = np.asarray(x_root, dtype=np.float64).reshape(rhs_s.shape)

    nan_or_inf = not (np.all(np.isfinite(xt)) and np.all(np.isfinite(xr)))

    rt, rr = structural_matvec(d64, e64, c64, r64, xt, xr)
    res_t = rt - rhs_b
    res_r = rr - rhs_s
    res_norm = float(np.sqrt(np.sum(res_t**2) + np.sum(res_r**2)))
    rhs_norm = float(np.sqrt(np.sum(rhs_b**2) + np.sum(rhs_s**2)))
    x_norm = float(np.sqrt(np.sum(xt**2) + np.sum(xr**2)))

    if a_norm is None:
        a_norm = estimate_two_norm(d64, e64, c64, r64)

    # componentwise backward error |res_i| / (|A| |x| + |r|)_i, computed
    # structurally with absolute-value blocks and a safe denominator
    at, ar = structural_matvec(np.abs(d64), np.abs(e64), np.abs(c64),
                               np.abs(r64), np.abs(xt), np.abs(xr))
    den_t = at + np.abs(rhs_b)
    den_r = ar + np.abs(rhs_s)
    tiny = np.finfo(np.float64).tiny
    comp = max(
        float(np.max(np.abs(res_t) / np.maximum(den_t, tiny))) if res_t.size else 0.0,
        float(np.max(np.abs(res_r) / np.maximum(den_r, tiny))) if res_r.size else 0.0,
    )

    w_true = problem.exact_solution.tail
    y_true = problem.exact_solution.root
    from src.general_arrow.problem import TreeVector
    shape = matrix.shape
    err = TreeVector(shape, xt - w_true,
                     (xr - y_true).reshape(y_true.shape)).flat()
    z_true = TreeVector(shape, w_true, y_true).flat()

    return {
        "scaled_residual": res_norm / max(a_norm * x_norm + rhs_norm, tiny),
        "rhs_relative_residual": res_norm / max(rhs_norm, tiny),
        "forward_error": float(np.linalg.norm(err) /
                               max(np.linalg.norm(z_true), tiny)),
        "max_componentwise_backward_error": comp,
        "nan_or_inf": nan_or_inf,
        "a_norm_used": float(a_norm),
    }

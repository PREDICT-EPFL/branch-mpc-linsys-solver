"""Benchmark/test problem fixtures: specs and the reproducible generator.

``ProblemSpec`` and ``GeneratedProblem`` describe randomly generated SPD
scenario-tree systems; the ``factor`` mode builds every tail matrix from
a known lower block-bidiagonal factor (``K_i = L_i L_i^T``) and the root
matrix from a prescribed updated-root target ``S_0 = Z Z^T + delta I``,
so the assembled system is SPD by construction and two exact invariants
are available for validation:

- the block-Cholesky root update of the generated system equals ``S_0``;
- the right-hand side equals ``A @ z_true`` for a known ``z_true``.

The known construction factors never leave this module; solvers only see
the :class:`src.problem.TreeMatrix` block data and the right-hand sides.
This module is benchmark/test support: core solver code never imports it.
"""

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from src.dense_arrow.problem import (
    TreeMatrix,
    TreeShape,
    TreeVector,
    structural_matvec,
)
from baselines import reference as validation


_COUPLING_PATTERNS = ("dense", "root_only", "terminal_only", "banded_root")
_GENERATOR_MODES = ("factor", "diagonal_dominant")


@dataclass(frozen=True)
class ProblemSpec:
    """Dimensions and generator parameters of one scenario-tree system.

    The four dimension fields keep their historical names (``horizon``,
    ``block_size``, ...) because they are serialized into benchmark
    records and case identifiers; :attr:`shape` exposes the same
    dimensions as the canonical :class:`src.problem.TreeShape` used by
    the solver.

    Parameters
    ----------
    num_tails : int
        Number of independent scenario tails ``B`` (>= 1).
    horizon : int
        Number of stage blocks per tail ``T`` (>= 1).
    block_size : int
        Tail-stage block dimension ``n_b`` (>= 1).
    root_dim : int
        Dimension ``n_r`` of the shared root variable ``y`` (>= 1).
    num_rhs : int
        Number of right-hand-side columns generated with the system.
    precision : str
        ``"float64"`` (default) or ``"float32"``.
    coupling_pattern : str
        ``"dense"`` (default), ``"root_only"``, ``"terminal_only"`` or
        ``"banded_root"``.
    band_width : int
        Width of the root range each stage couples to when
        ``coupling_pattern == "banded_root"``.
    rho_tail : float
        Scale of the sub-diagonal factor blocks (tail coupling strength).
    rho_sep : float
        Scale of the tail-to-root coupling blocks.
    condition_target : float or None
        Approximate target condition number of the assembled system.
        ``None`` (default) selects the well-conditioned construction.
    generator_mode : str
        ``"factor"`` (default, SPD by construction with known root
        complement) or ``"diagonal_dominant"`` (ablation).
    seed : int
        Seed for all random draws; the same seed regenerates identical
        data.
    """

    num_tails: int
    horizon: int
    block_size: int
    root_dim: int
    num_rhs: int = 1
    precision: str = "float64"
    coupling_pattern: str = "dense"
    band_width: int = 8
    rho_tail: float = 0.5
    rho_sep: float = 1.0
    condition_target: float | None = None
    generator_mode: str = "factor"
    seed: int = 0

    def __post_init__(self):
        # shape fields are validated by the TreeShape constructor
        _ = self.shape
        if self.num_rhs < 1:
            raise ValueError(f"num_rhs must be >= 1, got {self.num_rhs}")
        if self.coupling_pattern not in _COUPLING_PATTERNS:
            raise ValueError(f"coupling_pattern must be one of {_COUPLING_PATTERNS}")
        if self.generator_mode not in _GENERATOR_MODES:
            raise ValueError(f"generator_mode must be one of {_GENERATOR_MODES}")
        if self.band_width < 1:
            raise ValueError(f"band_width must be >= 1, got {self.band_width}")

    @property
    def shape(self) -> TreeShape:
        """The structural dimensions as the canonical :class:`TreeShape`."""
        return TreeShape(num_tails=self.num_tails,
                         num_stages=self.horizon,
                         tail_block_dim=self.block_size,
                         root_dim=self.root_dim,
                         precision=self.precision)

    @property
    def np_dtype(self):
        """NumPy dtype matching :attr:`precision`."""
        return self.shape.np_dtype

    @property
    def tail_dimension(self) -> int:
        """Number of scalar tail unknowns, ``B * T * n_b``."""
        return self.shape.tail_dimension

    @property
    def total_dimension(self) -> int:
        """Total number of scalar unknowns, ``B * T * n_b + n_r``."""
        return self.shape.total_dimension

    @property
    def nnz_lower(self) -> int:
        """Stored entries in the lower triangle of the assembled matrix."""
        return self.shape.nnz_lower

    def to_dict(self) -> dict:
        """Plain-dict form used in benchmark records and case identifiers
        (field names are part of the stable record schema)."""
        return asdict(self)


@dataclass
class GeneratedProblem:
    """A generated SPD scenario-tree system with its known ground truth.

    ``matrix`` and ``rhs`` are what a solver consumes; ``exact_solution``
    and ``root_target`` exist only because the generator constructs the
    system from known factors -- they are for validation and must never be
    given to a timed solver.  ``matrix``/``rhs`` arrays are in the spec's
    precision; the ground truth stays in FP64 for accurate error
    evaluation.
    """

    spec: ProblemSpec
    matrix: TreeMatrix
    rhs: TreeVector
    exact_solution: TreeVector
    root_target: np.ndarray    # (n_r, n_r) exact updated-root target
    gamma: float                # ||sum_i C_i K_i^-1 C_i^T||_F / ||R||_F
    kappa_estimate: float | None = None
    kappa_method: str = "not_computed"
    generation_seconds: float = 0.0
    extras: dict = field(default_factory=dict)

    @property
    def shape(self) -> TreeShape:
        """The structural dimensions of the generated system."""
        return self.matrix.shape


# Fixed sub-stream offsets so every draw has a stable, documented seed.
_SEED_TAIL, _SEED_COUPLING, _SEED_ROOT, _SEED_TRUTH, _SEED_KAPPA = range(5)


def _rng(spec: ProblemSpec, stream: int) -> np.random.Generator:
    return np.random.default_rng([spec.seed, stream])


def _coupling_mask(spec: ProblemSpec) -> np.ndarray:
    """Stage-by-root mask (T, n_r) implementing the coupling
    pattern."""
    T, n_r, w = spec.horizon, spec.root_dim, spec.band_width
    mask = np.zeros((T, n_r))
    if spec.coupling_pattern == "dense":
        mask[:] = 1.0
    elif spec.coupling_pattern == "root_only":
        mask[0, :] = 1.0
    elif spec.coupling_pattern == "terminal_only":
        mask[T - 1, :] = 1.0
    elif spec.coupling_pattern == "banded_root":
        w = min(w, n_r)
        span = max(T - 1, 1)
        for t in range(T):
            start = int(round(t * (n_r - w) / span)) if n_r > w else 0
            mask[t, start:start + w] = 1.0
    return mask


def _tail_factors(spec: ProblemSpec):
    """Draw the known tail factors (L diagonal blocks, F sub-diagonal
    blocks) and the per-block conditioning scales."""
    B, T, n_b = spec.num_tails, spec.horizon, spec.block_size
    rng = _rng(spec, _SEED_TAIL)

    L = np.tril(rng.standard_normal((B, T, n_b, n_b)) / np.sqrt(n_b), k=-1)
    idx = np.arange(n_b)
    L[:, :, idx, idx] = rng.uniform(1.0, 2.0, size=(B, T, n_b))

    if spec.condition_target is not None and spec.condition_target > 1.0:
        # Geometrically spaced block scales in [kappa^-1/2, 1]; K blocks
        # then span eigenvalue scales [kappa^-1, O(1)], putting the
        # assembled condition number near the target (reported as an
        # estimate only).
        ladder = np.geomspace(1.0 / np.sqrt(spec.condition_target), 1.0, B * T)
        scales = rng.permutation(ladder).reshape(B, T)
    else:
        scales = np.ones((B, T))
    L *= scales[:, :, None, None]

    F = np.empty((B, max(T - 1, 0), n_b, n_b))
    if T > 1:
        F[:] = rng.standard_normal((B, T - 1, n_b, n_b)) * (spec.rho_tail
                                                            / np.sqrt(n_b))
        # Sub-diagonal factors inherit the geometric scale of their stage
        # so ill-conditioning does not blow up the off-diagonal coupling.
        F *= scales[:, 1:, None, None]
    return L, F, scales


def _tail_blocks_from_factors(L, F):
    """Exact block-tridiagonal blocks of ``K_i = L_i L_i^T``."""
    B, T, n_b, _ = L.shape
    LT = np.swapaxes(L, -1, -2)
    D = L @ LT
    if T > 1:
        D[:, 1:] += F @ np.swapaxes(F, -1, -2)
        E = F @ LT[:, :-1]
    else:
        E = np.zeros((B, 0, n_b, n_b))
    return D, E


def _forward_substitute_known(L, F, rhs):
    """Solve ``L_i G = rhs`` with the known block-bidiagonal factor."""
    B, T, n_b, nrhs = rhs.shape
    G = np.empty_like(rhs)
    G[:, 0] = np.linalg.solve(L[:, 0], rhs[:, 0])
    for t in range(1, T):
        G[:, t] = np.linalg.solve(L[:, t], rhs[:, t] - F[:, t - 1] @ G[:, t - 1])
    return G


def _diagonal_dominant_blocks(spec: ProblemSpec, C_T):
    """Ablation generator: strictly diagonally dominant construction."""
    B, T, n_b = spec.num_tails, spec.horizon, spec.block_size
    rng = _rng(spec, _SEED_TAIL)
    sym = rng.standard_normal((B, T, n_b, n_b)) / np.sqrt(n_b)
    sym = 0.5 * (sym + np.swapaxes(sym, -1, -2))
    if T > 1:
        E = rng.standard_normal((B, T - 1, n_b, n_b)) * (spec.rho_tail
                                                         / np.sqrt(n_b))
    else:
        E = np.zeros((B, 0, n_b, n_b))
    D = sym.copy()
    # scalar-row absolute sums of every off-diagonal contribution
    row = np.abs(sym).sum(axis=-1)
    if T > 1:
        row[:, 1:] += np.abs(E).sum(axis=-1)
        row[:, :-1] += np.abs(np.swapaxes(E, -1, -2)).sum(axis=-1)
    row += np.abs(C_T).sum(axis=-1)
    idx = np.arange(n_b)
    D[:, :, idx, idx] += row + 1.0
    return D, E


def _make_root(spec: ProblemSpec, root_update):
    """Updated-root target ``S_0`` and root matrix ``R``."""
    n_r = spec.root_dim
    rng = _rng(spec, _SEED_ROOT)
    Z = rng.standard_normal((n_r, n_r)) / np.sqrt(n_r)
    delta = 0.1
    root_target = Z @ Z.T + delta * np.eye(n_r)
    R = root_target + root_update
    R = 0.5 * (R + R.T)
    return root_target, R


def _estimate_condition(spec, D, E, C_T, R, solve_fn, iters=25):
    """Estimate ``kappa_2(A)`` as ``lambda_max / lambda_min``.

    ``lambda_max`` comes from power iteration with the structural matvec,
    ``lambda_min`` from inverse iteration using generator-known factors.
    Documented as an estimate, never an exact value.
    """
    B, T, n_b, n_r = spec.shape.dims()
    rng = _rng(spec, _SEED_KAPPA)
    xt = rng.standard_normal((B, T, n_b, 1))
    xr = rng.standard_normal((n_r, 1))
    lam_max = 1.0
    for _ in range(iters):
        yt, yr = structural_matvec(D, E, C_T, R, xt, xr)
        lam_max = float(np.sqrt(np.sum(yt * yt) + np.sum(yr * yr)))
        xt, xr = yt / lam_max, yr / lam_max

    xt = rng.standard_normal((B, T, n_b, 1))
    xr = rng.standard_normal((n_r, 1))
    mu = 1.0
    for _ in range(iters):
        yt, yr = solve_fn(xt, xr)
        mu = float(np.sqrt(np.sum(yt * yt) + np.sum(yr * yr)))
        xt, xr = yt / mu, yr / mu
    lam_min = 1.0 / mu
    return lam_max / lam_min


def generate_problem(spec: ProblemSpec, estimate_condition=True,
                     exact_condition_dim=1500) -> GeneratedProblem:
    """Generate a reproducible SPD scenario-tree system for ``spec``.

    All construction happens in FP64; the matrix block data and
    right-hand sides are cast to ``spec.precision`` at the end, while the
    ground truth (``exact_solution``, ``root_target``) stays in FP64 for
    accurate error evaluation.

    With ``estimate_condition=True`` the condition number is computed
    exactly (dense) when the total dimension is at most
    ``exact_condition_dim`` and estimated by power/inverse iteration
    otherwise; set it to ``False`` to skip this (fastest).
    """
    t_start = time.perf_counter()
    B, T, n_b, n_r = spec.shape.dims()

    # ---- coupling blocks C^T (stage-rows layout) ---------------------------
    rng_c = _rng(spec, _SEED_COUPLING)
    C_T = rng_c.standard_normal((B, T, n_b, n_r))
    C_T *= spec.rho_sep / np.sqrt(B * T * n_b)
    C_T *= _coupling_mask(spec)[None, :, None, :]

    # ---- tail matrices and the exact root update --------------------------
    if spec.generator_mode == "factor":
        L, F, scales = _tail_factors(spec)
        if spec.condition_target is not None:
            # Couple each stage proportionally to its local stiffness so
            # the root update stays O(rho_sep^2) despite ill conditioning.
            C_T *= scales[:, :, None, None]
        D, E = _tail_blocks_from_factors(L, F)
        G = _forward_substitute_known(L, F, C_T)
        root_update = np.einsum("btij,btil->jl", G, G)

        def tail_solve(rhs):
            u = _forward_substitute_known(L, F, rhs)
            # backward pass with the known factor transposed
            x = np.empty_like(u)
            LT = np.swapaxes(L, -1, -2)
            x[:, T - 1] = np.linalg.solve(LT[:, T - 1], u[:, T - 1])
            for t in range(T - 2, -1, -1):
                s = u[:, t] - np.swapaxes(F[:, t], -1, -2) @ x[:, t + 1]
                x[:, t] = np.linalg.solve(LT[:, t], s)
            return x
    else:  # diagonal_dominant ablation
        D, E = _diagonal_dominant_blocks(spec, C_T)
        L_diag, L_sub = validation.chain_cholesky(D, E)
        G = validation.chain_forward(L_diag, L_sub, C_T)
        root_update = np.einsum("btij,btil->jl", G, G)

        def tail_solve(rhs):
            return validation.chain_solve(L_diag, L_sub, rhs)

    root_update = 0.5 * (root_update + root_update.T)
    root_target, R = _make_root(spec, root_update)
    gamma = float(np.linalg.norm(root_update) / np.linalg.norm(R))

    # ---- true solution and structural right-hand side ----------------------
    rng_x = _rng(spec, _SEED_TRUTH)
    w_true = rng_x.standard_normal((B, T, n_b, spec.num_rhs))
    y_true = rng_x.standard_normal((n_r, spec.num_rhs))
    r, q = structural_matvec(D, E, C_T, R, w_true, y_true)

    # ---- condition number ---------------------------------------------------
    kappa, kappa_method = None, "not_computed"
    if estimate_condition:
        if spec.total_dimension <= exact_condition_dim:
            # exact kappa for small systems, densified from the sparse
            # lower triangle (no dense assembly path exists in src)
            import scipy.sparse as sp
            lower = TreeMatrix(spec.shape, D=D, E=E, C_T=C_T,
                               R=R).to_csr_lower()
            A = (lower + sp.tril(lower, k=-1).T).toarray()
            kappa = float(np.linalg.cond(A, 2))
            kappa_method = "exact_dense"
        else:
            Ls = np.linalg.cholesky(root_target)

            def full_solve(rt, rr):
                u = tail_solve(rt)
                s = rr - np.einsum("btim,btiq->mq", C_T, u)
                y = np.linalg.solve(Ls.T, np.linalg.solve(Ls, s))
                w = u - tail_solve(np.einsum("btim,mq->btiq", C_T, y))
                return w, y

            kappa = _estimate_condition(spec, D, E, C_T, R, full_solve)
            kappa_method = "power_iteration_estimate"

    dt = spec.np_dtype
    shape = spec.shape
    matrix = TreeMatrix(shape, D=D.astype(dt), E=E.astype(dt),
                        C_T=C_T.astype(dt), R=R.astype(dt))
    return GeneratedProblem(
        spec=spec,
        matrix=matrix,
        rhs=TreeVector(shape, r.astype(dt), q.astype(dt)),
        exact_solution=TreeVector(shape, w_true, y_true),
        root_target=root_target,
        gamma=gamma,
        kappa_estimate=kappa,
        kappa_method=kappa_method,
        generation_seconds=time.perf_counter() - t_start,
    )


# --------------------------------------------------------------------------
# Scenario optimal-control QP in conventional sparse form (ADMM stage 6)
# --------------------------------------------------------------------------
def generate_scenario_qp(num_tails=3, num_stages=6, nx=6, nu=2, seed=0):
    """A scenario-based stochastic optimal-control QP in ordinary CSC
    form, whose ADMM normal matrix ``P + rho A'A`` has the supported
    one-level block-arrow structure under the canonical variable order.

    Variables: per tail ``b`` and stage ``t`` a block
    ``(x_t (nx), u_t (nu))``, tails consecutive, followed by the
    shared first-stage control ``u_s (nu)`` (the root).
    Constraints: fixed initial state per tail, per-tail linear
    dynamics ``x_{t+1} = Ad_b x_t + Bd_b u_t``, the non-anticipativity
    coupling ``u_0(b) = u_s``, and box bounds on every control.  The
    cost is strictly convex block-diagonal (with a small within-stage
    ``x``/``u`` cross term), so ``K_rho`` is SPD for every ``rho > 0``.

    Returns ``(P, q, A, l, u, meta)`` with CSC ``P`` (upper triangle),
    CSC ``A``, dense FP64 vectors, and ``meta`` recording the tree
    dimensions ``(B, T, n_b, n_r)`` the analyzer should recover.
    """
    import scipy.sparse as sp
    rng = np.random.default_rng(seed)
    B, T = int(num_tails), int(num_stages)
    if B < 2 or T < 2:
        raise ValueError("need num_tails >= 2 and num_stages >= 2")
    n_b = nx + nu
    L = T * n_b
    n = B * L + nu

    def x_idx(b, t):
        return b * L + t * n_b

    def u_idx(b, t):
        return b * L + t * n_b + nx

    us0 = B * L  # shared control offset

    # ---- cost: block-diagonal SPD upper triangle ---------------------------
    rows, cols, vals = [], [], []
    for b in range(B):
        for t in range(T):
            for i in range(nx):
                rows.append(x_idx(b, t) + i); cols.append(x_idx(b, t) + i)
                vals.append(1.0 + rng.uniform(0.0, 1.0))
            for i in range(nu):
                rows.append(u_idx(b, t) + i); cols.append(u_idx(b, t) + i)
                vals.append(0.5 + rng.uniform(0.0, 0.5))
            # small within-stage cross terms (stay inside the D block;
            # they also keep every tail internally connected, which
            # the conservative analyzer requires)
            for i in range(nu):
                rows.append(x_idx(b, t)); cols.append(u_idx(b, t) + i)
                vals.append(0.05)
    for i in range(nu):
        rows.append(us0 + i); cols.append(us0 + i)
        vals.append(0.5 + rng.uniform(0.0, 0.5))
    P = sp.csc_matrix((vals, (rows, cols)), shape=(n, n))
    P.sum_duplicates(); P.sort_indices()

    q = rng.standard_normal(n) * 0.1

    # ---- constraints --------------------------------------------------------
    a_rows, a_cols, a_vals, lo, hi = [], [], [], [], []
    row = 0
    for b in range(B):
        Ad = np.eye(nx) + 0.05 * rng.standard_normal((nx, nx)) / np.sqrt(nx)
        Bd = rng.standard_normal((nx, nu)) / np.sqrt(nu)
        x_init = rng.standard_normal(nx) * 0.5
        # fixed initial state
        for i in range(nx):
            a_rows.append(row); a_cols.append(x_idx(b, 0) + i)
            a_vals.append(1.0)
            lo.append(x_init[i]); hi.append(x_init[i])
            row += 1
        # dynamics x_{t+1} = Ad x_t + Bd u_t
        for t in range(T - 1):
            for i in range(nx):
                a_rows.append(row); a_cols.append(x_idx(b, t + 1) + i)
                a_vals.append(-1.0)
                for j in range(nx):
                    a_rows.append(row); a_cols.append(x_idx(b, t) + j)
                    a_vals.append(Ad[i, j])
                for j in range(nu):
                    a_rows.append(row); a_cols.append(u_idx(b, t) + j)
                    a_vals.append(Bd[i, j])
                lo.append(0.0); hi.append(0.0)
                row += 1
        # non-anticipativity: u_0(b) = u_s
        for i in range(nu):
            a_rows.append(row); a_cols.append(u_idx(b, 0) + i)
            a_vals.append(1.0)
            a_rows.append(row); a_cols.append(us0 + i)
            a_vals.append(-1.0)
            lo.append(0.0); hi.append(0.0)
            row += 1
        # control box bounds
        for t in range(T):
            for i in range(nu):
                a_rows.append(row); a_cols.append(u_idx(b, t) + i)
                a_vals.append(1.0)
                lo.append(-1.5); hi.append(1.5)
                row += 1
    for i in range(nu):
        a_rows.append(row); a_cols.append(us0 + i)
        a_vals.append(1.0)
        lo.append(-1.5); hi.append(1.5)
        row += 1
    A = sp.csc_matrix((a_vals, (a_rows, a_cols)), shape=(row, n))
    A.sum_duplicates(); A.sort_indices()

    meta = {"num_tails": B, "num_stages": T, "tail_block_dim": n_b,
            "root_dim": nu}
    return (P, np.asarray(q), A, np.asarray(lo, dtype=np.float64),
            np.asarray(hi, dtype=np.float64), meta)

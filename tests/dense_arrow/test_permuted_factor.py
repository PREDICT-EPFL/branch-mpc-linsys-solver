"""Direct-factor reconstruction tests (plan 3, stage 1).

The principal invariant: Pi Phi Pi^T = L_hat L_hat^T with the factor
built structurally from per-tail permuted tail factors, root coupling
columns M_i, and the root diagonal update.  These tests reconstruct
dense matrices from the structured blocks on small cases; they are the
direct validation of the factor representation, independent of solve
residuals.
"""

import numpy as np
import pytest

from tests.general_arrow.permuted_factor_reference import (
    TailPermutation,
    assemble_dense_system,
    extract_tail_factor_dense,
    global_permutation,
    reference_permuted_factor,
    socu_level_segments,
    stage_elimination_levels,
    tail_permutation,
)

wp = pytest.importorskip("warp")


def _factor(engine):
    """Refresh and factor all tails (test helper; the production solver
    uses the fused factor-plus-forward launch instead)."""
    from socu.block_tridiag_solver import create_cholesky_factor_launch
    engine.refresh()
    create_cholesky_factor_launch(engine.diag_factor,
                                  engine.offdiag_factor,
                                  dtype=engine.dtype,
                                  device=engine.device)()


def _problem(B=3, T=8, n_b=8, n_r=4, seed=0, precision="float64"):
    from experiments.general_arrow.benchmarks.problems import ProblemSpec, generate_problem
    spec = ProblemSpec(num_tails=B, horizon=T, block_size=n_b,
                       root_dim=n_r, num_rhs=1, precision=precision,
                       seed=seed)
    return generate_problem(spec, estimate_condition=False)


# ------------------------------------------------------------ permutation
def test_elimination_levels_and_order():
    levels = stage_elimination_levels(8)
    assert list(levels) == [0, 1, 0, 2, 0, 1, 0, 3]
    order = tail_permutation(8)
    assert list(order) == [0, 2, 4, 6, 1, 5, 3, 7]
    # non-power-of-two horizon
    assert list(tail_permutation(5)) == [0, 2, 4, 1, 3]
    assert list(tail_permutation(1)) == [0]


def test_permutation_is_bijection():
    for T in (1, 2, 3, 5, 8, 13, 32, 100, 128):
        perm = TailPermutation.create(T)
        assert sorted(perm.order) == list(range(T))
        assert np.array_equal(perm.order[perm.position], np.arange(T))
        rows = perm.row_order(4)
        assert sorted(rows) == list(range(4 * T))


def test_socu_segments_match_storage_length():
    from socu.block_tridiag_solver import calculate_off_diag_storage_len
    for T in (1, 2, 3, 5, 8, 13, 32, 100, 128):
        segs = socu_level_segments(T)
        total = sum(length for _, _, length in segs)
        assert total == calculate_off_diag_storage_len(T)


# ---------------------------------------------------- CPU reference factor
@pytest.mark.parametrize("B,T,n_b,n_r", [(1, 1, 2, 2), (2, 4, 3, 2),
                                         (3, 8, 4, 5), (2, 13, 3, 4)])
def test_reference_factor_reconstructs_permuted_system(B, T, n_b, n_r):
    problem = _problem(B=B, T=T, n_b=max(n_b, 2), n_r=n_r, seed=1)
    matrix = problem.matrix
    Phi = assemble_dense_system(matrix)
    p = global_permutation(matrix.shape)
    Phi_hat = Phi[np.ix_(p, p)]
    L_hat = reference_permuted_factor(matrix)
    # the structured factor is lower triangular and reconstructs the
    # permuted matrix: the principal plan-3 invariant
    assert np.allclose(L_hat, np.tril(L_hat))
    err = np.abs(Phi_hat - L_hat @ L_hat.T).max()
    scale = np.abs(Phi_hat).max()
    assert err <= 1e-11 * scale
    # and it IS the unique Cholesky factor (positive diagonal)
    assert np.allclose(L_hat, np.linalg.cholesky(Phi_hat), atol=1e-9 * scale)


# ------------------------------------------------- GPU SOCU tail extraction
@pytest.mark.gpu
@pytest.mark.parametrize("T", [1, 2, 4, 8, 16, 13, 32])
def test_socu_storage_reconstructs_permuted_tail_factor(T):
    """SOCU's in-place factor storage holds exactly the permuted
    Cholesky factor blocks of each tail: dense reconstruction equals
    chol(Pi K Pi^T)."""
    from src.general_arrow.socu import TailEngine as SocuTailEngine
    B, n_b = 3, 8
    problem = _problem(B=B, T=T, n_b=n_b, n_r=2, seed=2)
    D = np.asarray(problem.matrix.D, dtype=np.float64)
    E = np.asarray(problem.matrix.E, dtype=np.float64)
    engine = SocuTailEngine(B, T, n_b, wp.float64, "cuda:0")
    engine.stage(D, E)
    _factor(engine)
    wp.synchronize_device("cuda:0")
    diag = engine.diag_factor.numpy()
    off = engine.offdiag_factor.numpy()
    perm = TailPermutation.create(T)
    rows = perm.row_order(n_b)
    for b in range(B):
        K = np.zeros((T * n_b, T * n_b))
        for t in range(T):
            r = t * n_b
            K[r:r + n_b, r:r + n_b] = D[b, t]
            if t + 1 < T:
                K[r + n_b:r + 2 * n_b, r:r + n_b] = E[b, t]
                K[r:r + n_b, r + n_b:r + 2 * n_b] = E[b, t].T
        K_hat = K[np.ix_(rows, rows)]
        L_gpu = extract_tail_factor_dense(diag[b], off[b], perm)
        err = np.abs(K_hat - L_gpu @ L_gpu.T).max()
        assert err <= 1e-10 * np.abs(K_hat).max(), f"tail {b}, T={T}"
        assert np.allclose(L_gpu, np.tril(L_gpu))


@pytest.mark.gpu
def test_full_factor_invariant_via_gpu_tails():
    """Complete-factor invariant with GPU tail factors: assemble L_hat
    from SOCU storage plus CPU-computed M_i and L_R, verify
    Pi Phi Pi^T = L_hat L_hat^T."""
    import scipy.linalg as sla
    from src.general_arrow.socu import TailEngine as SocuTailEngine
    B, T, n_b, n_r = 2, 8, 8, 3
    problem = _problem(B=B, T=T, n_b=n_b, n_r=n_r, seed=3)
    matrix = problem.matrix
    D = np.asarray(matrix.D, dtype=np.float64)
    E = np.asarray(matrix.E, dtype=np.float64)
    C_T = np.asarray(matrix.C_T, dtype=np.float64)
    R = np.asarray(matrix.R, dtype=np.float64)

    engine = SocuTailEngine(B, T, n_b, wp.float64, "cuda:0")
    engine.stage(D, E)
    _factor(engine)
    wp.synchronize_device("cuda:0")
    diag = engine.diag_factor.numpy()
    off = engine.offdiag_factor.numpy()

    perm = TailPermutation.create(T)
    rows = perm.row_order(n_b)
    n = B * T * n_b + n_r
    L = np.zeros((n, n))
    root_update = np.zeros((n_r, n_r))
    for b in range(B):
        L_hat = extract_tail_factor_dense(diag[b], off[b], perm)
        C_hat_T = C_T[b].reshape(T * n_b, n_r)[rows]
        M_T = sla.solve_triangular(L_hat, C_hat_T, lower=True)
        base = b * T * n_b
        L[base:base + T * n_b, base:base + T * n_b] = L_hat
        L[-n_r:, base:base + T * n_b] = M_T.T
        root_update += M_T.T @ M_T
    L[-n_r:, -n_r:] = np.linalg.cholesky(R - root_update)

    Phi = assemble_dense_system(matrix)
    p = global_permutation(matrix.shape)
    Phi_hat = Phi[np.ix_(p, p)]
    err = np.abs(Phi_hat - L @ L.T).max()
    assert err <= 1e-10 * np.abs(Phi_hat).max()


# --------------------------------------------- solver-boundary factor tests
@pytest.mark.gpu
@pytest.mark.parametrize("n_r", [2, 16, 64])
def test_solver_factor_blocks_match_reference(n_r):
    """The solver's _PermutedTreeFactor blocks are exactly the permuted
    factor: root_coupling_T = M_i^T (physical row order), root_diagonal
    reconstructs the root update, and the complete structured factor
    reconstructs Pi Phi Pi^T."""
    import scipy.linalg as sla
    from src.general_arrow.solver import Solver
    B, T, n_b = 3, 8, 8
    problem = _problem(B=B, T=T, n_b=n_b, n_r=n_r, seed=4)
    matrix = problem.matrix
    solver = Solver(matrix.shape)
    solver.update(matrix)
    solver.factorize()
    wp.synchronize_device("cuda:0")
    perm = TailPermutation.create(T)
    rows = perm.row_order(n_b)

    D = np.asarray(matrix.D, dtype=np.float64)
    E = np.asarray(matrix.E, dtype=np.float64)
    C_T = np.asarray(matrix.C_T, dtype=np.float64)
    R = np.asarray(matrix.R, dtype=np.float64)
    diag = solver._socu.diag_factor.numpy()
    off = solver._socu.offdiag_factor.numpy()
    M_T_gpu = solver._root_coupling_T.numpy().reshape(B, T * n_b, n_r)
    L_R = np.tril(solver._root_diagonal.numpy())

    n = B * T * n_b + n_r
    L = np.zeros((n, n))
    root_update = np.zeros((n_r, n_r))
    scale = max(np.abs(D).max(), np.abs(R).max())
    for b in range(B):
        L_hat = extract_tail_factor_dense(diag[b], off[b], perm)
        # reference M^T from the extracted tail factor
        K = np.zeros((T * n_b, T * n_b))
        for t in range(T):
            r = t * n_b
            K[r:r + n_b, r:r + n_b] = D[b, t]
            if t + 1 < T:
                K[r + n_b:r + 2 * n_b, r:r + n_b] = E[b, t]
                K[r:r + n_b, r + n_b:r + 2 * n_b] = E[b, t].T
        C_hat_T = C_T[b].reshape(T * n_b, n_r)[rows]
        M_T_ref = sla.solve_triangular(np.linalg.cholesky(
            K[np.ix_(rows, rows)]), C_hat_T, lower=True)
        # GPU root_coupling_T rows are physical order: permute to check
        M_T = M_T_gpu[b][rows]
        assert np.abs(M_T - M_T_ref).max() <= 1e-8 * max(scale, 1.0)
        base = b * T * n_b
        L[base:base + T * n_b, base:base + T * n_b] = L_hat
        L[-n_r:, base:base + T * n_b] = M_T.T
        root_update += M_T.T @ M_T
    # root diagonal: L_R L_R^T = R - sum M M^T
    assert np.abs(L_R @ L_R.T - (R - root_update)).max() <= 1e-8 * scale
    L[-n_r:, -n_r:] = L_R

    Phi = assemble_dense_system(matrix)
    p = global_permutation(matrix.shape)
    Phi_hat = Phi[np.ix_(p, p)]
    assert np.abs(Phi_hat - L @ L.T).max() <= 1e-8 * scale


@pytest.mark.gpu
def test_lifecycle_state_machine():
    from src.general_arrow.solver import Solver
    problem = _problem(B=2, T=4, n_b=8, n_r=2, seed=5)
    solver = Solver(problem.matrix.shape)
    # factorize before update fails before device work
    with pytest.raises(RuntimeError, match="update"):
        solver.factorize()
    # solve before factorize fails
    solver.update(problem.matrix)
    with pytest.raises(RuntimeError, match="factorize"):
        solver.solve(problem.rhs)
    solver.factorize()
    x1 = solver.solve(problem.rhs)
    # update() invalidates the factor without factorizing implicitly
    solver.update(problem.matrix)
    with pytest.raises(RuntimeError, match="factoriz"):
        solver.solve(problem.rhs)
    # repeated factorize refreshes overwritten buffers (same solution)
    solver.factorize()
    x2 = solver.solve(problem.rhs)
    a, b = x1.numpy(), x2.numpy()
    tol = 100 * np.finfo(np.float64).eps
    assert np.abs(a.tail - b.tail).max() <= tol
    assert np.abs(a.root - b.root).max() <= tol

"""Equivalence tests for the SOCU partial-solve API patch (plan-2 section 4).

These tests validate the upstream-candidate launch builders in
:mod:`src.socu_patch` against SOCU's existing full-solve and factor
builders, on both SOCU code paths (fused ``n < 32`` and blocked
``n >= 32``), in FP64 and FP32, batched, with multiple right-hand sides.
"""

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from socu.block_tridiag_solver import (  # noqa: E402
    calculate_off_diag_storage_len,
    create_cholesky_factor_launch,
    create_cholesky_solve_launch,
)
from src.socu_patch import (  # noqa: E402
    create_cholesky_backward_solve_launch,
    create_cholesky_factor_and_forward_solve_launch,
    create_cholesky_forward_solve_launch,
)
from benchmarks.problems import ProblemSpec, generate_problem  # noqa: E402
from src import validation  # noqa: E402

pytestmark = pytest.mark.gpu

# (B, N, n) covering the fused (n < 32) and blocked (n >= 32) SOCU paths,
# N = 1, and non-power-of-two horizons
CASES = [(3, 6, 16), (2, 5, 32), (4, 1, 16), (2, 9, 16), (2, 8, 64)]


def _factored_system(B, N, n, q, wp_dtype, seed=3):
    spec = ProblemSpec(num_branches=B, horizon=N, block_size=n,
                       separator_dim=4, seed=seed,
                       precision="float64" if wp_dtype == wp.float64
                       else "float32")
    s = generate_problem(spec)
    n_off = calculate_off_diag_storage_len(N)
    L = wp.array(s.matrix.D, dtype=wp_dtype, device="cuda:0")
    E = wp.zeros((B, n_off, n, n), dtype=wp_dtype, device="cuda:0")
    if N > 1:
        wp.copy(E[:, :N - 1], wp.array(s.matrix.E, dtype=wp_dtype,
                                       device="cuda:0"))
    rng = np.random.default_rng(seed)
    rhs = rng.standard_normal((B, N, n, q))
    return s, L, E, rhs


@pytest.mark.parametrize("B,N,n", CASES)
@pytest.mark.parametrize("wp_dtype", [wp.float64, wp.float32])
def test_forward_then_backward_equals_full_solve(B, N, n, wp_dtype):
    q = 3
    s, L, E, rhs = _factored_system(B, N, n, q, wp_dtype)
    create_cholesky_factor_launch(L, E, dtype=wp_dtype, device="cuda:0")()

    x_full = wp.array(rhs, dtype=wp_dtype, device="cuda:0")
    create_cholesky_solve_launch(L, E, x_full, dtype=wp_dtype,
                                 device="cuda:0")()

    x_split = wp.array(rhs, dtype=wp_dtype, device="cuda:0")
    create_cholesky_forward_solve_launch(L, E, x_split, dtype=wp_dtype,
                                         device="cuda:0")()
    create_cholesky_backward_solve_launch(L, E, x_split, dtype=wp_dtype,
                                          device="cuda:0")()
    wp.synchronize()
    # The fused path (n < 32) is bitwise reproducible; the blocked
    # multi-stream path (n >= 32) is not run-to-run bitwise deterministic
    # even for upstream's own full solve (epsilon-level FMA-order
    # variation), so compare at a tight scale-aware tolerance.
    eps = np.finfo(np.float64 if wp_dtype == wp.float64 else np.float32).eps
    scale = np.abs(x_full.numpy()).max()
    assert np.allclose(x_full.numpy(), x_split.numpy(),
                       atol=100 * eps * max(scale, 1.0))


@pytest.mark.parametrize("B,N,n", CASES)
def test_factor_and_forward_equals_separate_calls(B, N, n):
    q = 2
    s, L, E, rhs = _factored_system(B, N, n, q, wp.float64)
    L2, E2 = wp.clone(L), wp.clone(E)

    x_fused = wp.array(rhs, dtype=wp.float64, device="cuda:0")
    create_cholesky_factor_and_forward_solve_launch(
        L, E, x_fused, dtype=wp.float64, device="cuda:0")()

    x_sep = wp.array(rhs, dtype=wp.float64, device="cuda:0")
    create_cholesky_factor_launch(L2, E2, dtype=wp.float64, device="cuda:0")()
    create_cholesky_forward_solve_launch(L2, E2, x_sep, dtype=wp.float64,
                                         device="cuda:0")()
    wp.synchronize()
    assert np.allclose(L.numpy(), L2.numpy(), atol=1e-12)
    assert np.allclose(x_fused.numpy(), x_sep.numpy(), atol=1e-12)


@pytest.mark.parametrize("B,N,n", CASES)
def test_forward_operator_identity(B, N, n):
    """F^T F = K^{-1}: the identity the factor-space Schur update relies
    on.  Checked as (F r1)^T (F r2) == r1^T K^{-1} r2 for random vectors."""
    q = 2
    s, L, E, rhs = _factored_system(B, N, n, q, wp.float64)
    create_cholesky_factor_launch(L, E, dtype=wp.float64, device="cuda:0")()

    x = wp.array(rhs, dtype=wp.float64, device="cuda:0")
    create_cholesky_forward_solve_launch(L, E, x, dtype=wp.float64,
                                         device="cuda:0")()
    wp.synchronize()
    W = x.numpy()  # (B, N, n, q) forward-transformed columns

    Ld, Lsub = validation.chain_cholesky(s.matrix.D, s.matrix.E)
    Kinv_r = validation.chain_solve(Ld, Lsub, rhs)
    # Gram matrices per branch: W^T W vs rhs^T K^-1 rhs
    got = np.einsum("bkiq,bkip->bqp", W, W)
    ref = np.einsum("bkiq,bkip->bqp", rhs, Kinv_r)
    assert np.allclose(got, ref, atol=1e-9 * max(1.0, np.abs(ref).max()))

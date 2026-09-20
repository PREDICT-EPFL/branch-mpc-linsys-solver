"""Symbolic plan and normal-matrix assembly plan (CPU only)."""

import numpy as np
import pytest
import scipy.sparse as sp

from admm.problem import build_plan, _validate_vectors
from experiments.general_arrow.benchmarks.problems import generate_scenario_qp


def _qp(n=10, m=14, seed=1):
    rng = np.random.default_rng(seed)
    P_full = rng.standard_normal((n, n))
    P_full = P_full @ P_full.T + n * np.eye(n)
    P_full[np.abs(P_full) < 1.0] = 0.0  # sparsify
    np.fill_diagonal(P_full, np.abs(np.diag(P_full)) + 1.0)
    P = sp.csc_matrix(np.triu(P_full))
    A_d = rng.standard_normal((m, n))
    A_d[np.abs(A_d) < 0.8] = 0.0
    A = sp.csc_matrix(A_d)
    return P, A


def test_assembly_plan_matches_scipy():
    P, A = _qp()
    rho = 2.5
    plan = build_plan(P, A)
    K_vals = plan.assemble_K_host(P.data, A.data,
                                  np.full(A.shape[0], rho))
    K = sp.csr_matrix((K_vals, plan.K_indices, plan.K_indptr),
                      shape=(plan.n, plan.n))
    P_full = sp.triu(P) + sp.triu(P, k=1).T
    ref = sp.tril(P_full + rho * (A.T @ A)).tocsr()
    ref.sort_indices()
    assert np.allclose(K.toarray(), ref.toarray(), atol=1e-13)


def test_assembly_plan_tracks_value_updates():
    P, A = _qp(seed=2)
    plan = build_plan(P, A)
    rng = np.random.default_rng(3)
    new_P = rng.standard_normal(P.nnz)
    new_A = rng.standard_normal(A.nnz)
    K_vals = plan.assemble_K_host(new_P, new_A,
                                  np.ones(A.shape[0]))
    P2 = sp.csc_matrix((new_P, P.indices, P.indptr), shape=P.shape)
    A2 = sp.csc_matrix((new_A, A.indices, A.indptr), shape=A.shape)
    P2_full = sp.triu(P2) + sp.triu(P2, k=1).T
    ref = sp.tril(P2_full + (A2.T @ A2)).tocsr()
    ref.sort_indices()
    K = sp.csr_matrix((K_vals, plan.K_indices, plan.K_indptr),
                      shape=(plan.n, plan.n))
    assert np.allclose(K.toarray(), ref.toarray(), atol=1e-13)


def test_csc_to_csr_value_map():
    _, A = _qp(seed=4)
    plan = build_plan(sp.identity(A.shape[1], format="csc").astype(float), A)
    Ar = A.tocsr()
    Ar.sort_indices()
    assert np.allclose(np.asarray(A.data)[plan.csc_to_csr], Ar.data)


def test_input_validation():
    P, A = _qp()
    n, m = P.shape[0], A.shape[0]
    with pytest.raises(TypeError, match="csc"):
        build_plan(P.tocsr(), A)
    with pytest.raises(ValueError, match="upper"):
        build_plan(sp.csc_matrix(sp.tril(P.T)), A)
    with pytest.raises(ValueError, match="columns"):
        build_plan(P, sp.csc_matrix(np.ones((m, n + 1))))
    with pytest.raises(ValueError, match="length"):
        _validate_vectors(np.zeros(n + 1), np.zeros(m), np.zeros(m), n, m)
    with pytest.raises(ValueError, match="l <= u"):
        _validate_vectors(np.zeros(n), np.ones(m), np.zeros(m), n, m)


def test_scenario_qp_is_valid_input():
    P, q, A, l, u, meta = generate_scenario_qp(num_tails=3, num_stages=4,
                                               nx=4, nu=2, seed=7)
    plan = build_plan(P, A)
    K_vals = plan.assemble_K_host(P.data, A.data, np.ones(A.shape[0]))
    P_full = sp.triu(P) + sp.triu(P, k=1).T
    K_dense = sp.csr_matrix((K_vals, plan.K_indices, plan.K_indptr),
                            shape=(plan.n, plan.n)).toarray()
    K_dense = K_dense + np.tril(K_dense, -1).T
    ref = (P_full + (A.T @ A)).toarray()
    assert np.allclose(K_dense, ref, atol=1e-12)
    np.linalg.cholesky(ref)  # SPD as promised

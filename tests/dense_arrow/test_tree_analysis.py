"""Conservative tree-structure analyzer (CPU only)."""

import numpy as np
import pytest
import scipy.sparse as sp

from admm.problem import (
    TreeStructureError,
    analyze_tree_structure,
    build_plan,
)
from experiments.general_arrow.benchmarks.problems import generate_scenario_qp


def test_scenario_qp_recognized_with_expected_dimensions():
    P, q, A, l, u, meta = generate_scenario_qp(num_tails=3, num_stages=5,
                                               nx=6, nu=2, seed=0)
    plan = build_plan(P, A)
    tree = analyze_tree_structure(plan)
    assert tree.shape.num_tails == meta["num_tails"]
    assert tree.shape.num_stages == meta["num_stages"]
    assert tree.shape.tail_block_dim == meta["tail_block_dim"]
    assert tree.shape.root_dim == meta["root_dim"]


def test_packing_reconstructs_dense_k():
    """Scattering the K lower values through the packing maps rebuilds
    exactly the dense symmetric K."""
    P, q, A, l, u, meta = generate_scenario_qp(num_tails=2, num_stages=4,
                                               nx=4, nu=2, seed=1)
    rho = 1.7
    plan = build_plan(P, A)
    tree = analyze_tree_structure(plan)
    K_vals = plan.assemble_K_host(P.data, A.data,
                                  np.full(A.shape[0], rho))
    B, T, n_b, n_r = tree.shape.dims()
    D = np.zeros(B * T * n_b * n_b)
    E = np.zeros(max(B * (T - 1) * n_b * n_b, 1))
    C = np.zeros(B * T * n_b * n_r)
    R = np.zeros(n_r * n_r)
    for buf, src, dst in ((D, tree.d_src, tree.d_dst),
                          (E, tree.e_src, tree.e_dst),
                          (C, tree.c_src, tree.c_dst),
                          (R, tree.r_src, tree.r_dst)):
        buf[dst] = K_vals[src]
    # rebuild the dense matrix from the blocks
    n = tree.shape.total_dimension
    off = tree.shape.tail_dimension
    K_dense = np.zeros((n, n))
    D = D.reshape(B, T, n_b, n_b)
    E = E.reshape(B, max(T - 1, 0), n_b, n_b) if T > 1 else E
    C = C.reshape(B, T, n_b, n_r)
    for b in range(B):
        for t in range(T):
            r0 = (b * T + t) * n_b
            K_dense[r0:r0 + n_b, r0:r0 + n_b] = D[b, t]
            if t < T - 1:
                K_dense[r0 + n_b:r0 + 2 * n_b, r0:r0 + n_b] = E[b, t]
                K_dense[r0:r0 + n_b, r0 + n_b:r0 + 2 * n_b] = E[b, t].T
            K_dense[r0:r0 + n_b, off:] = C[b, t]
            K_dense[off:, r0:r0 + n_b] = C[b, t].T
    K_dense[off:, off:] = R.reshape(n_r, n_r)
    P_full = sp.triu(P) + sp.triu(P, k=1).T
    ref = (P_full + rho * (A.T @ A)).toarray()
    assert np.allclose(K_dense, ref, atol=1e-12)


def test_unstructured_qp_rejected_with_reason():
    rng = np.random.default_rng(2)
    n, m = 12, 16
    Pd = rng.standard_normal((n, n))
    P = sp.csc_matrix(np.triu(Pd @ Pd.T + n * np.eye(n)))
    A = sp.csc_matrix(rng.standard_normal((m, n)))  # dense coupling
    plan = build_plan(P, A)
    with pytest.raises(TreeStructureError, match="root|tails"):
        analyze_tree_structure(plan)


def test_single_chain_rejected():
    """One tail (no scenario split) is not a tree problem."""
    n = 12
    diags = [np.ones(n), 0.3 * np.ones(n - 1)]
    K = sp.diags(diags, [0, 1], format="csc") + sp.diags(diags, [0, -1],
                                                         format="csc")
    P = sp.csc_matrix(sp.triu(K))
    A = sp.csc_matrix(sp.identity(n, format="csc"))
    plan = build_plan(P, A)
    with pytest.raises(TreeStructureError):
        analyze_tree_structure(plan)

"""GPU solver correctness: against dense solve, the CPU reference, the
generator ground truth, and the general solver on the converted
matrix."""

import numpy as np
import pytest
import scipy.sparse as sp

from src.endpoint_tree import (
    EndpointTreeShape,
    EndpointTreeSolver,
    EndpointTreeVector,
)
from tests.endpoint_tree.problems import generate_endpoint_problem
from src.endpoint_tree.reference import factorize_reference, solve_reference

wp = pytest.importorskip("warp")
pytestmark = pytest.mark.gpu

SHAPES = [EndpointTreeShape(1, 1, 3, 2), EndpointTreeShape(1, 6, 4, 3),
          EndpointTreeShape(5, 3, 4, 6),
          EndpointTreeShape(3, 8, 9, 9),      # drone-like odd block
          EndpointTreeShape(4, 5, 6, 16),
          EndpointTreeShape(8, 16, 10, 12)]


def _solve_gpu(p, graph=True):
    solver = EndpointTreeSolver(p.shape)
    solver.update(p.matrix)
    solver.factorize()
    return solver, solver.solve(p.rhs, graph=graph)


def _err(x, ref):
    return (np.linalg.norm(x - ref)
            / max(np.linalg.norm(ref), 1e-300))


@pytest.mark.parametrize("shape", SHAPES)
def test_gpu_matches_dense_and_truth(shape):
    p = generate_endpoint_problem(shape, seed=21)
    solver, x = _solve_gpu(p)
    xf = np.concatenate([x.tail.numpy().reshape(-1), x.root.numpy()])
    A = p.matrix.to_csr_lower(dtype=np.float64)
    K = (A + sp.tril(A, k=-1).T).toarray()
    x_dense = np.linalg.solve(K, p.rhs.flat())
    assert _err(xf, x_dense) < 1e-11
    assert _err(xf, p.exact_solution.flat()) < 1e-10
    r = K @ xf - p.rhs.flat()
    assert np.linalg.norm(r) / np.linalg.norm(p.rhs.flat()) < 1e-12


@pytest.mark.parametrize("shape", SHAPES[:4])
def test_gpu_matches_cpu_reference(shape):
    p = generate_endpoint_problem(shape, seed=22)
    _, x = _solve_gpu(p)
    x_ref = solve_reference(factorize_reference(p.matrix), p.rhs)
    xf = np.concatenate([x.tail.numpy().reshape(-1), x.root.numpy()])
    assert _err(xf, x_ref.flat()) < 1e-11


def test_gpu_matches_general_solver_on_converted_matrix():
    shape = EndpointTreeShape(6, 8, 6, 16)
    p = generate_endpoint_problem(shape, seed=23)
    _, x = _solve_gpu(p)
    from src.endpoint_tree._reuse import TreeVector
    from src.dense_arrow.solver import Solver as GeneralSolver
    general = p.matrix.to_general_tree_matrix()
    gs = GeneralSolver(general.shape)
    gs.update(general)
    gs.factorize()
    grhs = TreeVector(general.shape,
                      np.asarray(p.rhs.tail)[..., None].copy(),
                      np.asarray(p.rhs.root)[:, None].copy())
    gx = gs.solve(grhs)
    xf = np.concatenate([x.tail.numpy().reshape(-1), x.root.numpy()])
    gf = np.concatenate([gx.tail.numpy().reshape(-1),
                         gx.root.numpy().reshape(-1)])
    assert _err(xf, gf) < 1e-11


def test_graph_and_eager_paths_agree():
    shape = EndpointTreeShape(4, 6, 5, 7)
    p = generate_endpoint_problem(shape, seed=24)
    solver = EndpointTreeSolver(shape)
    solver.update(p.matrix)
    solver.factorize()
    xg = solver.solve(p.rhs, graph=True)
    xg_t, xg_r = xg.tail.numpy().copy(), xg.root.numpy().copy()
    xe = solver.solve(p.rhs, graph=False)
    # SOCU's substitutions accumulate with atomics, so two
    # separate executions may differ in the last ulps
    np.testing.assert_allclose(xg_t, xe.tail.numpy(),
                               rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(xg_r, xe.root.numpy(),
                               rtol=1e-12, atol=1e-13)


def test_repeated_update_factorize_solve():
    shape = EndpointTreeShape(3, 5, 4, 6)
    solver = EndpointTreeSolver(shape)
    for seed in (31, 32, 33):
        p = generate_endpoint_problem(shape, seed=seed)
        solver.update(p.matrix)
        solver.factorize()
        x = solver.solve(p.rhs)
        xf = np.concatenate([x.tail.numpy().reshape(-1),
                             x.root.numpy()])
        assert _err(xf, p.exact_solution.flat()) < 1e-10


def test_float32_solve():
    shape = EndpointTreeShape(3, 6, 4, 5, precision="float32")
    p = generate_endpoint_problem(shape, seed=25)
    _, x = _solve_gpu(p)
    xf = np.concatenate([x.tail.numpy().reshape(-1), x.root.numpy()])
    assert _err(xf, p.exact_solution.flat()) < 1e-3


@pytest.mark.parametrize("T", [2, 5, 8, 9, 16, 21, 33])
def test_connector_path_matches_dense_socu_transform(T):
    """The stored H_path must equal the dense SOCU forward transform of
    the connector on its support, and the reported path must equal the
    numerical support (plan section 8)."""
    shape = EndpointTreeShape(3, T, 4, 5)
    p = generate_endpoint_problem(shape, seed=70 + T)
    solver = EndpointTreeSolver(shape)
    solver.update(p.matrix)
    solver.factorize()
    diag = solver.factor_diagnostics()
    path = diag["path"]
    H_path = diag["H_path"]

    # dense reference: SOCU forward substitution on a zero-padded
    # full-prefix RHS holding the connector at slot P-1
    from src.endpoint_tree._reuse import TailEngine
    P, n_p = T - 1, solver.padded_block_dim
    from src.endpoint_tree.kernels.tail import pad_blocks
    D, E, G = pad_blocks(np.asarray(p.matrix.D), np.asarray(p.matrix.E),
                         np.asarray(p.matrix.G_T), n_p)
    engine = TailEngine(3, P, n_p, wp.float64, "cuda:0")
    engine.stage(D[:, :P], E[:, :P - 1])
    dense = np.zeros((3, P, n_p, n_p))
    dense[:, P - 1] = np.swapaxes(E[:, P - 1], -1, -2)
    x = wp.array(dense, dtype=wp.float64, device="cuda:0")
    fwd = engine.build_forward_launch(x)
    from src.endpoint_tree._reuse import create_cholesky_factor_launch
    fac = create_cholesky_factor_launch(engine.diag_factor,
                                        engine.offdiag_factor,
                                        dtype=wp.float64,
                                        device=engine.device)
    engine.refresh()
    fac()
    fwd()
    wp.synchronize_device("cuda:0")
    H_dense = x.numpy()
    # numerical support == reported path
    nz = [k for k in range(P)
          if np.abs(H_dense[:, k]).max() > 1e-13]
    assert nz == list(path)
    # values agree block for block
    for pos, slot in enumerate(path):
        np.testing.assert_allclose(H_path[:, pos], H_dense[:, slot],
                                   rtol=1e-11, atol=1e-12)
    assert len(path) <= 2 * int(P).bit_length() + 1


def test_condensed_node_and_root_match_reference():
    """Dbar (as J J^T) and the root Schur complement match the CPU
    leaf-to-root reference and the generator target."""
    shape = EndpointTreeShape(4, 7, 4, 6)
    p = generate_endpoint_problem(shape, seed=77)
    solver = EndpointTreeSolver(shape)
    solver.update(p.matrix)
    solver.factorize()
    diag = solver.factor_diagnostics()
    ref = factorize_reference(p.matrix)
    n_b = shape.tail_block_dim
    J = diag["J"][:, :n_b, :n_b]
    Dbar_gpu = np.einsum("bij,bkj->bik", J, J)
    Dbar_ref = np.einsum("bij,bkj->bik", ref.L[:, -1], ref.L[:, -1])
    np.testing.assert_allclose(Dbar_gpu, Dbar_ref, rtol=1e-10,
                               atol=1e-12)
    L_R = np.tril(diag["S_factor"])
    np.testing.assert_allclose(L_R @ L_R.T, p.root_schur_target,
                               rtol=1e-9, atol=1e-10)


def test_padding_leaves_logical_solution_unchanged():
    """Odd drone-like block (padded internally to the SOCU-aligned
    size): the logical solution equals the CPU reference exactly, and
    the padded dimension is reported."""
    shape = EndpointTreeShape(3, 6, 9, 9)
    p = generate_endpoint_problem(shape, seed=78)
    solver = EndpointTreeSolver(shape)
    assert solver.padded_block_dim == 10  # fp64 alignment: even
    solver.update(p.matrix)
    solver.factorize()
    x = solver.solve(p.rhs)
    x_ref = solve_reference(factorize_reference(p.matrix), p.rhs)
    xf = np.concatenate([x.tail.numpy().reshape(-1), x.root.numpy()])
    assert _err(xf, x_ref.flat()) < 1e-11


def test_graph_node_counts_reported():
    shape = EndpointTreeShape(4, 9, 4, 5)
    p = generate_endpoint_problem(shape, seed=79)
    solver = EndpointTreeSolver(shape)
    solver.update(p.matrix)
    solver.factorize()
    solver.solve(p.rhs)
    assert solver.factor_graph_nodes and solver.factor_graph_nodes > 3
    assert solver.solve_graph_nodes and solver.solve_graph_nodes > 3
    # boundary work is batched: node counts do not scale with B within
    # one root-kernel regime (B=16 stays on the fused-root side)
    big = EndpointTreeShape(16, 9, 4, 5)
    p2 = generate_endpoint_problem(big, seed=80)
    s2 = EndpointTreeSolver(big)
    s2.update(p2.matrix)
    s2.factorize()
    s2.solve(p2.rhs)
    assert s2.factor_graph_nodes == solver.factor_graph_nodes
    assert s2.solve_graph_nodes == solver.solve_graph_nodes
    # a large batch switches to the root fallback (init copy + atomic
    # reduction + tile kernel replacing the single fused launch):
    # exactly two extra nodes per pipeline, independent of B
    huge = EndpointTreeShape(64, 9, 4, 5)
    p3 = generate_endpoint_problem(huge, seed=81)
    s3 = EndpointTreeSolver(huge)
    s3.update(p3.matrix)
    s3.factorize()
    s3.solve(p3.rhs)
    assert s3.factor_graph_nodes == solver.factor_graph_nodes + 2
    assert s3.solve_graph_nodes == solver.solve_graph_nodes + 2

"""API-contract tests: shape/lifecycle validation, output ownership,
and prepared-solve management (review-driven; no numerics here)."""

import numpy as np
import pytest

from experiments.dense_arrow.benchmarks.problems import ProblemSpec, generate_problem
from src.dense_arrow import TreeMatrix, TreeShape, TreeVector
from src.dense_arrow.problem import tree_vector_from_arrays

SMALL = ProblemSpec(num_tails=3, horizon=5, block_size=16,
                    root_dim=8, seed=7)


# ------------------------------------------------------------- CPU contracts
def test_tree_shape_validation_and_spec_mapping():
    with pytest.raises(ValueError):
        TreeShape(0, 4, 4, 4)
    with pytest.raises(ValueError):
        TreeShape(1, 4, 4, 4, precision="float16")
    shape = SMALL.shape
    assert shape == TreeShape(3, 5, 16, 8)
    assert shape.dims() == (3, 5, 16, 8)
    assert shape.tail_dimension == 3 * 5 * 16
    assert shape.total_dimension == 3 * 5 * 16 + 8
    assert shape.nnz_lower == SMALL.nnz_lower


def test_tree_matrix_shape_checks():
    p = generate_problem(SMALL)
    m = p.matrix
    with pytest.raises(ValueError, match="C_T"):
        TreeMatrix(p.shape, D=m.D, E=m.E, C_T=m.C_T[:, :, :, :4], R=m.R)
    with pytest.raises(ValueError, match="R"):
        TreeMatrix(p.shape, D=m.D, E=m.E, C_T=m.C_T, R=m.R[:4, :4])


def test_tree_vector_shape_checks_and_helpers():
    shape = SMALL.shape
    B, T, n_b, n_r = shape.dims()
    with pytest.raises(ValueError, match="4D"):
        TreeVector(shape, np.zeros((B, T, n_b)), np.zeros((n_r, 1)))
    with pytest.raises(ValueError, match="root"):
        TreeVector(shape, np.zeros((B, T, n_b, 2)), np.zeros((n_r, 3)))
    vec = tree_vector_from_arrays(shape, np.zeros((B, T, n_b)),
                                  np.zeros(n_r))
    assert vec.nrhs == 1
    assert vec.tail.shape == (B, T, n_b, 1)


# -------------------------------------------------------------- GPU contracts
pytest.importorskip("warp")


@pytest.mark.gpu
def test_lifecycle_errors():
    from src.dense_arrow.solver import Solver
    problem = generate_problem(SMALL)
    solver = Solver(problem.shape)
    with pytest.raises(RuntimeError, match="update"):
        solver.factorize()
    solver.update(problem.matrix)
    with pytest.raises(RuntimeError, match="factorize"):
        solver.solve(problem.rhs)
    solver.factorize()
    solver.solve(problem.rhs)
    # updating invalidates the factor until factorize() again
    solver.update(problem.matrix)
    with pytest.raises(RuntimeError, match="factoriz"):
        solver.solve(problem.rhs)


@pytest.mark.gpu
def test_shape_mismatch_rejected_before_gpu_work():
    from src.dense_arrow.solver import Solver
    problem = generate_problem(SMALL)
    other = generate_problem(ProblemSpec(num_tails=2, horizon=5,
                                         block_size=16, root_dim=8,
                                         seed=1))
    solver = Solver(problem.shape)
    with pytest.raises(ValueError, match="shape"):
        solver.update(other.matrix)
    solver.update(problem.matrix)
    solver.factorize()
    with pytest.raises(ValueError, match="shape"):
        solver.solve(other.rhs)


@pytest.mark.gpu
def test_solve_returns_owned_output():
    import warp as wp
    from src.dense_arrow.solver import Solver
    problem = generate_problem(SMALL)
    solver = Solver(problem.shape)
    solver.update(problem.matrix)
    solver.factorize()
    a = solver.solve(problem.rhs)
    b = solver.solve(problem.rhs)
    assert a.tail.ptr != b.tail.ptr  # distinct device buffers
    a.tail.zero_()  # mutating one result must not affect the other
    wp.synchronize_device("cuda:0")
    assert float(np.abs(b.numpy().tail).max()) > 0.0


@pytest.mark.gpu
def test_bound_solve_binding():
    """Device rhs/out pointers bind into the solve graph; the binding is
    reused across calls and rebuilt when nrhs or pointers change."""
    import warp as wp
    from src.dense_arrow.solver import Solver
    problem = generate_problem(SMALL)
    solver = Solver(problem.shape)
    solver.update(problem.matrix)
    solver.factorize()
    B, T, n_b, n_r = problem.shape.dims()
    dt = wp.float64

    def dev_rhs():
        return TreeVector(
            problem.shape,
            wp.array(np.ascontiguousarray(problem.rhs.tail), dtype=dt,
                     device="cuda:0"),
            wp.array(np.ascontiguousarray(problem.rhs.root), dtype=dt,
                     device="cuda:0"))

    out = TreeVector(problem.shape,
                     wp.zeros((B, T, n_b, 1), dtype=dt, device="cuda:0"),
                     wp.zeros((n_r, 1), dtype=dt, device="cuda:0"))
    rhs = dev_rhs()
    got = solver.solve(rhs, out=out)
    assert got is out
    binding = solver._binding
    # same pointers: binding object is reused
    rhs2 = dev_rhs()
    wp.copy(rhs.tail, rhs2.tail)
    wp.copy(rhs.root, rhs2.root)
    solver.solve(rhs, out=out)
    assert solver._binding is binding
    ref = solver.solve(problem.rhs).numpy()  # host path (staged copy)
    assert solver._binding is not binding    # rebound: different key
    host = out.numpy()
    assert np.allclose(host.tail, ref.tail, atol=1e-10)
    # mismatched widths are rejected before any GPU work
    with pytest.raises(ValueError, match="columns"):
        solver.solve(dev_rhs(), out=TreeVector(
            problem.shape,
            wp.zeros((B, T, n_b, 3), dtype=dt, device="cuda:0"),
            wp.zeros((n_r, 3), dtype=dt, device="cuda:0")))


def test_tree_matrix_from_c_orientation():
    p = generate_problem(SMALL)
    m = p.matrix
    C = np.swapaxes(m.C_T, -1, -2)  # bottom-left blocks (B, T, n_r, n_b)
    rebuilt = TreeMatrix.from_C(p.shape, m.D, m.E, C, m.R)
    assert np.array_equal(rebuilt.C_T, m.C_T)
    with pytest.raises(ValueError, match="C"):
        TreeMatrix.from_C(p.shape, m.D, m.E, m.C_T, m.R)  # wrong layout


def test_matvec_method_and_alias_safety():
    p = generate_problem(SMALL)
    m = p.matrix
    x = p.exact_solution
    y = m.matvec(x)  # allocating path reproduces the generated rhs
    assert np.allclose(y.tail, p.rhs.tail, atol=1e-10)
    assert np.allclose(y.root, p.rhs.root, atol=1e-10)
    out = TreeVector(p.shape, np.zeros_like(x.tail),
                     np.zeros_like(x.root))
    got = m.matvec(x, out=out)  # disjoint out
    assert got is out and np.allclose(out.tail, y.tail, atol=1e-12)
    aliased = TreeVector(p.shape, x.tail.copy(), x.root.copy())
    m.matvec(aliased, out=aliased)  # out is x: must equal A @ original x
    assert np.allclose(aliased.tail, y.tail, atol=1e-12)
    assert np.allclose(aliased.root, y.root, atol=1e-12)
    with pytest.raises(ValueError, match="nrhs|shape"):
        m.matvec(x, out=TreeVector(
            p.shape, np.zeros((*x.tail.shape[:3], 3)),
            np.zeros((x.root.shape[0], 3))))


def test_to_csr_lower_contract():
    import scipy.sparse as sp
    from src.dense_arrow.problem import TreeVector
    p = generate_problem(SMALL)
    m = p.matrix
    lower = m.to_csr_lower()
    assert lower.has_sorted_indices
    # strictly lower-triangular storage plus the diagonal
    coo = lower.tocoo()
    assert (coo.row >= coo.col).all()
    # symmetrized CSR realizes the same operator as matrix.matvec
    A = lower + sp.tril(lower, k=-1).T
    rng = np.random.default_rng(0)
    x = TreeVector(p.shape,
                   rng.standard_normal(p.rhs.tail.shape),
                   rng.standard_normal(p.rhs.root.shape))
    y = m.matvec(x)
    z = A @ x.flat()
    assert np.allclose(z, y.flat(), atol=1e-12 * max(1.0, np.abs(z).max()))
    lower32 = m.to_csr_lower(dtype=np.float32)
    assert lower32.dtype == np.float32


def test_zero_nrhs_rejected():
    shape = SMALL.shape
    B, T, n_b, n_r = shape.dims()
    with pytest.raises(ValueError, match="nrhs"):
        TreeVector(shape, np.zeros((B, T, n_b, 0)), np.zeros((n_r, 0)))
    with pytest.raises(ValueError, match="3D or 4D"):
        tree_vector_from_arrays(shape, np.zeros((B, T)), np.zeros(n_r))

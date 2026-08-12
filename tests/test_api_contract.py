"""API-contract tests: shape/lifecycle validation, output ownership,
and prepared-solve management (review-driven; no numerics here)."""

import numpy as np
import pytest

from benchmarks.problems import ProblemSpec, generate_problem
from src import (
    TreeMatrix,
    TreeShape,
    TreeVector,
    tree_vector_from_arrays,
)

SMALL = ProblemSpec(num_branches=3, horizon=5, block_size=16,
                    separator_dim=8, seed=7)


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
    B, T, n_b, n_y = shape.dims()
    with pytest.raises(ValueError, match="4D"):
        TreeVector(shape, np.zeros((B, T, n_b)), np.zeros((n_y, 1)))
    with pytest.raises(ValueError, match="separator"):
        TreeVector(shape, np.zeros((B, T, n_b, 2)), np.zeros((n_y, 3)))
    vec = tree_vector_from_arrays(shape, np.zeros((B, T, n_b)),
                                  np.zeros(n_y))
    assert vec.nrhs == 1
    assert vec.branch.shape == (B, T, n_b, 1)


# -------------------------------------------------------------- GPU contracts
pytest.importorskip("warp")


@pytest.mark.gpu
def test_lifecycle_errors():
    from src.solver import TreeSolver
    problem = generate_problem(SMALL)
    solver = TreeSolver(problem.shape)
    with pytest.raises(RuntimeError, match="stage_matrix"):
        solver.factorize()
    with pytest.raises(RuntimeError, match="factorize"):
        solver.compute_min_pivot()
    with pytest.raises(TypeError, match="TreeMatrix"):
        solver.stage_matrix(problem.matrix.D)
    solver.stage_matrix(problem.matrix)
    solver.factorize()
    solver.solve(problem.rhs)
    # re-staging invalidates the factored state until factorize() again
    solver.stage_matrix(problem.matrix)
    with pytest.raises(RuntimeError, match="factoriz"):
        solver.solve(problem.rhs)


@pytest.mark.gpu
def test_shape_mismatch_rejected_before_gpu_work():
    from src.solver import TreeSolver
    problem = generate_problem(SMALL)
    other = generate_problem(ProblemSpec(num_branches=2, horizon=5,
                                         block_size=16, separator_dim=8,
                                         seed=1))
    solver = TreeSolver(problem.shape)
    with pytest.raises(ValueError, match="shape"):
        solver.stage_matrix(other.matrix)
    solver.stage_matrix(problem.matrix)
    solver.factorize()
    with pytest.raises(ValueError, match="shape"):
        solver.solve(other.rhs)


@pytest.mark.gpu
def test_solve_returns_owned_output():
    from src.solver import TreeSolver
    problem = generate_problem(SMALL)
    solver = TreeSolver(problem.shape)
    solver.stage_matrix(problem.matrix)
    solver.factorize()
    a = solver.solve(problem.rhs)
    b = solver.solve(problem.rhs)
    assert a.branch.ptr != b.branch.ptr  # distinct device buffers
    a.branch.zero_()  # mutating one result must not affect the other
    solver.synchronize()
    assert float(np.abs(b.numpy().branch).max()) > 0.0


@pytest.mark.gpu
def test_prepared_solve_binding():
    from src.solver import TreeSolver
    problem = generate_problem(SMALL)
    solver = TreeSolver(problem.shape)
    solver.stage_matrix(problem.matrix)
    solver.factorize()
    rhs_dev = solver.upload_rhs(problem.rhs)
    out = solver.create_solution(problem.rhs.nrhs)
    prepared = solver.prepare_solve(problem.rhs.nrhs, out=out)
    got = prepared.solve_into(rhs_dev)
    assert got is out is prepared.out
    ref = solver.solve(problem.rhs).numpy()
    host = out.numpy()
    assert np.allclose(host.branch, ref.branch, atol=1e-12)
    with pytest.raises(ValueError, match="columns"):
        prepared.solve_into(solver.upload_rhs(
            TreeVector(problem.shape,
                       np.zeros((*problem.rhs.branch.shape[:3], 3)),
                       np.zeros((problem.rhs.separator.shape[0], 3)))))
    with pytest.raises(ValueError, match="columns"):
        solver.prepare_solve(2, out=out)
    solver.close()


@pytest.mark.gpu
def test_context_manager():
    from src.solver import TreeSolver
    problem = generate_problem(SMALL)
    with TreeSolver(SMALL.shape) as solver:
        assert solver.shape == problem.shape
        solver.stage_matrix(problem.matrix)
        solver.factorize(check=True)


def test_tree_matrix_from_c_orientation():
    p = generate_problem(SMALL)
    m = p.matrix
    C = np.swapaxes(m.C_T, -1, -2)  # bottom-left blocks (B, T, n_y, n_b)
    rebuilt = TreeMatrix.from_C(p.shape, m.D, m.E, C, m.R)
    assert np.array_equal(rebuilt.C_T, m.C_T)
    with pytest.raises(ValueError, match="C"):
        TreeMatrix.from_C(p.shape, m.D, m.E, m.C_T, m.R)  # wrong layout


def test_matvec_method_and_alias_safety():
    p = generate_problem(SMALL)
    m = p.matrix
    x = p.exact_solution
    y = m.matvec(x)  # allocating path reproduces the generated rhs
    assert np.allclose(y.branch, p.rhs.branch, atol=1e-10)
    assert np.allclose(y.separator, p.rhs.separator, atol=1e-10)
    out = TreeVector(p.shape, np.zeros_like(x.branch),
                     np.zeros_like(x.separator))
    got = m.matvec(x, out=out)  # disjoint out
    assert got is out and np.allclose(out.branch, y.branch, atol=1e-12)
    aliased = TreeVector(p.shape, x.branch.copy(), x.separator.copy())
    m.matvec(aliased, out=aliased)  # out is x: must equal A @ original x
    assert np.allclose(aliased.branch, y.branch, atol=1e-12)
    assert np.allclose(aliased.separator, y.separator, atol=1e-12)
    with pytest.raises(ValueError, match="nrhs|shape"):
        m.matvec(x, out=TreeVector(
            p.shape, np.zeros((*x.branch.shape[:3], 3)),
            np.zeros((x.separator.shape[0], 3))))


def test_to_csr_lower_contract():
    import scipy.sparse as sp
    from src.problem import TreeVector
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
                   rng.standard_normal(p.rhs.branch.shape),
                   rng.standard_normal(p.rhs.separator.shape))
    y = m.matvec(x)
    z = A @ x.flat()
    assert np.allclose(z, y.flat(), atol=1e-12 * max(1.0, np.abs(z).max()))
    lower32 = m.to_csr_lower(dtype=np.float32)
    assert lower32.dtype == np.float32


def test_zero_nrhs_rejected():
    shape = SMALL.shape
    B, T, n_b, n_y = shape.dims()
    with pytest.raises(ValueError, match="nrhs"):
        TreeVector(shape, np.zeros((B, T, n_b, 0)), np.zeros((n_y, 0)))
    with pytest.raises(ValueError, match="3D or 4D"):
        tree_vector_from_arrays(shape, np.zeros((B, T)), np.zeros(n_y))


@pytest.mark.gpu
def test_closed_state_and_upload_validation():
    import warp as wp
    from src.solver import TreeSolver
    problem = generate_problem(SMALL)
    solver = TreeSolver(problem.shape)
    solver.stage_matrix(problem.matrix)
    solver.factorize()
    prepared = solver.prepare_solve(1)
    # wrong-dtype device rhs is rejected before any GPU work
    wrong = TreeVector(
        problem.shape,
        wp.zeros(problem.rhs.branch.shape, dtype=wp.float32, device="cuda:0"),
        wp.zeros(problem.rhs.separator.shape, dtype=wp.float32,
                 device="cuda:0"))
    with pytest.raises(TypeError, match="dtype"):
        solver.upload_rhs(wrong)
    with pytest.raises(ValueError, match="nrhs"):
        solver.create_solution(0)
    solver.close()
    for call in (lambda: solver.stage_matrix(problem.matrix),
                 lambda: solver.factorize(),
                 lambda: solver.prepare_solve(1),
                 lambda: solver.solve(problem.rhs),
                 lambda: prepared.solve_into(problem.rhs)):
        with pytest.raises(RuntimeError):
            call()

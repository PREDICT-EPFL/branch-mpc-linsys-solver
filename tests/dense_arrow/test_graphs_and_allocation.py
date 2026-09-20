"""CUDA-graph capture, warm-path allocation, and invalidation rules
(plan 3, stage 6)."""

import numpy as np
import pytest

wp = pytest.importorskip("warp")
pytestmark = pytest.mark.gpu

DEV = "cuda:0"


def _problem(B=11, T=16, n_b=8, n_r=2, nrhs=1, seed=0):
    from experiments.dense_arrow.benchmarks.problems import ProblemSpec, generate_problem
    spec = ProblemSpec(num_tails=B, horizon=T, block_size=n_b,
                       root_dim=n_r, num_rhs=nrhs,
                       precision="float64", seed=seed)
    return generate_problem(spec, estimate_condition=False)


def _dev_vec(shape, tail, root):
    from src.dense_arrow.problem import TreeVector
    return TreeVector(shape,
                      wp.array(np.ascontiguousarray(tail),
                               dtype=wp.float64, device=DEV),
                      wp.array(np.ascontiguousarray(root),
                               dtype=wp.float64, device=DEV))


def _refresh(rhs, problem):
    """Restore the bound RHS values (each solve consumes them)."""
    src = _dev_vec(problem.matrix.shape, problem.rhs.tail,
                   problem.rhs.root)
    wp.copy(rhs.tail, src.tail)
    wp.copy(rhs.root, src.root)


def _warm_solver(problem):
    from src.dense_arrow.solver import Solver
    shape = problem.matrix.shape
    solver = Solver(shape)
    solver.update(problem.matrix)
    solver.factorize()
    rhs = _dev_vec(shape, problem.rhs.tail, problem.rhs.root)
    out = _dev_vec(shape, np.zeros_like(problem.rhs.tail),
                   np.zeros_like(problem.rhs.root))
    solver.solve(rhs, out=out)  # builds + captures the solve binding
    solver.factorize()
    _refresh(rhs, problem)
    solver.solve(rhs, out=out)  # out now holds the true solution
    wp.synchronize_device(DEV)
    return solver, rhs, out


def test_warm_calls_allocate_zero_bytes():
    """After the first factorize/solve pair, warm calls (including the
    first replay of each captured graph) allocate no device memory."""
    problem = _problem()
    solver, rhs, out = _warm_solver(problem)
    before = wp.get_mempool_used_mem_current(DEV)
    for _ in range(3):
        solver.factorize()
        solver.solve(rhs, out=out)
    wp.synchronize_device(DEV)
    after = wp.get_mempool_used_mem_current(DEV)
    assert after == before, f"warm path allocated {after - before} bytes"


def test_graphs_survive_value_updates():
    """update() changes values in fixed buffers: captured graphs stay
    valid and produce the updated system's solution."""
    from src.dense_arrow.problem import TreeMatrix
    problem = _problem(seed=1)
    solver, rhs, out = _warm_solver(problem)
    # scale the whole system: solution of (2 Phi) x = r is x/2
    m = problem.matrix
    scaled = TreeMatrix(m.shape,
                        D=np.asarray(m.D) * 2.0, E=np.asarray(m.E) * 2.0,
                        C_T=np.asarray(m.C_T) * 2.0,
                        R=np.asarray(m.R) * 2.0)
    x_before = out.numpy()
    before = wp.get_mempool_used_mem_current(DEV)
    solver.update(scaled)
    solver.factorize()
    _refresh(rhs, problem)  # the solve consumed the bound rhs
    solver.solve(rhs, out=out)
    wp.synchronize_device(DEV)
    x_after = out.numpy()
    assert np.allclose(x_after.tail, x_before.tail / 2.0, atol=1e-10)
    assert np.allclose(x_after.root, x_before.root / 2.0,
                       atol=1e-10)


def test_bound_and_staged_paths_agree():
    """The zero-copy bound-pointer path and the staged host-rhs
    convenience path produce the same solution (up to SOCU's
    epsilon-level atomic-order nondeterminism, plan 9.4)."""
    problem = _problem(B=5, T=13, n_r=4, seed=2)
    solver, rhs, out = _warm_solver(problem)
    a = out.numpy()
    b = solver.solve(problem.rhs).numpy()
    tol = 100 * np.finfo(np.float64).eps
    scale = max(np.abs(b.tail).max(), 1.0)
    assert np.abs(a.tail - b.tail).max() <= tol * scale
    assert np.abs(a.root - b.root).max() <= tol * scale


def test_separate_factor_and_solve_graphs():
    """Factorization and solve replay independently: two factorizations
    between solves and two solves per factorization are both valid."""
    problem = _problem(seed=4)
    solver, rhs, out = _warm_solver(problem)
    solver.factorize()
    solver.factorize()
    _refresh(rhs, problem)
    solver.solve(rhs, out=out)
    x1 = out.numpy()
    _refresh(rhs, problem)
    solver.solve(rhs, out=out)
    x2 = out.numpy()
    tol = 100 * np.finfo(np.float64).eps
    assert np.abs(x1.tail - x2.tail).max() <= tol


def test_single_refresh_per_factorization():
    """One update() followed by one factorize() performs exactly one
    pristine-to-factor refresh of each tail buffer (plan 4: staging must
    not duplicate the device copy that factorize() owns)."""
    from src.dense_arrow.solver import Solver
    problem = _problem()
    solver = Solver(problem.matrix.shape)

    copies = []
    orig_copy = wp.copy

    def counting_copy(dst, src, *a, **k):
        copies.append(dst.ptr if hasattr(dst, "ptr") else None)
        return orig_copy(dst, src, *a, **k)

    wp.copy = counting_copy
    try:
        solver.update(problem.matrix)
        diag_ptr = solver._socu.diag_factor.ptr
        off_ptr = solver._socu.offdiag_factor.ptr
        n_update = sum(p in (diag_ptr, off_ptr) for p in copies)
        copies.clear()
        solver._factorize_numeric(None)  # eager: every node visible
        n_factor = sum(p in (diag_ptr, off_ptr) for p in copies)
    finally:
        wp.copy = orig_copy
    wp.synchronize_device(DEV)
    assert n_update == 0, "update() must not refresh factor buffers"
    assert n_factor == 2, "factorize() must refresh each buffer once"


def test_root_output_store_inside_graph():
    """The warm solve is exactly one graph replay: no graph-external
    launches or copies remain (the root output store is the last graph
    node)."""
    problem = _problem()
    solver, rhs, out = _warm_solver(problem)
    _refresh(rhs, problem)
    calls = []
    orig = (wp.copy, wp.launch, wp.launch_tiled)
    wp.copy = lambda *a, **k: calls.append("copy") or orig[0](*a, **k)
    wp.launch = lambda *a, **k: calls.append("launch") or orig[1](*a, **k)
    wp.launch_tiled = (
        lambda *a, **k: calls.append("tiled") or orig[2](*a, **k))
    try:
        solver.solve(rhs, out=out)
    finally:
        wp.copy, wp.launch, wp.launch_tiled = orig
    wp.synchronize_device(DEV)
    assert calls == [], f"graph-external work in warm solve: {calls}"
    # and the root output is still correct
    assert np.allclose(out.root.numpy(),
                       problem.exact_solution.root, atol=1e-8)


def test_rebind_on_changed_out_root_pointer():
    """Changing only the output root pointer rebinds the solve graph and
    writes the new buffer (the binding key covers every pointer)."""
    problem = _problem()
    solver, rhs, out = _warm_solver(problem)
    shape = problem.matrix.shape
    out2 = _dev_vec(shape, np.asarray(out.tail.numpy()),
                    np.zeros_like(problem.rhs.root))
    _refresh(rhs, problem)
    sol = solver.solve(rhs, out=out2)
    wp.synchronize_device(DEV)
    assert np.allclose(out2.root.numpy(), problem.exact_solution.root,
                       atol=1e-8)
    assert sol.root.ptr == out2.root.ptr


def test_rebind_on_changed_rhs_root_pointer():
    """Changing only the RHS root pointer rebinds and solves with the
    new root RHS values."""
    problem = _problem()
    solver, rhs, out = _warm_solver(problem)
    shape = problem.matrix.shape
    _refresh(rhs, problem)
    sep2 = wp.array(np.ascontiguousarray(problem.rhs.root),
                    dtype=wp.float64, device=DEV)
    from src.dense_arrow.problem import TreeVector
    rhs2 = TreeVector(shape, rhs.tail, sep2)
    solver.solve(rhs2, out=out)
    wp.synchronize_device(DEV)
    assert np.allclose(out.root.numpy(), problem.exact_solution.root,
                       atol=1e-8)
    assert np.allclose(out.tail.numpy(), problem.exact_solution.tail,
                       atol=1e-8)


def test_mixed_host_device_rhs_stages():
    """A mixed host/device RHS must not be classified zero-copy: it
    stages through internal buffers, leaves the caller arrays untouched,
    and still solves correctly."""
    from src.dense_arrow.problem import TreeVector
    problem = _problem()
    solver, rhs, out = _warm_solver(problem)
    shape = problem.matrix.shape
    tail_dev = wp.array(np.ascontiguousarray(problem.rhs.tail),
                          dtype=wp.float64, device=DEV)
    sep_host = np.ascontiguousarray(problem.rhs.root)
    mixed = TreeVector(shape, tail_dev, sep_host)
    sol = solver.solve(mixed, out=out).numpy()
    wp.synchronize_device(DEV)
    assert not solver._binding.zero_copy
    # staged path: the caller's device tail array is not consumed
    assert np.allclose(tail_dev.numpy(), problem.rhs.tail)
    assert np.allclose(sol.root, problem.exact_solution.root, atol=1e-8)
    assert np.allclose(sol.tail, problem.exact_solution.tail, atol=1e-8)


def test_aliased_out_root_skips_store():
    """out.root may alias the RHS root buffer (the root solve already
    ran in place there); the graph then skips the store and the result
    is still correct."""
    from src.dense_arrow.problem import TreeVector
    problem = _problem()
    shape = problem.matrix.shape
    from src.dense_arrow.solver import Solver
    solver = Solver(shape)
    solver.update(problem.matrix)
    solver.factorize()
    rhs = _dev_vec(shape, problem.rhs.tail, problem.rhs.root)
    out_tail = wp.zeros(rhs.tail.shape, dtype=wp.float64, device=DEV)
    out = TreeVector(shape, out_tail, rhs.root)  # aliased root
    solver.solve(rhs, out=out)
    _refresh(rhs, problem)
    solver.solve(rhs, out=out)
    wp.synchronize_device(DEV)
    assert np.allclose(out.root.numpy(), problem.exact_solution.root,
                       atol=1e-8)
    assert np.allclose(out.tail.numpy(), problem.exact_solution.tail,
                       atol=1e-8)


def test_convenience_path_preserves_device_rhs():
    """solve(rhs) with a device RHS and out=None must stage (never
    consume the caller's arrays): the RHS survives and repeated calls
    return the same correct solution."""
    problem = _problem()
    from src.dense_arrow.solver import Solver
    solver = Solver(problem.matrix.shape)
    solver.update(problem.matrix)
    solver.factorize()
    rhs = _dev_vec(problem.matrix.shape, problem.rhs.tail,
                   problem.rhs.root)
    x1 = solver.solve(rhs).numpy()
    x2 = solver.solve(rhs).numpy()  # same pointers, cached binding
    wp.synchronize_device(DEV)
    assert not solver._binding.zero_copy
    assert np.allclose(rhs.tail.numpy(), problem.rhs.tail)
    assert np.allclose(rhs.root.numpy(), problem.rhs.root)
    for x in (x1, x2):
        assert np.allclose(x.tail, problem.exact_solution.tail, atol=1e-8)
        assert np.allclose(x.root, problem.exact_solution.root, atol=1e-8)
    assert np.array_equal(x1.tail, x2.tail)

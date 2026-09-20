"""Tile-kernel capability probes (moved out of production construction,
plan 3 section 15.3): Warp tile kernels whose shapes exceed device
shared memory fail to launch with only a warning, so probe the largest
supported specializations numerically and fail loudly here instead."""

import numpy as np
import pytest

from src.general_arrow.kernels import BLOCK_DIM, TILE_M, num_root_tiles

wp = pytest.importorskip("warp")
pytestmark = pytest.mark.gpu

DEV = "cuda:0"


@pytest.mark.parametrize("n_r,precision", [(64, "float64"), (128, "float64"),
                                           (256, "float64"),
                                           (128, "float32")])
def test_chunked_coupling_kernel_launches(n_r, precision):
    from src.general_arrow._utils import wp_dtype
    from src.general_arrow.kernels.coupling import (create_chunked_root_update_kernel,
                                      num_row_chunks)
    dt = wp_dtype(precision)
    T, n_b = 8, 16
    rows = T * n_b
    mt = num_root_tiles(n_r)
    U = wp.zeros((1, rows, n_r), dtype=dt, device=DEV)
    U.fill_(1.0)
    out = wp.zeros((1, n_r, n_r), dtype=dt, device=DEV)
    wp.launch_tiled(create_chunked_root_update_kernel(dt), dim=[1, mt, mt],
                    inputs=[num_row_chunks(rows), U, U, out],
                    block_dim=BLOCK_DIM, device=DEV)
    wp.synchronize_device(DEV)
    got = out.numpy()[0]
    assert abs(float(got[0, 0]) - rows) <= 1e-3
    assert abs(float(got[n_r - 1, n_r - 1]) - rows) <= 1e-3


@pytest.mark.parametrize("n_r", [1, 8, 15, 64])
def test_single_tile_root_factor_launches(n_r):
    from src.general_arrow.kernels.root import create_root_factor_kernel
    S = wp.array(4.0 * np.eye(n_r), dtype=wp.float64, device=DEV)
    wp.launch_tiled(create_root_factor_kernel(n_r, wp.float64), dim=[1],
                    inputs=[S], block_dim=BLOCK_DIM, device=DEV)
    wp.synchronize_device(DEV)
    assert abs(float(S.numpy()[0, 0]) - 2.0) <= 1e-3


def test_big_block_end_to_end():
    """The largest supported tail block size solves correctly (a
    silently failed tile launch would corrupt the solution)."""
    from experiments.general_arrow.benchmarks.problems import ProblemSpec, generate_problem
    from baselines import reference
    from src.general_arrow.solver import Solver
    spec = ProblemSpec(num_tails=2, horizon=4, block_size=64,
                       root_dim=64, seed=0)
    problem = generate_problem(spec, estimate_condition=False)
    solver = Solver(problem.shape)
    solver.update(problem.matrix)
    solver.factorize()
    x = solver.solve(problem.rhs).numpy()
    metrics = reference.compute_metrics(problem, x.tail, x.root)
    assert metrics["forward_error"] <= 1e-8


@pytest.mark.slow
def test_oversized_nrhs_raises_instead_of_zeros():
    """An nrhs whose tile kernels exceed device shared memory must
    raise, never return a zero solution.  Warp 1.16 raises RuntimeError
    at kernel compile (mathdx LTO shared-memory estimate) or launch
    ("invalid argument"), so no sentinel probe layer is needed --
    verified empirically for both the SOCU substitution kernels and the
    project's chunked tile kernels; this test locks the loud-failure
    contract in."""
    from experiments.general_arrow.benchmarks.problems import ProblemSpec, generate_problem
    from src.general_arrow.problem import TreeVector
    from src.general_arrow.solver import Solver
    wp = pytest.importorskip("warp")
    spec = ProblemSpec(num_tails=2, horizon=4, block_size=16,
                       root_dim=16, num_rhs=1024, precision="float64",
                       seed=0)
    problem = generate_problem(spec, estimate_condition=False)
    solver = Solver(spec.shape)
    solver.update(problem.matrix)
    solver.factorize()
    rhs = TreeVector(spec.shape,
                     wp.array(problem.rhs.tail, dtype=wp.float64,
                              device="cuda:0"),
                     wp.array(problem.rhs.root, dtype=wp.float64,
                              device="cuda:0"))
    out = TreeVector(spec.shape,
                     wp.zeros(rhs.tail.shape, dtype=wp.float64,
                              device="cuda:0"),
                     wp.zeros(rhs.root.shape, dtype=wp.float64,
                              device="cuda:0"))
    with pytest.raises((RuntimeError, ValueError)):
        solver.solve(rhs, out=out)

"""Two-level permuted structured Cholesky solver for one-level
scenario-tree SPD systems.

Public API::

    from src.dense_arrow import TreeShape, TreeMatrix, TreeVector, Solver

Typical usage::

    solver = Solver(matrix.shape, device="cuda:0")
    solver.update(matrix)
    solver.factorize()
    x = solver.solve(rhs, out=x_workspace)

``Solver`` is imported lazily so that CPU-only work (references, hosts
without CUDA) does not initialize Warp.  Problem generation lives in
:mod:`experiments.dense_arrow.benchmarks.problems`; comparison solvers (cuDSS, CPU references)
in :mod:`baselines`.
"""

from src.dense_arrow.problem import TreeMatrix, TreeShape, TreeVector

__all__ = ["TreeShape", "TreeMatrix", "TreeVector", "Solver", "TreeSolver"]


def __getattr__(name):
    if name in ("Solver", "TreeSolver"):
        from src.dense_arrow.solver import Solver
        return Solver
    raise AttributeError(f"module 'src.dense_arrow' has no attribute {name!r}")

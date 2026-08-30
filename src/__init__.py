"""Two-level permuted structured Cholesky solver for one-level
scenario-tree SPD systems.

Public API::

    from src import TreeShape, TreeMatrix, TreeVector, Solver

Typical usage::

    solver = Solver(matrix.shape, device="cuda:0")
    solver.update(matrix)
    solver.factorize()
    x = solver.solve(rhs, out=x_workspace)

``Solver`` is imported lazily so that CPU-only work (references, hosts
without CUDA) does not initialize Warp.  Problem generation lives in
:mod:`benchmarks.problems`; comparison solvers (cuDSS, CPU references)
in :mod:`baselines`.
"""

from src.problem import TreeMatrix, TreeShape, TreeVector

__all__ = ["TreeShape", "TreeMatrix", "TreeVector", "Solver", "TreeSolver"]


def __getattr__(name):
    if name in ("Solver", "TreeSolver"):
        from src.solver import Solver
        return Solver
    raise AttributeError(f"module 'src' has no attribute {name!r}")

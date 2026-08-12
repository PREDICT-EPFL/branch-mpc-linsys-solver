"""GPU one-level scenario-tree Cholesky solver.

This package solves symmetric positive definite (SPD) linear systems with
one-level scenario-tree structure: ``B`` independent block-tridiagonal
branches (``T`` stage blocks of size ``n_b`` each) coupled to a shared
separator of dimension ``n_y``.  The structured GPU solver factors all
branch chains in parallel with the SOCU batched block-tridiagonal
Cholesky, forms the dense separator Schur complement on the GPU, and
never assembles the global sparse matrix.

Stable public API (everything else is submodule-level)::

    from src import (
        TreeShape, TreeMatrix, TreeVector,
        TreeSolver, PreparedSolve,
    )

Typical usage::

    solver = TreeSolver(matrix.shape, device="cuda:0")
    solver.stage_matrix(matrix)
    solver.factorize(check=True)
    solution = solver.solve(rhs)

``TreeSolver``/``PreparedSolve`` are imported lazily so that CPU-only
work (validation, CPU references) does not initialize Warp.  Random
problem generation is benchmark/test support and lives in
:mod:`benchmarks.problems`; comparison solvers (cuDSS, CPU references)
live in the top-level :mod:`baselines` package.
"""

from src.problem import (
    TreeMatrix,
    TreeShape,
    TreeVector,
    tree_vector_from_arrays,
)

__all__ = [
    "TreeShape",
    "TreeMatrix",
    "TreeVector",
    "tree_vector_from_arrays",
    "TreeSolver",
    "PreparedSolve",
    "SolverStats",
]

__version__ = "0.3.0"

_LAZY = {
    "TreeSolver": ("src.solver", "TreeSolver"),
    "PreparedSolve": ("src.solver", "PreparedSolve"),
    "SolverStats": ("src.solver", "SolverStats"),
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        module, attr = _LAZY[name]
        return getattr(importlib.import_module(module), attr)
    raise AttributeError(f"module 'src' has no attribute {name!r}")

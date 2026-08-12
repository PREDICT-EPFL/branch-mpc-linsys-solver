"""CPU reference solvers (correctness references, not speed baselines).

Two references with different backends:

- :func:`solve_sparse_direct` -- SciPy sparse direct solve on the
  assembled sparse ``A`` (small/medium systems);
- :func:`solve_structured_cpu` -- the structured Schur-complement algorithm
  in plain NumPy (validates the exact equations the GPU solver implements).

All take a :class:`~src.problem.GeneratedProblem` and return the
solution parts ``(x_branch, x_separator)`` as NumPy arrays.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from src import validation
from src.problem import TreeVector


def _blocks64(problem):
    m = problem.matrix
    return (m.D.astype(np.float64), m.E.astype(np.float64),
            m.C_T.astype(np.float64), m.R.astype(np.float64))


def _rhs64(problem):
    return (problem.rhs.branch.astype(np.float64),
            problem.rhs.separator.astype(np.float64))


def _split(problem, z):
    v = TreeVector.from_flat(problem.shape, z)
    return v.branch, v.separator


def solve_sparse_direct(problem):
    """Solve with SciPy's sparse direct factorization (SuperLU) on the
    assembled sparse matrix."""
    lower = problem.matrix.to_csr_lower(dtype=np.float64)
    A = (lower + sp.tril(lower, k=-1).T).tocsc()
    rt, rr = _rhs64(problem)
    r = TreeVector(problem.shape, rt, rr).flat()
    lu = spla.splu(A)
    z = lu.solve(r)
    return _split(problem, z)


def solve_structured_cpu(problem, return_schur=False):
    """Solve with the structured Schur-complement algorithm on the CPU.

    This is the NumPy twin of the GPU solver: batched branch Cholesky,
    ``X = K^{-1} C^T`` columnwise, ``S = R - sum_i C_i X_i``, root solve,
    and recovery.  With ``return_schur=True`` also returns the Schur
    complement ``S``.
    """
    D, E, C_T, R = _blocks64(problem)
    r, q = _rhs64(problem)
    return validation.structured_solve_cpu(D, E, C_T, R, r, q,
                                           return_schur=return_schur)

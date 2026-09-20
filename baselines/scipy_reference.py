"""CPU reference solver (a correctness reference, not a speed baseline).

:func:`solve_structured_cpu` is the structured block-Cholesky algorithm
in plain NumPy, validating the exact equations the GPU solver
implements.  It takes a
:class:`~experiments.general_arrow.benchmarks.problems.GeneratedProblem`
and returns the solution parts ``(x_tail, x_root)`` as NumPy arrays.
"""

import numpy as np

from baselines import reference as validation
from src.general_arrow.problem import TreeVector


def _blocks64(problem):
    m = problem.matrix
    return (m.D.astype(np.float64), m.E.astype(np.float64),
            m.C_T.astype(np.float64), m.R.astype(np.float64))


def _rhs64(problem):
    return (problem.rhs.tail.astype(np.float64),
            problem.rhs.root.astype(np.float64))


def _split(problem, z):
    v = TreeVector.from_flat(problem.shape, z)
    return v.tail, v.root


def solve_structured_cpu(problem, return_root_update=False):
    """Solve with the structured block-Cholesky root-update algorithm
    on the CPU.

    This is the NumPy twin of the GPU solver: batched tail Cholesky,
    ``X = K^{-1} C^T`` columnwise, ``S = R - sum_i C_i X_i``, root solve,
    and recovery.  With ``return_root_update=True`` also returns the
    complement ``S``.
    """
    D, E, C_T, R = _blocks64(problem)
    r, q = _rhs64(problem)
    return validation.structured_solve_cpu(D, E, C_T, R, r, q,
                                           return_root_update=return_root_update)

"""Thin adapter over the upstream SOCU block-tridiagonal solver.

All tail factorization and substitution work goes through the public
batch interface of ``socu.block_tridiag_solver`` (plus the fused
factor-and-forward launch in :mod:`src._socu_fused`).  This module does
layout conversion, buffer allocation, refresh, and launch construction
only -- no factorization or solve math.

SOCU storage contract (upstream):

- diagonal buffer ``(B, T, n_b, n_b)``, overwritten in place by the
  permuted factor's diagonal blocks;
- off-diagonal buffer ``(B, n_off, n_b, n_b)`` with
  ``n_off = calculate_off_diag_storage_len(T)``: the first ``T - 1``
  slots hold the physical sub-diagonal blocks ``E``, the rest is
  cyclic-reduction fill workspace (zeroed here) that the factorization
  overwrites with the permuted factor's off-diagonal blocks;
- solve buffers ``(B, T, n_b, nrhs)``, overwritten with the result;
  input and output stay in physical stage order;
- block sizes must be aligned to SOCU's padding multiples.
"""

import warp as wp
from socu.block_tridiag_solver import (
    calculate_off_diag_storage_len,
    create_backward_substitution_launch,
    create_forward_substitution_launch,
    optimal_problem_settings,
)

from src.dense_arrow._utils import Launch, copy_into


def is_block_size_aligned(block_dim: int, dtype) -> bool:
    """True if ``block_dim`` satisfies SOCU's padding multiples for both
    the factor and solve phases."""
    settings = optimal_problem_settings(block_dim, dtype)
    return all(block_dim % settings["pad_multiple"][phase] == 0
               for phase in ("factor", "solve"))


class TailEngine:
    """Batched tail factor/solve launches bound to persistent buffers.

    ``diag_factor``/``offdiag_factor`` are overwritten in place by every
    factorization (they hold the permuted tail factors afterwards);
    ``stage()`` keeps pristine copies only -- the one refresh of the
    mutable factor buffers happens inside the factorization pipeline
    (``refresh()``), so staging new values never duplicates the
    device-to-device copy.  All launch construction happens here, outside
    any timed region; the returned callables allocate nothing.
    """

    def __init__(self, num_tails, num_stages, tail_block_dim, dtype,
                 device):
        if not is_block_size_aligned(tail_block_dim, dtype):
            raise ValueError(
                f"tail block dimension {tail_block_dim} is not aligned "
                f"to SOCU's padding multiples for this dtype")
        B, T, n_b = int(num_tails), int(num_stages), int(tail_block_dim)
        self.num_stages = T
        self.dtype = dtype
        self.device = wp.get_device(device)
        self.n_off = calculate_off_diag_storage_len(T)
        self.diag_factor = wp.zeros((B, T, n_b, n_b), dtype=dtype,
                                    device=self.device)
        self.offdiag_factor = wp.zeros((B, self.n_off, n_b, n_b),
                                       dtype=dtype, device=self.device)
        self._diag0 = wp.zeros_like(self.diag_factor)
        self._offdiag0 = wp.zeros_like(self.offdiag_factor)

    def stage(self, D, E):
        """Copy new block values into the pristine buffers (fill slots
        zeroed).  Does not touch the mutable factor buffers; the
        factorization pipeline refreshes them exactly once per factor."""
        T = self.num_stages
        copy_into(self._diag0, D, "D")
        self._offdiag0.zero_()
        if T > 1:
            copy_into(self._offdiag0[:, :T - 1], E, "E")

    def refresh(self):
        """Restore factor buffers from pristine data (factorization
        overwrites them in place)."""
        wp.copy(self.diag_factor, self._diag0)
        wp.copy(self.offdiag_factor, self._offdiag0)

    def build_forward_launch(self, x) -> Launch:
        """Forward substitution launch bound to ``x`` (in place)."""
        return create_forward_substitution_launch(
            self.diag_factor, self.offdiag_factor, x, dtype=self.dtype,
            device=self.device)

    def build_backward_launch(self, x) -> Launch:
        """Backward substitution launch bound to ``x`` (in place)."""
        return create_backward_substitution_launch(
            self.diag_factor, self.offdiag_factor, x, dtype=self.dtype,
            device=self.device)

    def build_factor_forward_launch(self, x) -> Launch:
        """Fused factorization + forward transform launch bound to
        ``x`` (the root-coupling buffer)."""
        from src.dense_arrow._socu_fused import (
            create_cholesky_factor_and_forward_solve_launch)
        return create_cholesky_factor_and_forward_solve_launch(
            self.diag_factor, self.offdiag_factor, x, dtype=self.dtype,
            device=self.device)

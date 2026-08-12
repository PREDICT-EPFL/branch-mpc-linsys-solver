"""Thin adapter over the upstream SOCU block-tridiagonal solver.

All branch factorization and solve work goes through the public batch
interface of ``socu.block_tridiag_solver`` from
https://github.com/PREDICT-EPFL/socu (pinned commit below).  This module
performs only layout conversion, launch construction, factor refresh, and
metadata collection -- it contains no factorization or solve math.

Name mapping (``D_to_socu_L``): this project's diagonal blocks are called
``D`` everywhere (equations, generated data, results).  SOCU names its
mutable diagonal-block buffer ``L`` because the factorization overwrites
it in place with the Cholesky factor; that name appears only at this call
boundary.

SOCU storage contract (upstream):

- diagonal buffer of shape ``(B, T, n_b, n_b)``, overwritten by the
  factor;
- off-diagonal buffer of shape ``(B, n_off, n_b, n_b)`` where
  ``n_off = calculate_off_diag_storage_len(T) >= T - 1``; the first
  ``T - 1`` slots hold the physical sub-diagonal blocks ``E`` and the
  rest is nested-dissection fill-in workspace (must be zeroed);
- solve buffers of shape ``(B, T, n_b, nrhs)``, overwritten with the
  solution;
- block arrays and solve input/output stay in physical stage order;
- block sizes must be aligned to SOCU's padding multiples (checked here;
  the benchmark sweeps use aligned sizes only).
"""

import importlib.metadata
import subprocess
from pathlib import Path

import warp as wp
from socu.block_tridiag_solver import (
    calculate_off_diag_storage_len,
    calculate_recursive_iterations,
    create_backward_substitution_launch,
    create_cholesky_factor_launch,
    create_forward_substitution_launch,
    optimal_problem_settings,
)

from src.utils import copy_into
from src.runtime import Launch

#: SOCU source used for the experiments (fallback when the installed
#: package carries no git metadata): local checkout of branch
#: feature/sequential, installed from ~/Documents/gpu_code/socu.
SOCU_COMMIT_PIN = "2ff9dc0 (feature/sequential)"


def socu_version() -> str:
    """Installed SOCU package version, or ``"unknown"``."""
    try:
        return importlib.metadata.version("socu")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def socu_commit() -> str:
    """Git commit of the installed SOCU package when its installation
    directory is a git worktree (editable installs); otherwise the pinned
    source description :data:`SOCU_COMMIT_PIN`."""
    try:
        import socu
        pkg_dir = Path(socu.__file__).resolve().parent
        out = subprocess.run(
            ["git", "-C", str(pkg_dir), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:  # noqa: BLE001 - metadata only, never fatal
        pass
    return SOCU_COMMIT_PIN


def check_block_size_alignment(block_dim: int, dtype) -> dict:
    """Validate a block dimension against SOCU's padding multiples.

    SOCU's Warp interface does not pad arbitrary block sizes
    automatically; this project restricts experiments to aligned sizes
    rather than padding (a documented deviation).  Raises ``ValueError``
    for unaligned sizes.  Returns the upstream settings dict for metadata.
    """
    settings = optimal_problem_settings(block_dim, dtype)
    for phase in ("factor", "solve"):
        mult = settings["pad_multiple"][phase]
        if block_dim % mult != 0:
            raise ValueError(
                f"block dimension {block_dim} is not a multiple of SOCU's "
                f"{phase} padding multiple {mult} for this dtype; choose an "
                f"aligned size (the required sweeps use aligned sizes only)")
    return settings


def is_block_size_aligned(block_dim: int, dtype) -> bool:
    """True if ``block_dim`` satisfies SOCU's padding multiples (such
    sizes can be factored/solved by SOCU, including the dense root Schur
    complement treated as a one-stage chain)."""
    try:
        check_block_size_alignment(block_dim, dtype)
        return True
    except ValueError:
        return False


class SocuTailEngine:
    """Batched branch factor/solve engine bound to persistent device
    buffers.

    All ``build_*_launch`` methods construct a launch bound to a caller
    buffer and return a callable; calling it enqueues the GPU work.
    Construction happens once outside timed regions; the callables
    allocate nothing.

    Parameters
    ----------
    num_branches, num_stages, branch_block_dim : int
        Problem dimensions ``B``, ``T``, ``n_b``.
    dtype : warp dtype
        ``wp.float64`` or ``wp.float32``.
    device : warp device

    Every launch covers all branches at once (SOCU's batch dimension)
    and uses SOCU's parallel (cyclic-reduction) algorithm, the upstream
    default.
    """

    def __init__(self, num_branches, num_stages, branch_block_dim, dtype,
                 device):
        self.num_branches = int(num_branches)
        self.num_stages = int(num_stages)
        self.branch_block_dim = int(branch_block_dim)
        self.dtype = dtype
        self.device = wp.get_device(device)
        self.settings = check_block_size_alignment(self.branch_block_dim,
                                                   dtype)

        B, T, n_b = (self.num_branches, self.num_stages,
                     self.branch_block_dim)
        self.n_off = calculate_off_diag_storage_len(T)
        # D_to_socu_L: our diagonal blocks D are staged into the buffer
        # SOCU calls L (overwritten in place with the Cholesky factor).
        self.diag_factor = wp.zeros((B, T, n_b, n_b), dtype=dtype,
                                    device=self.device)
        self.offdiag_factor = wp.zeros((B, self.n_off, n_b, n_b),
                                       dtype=dtype, device=self.device)
        # pristine copies so repeated factorizations can refresh in place
        self._diag0 = wp.zeros_like(self.diag_factor)
        self._offdiag0 = wp.zeros_like(self.offdiag_factor)

        self._factor_launch = create_cholesky_factor_launch(
            self.diag_factor, self.offdiag_factor, dtype=dtype,
            device=self.device)

    # ---------------------------------------------------------------- data
    def stage(self, D, E):
        """Copy the block data into the engine's pristine buffers.

        ``D`` is ``(B, T, n_b, n_b)`` and ``E`` ``(B, T-1, n_b, n_b)``,
        NumPy or Warp arrays.  The fill-in workspace slots are zeroed.
        Call :meth:`refresh` (or ``factor(refresh=True)``) afterwards.
        """
        T = self.num_stages
        copy_into(self._diag0, D, "D")
        self._offdiag0.zero_()
        if T > 1:
            copy_into(self._offdiag0[:, :T - 1], E, "E")
        self.refresh()

    def refresh(self):
        """Restore the factor buffers from the pristine block data
        (device-to-device copies; required before re-factorizing because
        SOCU factors in place)."""
        wp.copy(self.diag_factor, self._diag0)
        wp.copy(self.offdiag_factor, self._offdiag0)

    # ------------------------------------------------------------- compute
    def factor(self, refresh=True):
        """Factor all branch matrices (one batched SOCU invocation)."""
        if refresh:
            self.refresh()
        self._factor_launch()

    def build_forward_launch(self, x) -> Launch:
        """Build a forward-substitution-only launch bound to ``x``
        (the transform ``x <- F x`` with ``F^T F = K^{-1}``)."""
        return create_forward_substitution_launch(
            self.diag_factor, self.offdiag_factor, x, dtype=self.dtype,
            device=self.device)

    def build_backward_launch(self, x) -> Launch:
        """Build a backward-substitution-only launch bound to ``x``
        (``x <- F^T x``)."""
        return create_backward_substitution_launch(
            self.diag_factor, self.offdiag_factor, x, dtype=self.dtype,
            device=self.device)

    def build_factor_forward_launch(self, x) -> Launch:
        """Build the fused factorization-plus-forward-transform launch
        bound to ``x``."""
        from src.socu_patch import (
            create_cholesky_factor_and_forward_solve_launch)
        return create_cholesky_factor_and_forward_solve_launch(
            self.diag_factor, self.offdiag_factor, x, dtype=self.dtype,
            device=self.device)

    # ------------------------------------------------------------ metadata
    def stats(self) -> dict:
        """Structure and configuration metadata recorded with benchmarks."""
        T = self.num_stages
        return {
            "socu_version": socu_version(),
            "socu_commit": socu_commit(),
            "socu_levels": calculate_recursive_iterations(T),
            "socu_off_diag_storage_blocks": self.n_off,
            "socu_fill_in_blocks": self.n_off - max(T - 1, 0),
            "socu_block_dim": self.settings["block_dim"],
            "socu_block_size": self.settings["block_size"],
            "socu_pad_multiple": self.settings["pad_multiple"],
            "logical_block_size": self.branch_block_dim,
            "padded_block_size": self.branch_block_dim,  # aligned sizes only
        }

    def workspace_bytes(self) -> int:
        """Device bytes held by the engine's persistent buffers."""
        return (self.diag_factor.capacity + self.offdiag_factor.capacity
                + self._diag0.capacity + self._offdiag0.capacity)

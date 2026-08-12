"""Persistent matrix-side device buffers and capability probes.

The capability probes convert Warp's silent launch failures (a kernel
whose tile shapes exceed device shared memory only warns and leaves its
output unchanged) into immediate errors; successful probes are cached
per device and kernel specialization so repeated solver constructions
skip them.  Only the kernels the constructed solver actually uses are
probed.  Solve-side buffers live on :class:`src.solver.PreparedSolve`.
"""

import numpy as np
import warp as wp

from src.utils import copy_into
from src.kernels import BLOCK_DIM, num_separator_tiles
from src.kernels.root import create_root_factor_kernel
from src.kernels.schur import (
    create_chunked_recover_kernel,
    create_chunked_schur_kernel,
    create_chunked_schur_rhs_kernel,
    num_row_chunks,
)
from src.problem import TreeShape
from src.socu_adapter import calculate_off_diag_storage_len


class MatrixBuffers:
    """Matrix-side persistent device buffers of one solver instance.

    ``G`` holds the transformed coupling ``G = F C^T`` (columnwise).
    ``S_factor4`` is 4D so SOCU can factor the root as a one-branch,
    one-stage chain; ``S_factor`` is its 2D view.
    """

    def __init__(self, shape: TreeShape, dtype, device):
        B, T, n_b, n_y = shape.dims()
        dt, dev = dtype, device
        self.C_T = wp.zeros((B, T, n_b, n_y), dtype=dt, device=dev)
        self.G = wp.zeros((B, T, n_b, n_y), dtype=dt, device=dev)
        # contiguous (B, T*n_b, n_y) views for the chunked Schur kernels
        self.C_T3 = self.C_T.reshape((B, T * n_b, n_y))
        self.G3 = self.G.reshape((B, T * n_b, n_y))
        self.R = wp.zeros((n_y, n_y), dtype=dt, device=dev)
        self.S = wp.zeros((n_y, n_y), dtype=dt, device=dev)
        self.S_factor4 = wp.zeros((1, 1, n_y, n_y), dtype=dt, device=dev)
        self.S_factor = self.S_factor4.reshape((n_y, n_y))
        self.root_E = wp.zeros((1, calculate_off_diag_storage_len(1), n_y, n_y),
                               dtype=dt, device=dev)
        self.contrib = wp.zeros((B, n_y, n_y), dtype=dt, device=dev)
        self.tail_status = wp.zeros(1, dtype=wp.int32, device=dev)
        self.root_status = wp.zeros(1, dtype=wp.int32, device=dev)
        self.min_pivot = wp.zeros(1, dtype=dt, device=dev)

    def device_bytes(self) -> int:
        return sum(buf.capacity for buf in (
            self.C_T, self.G, self.R, self.S, self.S_factor4, self.root_E,
            self.contrib, self.tail_status, self.root_status,
            self.min_pivot))


# --------------------------------------------------------------------------
# Capability probes (cached per device and kernel specialization)
# --------------------------------------------------------------------------
_PROBED_MATRIX = set()
_PROBED_RHS = set()


def _device_key(device):
    """Cache key part identifying the kernel environment: device, its
    compute capability, and the Warp version (kernels recompile across
    Warp releases, so cached probe results must not survive them)."""
    dev = wp.get_device(device)
    return (str(dev), getattr(dev, "arch", None), wp.__version__)


def _tile_limit_error(what, shape: TreeShape):
    return ValueError(
        f"Warp {what} cannot launch for n_b={shape.branch_block_dim}, "
        f"n_y={shape.separator_dim}, precision={shape.precision} "
        f"(tile shared-memory limit on this device)")


def probe_matrix_kernels(bufs: MatrixBuffers, shape: TreeShape, dtype,
                         device, root_via_socu: bool):
    """Launch each matrix-side Warp tile kernel once with known inputs and
    verify the output numerically, raising a descriptive error when a
    kernel exceeds the device's shared-memory limits.  Successful probes
    are cached; the buffers are left zeroed."""
    B, T, n_b, n_y = shape.dims()
    key = (_device_key(device), n_b, n_y, shape.precision, root_via_socu)
    if key in _PROBED_MATRIX:
        return
    mt = num_separator_tiles(n_y)
    # with U = V = 1, every entry of the branch-0 contribution is T * n_b
    bufs.C_T.fill_(1.0)
    bufs.G.fill_(1.0)
    wp.launch_tiled(create_chunked_schur_kernel(dtype), dim=[1, mt, mt],
                    inputs=[num_row_chunks(T * n_b), bufs.C_T3, bufs.G3,
                            bufs.contrib],
                    block_dim=BLOCK_DIM, device=device)
    wp.synchronize_device(device)
    expected = float(T * n_b)
    got = bufs.contrib.numpy()[0]
    if (abs(float(got[0, 0]) - expected) > 1e-3
            or abs(float(got[n_y - 1, n_y - 1]) - expected) > 1e-3):
        raise _tile_limit_error("schur tile kernel", shape)
    bufs.C_T.zero_(); bufs.G.zero_()
    bufs.contrib.zero_()
    if not root_via_socu:
        probe = (4.0 * np.eye(n_y)).astype(shape.np_dtype)
        copy_into(bufs.S_factor, probe, "probe")
        wp.launch_tiled(create_root_factor_kernel(n_y, dtype), dim=[1],
                        inputs=[bufs.S_factor], block_dim=BLOCK_DIM, device=device)
        wp.synchronize_device(device)
        if abs(float(bufs.S_factor.numpy()[0, 0]) - 2.0) > 1e-3:
            raise _tile_limit_error("root factor kernel", shape)
        bufs.S_factor.zero_()
    _PROBED_MATRIX.add(key)


def probe_rhs_kernels(bufs: MatrixBuffers, rhs3, contrib, s, out3,
                      shape: TreeShape, dtype, device):
    """Numerical probe of the ``nrhs``-specialized tile kernels of one
    prepared solve (see :func:`probe_matrix_kernels`); output buffers are
    NaN sentinels that a successful launch overwrites, and are zeroed
    afterwards."""
    B, T, n_b, n_y = shape.dims()
    nrhs = rhs3.shape[2]
    key = (_device_key(device), n_b, n_y, nrhs, shape.precision)
    if key in _PROBED_RHS:
        return
    mt = num_separator_tiles(n_y)
    nan = float("nan")
    contrib.fill_(nan)
    out3.fill_(nan)
    chunks = num_row_chunks(T * n_b)
    wp.launch_tiled(create_chunked_schur_rhs_kernel(nrhs, dtype),
                    dim=[1, mt],
                    inputs=[chunks, bufs.C_T3, rhs3, contrib],
                    block_dim=BLOCK_DIM, device=device)
    wp.launch_tiled(create_chunked_recover_kernel(nrhs, dtype),
                    dim=[1, chunks],
                    inputs=[mt, bufs.G3, rhs3, s, out3],
                    block_dim=BLOCK_DIM, device=device)
    wp.synchronize_device(device)
    ok = (np.isfinite(contrib.numpy()[0, n_y - 1, nrhs - 1])
          and np.isfinite(out3.numpy()[0, 0, nrhs - 1]))
    if not ok:
        raise _tile_limit_error(f"rhs tile kernels (nrhs={nrhs})", shape)
    contrib.zero_()
    out3.zero_()
    _PROBED_RHS.add(key)


def estimate_solver_bytes(shape: TreeShape, num_rhs=1) -> int:
    """Predicted persistent device memory of a solver for ``shape`` (used
    by the benchmark memory guard before construction)."""
    B, T, n_b, n_y = shape.dims()
    nrhs = num_rhs
    itemsize = 8 if shape.precision == "float64" else 4
    n_off = calculate_off_diag_storage_len(T)
    blocks = 2 * B * T * n_b * n_b + 2 * B * n_off * n_b * n_b  # factors + pristine
    blocks += 2 * B * T * n_b * n_y                             # C_T and G
    blocks += 4 * n_y * n_y + B * n_y * n_y                     # R, S, factor, contrib
    blocks += 2 * B * T * n_b * nrhs + n_y * nrhs + B * n_y * nrhs
    return blocks * itemsize

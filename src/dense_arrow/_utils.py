"""Internal device/dtype/copy helpers (shared by solver and adapter)."""

from typing import Callable

import numpy as np
import warp as wp

#: A prepared GPU launch: calling it enqueues work on the device stream
#: and returns immediately (no synchronization, no allocation).
Launch = Callable[[], None]

_NP_OF_WP = {wp.float64: np.float64, wp.float32: np.float32}

_initialized = False


def init_warp():
    """Initialize Warp once per process (idempotent, no global config
    changes)."""
    global _initialized
    if not _initialized:
        wp.init()
        _initialized = True


def require_cuda_device(device):
    """Resolve ``device`` to a CUDA device, raising for CPU devices."""
    init_warp()
    dev = wp.get_device(device)
    if not dev.is_cuda:
        raise ValueError(
            f"a CUDA device is required, got {dev} (CPU reference solvers "
            f"live in baselines)")
    return dev


def wp_dtype(precision: str):
    """Map a precision string to the Warp scalar dtype."""
    if precision == "float64":
        return wp.float64
    if precision == "float32":
        return wp.float32
    raise ValueError("precision must be 'float64' or 'float32'")


def copy_into(dst: wp.array, src, name="array"):
    """Copy ``src`` (NumPy host array or same-device Warp array) into the
    persistent buffer ``dst``; the shape must match exactly."""
    if isinstance(src, np.ndarray):
        if tuple(src.shape) != tuple(dst.shape):
            raise ValueError(f"{name}: expected shape {tuple(dst.shape)}, "
                             f"got {tuple(src.shape)}")
        src = wp.array(np.ascontiguousarray(src, dtype=_NP_OF_WP[dst.dtype]),
                       dtype=dst.dtype, device=dst.device)
    elif isinstance(src, wp.array):
        if src.dtype != dst.dtype:
            raise TypeError(f"{name}: expected dtype {dst.dtype}, "
                            f"got {src.dtype}")
        if tuple(src.shape) != tuple(dst.shape):
            raise ValueError(f"{name}: expected shape {tuple(dst.shape)}, "
                             f"got {tuple(src.shape)}")
    else:
        raise TypeError(f"{name}: expected a NumPy or Warp array, "
                        f"got {type(src).__name__}")
    wp.copy(dst, src)

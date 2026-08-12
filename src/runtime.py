"""Warp runtime helpers shared by the GPU modules.

Library code must not modify process-global configuration (logging policy
belongs to the application), so this module only initializes Warp
idempotently and resolves devices with library-specific error messages.
"""

from typing import Callable

import warp as wp

#: A prepared GPU launch: calling it enqueues work on the device stream
#: and returns immediately (no synchronization, no allocation).
Launch = Callable[[], None]

_initialized = False


def init_warp():
    """Initialize Warp once per process (idempotent, no global config
    changes)."""
    global _initialized
    if not _initialized:
        wp.init()
        _initialized = True


def require_cuda_device(device) -> "wp.context.Device":
    """Resolve ``device`` (name or Warp device) to a CUDA device, raising
    a descriptive error for CPU or unknown devices."""
    init_warp()
    dev = wp.get_device(device)
    if not dev.is_cuda:
        raise ValueError(
            f"a CUDA device is required, got {dev} (CPU reference solvers "
            f"live in baselines.scipy_reference)")
    return dev


def wp_dtype(precision: str):
    """Map a precision string to the Warp scalar dtype."""
    if precision == "float64":
        return wp.float64
    if precision == "float32":
        return wp.float32
    raise ValueError("precision must be 'float64' or 'float32'")

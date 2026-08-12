"""Internal stateless helpers: array validation and transfer.

Validation and transfer (shared by the solver and the SOCU adapter): all
checks run before any GPU work and raise domain-specific errors for
malformed rank, shape, dtype, or device instead of failing later with an
incidental indexing error.  Transfers are always explicit: these helpers
copy only when the caller asks for a copy.

The global scalar ordering of the unknown vector is
``z = (w_0, ..., w_{B-1}, y)``; flat/structured conversions live on
:class:`src.problem.TreeVector` (``flat`` / ``from_flat``).
"""

import numpy as np
import warp as wp

_NP_OF_WP = {wp.float64: np.float64, wp.float32: np.float32}


def as_device_array(src, dtype, device, name="array"):
    """Return ``src`` as a Warp array of ``dtype`` on ``device``.

    NumPy inputs are transferred (one host-to-device copy); Warp inputs
    are returned as-is after dtype and device checks (no copy).
    """
    if isinstance(src, np.ndarray):
        return wp.array(np.ascontiguousarray(src, dtype=_NP_OF_WP[dtype]),
                        dtype=dtype, device=device)
    if isinstance(src, wp.array):
        if src.dtype != dtype:
            raise TypeError(f"{name}: expected dtype {dtype}, got {src.dtype}")
        if src.device != wp.get_device(device):
            raise ValueError(f"{name}: expected device {device}, "
                             f"got {src.device}")
        return src
    raise TypeError(f"{name}: expected a NumPy or Warp array, "
                    f"got {type(src).__name__}")


def copy_into(dst: wp.array, src, name="array"):
    """Validate ``src`` against the destination buffer and copy it in.

    ``src`` may be a NumPy array (host-to-device transfer) or a Warp array
    on the same device (device-to-device copy); its shape must match
    ``dst`` exactly.
    """
    if isinstance(src, np.ndarray):
        np_dtype = _NP_OF_WP[dst.dtype]
        if tuple(src.shape) != tuple(dst.shape):
            raise ValueError(f"{name}: expected shape {tuple(dst.shape)}, "
                             f"got {tuple(src.shape)}")
        src = wp.array(np.ascontiguousarray(src, dtype=np_dtype),
                       dtype=dst.dtype, device=dst.device)
    else:
        src = as_device_array(src, dst.dtype, dst.device, name)
        if tuple(src.shape) != tuple(dst.shape):
            raise ValueError(f"{name}: expected shape {tuple(dst.shape)}, "
                             f"got {tuple(src.shape)}")
    wp.copy(dst, src)

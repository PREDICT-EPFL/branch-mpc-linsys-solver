"""NVIDIA cuDSS sparse direct Cholesky baseline (primary GPU baseline).

Driven through the raw cuDSS bindings shipped with nvmath-python (a
maintained Python binding of the installed cuDSS shared library), with all
device buffers held as Warp arrays -- no CuPy dependency and no Python
allocation or descriptor work inside timed regions.

The matrix is the assembled lower triangle of ``A`` in sorted CSR form,
declared symmetric positive definite with a lower view, so cuDSS runs a
Cholesky factorization.  Phases are exposed separately:

- ``create``  -- handle/config/data and matrix descriptor creation;
- ``plan``    -- reordering and symbolic factorization (analysis);
- ``factorize`` / ``solve`` -- numeric phases, re-executable without
  re-analysis (the CSR pattern is unchanged);
- ``free``    -- resource destruction.

If cuDSS or its Python binding is unavailable, ``CUDSS_AVAILABLE`` is
False and ``CUDSS_UNAVAILABLE_REASON`` records why; the benchmark then
reports an explicit unavailable status instead of silently omitting the
baseline.
"""

import os
import time

import numpy as np

CUDSS_AVAILABLE = True
CUDSS_UNAVAILABLE_REASON = ""
try:
    import warp as wp
    from nvmath.bindings import cudss
except Exception as exc:  # noqa: BLE001 - any import failure disables cuDSS
    CUDSS_AVAILABLE = False
    CUDSS_UNAVAILABLE_REASON = f"{type(exc).__name__}: {exc}"

# cudaDataType ABI values (stable CUDA library enum)
_CUDA_R_32F = 0
_CUDA_R_64F = 1
_CUDA_R_32I = 10


def cudss_version() -> str:
    """Version of the loaded cuDSS library, or 'unavailable'."""
    if not CUDSS_AVAILABLE:
        return "unavailable"
    try:
        major, minor, patch = (cudss.get_property(i) for i in range(3))
        return f"{major}.{minor}.{patch}"
    except Exception:  # noqa: BLE001
        return "unknown"


#: Reordering algorithms exposed for the ordering ablation
#: (cudssAlgType: 0 = default, 1 = BTF+COLAMD, 2 = COLAMD, 3 = AMD,
#:  4 = nested dissection, 5 = none).
ORDERINGS = {"default": 0, "amd": 3, "nested_dissection": 4}


def _find_mt_lib():
    """Locate the cuDSS multithreading layer library (threaded planning)."""
    import importlib.util
    for cuda_version in range(13, 10, -1):
        spec = importlib.util.find_spec(f"nvidia.cu{cuda_version}")
        if spec is None:
            continue
        for base in spec.submodule_search_locations or []:
            lib = os.path.join(base, "lib", "libcudss_mtlayer_gomp.so.0")
            if os.path.isfile(lib):
                return lib
    return None


def _config_set(config, param, value):
    """Set one cuDSS configuration parameter (host scalar)."""
    arr = np.zeros(1, dtype=cudss.get_config_param_dtype(param))
    arr[0] = value
    cudss.config_set(config, param, arr.ctypes.data, arr.dtype.itemsize)


class CudssCholesky:
    """cuDSS SPD Cholesky solver for one assembled system.

    Parameters
    ----------
    lower_csr : scipy.sparse.csr_matrix
        Lower triangle of ``A``, sorted indices, no duplicates.
    rhs : (dim,) or (dim, q) NumPy array
        Right-hand side(s); ``q > 1`` uses a column-major dense block.
    precision : str
        ``"float64"`` or ``"float32"``.
    ordering : str
        One of :data:`ORDERINGS` (``"default"``, ``"amd"``,
        ``"nested_dissection"``).
    ir_steps : int
        cuDSS iterative refinement steps during solve (0 = plain solve).
    device : str
        Warp CUDA device holding the buffers.

    Setup wall-clock times (seconds, synchronized) are recorded in
    ``self.setup_seconds`` with keys ``transfer``, ``create``, ``plan``.
    ``factorize``/``solve`` only enqueue GPU work on the given stream;
    time them with CUDA events.
    """

    def __init__(self, lower_csr, rhs, precision="float64",
                 ordering="default", ir_steps=0, device="cuda:0"):
        if not CUDSS_AVAILABLE:
            raise RuntimeError(f"cuDSS unavailable: {CUDSS_UNAVAILABLE_REASON}")
        if ordering not in ORDERINGS:
            raise ValueError(f"ordering must be one of {sorted(ORDERINGS)}")
        self._device = wp.get_device(device)
        self._wp_dtype = wp.float64 if precision == "float64" else wp.float32
        self._value_type = _CUDA_R_64F if precision == "float64" else _CUDA_R_32F
        self.ordering = ordering
        self.ir_steps = int(ir_steps)
        self.setup_seconds = {}

        dim = int(lower_csr.shape[0])
        rhs = np.asarray(rhs)
        q = 1 if rhs.ndim == 1 else int(rhs.shape[1])
        self._dim, self._q = dim, q
        np_dtype = np.float64 if precision == "float64" else np.float32

        t0 = time.perf_counter()
        dev = self._device
        self._indptr = wp.array(lower_csr.indptr.astype(np.int32),
                                dtype=wp.int32, device=dev)
        self._indices = wp.array(lower_csr.indices.astype(np.int32),
                                 dtype=wp.int32, device=dev)
        self.values = wp.array(lower_csr.data.astype(np_dtype),
                               dtype=self._wp_dtype, device=dev)
        # dense blocks are column-major: store the F-order flattening
        rhs_flat = np.asfortranarray(rhs.reshape(dim, q)).ravel(order="F")
        self._rhs = wp.array(rhs_flat.astype(np_dtype), dtype=self._wp_dtype,
                             device=dev)
        self._sol = wp.zeros(dim * q, dtype=self._wp_dtype, device=dev)
        wp.synchronize_device(dev)
        self.setup_seconds["transfer"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        self._handle = cudss.create()
        mt_lib = _find_mt_lib()
        if mt_lib:
            cudss.set_threading_layer(self._handle, mt_lib)
        self._config = cudss.config_create()
        _config_set(self._config, cudss.ConfigParam.REORDERING_ALG,
                    ORDERINGS[ordering])
        _config_set(self._config, cudss.ConfigParam.USE_SUPERPANELS, 1)
        _config_set(self._config, cudss.ConfigParam.IR_N_STEPS, self.ir_steps)
        self._data = cudss.data_create(self._handle)
        try:
            # nvmath >= 1.0: separate offset_type and index_type arguments
            self._a = cudss.matrix_create_csr(
                dim, dim, int(lower_csr.nnz),
                self._indptr.ptr, 0, self._indices.ptr, self.values.ptr,
                _CUDA_R_32I, _CUDA_R_32I, self._value_type,
                cudss.MatrixType.SPD, cudss.MatrixViewType.LOWER,
                cudss.IndexBase.ZERO)
        except TypeError:
            # nvmath 0.x: single index_type argument
            self._a = cudss.matrix_create_csr(
                dim, dim, int(lower_csr.nnz),
                self._indptr.ptr, 0, self._indices.ptr, self.values.ptr,
                _CUDA_R_32I, self._value_type,
                cudss.MatrixType.SPD, cudss.MatrixViewType.LOWER,
                cudss.IndexBase.ZERO)
        self._b = cudss.matrix_create_dn(dim, q, dim, self._rhs.ptr,
                                         self._value_type,
                                         cudss.Layout.COL_MAJOR)
        self._x = cudss.matrix_create_dn(dim, q, dim, self._sol.ptr,
                                         self._value_type,
                                         cudss.Layout.COL_MAJOR)
        wp.synchronize_device(dev)
        self.setup_seconds["create"] = time.perf_counter() - t0

    # ------------------------------------------------------------- phases
    def plan(self, stream_ptr=None):
        """Reordering + symbolic factorization (analysis).  Reused across
        numeric refactorizations with the same pattern."""
        t0 = time.perf_counter()
        self._execute(cudss.Phase.ANALYSIS, stream_ptr)
        wp.synchronize_device(self._device)
        self.setup_seconds["plan"] = time.perf_counter() - t0

    def factorize(self, stream_ptr=None):
        """Numeric factorization (enqueues on the stream; asynchronous)."""
        self._execute(cudss.Phase.FACTORIZATION, stream_ptr)

    def solve(self, stream_ptr=None):
        """Triangular solves (+ optional refinement); asynchronous."""
        self._execute(cudss.Phase.SOLVE, stream_ptr)

    def _execute(self, phase, stream_ptr):
        if stream_ptr is None:
            stream_ptr = self._device.stream.cuda_stream
        cudss.set_stream(self._handle, stream_ptr)
        cudss.execute(self._handle, phase, self._config, self._data,
                      self._a, self._x, self._b)

    def free(self):
        """Destroy cuDSS resources."""
        if self._handle is not None:
            for mat in (self._a, self._b, self._x):
                cudss.matrix_destroy(mat)
            cudss.data_destroy(self._handle, self._data)
            cudss.config_destroy(self._config)
            cudss.destroy(self._handle)
            self._handle = None

    # --------------------------------------------------------------- data
    @property
    def rhs(self):
        """Bound device right-hand-side buffer (column-major flat)."""
        return self._rhs

    def solution(self):
        """Copy the device solution to a NumPy array of shape (dim, q)."""
        flat = self._sol.numpy()
        return flat.reshape((self._dim, self._q), order="F")

    def memory_estimates(self):
        """cuDSS-reported (permanent, peak) device workspace bytes."""
        try:
            est = np.zeros(16, dtype=np.int64)
            written = np.zeros(1, dtype=np.uint64)
            cudss.data_get(self._handle, self._data,
                           cudss.DataParam.MEMORY_ESTIMATES,
                           est.ctypes.data, est.nbytes, written.ctypes.data)
            return int(est[0]), int(est[1])
        except Exception:  # noqa: BLE001
            return None, None

    def metadata(self) -> dict:
        return {
            "cudss_version": cudss_version(),
            "ordering": self.ordering,
            "matrix_type": "SPD",
            "matrix_view": "lower",
            "index_width": "int32",
            "precision": ("float64" if self._value_type == _CUDA_R_64F
                          else "float32"),
            "ir_steps": self.ir_steps,
            "nnz_lower": int(self.values.shape[0]),
        }

"""CPU sparse direct baselines: QDLDL, CHOLMOD, and Intel MKL PARDISO.

All consume the assembled lower triangle of the symmetric system (the
same input the cuDSS baseline receives) and expose the factorize/solve
phase split the benchmark runners time:

- :class:`QdldlDirect` wraps the ``qdldl`` package (the LDL^T solver
  OSQP builds on, AMD ordering).  Symbolic analysis happens once at
  construction; ``factorize()`` is the numeric-only refactorization an
  MPC or SCP loop repeats.
- :class:`CholmodDirect` wraps CHOLMOD via ``scikit-sparse`` (the
  supernodal sparse Cholesky used by SuiteSparse; the natural CPU
  reference for SPD systems, where QDLDL's simplicial LDL^T is not
  specialized).  Symbolic analysis happens once at construction;
  ``factorize()`` is the numeric-only refactorization.  Its factorization
  runs the supernodes sequentially and leans on a threaded BLAS inside
  each one, so it gains nothing from extra cores when the supernodes are
  small.
- :class:`PardisoDirect` calls Intel MKL PARDISO directly through
  ``ctypes``: the multicore reference, which schedules independent
  subtrees of the elimination tree across OpenMP threads.  Symbolic
  analysis happens once at construction; ``factorize()`` is the
  numeric-only refactorization.  Every array PARDISO reads or writes is
  prepared once and kept for the solver's lifetime, so the timed phases
  contain nothing but the MKL call.  See
  :attr:`PardisoDirect.DEFAULT_IPARM` for the control parameters and
  why each one is set.

All run in float64 only.
"""

import ctypes
import ctypes.util
import glob
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

try:
    import qdldl

    QDLDL_AVAILABLE = True
    QDLDL_UNAVAILABLE_REASON = ""
except ImportError as exc:  # pragma: no cover
    qdldl = None
    QDLDL_AVAILABLE = False
    QDLDL_UNAVAILABLE_REASON = str(exc)

try:
    from sksparse import cholmod as _cholmod

    CHOLMOD_AVAILABLE = True
    CHOLMOD_UNAVAILABLE_REASON = ""
except ImportError as exc:  # pragma: no cover
    _cholmod = None
    CHOLMOD_AVAILABLE = False
    CHOLMOD_UNAVAILABLE_REASON = str(exc)

def _load_mkl():
    """Return the MKL runtime library, or ``None`` if it is not installed.

    The single-dynamic-library ``mkl_rt`` ships with the ``mkl`` conda
    package; it is looked up the way the loader would, then in the
    active environment's ``lib`` directory as a fallback.
    """
    path = ctypes.util.find_library("mkl_rt")
    if path is None:
        candidates = sorted(glob.glob(
            os.path.join(sys.prefix, "lib", "libmkl_rt*.so*")))
        path = candidates[-1] if candidates else None
    return ctypes.CDLL(path) if path else None


_MKL = _load_mkl()
PARDISO_AVAILABLE = _MKL is not None
PARDISO_UNAVAILABLE_REASON = "" if PARDISO_AVAILABLE else "libmkl_rt not found"


def _full_from_lower(lower_csr):
    full = (lower_csr + sp.tril(lower_csr, k=-1).T).tocsc()
    full.sort_indices()
    return full


class QdldlDirect:
    """QDLDL LDL^T solver for one assembled SPD system.

    Parameters
    ----------
    lower_csr : scipy.sparse matrix
        Lower triangle of ``A``, sorted indices, no duplicates, no
        explicitly stored zeros.

    Construction performs the symbolic analysis plus one numeric
    factorization (wall time in ``self.setup_seconds["analyze"]``).
    ``factorize()`` repeats the numeric-only refactorization on the
    same values; ``solve(rhs)`` accepts ``(n,)`` or ``(n, q)`` arrays
    and returns the solution without touching the input.
    """

    def __init__(self, lower_csr):
        if not QDLDL_AVAILABLE:
            raise RuntimeError(
                f"qdldl unavailable: {QDLDL_UNAVAILABLE_REASON}")
        full = _full_from_lower(lower_csr.astype(np.float64))
        self._upper = sp.triu(full, format="csc")
        self._upper.sort_indices()
        self.setup_seconds = {}
        t0 = time.perf_counter()
        self._solver = qdldl.Solver(self._upper)
        self.setup_seconds["analyze"] = time.perf_counter() - t0

    def factorize(self):
        """Numeric-only refactorization with the current values."""
        self._solver.update(self._upper.data)

    def solve(self, rhs):
        """Solve for one vector or one column block."""
        rhs = np.asarray(rhs, dtype=np.float64)
        if rhs.ndim == 1:
            return self._solver.solve(rhs)
        out = np.empty_like(rhs)
        for j in range(rhs.shape[1]):
            out[:, j] = self._solver.solve(np.ascontiguousarray(rhs[:, j]))
        return out


class CholmodDirect:
    """CHOLMOD supernodal Cholesky solver for one assembled SPD system.

    Parameters
    ----------
    lower_csr : scipy.sparse matrix
        Lower triangle of ``A``, sorted indices, no duplicates, no
        explicitly stored zeros.

    Construction performs the fill-reducing ordering, symbolic
    analysis, and one numeric factorization (wall time in
    ``self.setup_seconds["analyze"]``).  ``factorize()`` repeats the
    numeric-only refactorization on the same values; ``solve(rhs)``
    accepts ``(n,)`` or ``(n, q)`` arrays and returns the solution
    without touching the input.  ``nnz_factor`` is the number of
    nonzeros in the computed Cholesky factor.
    """

    def __init__(self, lower_csr):
        if not CHOLMOD_AVAILABLE:
            raise RuntimeError(
                f"scikit-sparse unavailable: {CHOLMOD_UNAVAILABLE_REASON}")
        self._lower = sp.csc_array(lower_csr.astype(np.float64))
        self._lower.sort_indices()
        self.setup_seconds = {}
        t0 = time.perf_counter()
        self._factor = _cholmod.cho_factor(self._lower, lower=True)
        self.setup_seconds["analyze"] = time.perf_counter() - t0
        L = self._factor.L
        self.nnz_factor = int((L() if callable(L) else L).nnz)

    def factorize(self):
        """Numeric-only refactorization with the current values."""
        self._factor.factorize(self._lower)

    def solve(self, rhs):
        """Solve for one vector or one column block."""
        return self._factor.solve(np.asarray(rhs, dtype=np.float64))


class PardisoDirect:
    """Intel MKL PARDISO sparse Cholesky solver for one assembled SPD system.

    Parameters
    ----------
    lower_csr : scipy.sparse matrix
        Lower triangle of ``A`` (the same input the other baselines
        receive).  PARDISO's SPD mode reads the upper triangle, which is
        formed once at construction.
    iparm : dict, optional
        Overrides for :attr:`DEFAULT_IPARM`, keyed by the same 1-based
        index the MKL documentation uses.

    Construction performs the fill-reducing ordering, symbolic analysis,
    and one numeric factorization (wall time in
    ``self.setup_seconds["analyze"]``).  ``factorize()`` repeats the
    numeric-only refactorization on the same values; ``solve(rhs)``
    accepts ``(n,)`` or ``(n, q)`` arrays.  ``nnz_factor`` is the number
    of nonzeros in the computed Cholesky factor, ``mkl_version`` names
    the library in use, and ``iparm_used`` is the complete control
    table that was applied.  Call ``free()`` to release the MKL factor
    memory when the solver is no longer needed.

    The solution returned by ``solve()`` lives in a buffer the solver
    owns and is overwritten by the next call; copy it if it has to
    outlive that.  This mirrors the persistent device buffers of the GPU
    baselines, so a timed solve contains the triangular sweeps and
    nothing else.  For the same reason, the CSR index arrays are
    converted once here rather than on every call.

    Iterative refinement is off (``iparm(8) = 0``); this is the only
    place the configuration departs from MKL's own SPD defaults, and
    :attr:`DEFAULT_IPARM` explains why.

    The thread count comes from ``MKL_NUM_THREADS``.  Set
    ``MKL_THREADING_LAYER=GNU`` whenever CHOLMOD or SciPy is loaded in
    the same process: those link against libgomp, and mixing two OpenMP
    runtimes inside MKL can abort the process.
    """

    #: PARDISO control parameters, keyed by the 1-based index used in the
    #: MKL documentation.  MKL ignores every one of them unless
    #: ``iparm(1)`` is 1, so that flag comes first.  Every other entry
    #: restates what MKL picks for an SPD matrix on its own (read back
    #: from a run with ``iparm(1) = 0``), with ONE deliberate deviation:
    #: ``iparm(8) = 0`` turns iterative refinement OFF (MKL defaults to
    #: 2 steps).  See the marked entry below.
    DEFAULT_IPARM = {
        # apply the settings below; 0 would silently ignore all of them
        1: 1,
        # fill-reducing ordering: MKL's default, the OpenMP version of
        # METIS nested dissection.  2 is the serial variant (identical
        # fill here); 0 is minimum degree, which halves the fill on these
        # block-tridiagonal-arrow systems but yields thinner supernodes
        # and a slower triangular solve.
        2: 3,
        # ------------------------------------------------------------
        # ITERATIVE REFINEMENT OFF -- the only deviation from MKL's
        # defaults.  iparm(8) is the maximum number of refinement steps
        # and MKL defaults it to 2, which turns every solve into three
        # triangular sweeps plus residual evaluations, several times the
        # cost, while the unrefined residuals on these SPD systems
        # already sit at round-off.  Timing a refined solve against an
        # unrefined one would not be a fair comparison.  Set it back to
        # 2 (or -1 for MKL's own choice) to restore refinement.
        # ------------------------------------------------------------
        8: 0,
        # pivot perturbation exponent (eps = 1e-8), MKL's documented
        # value for symmetric matrices; the SPD Cholesky path does not
        # pivot, so this only matters if a pivot is numerically zero
        10: 8,
        # report nnz(L) back in iparm(18) once the analysis is done
        18: -1,
        # classic, rather than two-level, factorization (MKL's default)
        24: 0,
        # parallel forward/backward solve (MKL's default).  0 is the
        # PARALLEL algorithm; 1 would be the sequential one, which is
        # several times slower here at eight threads.
        25: 0,
        # zero-based (C-style) CSR indices, so SciPy's index arrays are
        # handed over as they are instead of being shifted by one
        35: 1,
    }

    _MSGLVL = 0  # no statistics printed by MKL

    def __init__(self, lower_csr, iparm=None):
        if not PARDISO_AVAILABLE:
            raise RuntimeError(
                f"MKL PARDISO unavailable: {PARDISO_UNAVAILABLE_REASON}")
        upper = sp.csr_matrix(lower_csr.astype(np.float64)).T.tocsr()
        upper.sort_indices()
        self.n = int(upper.shape[0])
        # the three CSR arrays PARDISO reads, fixed for the solver's life
        self._a = np.ascontiguousarray(upper.data, dtype=np.float64)
        self._ia = np.ascontiguousarray(upper.indptr, dtype=np.int32)
        self._ja = np.ascontiguousarray(upper.indices, dtype=np.int32)
        # PARDISO's opaque handle (64 pointers), control table and
        # (unused) user permutation
        self._pt = np.zeros(64, dtype=np.int64)
        self._iparm = np.zeros(64, dtype=np.int32)
        self._perm = np.zeros(1, dtype=np.int32)
        self.iparm_used = {**self.DEFAULT_IPARM, **(iparm or {})}
        for index, value in self.iparm_used.items():
            self._iparm[index - 1] = value
        # solver-owned right-hand-side and solution buffers, Fortran
        # order as PARDISO requires; resized only if a solve arrives
        # with a different number of columns
        self._b = np.zeros((self.n, 1), dtype=np.float64, order="F")
        self._x = np.zeros((self.n, 1), dtype=np.float64, order="F")
        self._mkl_pardiso = _MKL.pardiso
        self._mkl_pardiso.restype = None
        self.mkl_version = self._version_string()
        self.setup_seconds = {}
        t0 = time.perf_counter()
        self._call(12, self._b, self._x)  # analysis + numeric factorization
        self.setup_seconds["analyze"] = time.perf_counter() - t0
        self.nnz_factor = int(self._iparm[17])

    def factorize(self):
        """Numeric-only refactorization with the current values."""
        self._call(22, self._b, self._x)

    def solve(self, rhs):
        """Solve for one vector or one column block.

        Returns a view of the solver-owned solution buffer, shaped like
        ``rhs``; it is overwritten by the next ``solve()``.
        """
        rhs = np.asarray(rhs, dtype=np.float64)
        cols = rhs.shape[1] if rhs.ndim == 2 else 1
        b = np.asfortranarray(rhs.reshape(self.n, cols))
        if self._x.shape[1] != cols:
            self._x = np.zeros((self.n, cols), dtype=np.float64, order="F")
        self._call(33, b, self._x)
        return self._x if rhs.ndim == 2 else self._x[:, 0]

    def free(self):
        """Release the PARDISO factor memory."""
        self._call(-1, self._b, self._x)

    def _call(self, phase, b, x):
        """Run one PARDISO phase on the stored matrix."""
        c_int = ctypes.c_int32
        ref = ctypes.byref
        error = c_int(0)
        self._mkl_pardiso(
            self._pt.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            ref(c_int(1)),                     # maxfct
            ref(c_int(1)),                     # mnum
            ref(c_int(2)),                     # mtype: real SPD
            ref(c_int(phase)),
            ref(c_int(self.n)),
            self._a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._ia.ctypes.data_as(ctypes.POINTER(c_int)),
            self._ja.ctypes.data_as(ctypes.POINTER(c_int)),
            self._perm.ctypes.data_as(ctypes.POINTER(c_int)),
            ref(c_int(x.shape[1])),            # nrhs
            self._iparm.ctypes.data_as(ctypes.POINTER(c_int)),
            ref(c_int(self._MSGLVL)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ref(error))
        if error.value != 0:
            raise RuntimeError(
                f"MKL PARDISO phase {phase} failed with error "
                f"{error.value} (see the MKL PARDISO documentation)")

    @staticmethod
    def _version_string():
        buf = ctypes.create_string_buffer(256)
        _MKL.mkl_get_version_string(buf, 256)
        return buf.value.decode().strip()

"""Typed problem data: shapes, matrices, and structured vectors.

Canonical naming (used consistently in code, kernels, and tests):

- ``B``   -- number of tails (``num_tails``);
- ``T``   -- stages per tail (``num_stages``);
- ``n_b`` -- dimension of one tail-stage block (``tail_block_dim``);
- ``n_r`` -- root dimension (``root_dim``);
- ``D[i, t]`` -- diagonal blocks of the tail matrix ``K_i``;
- ``E[i, t]`` -- lower sub-diagonal blocks (block ``(t+1, t)``) of ``K_i``;
- ``C[i]``    -- the bottom-left tail-to-root coupling block of the
  arrow matrix, with stage blocks ``C[i, t]`` of shape ``(n_r, n_b)``.
  This project stores the transposed stage layout ``C_T[i, t]`` of shape
  ``(n_b, n_r)`` (the top-right blocks ``C[i]^T``), which is what the GPU
  kernels and the sparse assembly consume; build a matrix from
  bottom-left data with :meth:`TreeMatrix.from_C`.  Any buffer in the
  transposed layout is named ``C_T``, never ``C``;
- ``R``   -- the dense symmetric root (root) matrix;
- ``r``/``q`` -- tail/root right-hand sides; ``w``/``y`` -- the
  tail/root parts of the solution; ``nrhs`` -- number of
  right-hand-side columns (``q`` is reserved for the root RHS).

The public model, for tail ``i`` and root variable ``y``::

    K_i w_i + C_i^T y = r_i,
    sum_i C_i w_i + R y = q.

Generator-specific types (``ProblemSpec``, ``GeneratedProblem``) are
benchmark/test fixtures and live in :mod:`benchmarks.problems`.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

Precision = Literal["float64", "float32"]

_PRECISIONS = ("float64", "float32")


@dataclass(frozen=True, slots=True)
class TreeShape:
    """Structural dimensions of one scenario-tree system.

    This is the single source of truth for the solver's shape; matrices,
    vectors, and the solver all carry (or are checked against) the same
    ``TreeShape``, so dimensions are never repeated positionally.

    Parameters
    ----------
    num_tails : int
        Number of independent scenario tails ``B`` (>= 1).
    num_stages : int
        Number of stage blocks per tail ``T`` (>= 1).
    tail_block_dim : int
        Dimension ``n_b`` of one tail-stage block (>= 1).
    root_dim : int
        Dimension ``n_r`` of the shared root variable ``y`` (>= 1).
    precision : str
        ``"float64"`` (default) or ``"float32"``.
    """

    num_tails: int
    num_stages: int
    tail_block_dim: int
    root_dim: int
    precision: Precision = "float64"

    def __post_init__(self):
        for name in ("num_tails", "num_stages", "tail_block_dim",
                     "root_dim"):
            v = getattr(self, name)
            if not isinstance(v, (int, np.integer)) or v < 1:
                raise ValueError(f"{name} must be an int >= 1, got {v!r}")
        if self.precision not in _PRECISIONS:
            raise ValueError(f"precision must be one of {_PRECISIONS}")

    @property
    def np_dtype(self):
        """NumPy dtype matching :attr:`precision`."""
        return np.float64 if self.precision == "float64" else np.float32

    @property
    def tail_dimension(self) -> int:
        """Number of scalar tail unknowns, ``B * T * n_b``."""
        return self.num_tails * self.num_stages * self.tail_block_dim

    @property
    def total_dimension(self) -> int:
        """Total number of scalar unknowns, ``B * T * n_b + n_r``."""
        return self.tail_dimension + self.root_dim

    @property
    def nnz_lower(self) -> int:
        """Number of stored entries in the lower triangle of the assembled
        matrix (structural nonzeros, independent of coupling zeros)."""
        B, T, n_b, n_r = self.dims()
        tri = n_b * (n_b + 1) // 2
        return (B * (T * tri + (T - 1) * n_b * n_b)
                + B * T * n_b * n_r
                + n_r * (n_r + 1) // 2)

    def dims(self):
        """The tuple ``(B, T, n_b, n_r)`` for local binding in math-heavy
        code: ``B, T, n_b, n_r = shape.dims()``."""
        return (self.num_tails, self.num_stages, self.tail_block_dim,
                self.root_dim)


def _check_shape(name, array, expected):
    if tuple(array.shape) != tuple(expected):
        raise ValueError(f"{name}: expected shape {tuple(expected)}, "
                         f"got {tuple(array.shape)}")


def _require_host(name, *arrays):
    for a in arrays:
        if not isinstance(a, np.ndarray):
            raise TypeError(
                f"{name} requires NumPy host arrays; device-array matrices "
                f"must be copied to the host explicitly first")


@dataclass(frozen=True, slots=True)
class TreeMatrix:
    """The block data of one scenario-tree system matrix.

    Arrays may be NumPy host arrays or Warp device arrays; the container
    never copies or transfers them (staging into a solver is explicit).
    The coupling is stored in the transposed stage layout ``C_T``; use
    :meth:`from_C` to construct from bottom-left ``C`` blocks.

    Parameters
    ----------
    shape : TreeShape
    D : (B, T, n_b, n_b) array
        Diagonal blocks of the tail matrices (stored fully symmetric).
    E : (B, T-1, n_b, n_b) array
        Lower sub-diagonal blocks (block ``(t+1, t)`` of ``K_i``).
    C_T : (B, T, n_b, n_r) array
        Transposed coupling blocks: stage rows by root columns (the
        top-right arrow blocks ``C_i^T``).  The bottom-left block ``C_i``
        is its transpose; see the module docstring for the orientation
        contract.
    R : (n_r, n_r) array
        Dense symmetric root matrix.
    """

    shape: TreeShape
    D: object
    E: object
    C_T: object
    R: object

    def __post_init__(self):
        B, T, n_b, n_r = self.shape.dims()
        _check_shape("D", self.D, (B, T, n_b, n_b))
        _check_shape("E", self.E, (B, max(T - 1, 0), n_b, n_b))
        _check_shape("C_T", self.C_T, (B, T, n_b, n_r))
        _check_shape("R", self.R, (n_r, n_r))

    @classmethod
    def from_C(cls, shape: TreeShape, D, E, C, R) -> "TreeMatrix":
        """Construct from the mathematical bottom-left coupling blocks.

        ``C`` has stage layout ``(B, T, n_r, n_b)`` (rows in the
        root, columns in the tail stage); it is transposed on the
        host into the stored ``C_T`` layout (one copy; NumPy input only).
        """
        C = np.asarray(C)
        B, T, n_b, n_r = shape.dims()
        _check_shape("C", C, (B, T, n_r, n_b))
        C_T = np.ascontiguousarray(np.swapaxes(C, -1, -2))
        return cls(shape, D=D, E=E, C_T=C_T, R=R)

    # ------------------------------------------------------------- products
    def matvec(self, x: "TreeVector", out: "TreeVector | None" = None
               ) -> "TreeVector":
        """Structured matrix-vector product ``A @ x`` (host arrays).

        With ``out=None`` (default) a new host :class:`TreeVector` is
        allocated and returned.  With ``out`` given, the result is written
        there and ``out`` is returned; ``out`` may alias ``x`` (including
        ``out is x`` and partial overlap between component arrays) -- the
        product is computed into scratch storage first whenever any
        overlap is detected, because every output component reads original
        values of both input components.
        """
        _require_host("TreeMatrix.matvec", self.D)
        if not isinstance(x, TreeVector):
            raise TypeError("x must be a TreeVector")
        if x.shape != self.shape:
            raise ValueError(f"x shape {x.shape} does not match matrix "
                             f"shape {self.shape}")
        _require_host("TreeMatrix.matvec", x.tail, x.root)
        y_tail, y_root = structural_matvec(
            self.D, self.E, self.C_T, self.R, x.tail, x.root)
        if out is None:
            return TreeVector(self.shape, y_tail, y_root)
        if not isinstance(out, TreeVector):
            raise TypeError("out must be a TreeVector")
        if out.shape != self.shape or out.nrhs != x.nrhs:
            raise ValueError("out does not match the input shape/nrhs")
        _require_host("TreeMatrix.matvec", out.tail, out.root)
        # structural_matvec computed into fresh arrays, so aliasing
        # between x and out is safe by construction; copy the result in.
        np.copyto(out.tail, y_tail)
        np.copyto(out.root, y_root.reshape(out.root.shape))
        return out

    def to_csr_lower(self, dtype=None):
        """Assemble the lower triangle of ``A`` as a sorted SciPy CSR
        matrix with no duplicates and no explicit zeros (the format the
        cuDSS baseline consumes), in the global ordering
        ``z = (w_0, ..., w_{B-1}, y)``.

        Host-side and intended for validation and baselines, never the
        structured solve path; device-array matrices are rejected.
        ``dtype=None`` keeps the stored dtype; an explicit dtype converts
        the CSR values only.
        """
        import scipy.sparse as sp
        _require_host("TreeMatrix.to_csr_lower", self.D, self.E, self.C_T,
                      self.R)
        B, T, n_b, n_r = self.shape.dims()
        off = self.shape.tail_dimension
        dim = off + n_r
        dtype = dtype or self.D.dtype

        rows, cols, vals = [], [], []
        ii, jj = np.tril_indices(n_b)
        fi, fj = np.meshgrid(np.arange(n_b), np.arange(n_b), indexing="ij")
        fi, fj = fi.ravel(), fj.ravel()

        for b in range(B):
            base = b * T * n_b
            for t in range(T):
                r0 = base + t * n_b
                rows.append(r0 + ii); cols.append(r0 + jj)
                vals.append(self.D[b, t][ii, jj])
                if t < T - 1:
                    rows.append(r0 + n_b + fi); cols.append(r0 + fj)
                    vals.append(self.E[b, t].ravel())
        # root rows: the bottom-left C blocks are dense in the lower
        # triangle (the transpose of the stored stage-rows layout)
        ci, cj = np.meshgrid(np.arange(n_r), np.arange(B * T * n_b),
                             indexing="ij")
        rows.append(off + ci.ravel()); cols.append(cj.ravel())
        C_flat = self.C_T.reshape(B * T * n_b, n_r)
        vals.append(np.ascontiguousarray(C_flat.T).ravel())
        mi, mj = np.tril_indices(n_r)
        rows.append(off + mi); cols.append(off + mj)
        vals.append(self.R[mi, mj])

        A = sp.csr_matrix(
            (np.concatenate(vals).astype(dtype),
             (np.concatenate(rows), np.concatenate(cols))),
            shape=(dim, dim))
        A.sum_duplicates()
        A.sort_indices()
        # drop stored coupling zeros (sparse patterns) so nnz is real
        A.eliminate_zeros()
        return A


@dataclass(frozen=True, slots=True)
class TreeVector:
    """A structured (multi-)vector over the tree unknowns.

    One representation serves right-hand sides, solutions, and workspaces;
    the role of an instance is determined by where it is used.  For a
    right-hand side, ``tail`` holds ``r`` and ``root`` holds ``q``;
    for a solution they hold ``w`` and ``y``.  Arrays may be NumPy host
    arrays or Warp device arrays; the container never copies them.

    Parameters
    ----------
    shape : TreeShape
    tail : (B, T, n_b, nrhs) array
    root : (n_r, nrhs) array
    """

    shape: TreeShape
    tail: object
    root: object

    def __post_init__(self):
        B, T, n_b, n_r = self.shape.dims()
        if getattr(self.tail, "ndim", 0) != 4:
            raise ValueError("tail must be a 4D (B, T, n_b, nrhs) array; "
                             "reshape single vectors to nrhs = 1 explicitly")
        nrhs = self.tail.shape[3]
        if nrhs < 1:
            raise ValueError(f"nrhs must be >= 1, got {nrhs}")
        _check_shape("tail", self.tail, (B, T, n_b, nrhs))
        _check_shape("root", self.root, (n_r, nrhs))

    @property
    def nrhs(self) -> int:
        """Number of right-hand-side / solution columns."""
        return int(self.tail.shape[-1])

    def numpy(self) -> "TreeVector":
        """Copy to host: a new :class:`TreeVector` with NumPy arrays
        (device arrays are transferred; host arrays are copied)."""
        def to_np(a):
            return a.numpy() if hasattr(a, "numpy") else np.array(a)
        return TreeVector(self.shape, to_np(self.tail), to_np(self.root))

    def flat(self) -> np.ndarray:
        """Stack into one flat host ``(total_dimension, nrhs)`` array in
        the global ordering ``z = (w_0, ..., w_{B-1}, y)`` (host arrays
        only; no copy is avoided)."""
        _require_host("TreeVector.flat", self.tail, self.root)
        return np.concatenate(
            [self.tail.reshape(self.shape.tail_dimension, self.nrhs),
             self.root.reshape(-1, self.nrhs)], axis=0)

    @classmethod
    def from_flat(cls, shape: TreeShape, z) -> "TreeVector":
        """Split a flat host ``(total_dimension, nrhs)`` (or 1D) array in
        the global ordering back into a structured :class:`TreeVector`
        (views of ``z``; no copy)."""
        B, T, n_b, n_r = shape.dims()
        z = np.asarray(z)
        z = z.reshape(z.shape[0], -1)
        if z.shape[0] != shape.total_dimension:
            raise ValueError(f"expected {shape.total_dimension} rows, "
                             f"got {z.shape[0]}")
        nrhs = z.shape[1]
        off = shape.tail_dimension
        return cls(shape, z[:off].reshape(B, T, n_b, nrhs), z[off:])


def tree_vector_from_arrays(shape: TreeShape, tail, root) -> TreeVector:
    """Build a :class:`TreeVector` from arrays that may lack the trailing
    ``nrhs`` axis (``(B, T, n_b)`` tail and ``(n_r,)`` root inputs
    are viewed as one column; no data is copied for NumPy inputs)."""
    tail = np.asarray(tail) if not hasattr(tail, "device") else tail
    if tail.ndim not in (3, 4):
        raise ValueError(f"tail must be 3D or 4D, got {tail.ndim}D")
    if tail.ndim == 3:
        B, T, n_b = tail.shape
        tail = tail.reshape((B, T, n_b, 1))
    root = (np.asarray(root)
                 if not hasattr(root, "device") else root)
    nrhs = tail.shape[3]
    root = root.reshape((shape.root_dim, nrhs))
    return TreeVector(shape, tail, root)


# --------------------------------------------------------------------------
# Reference structural matvec (used by TreeMatrix.matvec, the generator,
# and validation; prefer the TreeMatrix.matvec method in new code)
# --------------------------------------------------------------------------
def structural_matvec(D, E, C_T, R, x_tail, x_root):
    """Compute ``A @ z`` from structured blocks (raw-array reference
    helper; the public operation is :meth:`TreeMatrix.matvec`).

    ``D (B, T, n_b, n_b)``, ``E (B, T-1, n_b, n_b)``,
    ``C_T (B, T, n_b, n_r)``, ``R (n_r, n_r)``;
    ``x_tail (B, T, n_b, nrhs)``, ``x_root (n_r, nrhs)``.
    Returns freshly allocated ``(y_tail, y_root)`` of the input
    shapes (never aliases its inputs).
    """
    B, T, n_b, nrhs = x_tail.shape
    n_r = x_root.shape[0]
    x_root = x_root.reshape(n_r, nrhs)

    y_tail = np.einsum("btij,btjq->btiq", D, x_tail)
    if T > 1:
        # E_t acts on stage t and lands on stage t+1; E_t^T acts on stage
        # t+1 and lands on stage t.
        y_tail[:, 1:] += np.einsum("btij,btjq->btiq", E, x_tail[:, :-1])
        y_tail[:, :-1] += np.einsum("btji,btjq->btiq", E, x_tail[:, 1:])
    y_tail += np.einsum("btim,mq->btiq", C_T, x_root)

    y_root = R @ x_root
    y_root += np.einsum("btim,btiq->mq", C_T, x_tail)
    return y_tail, y_root

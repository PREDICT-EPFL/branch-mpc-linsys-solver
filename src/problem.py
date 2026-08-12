"""Typed problem data: shapes, matrices, and structured vectors.

Canonical naming (used consistently in code, kernels, and tests):

- ``B``   -- number of branches (``num_branches``);
- ``T``   -- stages per branch (``num_stages``);
- ``n_b`` -- dimension of one branch-stage block (``branch_block_dim``);
- ``n_y`` -- separator dimension (``separator_dim``);
- ``D[i, t]`` -- diagonal blocks of the branch matrix ``K_i``;
- ``E[i, t]`` -- lower sub-diagonal blocks (block ``(t+1, t)``) of ``K_i``;
- ``C[i]``    -- the bottom-left branch-to-separator coupling block of the
  arrow matrix, with stage blocks ``C[i, t]`` of shape ``(n_y, n_b)``.
  This project stores the transposed stage layout ``C_T[i, t]`` of shape
  ``(n_b, n_y)`` (the top-right blocks ``C[i]^T``), which is what the GPU
  kernels and the sparse assembly consume; build a matrix from
  bottom-left data with :meth:`TreeMatrix.from_C`.  Any buffer in the
  transposed layout is named ``C_T``, never ``C``;
- ``R``   -- the dense symmetric separator (root) matrix;
- ``r``/``q`` -- branch/separator right-hand sides; ``w``/``y`` -- the
  branch/separator parts of the solution; ``nrhs`` -- number of
  right-hand-side columns (``q`` is reserved for the separator RHS).

The public model, for branch ``i`` and separator variable ``y``::

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
    num_branches : int
        Number of independent scenario branches ``B`` (>= 1).
    num_stages : int
        Number of stage blocks per branch ``T`` (>= 1).
    branch_block_dim : int
        Dimension ``n_b`` of one branch-stage block (>= 1).
    separator_dim : int
        Dimension ``n_y`` of the shared separator variable ``y`` (>= 1).
    precision : str
        ``"float64"`` (default) or ``"float32"``.
    """

    num_branches: int
    num_stages: int
    branch_block_dim: int
    separator_dim: int
    precision: Precision = "float64"

    def __post_init__(self):
        for name in ("num_branches", "num_stages", "branch_block_dim",
                     "separator_dim"):
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
        """Number of scalar branch unknowns, ``B * T * n_b``."""
        return self.num_branches * self.num_stages * self.branch_block_dim

    @property
    def total_dimension(self) -> int:
        """Total number of scalar unknowns, ``B * T * n_b + n_y``."""
        return self.tail_dimension + self.separator_dim

    @property
    def nnz_lower(self) -> int:
        """Number of stored entries in the lower triangle of the assembled
        matrix (structural nonzeros, independent of coupling zeros)."""
        B, T, n_b, n_y = self.dims()
        tri = n_b * (n_b + 1) // 2
        return (B * (T * tri + (T - 1) * n_b * n_b)
                + B * T * n_b * n_y
                + n_y * (n_y + 1) // 2)

    def dims(self):
        """The tuple ``(B, T, n_b, n_y)`` for local binding in math-heavy
        code: ``B, T, n_b, n_y = shape.dims()``."""
        return (self.num_branches, self.num_stages, self.branch_block_dim,
                self.separator_dim)


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
        Diagonal blocks of the branch matrices (stored fully symmetric).
    E : (B, T-1, n_b, n_b) array
        Lower sub-diagonal blocks (block ``(t+1, t)`` of ``K_i``).
    C_T : (B, T, n_b, n_y) array
        Transposed coupling blocks: stage rows by separator columns (the
        top-right arrow blocks ``C_i^T``).  The bottom-left block ``C_i``
        is its transpose; see the module docstring for the orientation
        contract.
    R : (n_y, n_y) array
        Dense symmetric separator matrix.
    """

    shape: TreeShape
    D: object
    E: object
    C_T: object
    R: object

    def __post_init__(self):
        B, T, n_b, n_y = self.shape.dims()
        _check_shape("D", self.D, (B, T, n_b, n_b))
        _check_shape("E", self.E, (B, max(T - 1, 0), n_b, n_b))
        _check_shape("C_T", self.C_T, (B, T, n_b, n_y))
        _check_shape("R", self.R, (n_y, n_y))

    @classmethod
    def from_C(cls, shape: TreeShape, D, E, C, R) -> "TreeMatrix":
        """Construct from the mathematical bottom-left coupling blocks.

        ``C`` has stage layout ``(B, T, n_y, n_b)`` (rows in the
        separator, columns in the branch stage); it is transposed on the
        host into the stored ``C_T`` layout (one copy; NumPy input only).
        """
        C = np.asarray(C)
        B, T, n_b, n_y = shape.dims()
        _check_shape("C", C, (B, T, n_y, n_b))
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
        _require_host("TreeMatrix.matvec", x.branch, x.separator)
        y_branch, y_separator = structural_matvec(
            self.D, self.E, self.C_T, self.R, x.branch, x.separator)
        if out is None:
            return TreeVector(self.shape, y_branch, y_separator)
        if not isinstance(out, TreeVector):
            raise TypeError("out must be a TreeVector")
        if out.shape != self.shape or out.nrhs != x.nrhs:
            raise ValueError("out does not match the input shape/nrhs")
        _require_host("TreeMatrix.matvec", out.branch, out.separator)
        # structural_matvec computed into fresh arrays, so aliasing
        # between x and out is safe by construction; copy the result in.
        np.copyto(out.branch, y_branch)
        np.copyto(out.separator, y_separator.reshape(out.separator.shape))
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
        B, T, n_b, n_y = self.shape.dims()
        off = self.shape.tail_dimension
        dim = off + n_y
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
        # separator rows: the bottom-left C blocks are dense in the lower
        # triangle (the transpose of the stored stage-rows layout)
        ci, cj = np.meshgrid(np.arange(n_y), np.arange(B * T * n_b),
                             indexing="ij")
        rows.append(off + ci.ravel()); cols.append(cj.ravel())
        C_flat = self.C_T.reshape(B * T * n_b, n_y)
        vals.append(np.ascontiguousarray(C_flat.T).ravel())
        mi, mj = np.tril_indices(n_y)
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
    right-hand side, ``branch`` holds ``r`` and ``separator`` holds ``q``;
    for a solution they hold ``w`` and ``y``.  Arrays may be NumPy host
    arrays or Warp device arrays; the container never copies them.

    Parameters
    ----------
    shape : TreeShape
    branch : (B, T, n_b, nrhs) array
    separator : (n_y, nrhs) array
    """

    shape: TreeShape
    branch: object
    separator: object

    def __post_init__(self):
        B, T, n_b, n_y = self.shape.dims()
        if getattr(self.branch, "ndim", 0) != 4:
            raise ValueError("branch must be a 4D (B, T, n_b, nrhs) array; "
                             "reshape single vectors to nrhs = 1 explicitly")
        nrhs = self.branch.shape[3]
        if nrhs < 1:
            raise ValueError(f"nrhs must be >= 1, got {nrhs}")
        _check_shape("branch", self.branch, (B, T, n_b, nrhs))
        _check_shape("separator", self.separator, (n_y, nrhs))

    @property
    def nrhs(self) -> int:
        """Number of right-hand-side / solution columns."""
        return int(self.branch.shape[-1])

    def numpy(self) -> "TreeVector":
        """Copy to host: a new :class:`TreeVector` with NumPy arrays
        (device arrays are transferred; host arrays are copied)."""
        def to_np(a):
            return a.numpy() if hasattr(a, "numpy") else np.array(a)
        return TreeVector(self.shape, to_np(self.branch), to_np(self.separator))

    def flat(self) -> np.ndarray:
        """Stack into one flat host ``(total_dimension, nrhs)`` array in
        the global ordering ``z = (w_0, ..., w_{B-1}, y)`` (host arrays
        only; no copy is avoided)."""
        _require_host("TreeVector.flat", self.branch, self.separator)
        return np.concatenate(
            [self.branch.reshape(self.shape.tail_dimension, self.nrhs),
             self.separator.reshape(-1, self.nrhs)], axis=0)

    @classmethod
    def from_flat(cls, shape: TreeShape, z) -> "TreeVector":
        """Split a flat host ``(total_dimension, nrhs)`` (or 1D) array in
        the global ordering back into a structured :class:`TreeVector`
        (views of ``z``; no copy)."""
        B, T, n_b, n_y = shape.dims()
        z = np.asarray(z)
        z = z.reshape(z.shape[0], -1)
        if z.shape[0] != shape.total_dimension:
            raise ValueError(f"expected {shape.total_dimension} rows, "
                             f"got {z.shape[0]}")
        nrhs = z.shape[1]
        off = shape.tail_dimension
        return cls(shape, z[:off].reshape(B, T, n_b, nrhs), z[off:])


def tree_vector_from_arrays(shape: TreeShape, branch, separator) -> TreeVector:
    """Build a :class:`TreeVector` from arrays that may lack the trailing
    ``nrhs`` axis (``(B, T, n_b)`` branch and ``(n_y,)`` separator inputs
    are viewed as one column; no data is copied for NumPy inputs)."""
    branch = np.asarray(branch) if not hasattr(branch, "device") else branch
    if branch.ndim not in (3, 4):
        raise ValueError(f"branch must be 3D or 4D, got {branch.ndim}D")
    if branch.ndim == 3:
        B, T, n_b = branch.shape
        branch = branch.reshape((B, T, n_b, 1))
    separator = (np.asarray(separator)
                 if not hasattr(separator, "device") else separator)
    nrhs = branch.shape[3]
    separator = separator.reshape((shape.separator_dim, nrhs))
    return TreeVector(shape, branch, separator)


# --------------------------------------------------------------------------
# Reference structural matvec (used by TreeMatrix.matvec, the generator,
# and validation; prefer the TreeMatrix.matvec method in new code)
# --------------------------------------------------------------------------
def structural_matvec(D, E, C_T, R, x_branch, x_separator):
    """Compute ``A @ z`` from structured blocks (raw-array reference
    helper; the public operation is :meth:`TreeMatrix.matvec`).

    ``D (B, T, n_b, n_b)``, ``E (B, T-1, n_b, n_b)``,
    ``C_T (B, T, n_b, n_y)``, ``R (n_y, n_y)``;
    ``x_branch (B, T, n_b, nrhs)``, ``x_separator (n_y, nrhs)``.
    Returns freshly allocated ``(y_branch, y_separator)`` of the input
    shapes (never aliases its inputs).
    """
    B, T, n_b, nrhs = x_branch.shape
    n_y = x_separator.shape[0]
    x_separator = x_separator.reshape(n_y, nrhs)

    y_branch = np.einsum("btij,btjq->btiq", D, x_branch)
    if T > 1:
        # E_t acts on stage t and lands on stage t+1; E_t^T acts on stage
        # t+1 and lands on stage t.
        y_branch[:, 1:] += np.einsum("btij,btjq->btiq", E, x_branch[:, :-1])
        y_branch[:, :-1] += np.einsum("btji,btjq->btiq", E, x_branch[:, 1:])
    y_branch += np.einsum("btim,mq->btiq", C_T, x_separator)

    y_separator = R @ x_separator
    y_separator += np.einsum("btim,btiq->mq", C_T, x_branch)
    return y_branch, y_separator

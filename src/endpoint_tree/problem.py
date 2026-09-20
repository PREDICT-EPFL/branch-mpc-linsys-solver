"""Problem containers for the endpoint-coupled scenario tree.

The structural contract: every scenario tail is a uniform block
tridiagonal chain stored in LEAF-TO-ROOT order (block ``k = 0`` is the
leaf, block ``k = T-1`` faces the root), and only that final block
couples to the shared root variables.  The coupling is therefore stored
as one ``(B, n_b, n_r)`` block per tail (``G_T``), never as the general
solver's ``(B, T, n_b, n_r)`` tensor.

Containers hold NumPy host arrays; the GPU solver copies them into its
persistent device buffers on ``update()``.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

_PRECISIONS = ("float64", "float32")


@dataclass(frozen=True, slots=True)
class EndpointTreeShape:
    """Dimensions of an endpoint-coupled tree system.

    ``num_stages`` counts the uniform algebraic blocks per scenario in
    leaf-to-root storage order; the application builder is responsible
    for mapping physical variables into uniform blocks (and documenting
    any padding it applies).
    """

    num_tails: int
    num_stages: int
    tail_block_dim: int
    root_dim: int
    precision: Literal["float64", "float32"] = "float64"

    def __post_init__(self):
        for name in ("num_tails", "num_stages", "tail_block_dim",
                     "root_dim"):
            v = getattr(self, name)
            if not isinstance(v, int) or v < 1:
                raise ValueError(f"{name} must be a positive int, got {v!r}")
        if self.precision not in _PRECISIONS:
            raise ValueError(f"precision must be one of {_PRECISIONS}")

    @property
    def np_dtype(self):
        """NumPy dtype matching :attr:`precision`."""
        return np.float64 if self.precision == "float64" else np.float32

    @property
    def tail_dimension(self) -> int:
        """Total number of tail unknowns ``B * T * n_b``."""
        return self.num_tails * self.num_stages * self.tail_block_dim

    @property
    def total_dimension(self) -> int:
        """Tail unknowns plus the root block."""
        return self.tail_dimension + self.root_dim

    @property
    def storage_order(self) -> str:
        """Fixed contract: block 0 is the leaf, block T-1 faces the
        root."""
        return "leaf_to_root"

    def dims(self):
        """``(B, T, n_b, n_r)`` convenience tuple."""
        return (self.num_tails, self.num_stages, self.tail_block_dim,
                self.root_dim)


def _check(name, arr, shape, dtype):
    a = np.asarray(arr)
    if tuple(a.shape) != tuple(shape):
        raise ValueError(f"{name}: expected shape {tuple(shape)}, got "
                         f"{tuple(a.shape)}")
    return np.ascontiguousarray(a, dtype=dtype)


@dataclass(frozen=True, slots=True)
class EndpointTreeMatrix:
    """SPD endpoint-coupled system in block storage.

    ``D`` holds the full symmetric diagonal blocks, ``E[k]`` the
    subdiagonal block coupling stage ``k`` to stage ``k+1`` (i.e. block
    row ``k+1``, block column ``k``), ``G_T`` the single root-facing
    coupling block per tail (tail rows by root columns, at stage
    ``T-1`` only), and ``R`` the full symmetric root block.
    """

    shape: EndpointTreeShape
    D: object       # (B, T, n_b, n_b)
    E: object       # (B, T-1, n_b, n_b), block (k+1, k)
    G_T: object     # (B, n_b, n_r)
    R: object       # (n_r, n_r)

    def __post_init__(self):
        B, T, n_b, n_r = self.shape.dims()
        dt = self.shape.np_dtype
        object.__setattr__(self, "D",
                           _check("D", self.D, (B, T, n_b, n_b), dt))
        object.__setattr__(self, "E",
                           _check("E", self.E,
                                  (B, max(T - 1, 0), n_b, n_b), dt))
        object.__setattr__(self, "G_T",
                           _check("G_T", self.G_T, (B, n_b, n_r), dt))
        object.__setattr__(self, "R",
                           _check("R", self.R, (n_r, n_r), dt))

    # ------------------------------------------------------------ algebra
    def matvec(self, x: "EndpointTreeVector",
               out: "EndpointTreeVector | None" = None
               ) -> "EndpointTreeVector":
        """Structured product ``K x`` on the host (validation use)."""
        if x.shape != self.shape:
            raise ValueError("vector shape does not match the matrix")
        B, T, n_b, n_r = self.shape.dims()
        xt = np.asarray(x.tail, dtype=np.float64)
        xr = np.asarray(x.root, dtype=np.float64)
        yt = np.einsum("btij,btj->bti", self.D.astype(np.float64), xt)
        if T > 1:
            E = self.E.astype(np.float64)
            yt[:, 1:] += np.einsum("btij,btj->bti", E, xt[:, :-1])
            yt[:, :-1] += np.einsum("btji,btj->bti", E, xt[:, 1:])
        G = self.G_T.astype(np.float64)
        yt[:, T - 1] += np.einsum("bij,j->bi", G, xr)
        yr = self.R.astype(np.float64) @ xr
        yr += np.einsum("bij,bi->j", G, xt[:, T - 1])
        result = EndpointTreeVector(self.shape,
                                    yt.astype(self.shape.np_dtype),
                                    yr.astype(self.shape.np_dtype))
        if out is not None:
            np.copyto(out.tail, result.tail)
            np.copyto(out.root, result.root)
            return out
        return result

    def to_csr_lower(self, dtype=None):
        """Assemble the lower triangle as a sorted SciPy CSR matrix (no
        duplicates, no explicit zeros) in the global leaf-to-root
        ordering ``[tail 1, ..., tail B, root]`` (validation and
        baseline input; never used by the specialized solver)."""
        import scipy.sparse as sp
        B, T, n_b, n_r = self.shape.dims()
        dt = dtype or self.shape.np_dtype
        n = self.shape.total_dimension
        rows, cols, vals = [], [], []

        def put(block, r0, c0, lower_of_diag=False):
            br, bc = block.shape
            for i in range(br):
                for j in range(bc):
                    if lower_of_diag and j > i:
                        continue
                    r, c = r0 + i, c0 + j
                    if r >= c:
                        rows.append(r)
                        cols.append(c)
                        vals.append(block[i, j])

        for b in range(B):
            base = b * T * n_b
            for k in range(T):
                put(self.D[b, k], base + k * n_b, base + k * n_b,
                    lower_of_diag=True)
            for k in range(T - 1):
                put(self.E[b, k], base + (k + 1) * n_b, base + k * n_b)
            put(self.G_T[b].T, B * T * n_b, base + (T - 1) * n_b)
        put(self.R, B * T * n_b, B * T * n_b, lower_of_diag=True)
        A = sp.csr_matrix(
            (np.asarray(vals, dtype=dt), (rows, cols)), shape=(n, n))
        A.sum_duplicates()
        A.eliminate_zeros()
        A.sort_indices()
        return A

    # -------------------------------------------------------- conversions
    def to_general_tree_matrix(self):
        """Adapter for validation and benchmarking against the general
        solver: embeds ``G_T`` at stage ``T-1`` of a zero-filled
        ``(B, T, n_b, n_r)`` coupling tensor.  The specialized solver
        never calls this."""
        from src.endpoint_tree._reuse import TreeMatrix, TreeShape
        B, T, n_b, n_r = self.shape.dims()
        C_T = np.zeros((B, T, n_b, n_r), dtype=self.shape.np_dtype)
        C_T[:, T - 1] = self.G_T
        shape = TreeShape(num_tails=B, num_stages=T, tail_block_dim=n_b,
                          root_dim=n_r, precision=self.shape.precision)
        return TreeMatrix(shape, D=self.D.copy(), E=self.E.copy(),
                          C_T=C_T, R=self.R.copy())

    @classmethod
    def from_general_tree_matrix(cls, matrix, *, atol=0.0):
        """Checked conversion from a general :class:`src.TreeMatrix`.

        Rejects the matrix if any coupling entry before the final tail
        block exceeds ``atol`` (silent projection is forbidden): the
        endpoint API is invalid for systems where earlier stages couple
        to the root -- those must use the general solver."""
        B, T, n_b, n_r = matrix.shape.dims()
        C_T = np.asarray(matrix.C_T)
        early = np.abs(C_T[:, :T - 1]) if T > 1 else np.zeros(1)
        worst = float(early.max()) if early.size else 0.0
        if worst > atol:
            raise ValueError(
                f"general matrix couples earlier stages to the root "
                f"(max |C_T[:, :-1]| = {worst:.3e} > atol = {atol:.3e}); "
                f"the endpoint API cannot represent it -- use the "
                f"general solver")
        shape = EndpointTreeShape(
            num_tails=B, num_stages=T, tail_block_dim=n_b, root_dim=n_r,
            precision=matrix.shape.precision)
        return cls(shape, D=np.asarray(matrix.D).copy(),
                   E=np.asarray(matrix.E).copy(),
                   G_T=C_T[:, T - 1].copy(),
                   R=np.asarray(matrix.R).copy())


@dataclass(frozen=True, slots=True)
class EndpointTreeVector:
    """One structured vector (the public workload is exactly one RHS)."""

    shape: EndpointTreeShape
    tail: object    # (B, T, n_b), leaf-to-root order
    root: object    # (n_r,)

    def __post_init__(self):
        B, T, n_b, n_r = self.shape.dims()
        import warp as wp
        if isinstance(self.tail, wp.array) or isinstance(self.root,
                                                         wp.array):
            # device vectors are validated by the solver binding
            return
        dt = self.shape.np_dtype
        object.__setattr__(self, "tail",
                           _check("tail", self.tail, (B, T, n_b), dt))
        object.__setattr__(self, "root",
                           _check("root", self.root, (n_r,), dt))

    def flat(self) -> np.ndarray:
        """Concatenate into one ``(total_dimension,)`` host vector in
        the global leaf-to-root ordering."""
        return np.concatenate(
            [np.asarray(self.tail).reshape(self.shape.tail_dimension),
             np.asarray(self.root)])

    @classmethod
    def from_flat(cls, shape: EndpointTreeShape,
                  z) -> "EndpointTreeVector":
        """Split one flat host vector back into (tail, root)."""
        z = np.asarray(z, dtype=shape.np_dtype).reshape(-1)
        if z.shape[0] != shape.total_dimension:
            raise ValueError(f"expected {shape.total_dimension} entries, "
                             f"got {z.shape[0]}")
        off = shape.tail_dimension
        B, T, n_b, _ = shape.dims()
        return cls(shape, z[:off].reshape(B, T, n_b), z[off:].copy())

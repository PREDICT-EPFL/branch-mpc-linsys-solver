"""Permutation / ordering helper for the one-level tree solver.

Phase 0 / Phase 1 finding (see ``socu/block_tridiag_solver.py``)
----------------------------------------------------------------
``socu`` implements the nested-dissection / cyclic-reduction Cholesky using a
*schedule* of strides rather than a physical reordering of the data.  Concretely
the factor/solve kernels always address a block through its **physical** index::

    L[batch_id, i]      # i is the natural block index 0 .. horizon-1
    E[batch_id, off]
    x[batch_id, i]

The block index ``i`` visited at a given elimination stride is computed as
``i = stride - 1 + tid * 2 * stride`` *inside* the kernels; the arrays
themselves are never permuted.  As a consequence:

* the right-hand side handed to ``create_cholesky_solve_launch`` is expected in
  **physical order** (block ``k`` of the input is physical node ``k``);
* the solution written back is likewise in **physical order**.

This is verified empirically by :func:`verify_identity_ordering` and by
``tests/test_block_tridiag_solver.py`` (which compares against a dense solve in
natural order without any permutation).

Therefore the "permuted block position" of the physical interface node is simply
the identity mapping and

    interface_pos = permuted_block_position(0, tail_length) == 0

for every tail length.  No dense permutation matrix and no ``perm``/``inv_perm``
index array is required.  We nevertheless keep this thin, well-documented layer
so that (a) the convention is explicit and testable, and (b) if a future
``socu`` release starts storing data in permuted order the whole tree solver only
needs this one function updated.

Convention used throughout the tree solver::

    inv_perm[old_index] = new_index      # new (storage) position of an old block

For the current ``socu`` storage layout ``inv_perm`` is the identity, hence
``interface_pos = inv_perm[0] = 0``.
"""

from functools import lru_cache


def permuted_block_position(orig_index: int, tail_length: int) -> int:
    """Return the storage position of physical tail block ``orig_index``.

    For the current ``socu`` block-tridiagonal backend the storage order equals
    the physical order (the nested-dissection permutation only reorders the
    *elimination schedule*, not the data), so this is the identity mapping.

    Parameters
    ----------
    orig_index:
        Physical (natural) block index, ``0 <= orig_index < tail_length``.
    tail_length:
        Number of blocks in the tail (``T``).

    Returns
    -------
    int
        The index at which ``orig_index`` lives in ``socu``'s block arrays.
    """
    if tail_length < 1:
        raise ValueError(f"tail_length must be >= 1, got {tail_length}")
    if not (0 <= orig_index < tail_length):
        raise ValueError(
            f"orig_index {orig_index} out of range for tail_length {tail_length}"
        )
    return orig_index


@lru_cache(maxsize=None)
def inverse_permutation(tail_length: int) -> tuple:
    """Return ``inv_perm`` with ``inv_perm[old_index] = new_index``.

    Identity for the current backend; returned as a tuple so it is hashable and
    cacheable.
    """
    if tail_length < 1:
        raise ValueError(f"tail_length must be >= 1, got {tail_length}")
    return tuple(range(tail_length))


def interface_position(tail_length: int) -> int:
    """Storage position of the physical first tail node (the root interface)."""
    return permuted_block_position(0, tail_length)


def verify_identity_ordering(tail_length: int, block_size: int = 3,
                             dtype=None, device="cuda", seed: int = 0) -> bool:
    """Empirically confirm that ``socu`` solve I/O uses physical block order.

    Builds a random SPD block-tridiagonal tail, solves ``K x = b`` with ``socu``
    and compares against a dense natural-order solve.  If they agree then the
    ``socu`` input/output ordering is the physical order and
    ``interface_pos == 0``.  Used by the permutation unit tests.
    """
    import numpy as np
    import warp as wp
    from socu.block_tridiag_solver import (
        create_cholesky_factor_launch,
        create_cholesky_solve_launch,
        calculate_off_diag_storage_len,
    )

    if dtype is None:
        dtype = wp.float64

    n, N = block_size, tail_length
    rng = np.random.default_rng(seed)

    Lc = np.zeros((N * n, N * n))
    for i in range(N):
        Lc[i * n:(i + 1) * n, i * n:(i + 1) * n] = np.tril(rng.random((n, n))) + 10 * np.eye(n)
        if i < N - 1:
            Lc[(i + 1) * n:(i + 2) * n, i * n:(i + 1) * n] = rng.random((n, n))
    A = Lc @ Lc.T

    L_np = np.zeros((N, n, n))
    E_np = np.zeros((calculate_off_diag_storage_len(N), n, n))
    for i in range(N):
        L_np[i] = A[i * n:(i + 1) * n, i * n:(i + 1) * n]
        if i < N - 1:
            E_np[i] = A[(i + 1) * n:(i + 2) * n, i * n:(i + 1) * n]

    b = rng.random(N * n)
    x_ref = np.linalg.solve(A, b)

    L = wp.from_numpy(L_np, dtype=dtype, device=device)
    E = wp.from_numpy(E_np, dtype=dtype, device=device)
    x = wp.from_numpy(b.reshape(N, n, 1), dtype=dtype, device=device)
    create_cholesky_factor_launch(L, E, device=device, dtype=dtype)()
    create_cholesky_solve_launch(L, E, x, device=device, dtype=dtype)()
    x_gpu = x.numpy().flatten()

    tol = 1e-8 if dtype == wp.float64 else 1e-3
    return bool(np.linalg.norm(x_gpu - x_ref) < tol)

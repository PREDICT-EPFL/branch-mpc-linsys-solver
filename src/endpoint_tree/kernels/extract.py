"""Single-kernel block extraction from a device-resident CSR matrix.

Given the lower triangle of an endpoint-arrow matrix in CSR form (the
same input the cuDSS baseline consumes) and the known structure
(number of tails B, stages T, uniform block width n_b, logical
last-block width n_x, root width n_r), one kernel scatters every
stored value into the solver's dense block buffers:

- D (B, T, n_b, n_b): symmetric stage diagonal blocks (both triangles
  written from the lower entries);
- E (B, T-1, n_b, n_b): sub-diagonal coupling blocks;
- G_T (B, n_b, n_r): root coupling, supported on the last stage block;
- R (n_r, n_r): the root block (both triangles written).

The buffers must be initialized once with ``init_padding`` (zeros plus
the identity dummies of the padded last block); afterwards repeated
extractions only overwrite the true nonzero slots, so no re-zeroing is
needed as long as the sparsity pattern is fixed.  The kernel is
launched over the matrix rows; each thread scatters one row's entries.
"""

from functools import lru_cache

import warp as wp


@lru_cache(maxsize=None)
def create_csr_extract_kernel(dtype=wp.float64):
    """One-launch scatter of a lower-CSR endpoint-arrow matrix into
    the dense block buffers (see module docstring)."""

    @wp.kernel
    def csr_extract_kernel(values: wp.array(dtype=dtype),      # type: ignore
                           colind: wp.array(dtype=wp.int32),   # type: ignore
                           rowoff: wp.array(dtype=wp.int32),   # type: ignore
                           tail_dim: int,
                           T: int,
                           blk: int,
                           nx: int,
                           D: wp.array4d(dtype=dtype),         # type: ignore
                           E: wp.array4d(dtype=dtype),         # type: ignore
                           G: wp.array3d(dtype=dtype),         # type: ignore
                           R: wp.array2d(dtype=dtype)):        # type: ignore
        r = wp.tid()
        tail_rows = tail_dim * D.shape[0]
        for j in range(rowoff[r], rowoff[r + 1]):
            c = int(colind[j])
            v = values[j]
            if r >= tail_rows:
                rr = r - tail_rows
                if c >= tail_rows:
                    cc = c - tail_rows
                    R[rr, cc] = v
                    R[cc, rr] = v
                else:
                    # root-tail coupling: supported on the last block
                    i = c / tail_dim
                    cl = c - i * tail_dim
                    G[i, cl - (T - 1) * blk, rr] = v
            else:
                i = r / tail_dim
                rl = r - i * tail_dim
                kb = rl / blk
                if kb > T - 1:
                    kb = T - 1
                rr = rl - kb * blk
                cl = c - i * tail_dim
                kc = cl / blk
                if kc > T - 1:
                    kc = T - 1
                cc = cl - kc * blk
                if kc == kb:
                    D[i, kb, rr, cc] = v
                    D[i, kb, cc, rr] = v
                elif kc == kb - 1:
                    E[i, kc, rr, cc] = v

    return csr_extract_kernel


def init_padding(D, num_logical_last):
    """One-time identity dummies on the padded part of the last stage
    block (call after allocating zeroed buffers)."""
    import numpy as np

    Dh = D.numpy()
    Dh[:, -1] = 0.0
    for j in range(num_logical_last, Dh.shape[2]):
        Dh[:, -1, j, j] = 1.0
    wp.copy(D, wp.array(Dh, dtype=D.dtype, device=D.device))


def extract_blocks(values, colind, rowoff, n_rows, tail_dim, T, blk,
                   nx, D, E, G, R, device=None):
    """Launch the extraction kernel once over all matrix rows."""
    kernel = create_csr_extract_kernel(values.dtype)
    wp.launch(kernel, dim=n_rows,
              inputs=[values, colind, rowoff, tail_dim, T, blk, nx,
                      D, E, G, R],
              device=device)

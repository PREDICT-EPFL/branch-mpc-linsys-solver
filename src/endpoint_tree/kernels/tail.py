"""SOCU-prefix connector machinery: the elimination-tree ancestor path
of the prefix endpoint, the path-sparse connector transform, and the
pad-aware pack/unpack kernels.

The SOCU prefix factors blocks ``0..P-1`` (``P = T-1``) with odd-even
cyclic reduction; its forward substitution runs level by level with
stride ``s = 2^level`` on physical stage slots (see
``socu.block_tridiag_solver``).  A right-hand side that is nonzero
only at the prefix endpoint ``P-1`` stays supported on the endpoint's
elimination-tree ancestor path -- ``O(log P)`` slots.
:func:`build_forward_path_program` replays SOCU's exact level rules on
the host and emits (a) the support slots and (b) the flat operation
program the path-sparse transform kernel executes per tail.  Solved
slots never receive later-level pushes (push targets are always
eliminated at strictly later levels), so the sequential per-tail replay
is exactly equivalent to SOCU's parallel forward substitution.
"""

from functools import lru_cache

import numpy as np
import warp as wp

from src.endpoint_tree.kernels import make_module

#: op codes of the path program
OP_SOLVE = 0        # H[dst] = L[l_slot]^{-1} H[dst]
OP_PUSH_NEXT = 1    # H[dst] -= E_f[od] @ H[src]
OP_PUSH_PREV = 2    # H[dst] -= E_f[od]^T @ H[src]


def build_forward_path_program(P: int):
    """Simulate SOCU's forward substitution for a source at slot
    ``P-1`` and return ``(path_slots, ops)``.

    ``path_slots`` is the sorted list of physical prefix slots the
    transform touches; ``ops`` is the ordered list of tuples
    ``(kind, src_pos, dst_pos, od_index, l_slot)`` with positions
    indexing into the compact path buffer.
    """
    if P < 1:
        return [], []
    support = {P - 1}
    raw = []
    s, off = 1, 0
    for _ in range(int(P).bit_length()):
        dim = (P + s) // (2 * s)
        for t in range(dim):
            i = s - 1 + t * 2 * s
            if i not in support:
                continue
            raw.append((OP_SOLVE, i, i, 0, i))
            if i // s < P // s - 1:
                raw.append((OP_PUSH_NEXT, i, i + s, off + i // s, 0))
                support.add(i + s)
            if i >= s:
                raw.append((OP_PUSH_PREV, i, i - s,
                            off + (i - s) // s, 0))
                support.add(i - s)
        off += P // s - 1
        s *= 2
    path = sorted(support)
    pos = {slot: k for k, slot in enumerate(path)}
    ops = [(kind, pos[a], pos[b], od, l_slot)
           for kind, a, b, od, l_slot in raw]
    return path, ops


@lru_cache(maxsize=None)
def _sub_func(dtype):
    @wp.func
    def sub2(a: dtype, b: dtype):
        return a - b

    return sub2


@lru_cache(maxsize=None)
def create_path_transform_kernel(n: int, dtype=wp.float64):
    """Execute the path program on the compact connector buffer
    ``H: (B, L_path, n, n)`` using the SOCU factor buffers
    (``Lf: (B, P, n, n)`` diagonal factor blocks, ``Ef`` the permuted
    off-diagonal/fill blocks).  One thread block per tail."""
    module = make_module(f"path_transform_{n}")
    sub2 = _sub_func(dtype)

    @wp.kernel(module=module)
    def path_transform_kernel(n_ops: int,
                              kind: wp.array(dtype=wp.int32),   # type: ignore
                              src: wp.array(dtype=wp.int32),    # type: ignore
                              dst: wp.array(dtype=wp.int32),    # type: ignore
                              od: wp.array(dtype=wp.int32),     # type: ignore
                              lslot: wp.array(dtype=wp.int32),  # type: ignore
                              Lf: wp.array4d(dtype=dtype),      # type: ignore
                              Ef: wp.array4d(dtype=dtype),      # type: ignore
                              H: wp.array4d(dtype=dtype)):      # type: ignore
        b = wp.tid()
        for j in range(n_ops):
            kd = kind[j]
            if kd == 0:
                Lt = wp.tile_load(Lf[b, lslot[j]], shape=(n, n))
                Ht = wp.tile_load(H[b, dst[j]], shape=(n, n))
                wp.tile_lower_solve_inplace(Lt, Ht)
                wp.tile_store(H[b, dst[j]], Ht)
            else:
                Et = wp.tile_load(Ef[b, od[j]], shape=(n, n))
                Hs = wp.tile_load(H[b, src[j]], shape=(n, n))
                prod = wp.tile_zeros(shape=(n, n), dtype=dtype)
                if kd == 1:
                    wp.tile_matmul(Et, Hs, prod)
                else:
                    EtT = wp.tile_transpose(Et)
                    wp.tile_matmul(EtT, Hs, prod)
                Hd = wp.tile_load(H[b, dst[j]], shape=(n, n))
                Hd = wp.tile_map(sub2, Hd, prod)
                wp.tile_store(H[b, dst[j]], Hd)

    return path_transform_kernel


@lru_cache(maxsize=None)
def create_pack_rhs_kernel(dtype=wp.float64):
    """Scatter the logical RHS tail ``(B, T, n_b)`` into the padded
    prefix solve buffer ``v: (B, P, n_p, 1)`` and the padded final-node
    RHS ``r0: (B, n_p)`` (padded rows zeroed)."""
    module = make_module("pack_rhs")

    @wp.kernel(module=module)
    def pack_rhs_kernel(n_b: int, P: int,
                        tail: wp.array3d(dtype=dtype),  # type: ignore
                        v: wp.array4d(dtype=dtype),     # type: ignore
                        r0: wp.array2d(dtype=dtype)):   # type: ignore
        b, k, j = wp.tid()   # k = 0..T-1 stages, j = 0..n_p-1
        val = dtype(0.0)
        if j < n_b:
            val = tail[b, k, j]
        if k < P:
            v[b, k, j, 0] = val
        else:
            r0[b, j] = val

    return pack_rhs_kernel


@lru_cache(maxsize=None)
def create_unpack_solution_kernel(dtype=wp.float64):
    """Gather the padded prefix solution ``v`` and final-node solution
    ``w0: (B, n_p)`` back into the logical out tail ``(B, T, n_b)``."""
    module = make_module("unpack_solution")

    @wp.kernel(module=module)
    def unpack_solution_kernel(P: int,
                               v: wp.array4d(dtype=dtype),    # type: ignore
                               w0: wp.array2d(dtype=dtype),   # type: ignore
                               tail: wp.array3d(dtype=dtype)):  # type: ignore
        b, k, j = wp.tid()   # j = 0..n_b-1 logical
        if k < P:
            tail[b, k, j] = v[b, k, j, 0]
        else:
            tail[b, k, j] = w0[b, j]

    return unpack_solution_kernel


def pad_blocks(D, E, G_T, n_p):
    """Host-side uniform padding to the SOCU-aligned block size: dummy
    diagonal slots carry identity on ``D`` and zeros everywhere else,
    so the logical solution is unchanged (tested)."""
    B, T, n_b, _ = D.shape
    n_r = G_T.shape[2]
    if n_p == n_b:
        return (np.ascontiguousarray(D), np.ascontiguousarray(E),
                np.ascontiguousarray(G_T))
    Dp = np.zeros((B, T, n_p, n_p), dtype=D.dtype)
    Dp[:, :, :n_b, :n_b] = D
    for j in range(n_b, n_p):
        Dp[:, :, j, j] = 1.0
    Ep = np.zeros((B, max(T - 1, 0), n_p, n_p), dtype=D.dtype)
    Ep[:, :, :n_b, :n_b] = E
    Gp = np.zeros((B, n_p, n_r), dtype=D.dtype)
    Gp[:, :n_b, :] = G_T
    return Dp, Ep, Gp

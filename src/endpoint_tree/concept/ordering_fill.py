"""Demonstrate the fill caused by chronological elimination order.

The same endpoint-coupled matrix is factored under two symmetric
permutations that differ only in the within-tail block order (the
shared root node is the stage x_0 and is last in both).  Stage
variables are labeled x_t^i (stage t of scenario i); each tail holds
x_N^i ... x_1^i, with x_1^i the block coupled to x_0:

- chronological ``x_1, ..., x_N``: the root-coupled block is the
  FIRST block of each tail, so eliminating it first propagates root
  coupling along the whole active front -- the factor acquires root
  fill at every later stage even though those stages had zero
  coupling;
- leaf-to-root ``x_N, ..., x_1``: the root-coupled block is
  eliminated LAST, so the factor's root coupling stays confined to the
  boundary blocks.

Numerical nonzeros are counted after eliminating explicit zeros.

Usage::

    python -m src.endpoint_tree.concept.ordering_fill

Writes experiments/endpoint_tree/results/endpoint_sequential[_T10].json and
experiments/endpoint_tree/figures/endpoint_sequential[_T10].pdf.
"""

import json
import os
from pathlib import Path

import numpy as np
import scipy.sparse as sp

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from src.endpoint_tree import EndpointTreeShape
from tests.endpoint_tree.problems import generate_endpoint_problem

# measurement outputs live outside the package, under experiments/
OUT = Path(__file__).resolve().parents[3] / "experiments/endpoint_tree"


def main():
    for T_stages in (8, 10):
        run(T_stages, "" if T_stages == 8 else f"_T{T_stages}")


def run(num_stages, out_suffix):
    shape = EndpointTreeShape(num_tails=2, num_stages=num_stages,
                              tail_block_dim=3, root_dim=3)
    B, T, n_b, n_r = shape.dims()
    p = generate_endpoint_problem(shape, seed=7)
    A = p.matrix.to_csr_lower(dtype=np.float64)
    K = (A + sp.tril(A, k=-1).T).toarray()
    n = K.shape[0]
    N_tail = B * T * n_b

    # storage block k of tail i is stage T-k (the root-coupled
    # endpoint x_1 is stored last); chronological reverses the block
    # order within every tail (the root node x_0 stays last)
    perm = np.arange(n)
    for b in range(B):
        base = b * T * n_b
        blocks = [np.arange(base + k * n_b, base + (k + 1) * n_b)
                  for k in range(T)]
        perm[base:base + T * n_b] = np.concatenate(blocks[::-1])

    def labels(stage_lists):
        out = []
        # scenarios are numbered from 1
        for i, stages in enumerate(stage_lists, start=1):
            out += [rf"$x_{{{t}}}^{{{i}}}$" for t in stages]
        # the shared root node is the stage x_0, drawn last
        return out + [r"$x_{0}$"]

    orders = {
        "temporal order": (K[np.ix_(perm, perm)],
                           labels([range(1, T + 1)] * B)),
        "reverse-temporal order": (K,
                           labels([range(T, 0, -1)] * B)),
    }

    # block id of every scalar index (stage blocks, then the root)
    block_id = np.empty(n, dtype=int)
    block_id[:N_tail] = np.arange(N_tail) // n_b
    block_id[N_tail:] = B * T
    on_diag = block_id[:, None] == block_id[None, :]

    # close-but-distinct colors: diagonal blocks dark, off-diagonal
    # blocks a lighter step of the same hue; the root block and its
    # coupling with the tails use a warm pair of the same style
    C_DIAG, C_OFF = "#1f3a6e", "#7d9cc9"
    C_ROOT, C_COUP = "#2d6b3c", "#d78f6c"
    C_FILL = "#c1272d"
    cmap = ListedColormap(["white", C_OFF, C_DIAG, C_COUP, C_ROOT,
                           C_FILL])
    in_root = np.arange(n) >= N_tail
    coupling = in_root[:, None] != in_root[None, :]
    root_both = in_root[:, None] & in_root[None, :]
    tick_pos = ([k * n_b + (n_b - 1) / 2 for k in range(B * T)]
                + [N_tail + (n_r - 1) / 2])

    plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                         "font.size": 9, "figure.dpi": 150,
                         "savefig.bbox": "tight"})
    fig, axes = plt.subplots(2, 2, figsize=(6.2, 6.8),
                             gridspec_kw={"hspace": 0.30,
                                          "wspace": 0.10})
    stats = {"shape": {"B": B, "T": T, "n_b": n_b, "n_r": n_r},
             "note": "same matrix up to a symmetric permutation; only "
                     "the elimination order changes", "orders": {}}
    for row, (name, (Kp, blk_labels)) in enumerate(orders.items()):
        L = np.linalg.cholesky(Kp)
        pat_K = np.abs(Kp) > 1e-14
        pat_L = np.abs(L) > 1e-14
        fill = pat_L & ~(pat_K | pat_K.T)
        root_rows = pat_L[N_tail:, :N_tail]
        rec = {
            "nnz_K": int(pat_K.sum()),
            "nnz_L": int(pat_L.sum()),
            "factor_nnz_in_root_columns": int(root_rows.sum()),
        }
        stats["orders"][name] = rec
        for col, (M, is_factor, tag, cnt) in enumerate((
                (pat_K, False, "matrix $K$", rec["nnz_K"]),
                (pat_L, True, "Cholesky factor $L$",
                 rec["nnz_L"]))):
            ax = axes[row, col]
            img = np.zeros((n, n), dtype=int)
            img[M & ~on_diag] = 1
            img[M & on_diag] = 2
            img[M & coupling] = 3
            img[M & root_both] = 4
            if is_factor:
                img[fill] = 5
            ax.imshow(img, cmap=cmap, vmin=0, vmax=5,
                      interpolation="nearest")
            # fine grid: every stage-block boundary
            for k in range(1, B * T):
                v = k * n_b - 0.5
                ax.axvline(v, color="0.88", lw=0.25, zorder=1)
                ax.axhline(v, color="0.88", lw=0.25, zorder=1)
            # tail boundaries
            for b in range(1, B):
                v = b * T * n_b - 0.5
                ax.axvline(v, color="0.45", lw=0.6, zorder=2)
                ax.axhline(v, color="0.45", lw=0.6, zorder=2)
            ax.axvline(N_tail - 0.5, color="0.45", lw=0.6, zorder=2)
            ax.axhline(N_tail - 0.5, color="0.45", lw=0.6, zorder=2)
            extra = (rf", fill-in $= {int(fill.sum())}$, "
                     rf"root-row nnz "
                     rf"$= {rec['factor_nnz_in_root_columns']}$"
                     if is_factor else "")
            ax.set_title(rf"{tag}: nnz $= {cnt}${extra}", fontsize=9)
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(blk_labels, fontsize=7)
            ax.set_yticks(tick_pos)
            ax.set_yticklabels(blk_labels, fontsize=7)
            ax.tick_params(length=0)
            if col == 0:
                ax.set_ylabel(rf"\textbf{{{name}}}", fontsize=10,
                              labelpad=6)
            for side in ax.spines.values():
                side.set_linewidth(0.6)
    handles = [
        Patch(facecolor=C_DIAG, label="tail diagonal blocks"),
        Patch(facecolor=C_OFF, label="tail off-diagonal blocks"),
        Patch(facecolor=C_ROOT, label="root block"),
        Patch(facecolor=C_COUP, label="root--tail coupling"),
        Patch(facecolor=C_FILL, label="fill-in"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.035))
    fig.tight_layout(rect=(0, 0.03, 1, 1), h_pad=1.6)
    (OUT / "figures").mkdir(exist_ok=True)
    (OUT / "results").mkdir(exist_ok=True)
    out_pdf = OUT / f"figures/endpoint_sequential{out_suffix}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    plt.close(fig)
    out_json = OUT / f"results/endpoint_sequential{out_suffix}.json"
    out_json.write_text(json.dumps(stats, indent=1))
    print("saved", out_pdf)
    print("wrote", out_json)
    for name, rec in stats["orders"].items():
        print(f"{name}: nnz(L) = {rec['nnz_L']}, root-column factor "
              f"nnz = {rec['factor_nnz_in_root_columns']}")


if __name__ == "__main__":
    main()

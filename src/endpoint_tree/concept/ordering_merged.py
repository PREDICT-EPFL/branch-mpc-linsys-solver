"""Merged ordering figure for the paper: sequential and parallel.

One 3x2 figure on the same endpoint-coupled matrix (B = 2 scenarios):

The shared root node is the stage x_0 itself, drawn last; every
tail therefore holds only its own stages x_N^i down to x_1^i, and
x_1^i is the single tail block coupled to x_0.

- row 1, temporal order: stages chronological, root-coupled x_1
  first -- the factor floods the root row with fill;
- row 2, reverse-temporal order: x_1 last in every tail -- zero fill;
- row 3, cyclic-reduction prefixes: the GPU elimination order (each
  tail's prefix in SOCU's cyclic-reduction order, the endpoint stage
  kept last) -- bounded fill on level neighbors and the O(log T)
  connector path, root rows still minimal.

The row name is a centered heading above each panel pair (no vertical
labels).  Compact layout intended for direct inclusion in the paper.

Usage::

    python -m src.endpoint_tree.concept.ordering_merged

Writes experiments/endpoint_tree/figures/endpoint_orderings.pdf (N = 9)
and endpoint_orderings_T8.pdf.
"""

import os
from pathlib import Path

import numpy as np
import scipy.sparse as sp

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

from src.endpoint_tree import EndpointTreeShape
from src.endpoint_tree.concept.ordering_idea import (
    C_COUP, C_DIAG, C_FILL, C_OFF, C_ROOT, cyclic_reduction_order)

# transparent zeros so background shading can show through
CMAP = ListedColormap(["none", C_OFF, C_DIAG, C_COUP, C_ROOT, C_FILL])
C_CR_BG = "0.92"
from tests.endpoint_tree.problems import generate_endpoint_problem

# measurement outputs live outside the package, under experiments/
OUT = Path(__file__).resolve().parents[3] / "experiments/endpoint_tree"


def build(num_stages):
    shape = EndpointTreeShape(num_tails=2, num_stages=num_stages,
                              tail_block_dim=3, root_dim=3)
    p = generate_endpoint_problem(shape, seed=7)
    A = p.matrix.to_csr_lower(dtype=np.float64)
    K = (A + sp.tril(A, k=-1).T).toarray()
    return shape, K


def render(shape, K0, out_name, grid="column", one_row=0):
    B, T, n_b, n_r = shape.dims()
    P = T - 1
    n = K0.shape[0]
    N_tail = B * T * n_b

    def perm_from_blocks(block_order):
        # storage block k of tail i is stage T-k; the shared
        # root node x_0 stays last
        perm = []
        for b in range(B):
            base = b * T * n_b
            for k in block_order:
                perm.extend(range(base + k * n_b,
                                  base + (k + 1) * n_b))
        perm.extend(range(B * T * n_b, n))
        return np.asarray(perm)

    def labels_from_blocks(block_order):
        # the shared root is the stage x_0, so a tail's own
        # stages run from x_T down to x_1
        stages = [T - k for k in block_order]
        out = []
        for i in range(B):
            out += [rf"$x_{{{t}}}^{{{i}}}$" for t in stages]
        return out + [r"$x_{0}$"]

    cr_order, cr_levels = cyclic_reduction_order(P)
    cr_full, cr_full_levels = cyclic_reduction_order(T)
    # shade mode: None, "prefix" (CR without the endpoint stage), or
    # "full" (CR over the whole tail, x_1 included)
    rows = (
        ("forward ordering, sequential in horizon",
         list(range(T - 1, -1, -1)), None),
        ("forward ordering, parallel in horizon",
         [T - 1 - t for t in cr_full], "full"),
        ("reverse ordering, sequential in horizon",
         list(range(T)), None),
        ("reverse ordering, parallel in horizon ($x_1$ not separated)",
         list(cr_full), "full"),
        ("reverse ordering, parallel in horizon ($x_1$ separated)",
         list(cr_order) + [P], "prefix"),
    )

    block_id = np.empty(n, dtype=int)
    block_id[:N_tail] = np.arange(N_tail) // n_b
    block_id[N_tail:] = B * T
    on_diag = block_id[:, None] == block_id[None, :]
    in_root = np.arange(n) >= N_tail
    coupling = in_root[:, None] != in_root[None, :]
    root_both = in_root[:, None] & in_root[None, :]
    tick_pos = ([k * n_b + (n_b - 1) / 2 for k in range(B * T)]
                + [N_tail + (n_r - 1) / 2])

    plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                         "font.size": 9, "figure.dpi": 150,
                         "savefig.bbox": "tight",
                         "hatch.linewidth": 0.35})
    if grid == "one":
        # a single standalone (matrix, factor) unit
        rows = (rows[one_row],)
    if grid == "2x2":
        # the four (matrix, factor) units the paper shows, in the paper's
        # own reading order: forward sequential, reverse sequential,
        # reverse parallel without and with the x_1 separation
        rows = (rows[0], rows[2], rows[3], rows[4])
        # a phantom spacer column separates the left and right units
        fig, axes = plt.subplots(
            2, 5, figsize=(13.0, 7.4),
            gridspec_kw={"hspace": 0.40, "wspace": 0.06,
                         "width_ratios": [1, 1, 0.22, 1, 1]})
        for a in axes[:, 2]:
            a.axis("off")
    else:
        fig, axes = plt.subplots(len(rows), 2,
                                 figsize=(6.2, 4.06 * len(rows)),
                                 gridspec_kw={"hspace": 0.46,
                                              "wspace": 0.04})
        axes = np.atleast_2d(axes)
    for row, (row_name, block_order, shade) in enumerate(rows):
        perm = perm_from_blocks(block_order)
        blk_labels = labels_from_blocks(block_order)
        K = K0[np.ix_(perm, perm)]
        L = np.linalg.cholesky(K)
        pat_K = np.abs(K) > 1e-14
        pat_L = np.abs(L) > 1e-14
        fill = pat_L & ~(pat_K | pat_K.T)
        panels = (
            (pat_K, False,
             rf"matrix: nnz $= {int(pat_K.sum())}$"),
            (pat_L, True,
             rf"factor: nnz $= {int(pat_L.sum())}$, "
             rf"fill-in $= {int(fill.sum())}$"),
        )
        for col, (M, is_factor, title) in enumerate(panels):
            if grid == "2x2":
                ax = axes[row // 2, (row % 2) * 3 + col]
            else:
                ax = axes[row, col]
            img = np.zeros((n, n), dtype=int)
            img[M & ~on_diag] = 1
            img[M & on_diag] = 2
            img[M & coupling] = 3
            img[M & root_both] = 4
            if is_factor:
                img[fill] = 5
            if shade is not None:
                # nested cumulative boxes over the cyclic-reduction
                # region (the prefix only, or the whole tail
                # including x_1^i): box k contains the blocks of
                # levels 0..k, so the ring between box k and box k-1
                # -- including the off-diagonal coupling blocks -- is
                # level k's territory.  The innermost box (level 0)
                # is the darkest; each enclosing box is lighter.
                lv = cr_levels if shade == "prefix" \
                    else cr_full_levels
                nl = len(lv)
                cum = np.cumsum(lv)
                # pale amber ladder (innermost level deepest), with
                # deep blue dashed level boundaries
                deep = np.array([0.952, 0.874, 0.686])
                light = np.array([0.990, 0.966, 0.902])
                for b in range(B):
                    lo = b * T * n_b - 0.5
                    for k in range(nl - 1, -1, -1):
                        f = k / max(nl - 1, 1)
                        fc = tuple(deep + (light - deep) * f)
                        w = cum[k] * n_b
                        ax.add_patch(Rectangle(
                            (lo, lo), w, w,
                            facecolor=fc, edgecolor="none",
                            zorder=-1))
                        # dashed boundary drawn ON TOP of the grid
                        ax.add_patch(Rectangle(
                            (lo, lo), w, w,
                            facecolor="none",
                            edgecolor="#1f3a6e", linewidth=0.45,
                            linestyle=(0, (3, 2)),
                            zorder=2.6))
                    # a light hatch over the whole region marks it as
                    # cyclic-reduction territory even at the palest
                    # ladder step
                    ax.add_patch(Rectangle(
                        (lo, lo), cum[-1] * n_b, cum[-1] * n_b,
                        facecolor="none", hatch="///",
                        edgecolor="#dcbf8e", linewidth=0.0,
                        zorder=-0.9))
            ax.imshow(img, cmap=CMAP, vmin=0, vmax=5,
                      interpolation="nearest")
            for k in range(1, B * T):
                v = k * n_b - 0.5
                ax.axvline(v, color="0.88", lw=0.25, zorder=1)
                ax.axhline(v, color="0.88", lw=0.25, zorder=1)
            if shade == "prefix":
                for b in range(B):
                    v = b * T * n_b + P * n_b - 0.5
                    ax.axvline(v, color="0.65", lw=0.45, zorder=2)
                    ax.axhline(v, color="0.65", lw=0.45, zorder=2)
            for b in range(1, B):
                v = b * T * n_b - 0.5
                ax.axvline(v, color="0.45", lw=0.6, zorder=2)
                ax.axhline(v, color="0.45", lw=0.6, zorder=2)
            ax.axvline(N_tail - 0.5, color="0.45", lw=0.6, zorder=2)
            ax.axhline(N_tail - 0.5, color="0.45", lw=0.6, zorder=2)
            ax.set_title(title, fontsize=9.5)
            if B * T <= 24:  # omit block labels on large horizons
                ax.set_xticks(tick_pos)
                ax.set_xticklabels(blk_labels, fontsize=5.5)
                ax.set_yticks(tick_pos)
                if col == 0:
                    ax.set_yticklabels(blk_labels, fontsize=5.5,
                                       rotation=90, va="center")
                else:
                    ax.set_yticklabels([])
            else:
                ax.set_xticks([])
                ax.set_yticks([])
            ax.tick_params(length=0)
            for side in ax.spines.values():
                side.set_linewidth(0.6)
        print(f"{row_name}: nnz(L) = {int(pat_L.sum())}, fill-in = "
              f"{int(fill.sum())}, root-row nnz = "
              f"{int(pat_L[N_tail:, :N_tail].sum())}")
    handles = [
        Patch(facecolor=C_DIAG, label="tail diagonal blocks"),
        Patch(facecolor=C_OFF, label="tail off-diagonal blocks"),
        Patch(facecolor=C_ROOT, label="root block"),
        Patch(facecolor=C_COUP, label="root--tail coupling"),
        Patch(facecolor=C_FILL, label="fill-in"),
        Patch(facecolor=(0.968, 0.912, 0.775), hatch="///",
              edgecolor="#1f3a6e", linewidth=0.45,
              linestyle=(0, (3, 2)),
              label="cyclic-reduction levels"),
    ]
    if grid == "2x2":
        fig.legend(handles=handles, loc="upper center", ncol=6,
                   frameon=False, fontsize=9,
                   bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(rect=(0, 0.03, 1, 0.93), h_pad=2.6)
        for row, (row_name, _, _) in enumerate(rows):
            p0 = axes[row // 2, (row % 2) * 3].get_position()
            p1 = axes[row // 2, (row % 2) * 3 + 1].get_position()
            xc = (p0.x0 + p1.x1) / 2
            fig.text(xc, p0.y1 + 0.034,
                     rf"\textbf{{{row_name}}}",
                     ha="center", va="bottom", fontsize=11)
            letter = "abcd"[row]
            fig.text(xc, p0.y0 - 0.040, rf"({letter})",
                     ha="center", va="top", fontsize=11)
    else:
        if grid != "one":  # standalone units carry no legend
            fig.legend(handles=handles, loc="upper center", ncol=3,
                       frameon=False, fontsize=8.5,
                       bbox_to_anchor=(0.5, 0.999))
            fig.tight_layout(rect=(0, 0, 1, 0.968), h_pad=2.4)
        else:
            fig.tight_layout(rect=(0, 0, 1, 0.88))
        # centered row headings above each panel pair
        head_off = 0.075 if grid == "one" else 0.021
        for row, (row_name, _, _) in enumerate(rows):
            p0 = axes[row, 0].get_position()
            p1 = axes[row, 1].get_position()
            fig.text((p0.x0 + p1.x1) / 2, p0.y1 + head_off,
                     rf"\textbf{{{row_name}}}", ha="center",
                     va="bottom", fontsize=11)
    (OUT / "figures").mkdir(exist_ok=True)
    out_pdf = OUT / f"figures/{out_name}"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("saved", out_pdf)


SINGLE_SLUGS = ("forward_sequential", "forward_parallel",
                "reverse_sequential", "reverse_parallel_nosep",
                "reverse_parallel_sep")


def main():
    # num_stages is the horizon N: each tail holds x_N^i ... x_1^i and
    # the shared root node x_0 is the single extra block at the end
    shape9, K9 = build(9)
    render(shape9, K9, "endpoint_orderings_2x2.pdf", grid="2x2")
    for i, slug in enumerate(SINGLE_SLUGS):
        render(shape9, K9, f"ordering_{slug}.pdf",
               grid="one", one_row=i)
    for num_stages, name in ((9, "endpoint_orderings.pdf"),
                             (8, "endpoint_orderings_T8.pdf"),
                             (6, "endpoint_orderings_N6.pdf"),
                             (20, "endpoint_orderings_N20.pdf")):
        shape, K = build(num_stages)
        render(shape, K, name)


if __name__ == "__main__":
    main()

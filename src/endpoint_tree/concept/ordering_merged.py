"""The paper's ordering figure: sparsity of the root-coupled matrix and
its Cholesky factor under four variable orderings.

One 2x2 figure on the same endpoint-coupled matrix (M = 2 scenarios,
N = 9 stages, 3x3 blocks).  The shared root node is the stage x_0
itself, drawn last; every tail therefore holds only its own stages
x_N^i down to x_1^i, and x_1^i is the single tail block coupled to x_0.
Each unit shows the permuted matrix next to its factor:

- (a) forward ordering, sequential in horizon: root-coupled x_1^i
  first -- the factor floods the root row with fill;
- (b) reverse ordering, sequential in horizon: x_1^i last in every
  tail -- zero fill;
- (c) reverse ordering, parallel in horizon, x_1^i inside the
  cyclic-reduction permutation -- fill on the level neighbors and
  root coupling leaking into the tails;
- (d) reverse ordering, parallel in horizon, x_1^i kept last -- the
  GPU elimination order: bounded fill, root rows still minimal.

Usage::

    python -m src.endpoint_tree.concept.ordering_merged      # from the repo root
    python src/endpoint_tree/concept/ordering_merged.py     # or as a file,
    python ordering_merged.py                               # from any directory

Writes endpoint_orderings_2x2.pdf next to this file.
"""

import os
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

# the repository root, so `from src...` and `from tests...` resolve
# when this file is run directly from any working directory
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.endpoint_tree import EndpointTreeShape  # noqa: E402
from tests.endpoint_tree.problems import generate_endpoint_problem  # noqa: E402

# the drawing is written next to this script
OUT = Path(__file__).resolve().parent / "endpoint_orderings_2x2.pdf"

# block colours: diagonal blocks dark, off-diagonal blocks a lighter step
# of the same hue; the root block and its coupling use a warm pair
C_DIAG, C_OFF = "#1f3a6e", "#7d9cc9"
C_ROOT, C_COUP = "#2d6b3c", "#d78f6c"
C_FILL = "#c1272d"
# transparent zeros so the level shading can show through
CMAP = ListedColormap(["none", C_OFF, C_DIAG, C_COUP, C_ROOT, C_FILL])


def cyclic_reduction_order(P):
    """SOCU's elimination order of a P-block chain: level l with
    stride s = 2^l eliminates blocks s-1 + t*2s.  Returns the block
    order and the number of blocks per level."""
    order, levels, seen = [], [], set()
    s = 1
    while len(order) < P:
        level = [i for i in range(s - 1, P, 2 * s) if i not in seen]
        order.extend(level)
        seen.update(level)
        levels.append(len(level))
        s *= 2
    return order, levels


def build(num_stages):
    """One SPD endpoint-coupled test matrix, assembled densely."""
    shape = EndpointTreeShape(num_tails=2, num_stages=num_stages,
                              tail_block_dim=3, root_dim=3)
    p = generate_endpoint_problem(shape, seed=7)
    A = p.matrix.to_csr_lower(dtype=np.float64)
    K = (A + sp.tril(A, k=-1).T).toarray()
    return shape, K


def render(shape, K0, out_pdf):
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

    cr_order, cr_levels = cyclic_reduction_order(P)
    cr_full, cr_full_levels = cyclic_reduction_order(T)
    # (name, storage block order, shading of the cyclic-reduction
    # region: None, "prefix" without x_1^i, or "full" with it)
    units = (
        ("forward ordering, sequential in horizon",
         list(range(T - 1, -1, -1)), None),
        ("reverse ordering, sequential in horizon",
         list(range(T)), None),
        ("reverse ordering, parallel in horizon ($x_1^i$ not separated)",
         list(cr_full), "full"),
        ("reverse ordering, parallel in horizon ($x_1^i$ separated)",
         list(cr_order) + [P], "prefix"),
    )

    block_id = np.empty(n, dtype=int)
    block_id[:N_tail] = np.arange(N_tail) // n_b
    block_id[N_tail:] = B * T
    on_diag = block_id[:, None] == block_id[None, :]
    in_root = np.arange(n) >= N_tail
    coupling = in_root[:, None] != in_root[None, :]
    root_both = in_root[:, None] & in_root[None, :]

    plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                         "font.size": 9, "figure.dpi": 150,
                         "savefig.bbox": "tight",
                         "hatch.linewidth": 0.35})
    # two units per row, a phantom spacer column between them; the
    # panels are square, so the width is what four panels plus the
    # spacer need at this height (no tick labels need room)
    fig, axes = plt.subplots(
        2, 5, figsize=(11.2, 7.4),
        gridspec_kw={"hspace": 0.21, "wspace": 0.05,
                     "width_ratios": [1, 1, 0.14, 1, 1]})
    for a in axes[:, 2]:
        a.axis("off")

    for u, (name, block_order, shade) in enumerate(units):
        perm = perm_from_blocks(block_order)
        K = K0[np.ix_(perm, perm)]
        L = np.linalg.cholesky(K)
        pat_K = np.abs(K) > 1e-14
        pat_L = np.abs(L) > 1e-14
        fill = pat_L & ~(pat_K | pat_K.T)
        panels = (
            (pat_K, False, rf"matrix: nnz $= {int(pat_K.sum())}$"),
            (pat_L, True, rf"factor: nnz $= {int(pat_L.sum())}$, "
                          rf"fill-in $= {int(fill.sum())}$"),
        )
        for col, (M, is_factor, title) in enumerate(panels):
            ax = axes[u // 2, (u % 2) * 3 + col]
            img = np.zeros((n, n), dtype=int)
            img[M & ~on_diag] = 1
            img[M & on_diag] = 2
            img[M & coupling] = 3
            img[M & root_both] = 4
            if is_factor:
                img[fill] = 5
            if shade is not None:
                # nested cumulative boxes over the cyclic-reduction
                # region: box k contains the blocks of levels 0..k, so
                # the ring between box k and box k-1 is level k's
                # territory.  The innermost box is the darkest step of
                # a pale amber ladder; deep blue dashed level boundaries
                # are drawn on top of the grid.
                lv = cr_levels if shade == "prefix" else cr_full_levels
                nl = len(lv)
                cum = np.cumsum(lv)
                deep = np.array([0.952, 0.874, 0.686])
                light = np.array([0.990, 0.966, 0.902])
                for b in range(B):
                    lo = b * T * n_b - 0.5
                    for k in range(nl - 1, -1, -1):
                        f = k / max(nl - 1, 1)
                        w = cum[k] * n_b
                        ax.add_patch(Rectangle(
                            (lo, lo), w, w,
                            facecolor=tuple(deep + (light - deep) * f),
                            edgecolor="none", zorder=-1))
                        ax.add_patch(Rectangle(
                            (lo, lo), w, w, facecolor="none",
                            edgecolor="#1f3a6e", linewidth=0.45,
                            linestyle=(0, (3, 2)), zorder=2.6))
                    # a light hatch marks the whole region as
                    # cyclic-reduction territory even at the palest step
                    ax.add_patch(Rectangle(
                        (lo, lo), cum[-1] * n_b, cum[-1] * n_b,
                        facecolor="none", hatch="///",
                        edgecolor="#dcbf8e", linewidth=0.0, zorder=-0.9))
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
            # the variable order is written under the unit instead
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_linewidth(0.6)
        print(f"{name}: nnz(L) = {int(pat_L.sum())}, fill-in = "
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
              label="horizon-level parallelism hierarchy"),
    ]
    # legend in two rows of three, a step above the panel-title size;
    # shrink until the rows are no wider than the span of the four
    # panels, so the legend never sets the page width itself.  It is
    # anchored to the top-row axes rather than to the figure edge, so
    # the gap does not depend on how the layout engine distributes
    # spare height.
    fig.tight_layout(rect=(0, 0.03, 1, 0.97), h_pad=2.6)
    left = axes[0, 0].get_position().x0
    right = axes[0, 4].get_position().x1
    span = (right - left) * fig.bbox.width
    top = axes[0, 0].get_position().y1 + 0.032
    size = 10.5
    while True:
        leg = fig.legend(handles=handles, loc="lower center", ncol=3,
                         frameon=False, fontsize=size,
                         handlelength=1.8, handleheight=1.1,
                         columnspacing=1.3, handletextpad=0.6,
                         bbox_to_anchor=(0.5, top))
        fig.canvas.draw()
        if leg.get_window_extent().width <= span or size <= 6.0:
            break
        leg.remove()
        size -= 0.25
    print(f"legend font {size:.2f} pt")

    # per unit: the variable order of one tail (the same for every
    # tail, root last) directly under the pair in the panel-title
    # font, and the plain "(a) name" subcaption below that
    tails = (",".join(str(i) for i in range(1, B + 1)) if B <= 3
             else rf"1,\ldots,{B}")
    for u, (name, block_order, _) in enumerate(units):
        p0 = axes[u // 2, (u % 2) * 3].get_position()
        p1 = axes[u // 2, (u % 2) * 3 + 1].get_position()
        xc = (p0.x0 + p1.x1) / 2
        stages = ", ".join(rf"x_{{{T - k}}}^{{i}}" for k in block_order)
        fig.text(xc, p0.y0 - 0.018,
                 rf"variable order: $\left({stages}\right)_{{i="
                 rf"{tails}}},\; x_0$",
                 ha="center", va="top", fontsize=9.5)
        fig.text(xc, p0.y0 - 0.052, rf"({'abcd'[u]}) {name}",
                 ha="center", va="top", fontsize=12)

    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("saved", out_pdf)


def main():
    # num_stages is the horizon N: each tail holds x_N^i ... x_1^i and
    # the shared root node x_0 is the single extra block at the end
    shape, K = build(9)
    render(shape, K, OUT)


if __name__ == "__main__":
    main()

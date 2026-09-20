"""Four-class variable ordering and sparsity-pattern visualizer.

Builds the block-structural pattern of a scenario--time optimization
problem with four variable classes (see plan.md):

- x[i,k]: scenario-local, time-local stage blocks (blue;
  called w in plan.md),
- y[i]:   scenario-wide blocks, one per scenario (orange),
- u[k]:   stage-wise inputs shared by all scenarios (green),
- t:      the fully global block (red).

For each requested ordering the program draws the original pattern,
the symbolically filled factor, a fill-only view (fill in magenta),
and the structural Schur complements after eliminating all stage
blocks and after eliminating complete scenarios.  Fill is computed by
graph-based symbolic elimination at the block level (never by
thresholding a numerical factor), and the report lists dimensions,
nonzero counts, fill ratios, and front sizes.  Nonzero counts refer to
the FULL symmetric matrix.

Usage::

    python visualize_four_class_structure.py \
        [--scenarios B] [--horizon N] [--nw 1] [--nu 1] [--ny 1]
        [--nt 1] [--ordering all|scenario-major|scenario-major-xuyt|
                  scenario-major[-xuyt]-cr|early-y|time-major|
                  four-class|amd]
        [--input-coupling same-stage|adjacent-stage]
        [--risk-stages all|terminal|last-q|k0,k1,...] [--risk-q q]
        [--u-temporal-coupling] [--direct-t-stage-coupling]
        [--block-level] [--output-dir figures]

Writes figures/four_class_<ordering>_B<B>_N<N>.pdf.
"""

import argparse
import os
from dataclasses import dataclass, field

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# class colors per plan section 8 (fill: dark gray, kept clearly
# apart from the red of the global block t)
COLORS = {"w": "#3a6fb0", "y": "#e08214", "u": "#2d6b3c",
          "t": "#c1272d", "fill": "#4d4d4d"}
CLASS_RANK = {"w": 0, "u": 1, "y": 2, "t": 3}


def cyclic_reduction_order(N):
    """Cyclic-reduction elimination order of an N-block chain: level l
    with stride s = 2^l eliminates positions s-1 + j*2s."""
    order, seen = [], set()
    s = 1
    while len(order) < N:
        level = [i for i in range(s - 1, N, 2 * s) if i not in seen]
        order.extend(level)
        seen.update(level)
        s *= 2
    return order


@dataclass
class Block:
    bid: int
    cls: str            # "w", "y", "u", or "t"
    i: int              # scenario index (-1 when not applicable)
    k: int              # stage index (-1 when not applicable)
    dim: int

    def label(self):
        if self.cls == "w":
            return rf"$x_{{{self.k}}}^{{{self.i}}}$"
        if self.cls == "y":
            return rf"$y_{{{self.i}}}$"
        if self.cls == "u":
            return rf"$u_{{{self.k}}}$"
        return r"$t$"


@dataclass
class FourClassStructure:
    """Block coupling graph of the four-class problem (plan section 3).

    Vertices are the blocks w[i,k], y[i], u[k], t; edges are the
    structural couplings selected by the options.  ``build_pattern``
    returns the adjacency sets, ``get_ordering`` the block permutation
    for each ordering of plan section 7.
    """

    B: int
    N: int
    n_w: int = 1
    n_u: int = 1
    n_y: int = 1
    n_t: int = 1
    input_coupling: str = "same-stage"
    risk_stages: str = "all"
    risk_q: int = 1
    u_temporal_coupling: bool = False
    direct_t_stage_coupling: bool = False
    blocks: list = field(default_factory=list, init=False)
    adj: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        B, N = self.B, self.N
        self.blocks = []
        for i in range(B):
            for k in range(N):
                self.blocks.append(Block(len(self.blocks), "w", i, k,
                                         self.n_w))
        for i in range(B):
            self.blocks.append(Block(len(self.blocks), "y", i, -1,
                                     self.n_y))
        for k in range(N):
            self.blocks.append(Block(len(self.blocks), "u", -1, k,
                                     self.n_u))
        self.blocks.append(Block(len(self.blocks), "t", -1, -1,
                                 self.n_t))
        self.adj = {b.bid: set() for b in self.blocks}
        self._build_edges()

    # ------------------------------------------------ block indexing
    def w(self, i, k):
        return i * self.N + k

    def y(self, i):
        return self.B * self.N + i

    def u(self, k):
        return self.B * self.N + self.B + k

    @property
    def t(self):
        return self.B * self.N + self.B + self.N

    def _connect(self, a, b):
        if a != b:
            self.adj[a].add(b)
            self.adj[b].add(a)

    def _risk_stage_list(self):
        if self.risk_stages == "all":
            return list(range(self.N))
        if self.risk_stages == "terminal":
            return [self.N - 1]
        if self.risk_stages == "last-q":
            return list(range(max(0, self.N - self.risk_q), self.N))
        return [int(k) for k in self.risk_stages.split(",")]

    def _build_edges(self):
        B, N = self.B, self.N
        for i in range(B):
            # 3.1 temporal chain
            for k in range(N - 1):
                self._connect(self.w(i, k), self.w(i, k + 1))
            # 3.2 stage-wise shared inputs
            for k in range(N):
                self._connect(self.w(i, k), self.u(k))
                if self.input_coupling == "adjacent-stage" and k > 0:
                    self._connect(self.w(i, k), self.u(k - 1))
            # 3.3 scenario-wide variables on the selected stages
            for k in self._risk_stage_list():
                self._connect(self.w(i, k), self.y(i))
                if self.direct_t_stage_coupling:
                    self._connect(self.w(i, k), self.t)
            # 3.4 fully global variable
            self._connect(self.y(i), self.t)
        # 3.5 optional input-rate coupling
        if self.u_temporal_coupling:
            for k in range(N - 1):
                self._connect(self.u(k), self.u(k + 1))

    def build_pattern(self):
        return self.adj, self.blocks

    # -------------------------------------------- orderings (sec. 7)
    def get_ordering(self, name):
        B, N = self.B, self.N
        if name == "scenario-major":
            order = []
            for i in range(B):
                order += [self.w(i, k) for k in range(N)]
                order.append(self.y(i))
            order += [self.u(k) for k in range(N)] + [self.t]
        elif name == "early-y":
            order = []
            for i in range(B):
                order.append(self.y(i))
                order += [self.w(i, k) for k in range(N)]
            order += [self.u(k) for k in range(N)] + [self.t]
        elif name == "time-major-reversed":
            order = []
            for k in range(N - 1, -1, -1):
                order += [self.w(i, k) for i in range(B)]
                order.append(self.u(k))
            order += [self.y(i) for i in range(B)] + [self.t]
        elif name == "time-major":
            order = []
            for k in range(N):
                order += [self.w(i, k) for i in range(B)]
                order.append(self.u(k))
            order += [self.y(i) for i in range(B)] + [self.t]
        elif name == "scenario-major-xuyt":
            order = ([self.w(i, k) for i in range(B)
                      for k in range(N)]
                     + [self.u(k) for k in range(N)]
                     + [self.y(i) for i in range(B)] + [self.t])
        elif name == "scenario-major-cr":
            cr = cyclic_reduction_order(N)
            order = []
            for i in range(B):
                order += [self.w(i, k) for k in cr]
                order.append(self.y(i))
            order += [self.u(k) for k in range(N)] + [self.t]
        elif name == "scenario-major-xuyt-cr":
            cr = cyclic_reduction_order(N)
            order = ([self.w(i, k) for i in range(B) for k in cr]
                     + [self.u(k) for k in range(N)]
                     + [self.y(i) for i in range(B)] + [self.t])
        elif name == "u-x-y-t":
            order = ([self.u(k) for k in range(N)]
                     + [self.w(i, k) for i in range(B)
                        for k in range(N)]
                     + [self.y(i) for i in range(B)] + [self.t])
        elif name == "four-class":
            order = ([self.w(i, k) for i in range(B)
                      for k in range(N)]
                     + [self.y(i) for i in range(B)]
                     + [self.u(k) for k in range(N)] + [self.t])
        elif name in ("amd", "nested-dissection"):
            # generic graph orderings (CHOLMOD AMD / METIS nested
            # dissection on the block pattern), NOT semantic orderings
            import scipy.sparse as sp
            from sksparse import cholmod as cm
            V = len(self.blocks)
            rows, cols = [], []
            for a, nbrs in self.adj.items():
                rows.append(a)
                cols.append(a)
                for b in nbrs:
                    rows.append(a)
                    cols.append(b)
            pat = sp.csc_array(
                (np.ones(len(rows)), (rows, cols)), shape=(V, V))
            if name == "amd":
                order = list(np.asarray(
                    cm.cho_factor(pat, beta=V, lower=True,
                                  order="amd").perm))
            else:
                order = list(np.asarray(cm.metis(pat)))
        else:
            raise ValueError(f"unknown ordering {name!r}")
        return order


# ------------------------------------------- symbolic elimination
def symbolic_elimination(adj, order, dims):
    """Block-level symbolic Cholesky by graph elimination (plan
    section 6): eliminating a vertex connects all its active
    neighbors; new edges are fill.  Returns the filled edge set, the
    fill edge set, and stats (scalar counts on the full symmetric
    matrix, plus the maximum scalar front size)."""
    active = {v: set(nbrs) for v, nbrs in adj.items()}
    fill = set()
    filled = {frozenset((a, b)) for a, nbrs in adj.items()
              for b in nbrs}
    max_front = 0
    for v in order:
        nbrs = active[v]
        max_front = max(max_front,
                        dims[v] + sum(dims[b] for b in nbrs))
        for a in nbrs:
            for b in nbrs:
                if a < b and b not in active[a]:
                    active[a].add(b)
                    active[b].add(a)
                    fill.add(frozenset((a, b)))
                    filled.add(frozenset((a, b)))
        for a in nbrs:
            active[a].discard(v)
        active[v] = set()

    def scalar_nnz(edges):
        return (sum(dims[v] ** 2 for v in adj)
                + 2 * sum(dims[a] * dims[b]
                          for e in edges for a, b in [tuple(e)]))

    orig_edges = {frozenset((a, b)) for a, nbrs in adj.items()
                  for b in nbrs}
    stats = {
        "nnz_original": scalar_nnz(orig_edges),
        "nnz_filled": scalar_nnz(filled),
        "fill_block_edges": len(fill),
        "fill_scalar": 2 * sum(dims[a] * dims[b]
                               for e in fill for a, b in [tuple(e)]),
        "max_front": max_front,
    }
    stats["fill_ratio"] = stats["nnz_filled"] / stats["nnz_original"]
    return filled, fill, stats


def symbolic_schur(adj, eliminate, dims):
    """Structural Schur complement: eliminate the given block set (the
    resulting pattern on the retained blocks is independent of the
    internal elimination order) and return (retained_edges,
    fill_edges_among_retained)."""
    active = {v: set(nbrs) for v, nbrs in adj.items()}
    elim = set(eliminate)
    for v in eliminate:
        nbrs = active[v]
        for a in nbrs:
            for b in nbrs:
                if a < b and b not in active[a]:
                    active[a].add(b)
                    active[b].add(a)
        for a in nbrs:
            active[a].discard(v)
        active[v] = set()
    retained = [v for v in adj if v not in elim]
    edges = {frozenset((a, b)) for a in retained for b in active[a]
             if b not in elim}
    orig = {frozenset((a, b)) for a in retained
            for b in adj[a] if b not in elim}
    return retained, edges, edges - orig


# ------------------------------------------------------ visualization
def _cell_color_index(cls_a, cls_b):
    """Color a coupling by its most global class (t > y > u > w)."""
    cls = max((cls_a, cls_b), key=lambda c: CLASS_RANK[c])
    return {"w": 1, "u": 2, "y": 3, "t": 4}[cls]


def _draw_panel(ax, order, blocks, edges, fill_edges, title,
                show_labels, gray_original=False):
    dims = [blocks[b].dim for b in order]
    off = np.concatenate([[0], np.cumsum(dims)])
    n = off[-1]
    pos = {b: j for j, b in enumerate(order)}
    img = np.zeros((n, n), dtype=int)
    for b in order:
        j = pos[b]
        c = (6 if gray_original
             else _cell_color_index(blocks[b].cls, blocks[b].cls))
        img[off[j]:off[j + 1], off[j]:off[j + 1]] = c
    for e in edges:
        a, b = tuple(e)
        if a not in pos or b not in pos:
            continue
        if e in fill_edges:
            c = 5
        elif gray_original:
            c = 6
        else:
            c = _cell_color_index(blocks[a].cls, blocks[b].cls)
        ja, jb = pos[a], pos[b]
        img[off[ja]:off[ja + 1], off[jb]:off[jb + 1]] = c
        img[off[jb]:off[jb + 1], off[ja]:off[ja + 1]] = c
    cmap = ListedColormap(["white", COLORS["w"], COLORS["u"],
                           COLORS["y"], COLORS["t"], COLORS["fill"],
                           "0.8"])
    ax.imshow(img, cmap=cmap, vmin=0, vmax=6,
              interpolation="nearest")
    for j in range(1, len(order)):
        prev, cur = blocks[order[j - 1]], blocks[order[j]]
        strong = (prev.cls != cur.cls or prev.i != cur.i)
        ax.axvline(off[j] - 0.5, color="0.45" if strong else "0.85",
                   lw=0.5 if strong else 0.25, zorder=1)
        ax.axhline(off[j] - 0.5, color="0.45" if strong else "0.85",
                   lw=0.5 if strong else 0.25, zorder=1)
    if show_labels:
        centers = [(off[j] + off[j + 1] - 1) / 2
                   for j in range(len(order))]
        labels = [blocks[b].label() for b in order]
        ax.set_xticks(centers)
        ax.set_xticklabels(labels, fontsize=5)
        ax.set_yticks(centers)
        ax.set_yticklabels(labels, fontsize=5)
        ax.tick_params(length=0)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    for side in ax.spines.values():
        side.set_linewidth(0.6)


def plot_patterns(model, ordering_name, out_dir):
    adj, blocks = model.build_pattern()
    dims = {b.bid: b.dim for b in blocks}
    order = model.get_ordering(ordering_name)
    orig_edges = {frozenset((a, b)) for a, nbrs in adj.items()
                  for b in nbrs}
    filled, fill, stats = symbolic_elimination(adj, order, dims)

    w_ids = [b.bid for b in blocks if b.cls == "w"]
    scen_ids = [b.bid for b in blocks if b.cls in ("w", "y")]
    ret_w, edges_w, fill_w = symbolic_schur(adj, w_ids, dims)
    ret_s, edges_s, fill_s = symbolic_schur(adj, scen_ids, dims)
    order_ret_w = [b for b in order if b in set(ret_w)]
    order_ret_s = [b for b in order if b in set(ret_s)]

    n = sum(dims.values())
    n_ut = sum(dims[b] for b in ret_s)
    schur_ut_nnz = (sum(dims[b] ** 2 for b in ret_s)
                    + 2 * sum(dims[a] * dims[b]
                              for e in edges_s for a, b in [tuple(e)]))
    stats.update({
        "n": n,
        "schur_wy_dim": sum(dims[b] for b in ret_w),
        "schur_wy_block_edges": len(edges_w),
        "schur_ut_dim": n_ut,
        "schur_ut_nnz": schur_ut_nnz,
        "schur_ut_density": schur_ut_nnz / n_ut ** 2,
    })

    show_labels = len(order) <= 40
    plt.rcParams.update({"font.family": "serif", "font.size": 9,
                         "figure.dpi": 150, "savefig.bbox": "tight"})
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6))
    _draw_panel(axes[0], order, blocks, orig_edges, set(),
                f"original pattern (n = {n}, "
                f"nnz = {stats['nnz_original']})", show_labels)
    _draw_panel(axes[1], order, blocks, filled, fill,
                f"symbolic factor (nnz = {stats['nnz_filled']}, "
                f"fill ratio = {stats['fill_ratio']:.2f})",
                show_labels)
    _draw_panel(axes[2], order, blocks, filled, fill,
                f"fill only ({stats['fill_block_edges']} block "
                f"edges, {stats['fill_scalar']} scalars)",
                show_labels, gray_original=True)
    handles = [Patch(facecolor=COLORS[c], label=lbl) for c, lbl in
               (("w", "$x_k^i$ stage blocks"),
                ("y", "$y_i$ scenario-wide"),
                ("u", "$u_k$ stage-wide"),
                ("t", "$t$ global"),
                ("fill", "symbolic fill"))]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.10))
    fig.suptitle(
        f"Four-class structure, ordering: {ordering_name}  "
        f"(B={model.B}, N={model.N}, "
        f"n_x={model.n_w}, n_u={model.n_u}, n_y={model.n_y}, "
        f"n_t={model.n_t}; input {model.input_coupling}, risk "
        f"{model.risk_stages}"
        + (", u-chain" if model.u_temporal_coupling else "") + ")",
        fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    os.makedirs(out_dir, exist_ok=True)
    base = (f"{out_dir}/four_class_{ordering_name}"
            f"_B{model.B}_N{model.N}")
    fig.savefig(f"{base}.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"saved {base}.pdf")
    return stats


def report(name, s):
    print(f"--- {name}")
    print(f"  dimension                  {s['n']}")
    print(f"  nnz original (full symm.)  {s['nnz_original']}")
    print(f"  nnz after elimination      {s['nnz_filled']}")
    print(f"  fill edges (blocks/scalar) {s['fill_block_edges']} / "
          f"{s['fill_scalar']}")
    print(f"  fill ratio                 {s['fill_ratio']:.3f}")
    print(f"  max scalar front           {s['max_front']}")
    print(f"  Schur dim after w elim     {s['schur_wy_dim']}")
    print(f"  Schur (u,t) dim / nnz      {s['schur_ut_dim']} / "
          f"{s['schur_ut_nnz']}")
    print(f"  Schur (u,t) density        "
          f"{s['schur_ut_density']:.3f}")


ORDERINGS = ("scenario-major", "scenario-major-xuyt",
             "scenario-major-cr", "scenario-major-xuyt-cr",
             "u-x-y-t", "early-y", "time-major",
             "time-major-reversed", "four-class")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenarios", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--nw", type=int, default=1)
    ap.add_argument("--nu", type=int, default=1)
    ap.add_argument("--ny", type=int, default=1)
    ap.add_argument("--nt", type=int, default=1)
    ap.add_argument("--ordering", default="all",
                    help="one of scenario-major, scenario-major-xuyt, "
                         "scenario-major[-xuyt]-cr, early-y, "
                         "nested-dissection, "
                         "time-major, four-class, amd, or 'all'")
    ap.add_argument("--input-coupling", default="same-stage",
                    choices=("same-stage", "adjacent-stage"))
    ap.add_argument("--risk-stages", default="all",
                    help="all | terminal | last-q | k0,k1,...")
    ap.add_argument("--risk-q", type=int, default=1)
    ap.add_argument("--u-temporal-coupling", action="store_true")
    ap.add_argument("--direct-t-stage-coupling", action="store_true")
    ap.add_argument("--block-level", action="store_true",
                    help="accepted for compatibility; the symbolic "
                         "elimination is always block-level")
    ap.add_argument("--output-dir", default="figures")
    args = ap.parse_args()

    model = FourClassStructure(
        args.scenarios, args.horizon, args.nw, args.nu, args.ny,
        args.nt, input_coupling=args.input_coupling,
        risk_stages=args.risk_stages, risk_q=args.risk_q,
        u_temporal_coupling=args.u_temporal_coupling,
        direct_t_stage_coupling=args.direct_t_stage_coupling)

    if args.ordering == "all":
        names = list(ORDERINGS)
        try:
            model.get_ordering("amd")
            names += ["amd", "nested-dissection"]
        except ImportError:
            print("amd/nested-dissection orderings skipped "
                  "(scikit-sparse missing)")
    else:
        names = [args.ordering]
    for name in names:
        stats = plot_patterns(model, name, args.output_dir)
        report(name, stats)


if __name__ == "__main__":
    main()

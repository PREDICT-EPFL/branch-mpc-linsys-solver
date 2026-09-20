"""Absolute latency along one axis of the (scenarios, horizon) grid.

Reads a results CSV written by quadrotor_endpoint_benchmark.py in
which exactly one of the two problem dimensions varies, prints the
speedup table, and draws median numerical factorization and
single-RHS triangular-solve latency against that dimension on a log
axis.  Which axis is drawn follows the CSV: a sweep at one fixed
horizon is plotted against the scenario count M, a sweep at one fixed
scenario count against the horizon N.  These are the paper's
absolute-time figures; ``--solvers`` selects which baselines appear,
so a figure can show a subset of what the CSV measured.

Usage (from this directory, socu env)::

    python quadrotor_cpu_baselines_plot.py \
        --input results/quadrotor_endpoint_times_T30.csv \
        --solvers endpoint cudss pardiso --tails 20 40 60 80 100 120 140 160 180 200 \
        --layout combined \
        --output figures/quadrotor_endpoint_times_T30_combined.pdf

    python quadrotor_cpu_baselines_plot.py \
        --input results/quadrotor_endpoint_times_M100.csv \
        --solvers endpoint cudss pardiso --layout combined \
        --output figures/quadrotor_endpoint_times_M100_combined.pdf
"""

import argparse
import csv
import os
import shutil
import statistics

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, NullFormatter  # noqa: E402

# Fixed series order, colours and markers.  Colour follows the solver,
# never its rank, so drawing a subset never repaints the others.  The
# slots are validated as a set: worst all-pairs separation is dE 16.3
# for normal vision and 9.2 under simulated CVD.  PARDISO deliberately
# does NOT use the yellow slot -- yellow against this orange measures
# dE 13.7, below the legibility floor, and the two read as one colour.
SERIES = [
    ("endpoint", "proposed", "#2a78d6", "o"),   # blue
    ("cudss", "cuDSS", "#eb6834", "s"),         # orange
    ("pardiso", "PARDISO", "#1baf7a", "D"),     # aqua
    ("cholmod", "CHOLMOD", "#4a3aa7", "^"),     # violet
]
# solid = factorization, dashed = triangular solve (combined layout)
PHASES = (("factor_ms", "factorization", "-"),
          ("solve_ms", "triangular solve", "--"))
INK, INK2, MUTED, GRID, AXIS = ("#0b0b0b", "#52514e", "#898781",
                                "#e1e0d9", "#c3c2b7")

# One IEEE column of the paper is \columnwidth = 245.72 TeX pt = 3.400
# in.  Authoring at exactly that width means \includegraphics[width=
# \linewidth] scales the figure 1:1 and the text renders at its true
# point size.
FIG_W = 3.45
COL_W = 3.40
# height of the combined single-panel figure; the two legend rows
# above the axes take about half an inch of it
COMB_H = 2.60
Y_TICKS = (0.1, 0.2, 0.5, 1, 2, 5, 10, 20)


def _rc():
    return {
        "text.usetex": shutil.which("latex") is not None,
        "font.family": "serif", "font.size": 10, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "figure.dpi": 150,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    }


def _finish_axis(ax, Ms, *, ylabel=True):
    """Recessive chrome, readable log ticks, clean scenario ticks."""
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    ax.set_yticks([t for t in Y_TICKS if lo <= t <= hi])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    if ylabel:
        ax.set_ylabel("time [ms]", labelpad=2)
    # recessive grid: solid hairlines, majors a touch stronger than the
    # log minors so the decade structure still reads first
    ax.grid(True, axis="y", which="major", color=GRID, lw=0.6, zorder=0)
    ax.grid(True, axis="y", which="minor", color=GRID, lw=0.35, zorder=0)
    ax.grid(True, axis="x", which="major", color=GRID, lw=0.5, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2.2, width=0.6, pad=1.5)
    step = 40 if all(m % 20 == 0 for m in Ms) else 50
    ax.set_xticks([m for m in range(step, max(Ms) + 1, step)] or Ms)


def _series_of(table, Ms, solver, key):
    xs = [M for M in Ms if solver in table[M]]
    return xs, [float(table[M][solver][key]) for M in xs]



def plot_stacked(table, xlabel, out, selected):
    """Factorization above, triangular solve below (two log panels)."""
    plt.rcParams.update(_rc())
    Ms = sorted(table)
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 3.05), sharex=True)
    for ax, (key, title) in zip(axes, (("factor_ms", "factorization"),
                                       ("solve_ms", "triangular solve"))):
        for solver, label, color, marker in SERIES:
            if solver not in selected:
                continue
            xs, ys = _series_of(table, Ms, solver, key)
            if xs:
                ax.plot(xs, ys, color=color, lw=1.3, marker=marker, ms=3.2,
                        mec="white", mew=0.6, label=label, zorder=3,
                        solid_joinstyle="round")
        # the panel name rides inside the axes, saving a title band
        ax.text(0.015, 0.94, title, transform=ax.transAxes, ha="left",
                va="top", fontsize=8, color=INK)
        _finish_axis(ax, Ms)
    axes[1].set_xlabel(xlabel, labelpad=1.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=len(handles),
               loc="upper center", bbox_to_anchor=(0.55, 1.03),
               handlelength=1.5, columnspacing=1.2, handletextpad=0.4)
    fig.subplots_adjust(hspace=0.12, top=0.92)
    _save(fig, out)


def plot_combined(table, xlabel, out, selected):
    """Both phases in one panel: colour names the solver, line style the
    phase (solid factorization, dashed triangular solve).  Sized to the
    paper column exactly, with the two legends stacked above the axes so
    neither can cover a curve."""
    plt.rcParams.update(_rc())
    Ms = sorted(table)
    fig, ax = plt.subplots(figsize=(COL_W, COMB_H))
    for key, _, style in PHASES:
        solve = key == "solve_ms"
        for solver, label, color, marker in SERIES:
            if solver not in selected:
                continue
            xs, ys = _series_of(table, Ms, solver, key)
            if xs:
                ax.plot(xs, ys, color=color, lw=1.5, ls=style, marker=marker,
                        ms=4.0, mew=1.0, label=label if not solve else None,
                        # open markers echo the dashed line: same solver,
                        # the cheaper phase
                        mfc="white" if solve else color,
                        mec=color if solve else "white", zorder=3)
    _finish_axis(ax, Ms)
    ax.set_xlabel(xlabel, labelpad=1.5)

    # two legend rows above the panel: colour = solver, line style = phase
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=len(handles),
               loc="upper center", bbox_to_anchor=(0.5, 1.008),
               handlelength=1.5, columnspacing=1.3, handletextpad=0.4)
    style_handles = [Line2D([], [], color=MUTED, lw=1.5, ls=st,
                            marker="o", ms=4.0, mew=1.0,
                            mfc="white" if st == "--" else MUTED,
                            mec=MUTED, label=name)
                     for _, name, st in PHASES]
    fig.legend(handles=style_handles, frameon=False, ncol=len(style_handles),
               loc="upper center", bbox_to_anchor=(0.5, 0.911),
               handlelength=2.2, columnspacing=1.3, handletextpad=0.4)

    # explicit margins instead of a tight bounding box, so the saved PDF
    # is exactly COL_W wide
    fig.subplots_adjust(left=0.165, right=0.985, top=0.81, bottom=0.19)
    _save(fig, out, tight=False)


def _save(fig, out, *, tight=True):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=300,
                **({"bbox_inches": "tight"} if tight else {}))
    plt.close(fig)
    print("saved", out)


def load(path, tails=None, horizons=None):
    """Read one results CSV and pick the dimension that varies.

    Returns the rows keyed by that dimension, its CSV column, and the
    (name, value) of the dimension held fixed.
    """
    rows = list(csv.DictReader(open(path)))
    if tails:
        rows = [r for r in rows if int(r["B"]) in set(tails)]
    if horizons:
        rows = [r for r in rows if int(r["T"]) in set(horizons)]
    assert rows, "no rows left after --tails/--horizons filtering"
    Bs = {int(r["B"]) for r in rows}
    Ts = {int(r["T"]) for r in rows}
    if len(Ts) == 1:
        return _by(rows, "B"), "B", ("N", Ts.pop())
    if len(Bs) == 1:
        return _by(rows, "T"), "T", ("M", Bs.pop())
    raise SystemExit(
        f"one dimension must be fixed, got {len(Bs)} scenario counts "
        f"and {len(Ts)} horizons; narrow with --tails / --horizons")


def _by(rows, key):
    table = {}
    for r in rows:
        table.setdefault(int(r[key]), {})[r["solver"]] = r
    return table


# axis label and table heading for each varying dimension
AXES = {"B": (r"scenarios $M$", "M"), "T": (r"horizon $N$", "N")}


def print_table(table, fixed, x_name, selected):
    keys = [s for s, _, _, _ in SERIES if s in selected]
    print(f"{fixed[0]}={fixed[1]}: median ms, factor / solve; "
          f"speedup = baseline / proposed")
    head = " ".join(f"{k:>15}" for k in keys[1:])
    print(f"{x_name:>4} {'proposed':>15} {head}")
    ratios = {k: ([], []) for k in keys[1:]}
    for M in sorted(table):
        v = table[M]
        fe, se = (float(v["endpoint"]["factor_ms"]),
                  float(v["endpoint"]["solve_ms"]))
        cells = [f"{fe:6.3f}/{se:6.3f}  "]
        for k in keys[1:]:
            if k not in v:
                cells.append(f"{'--':>15}")
                continue
            f, s = float(v[k]["factor_ms"]), float(v[k]["solve_ms"])
            ratios[k][0].append(f / fe)
            ratios[k][1].append(s / se)
            cells.append(f"{f:6.3f}/{s:6.3f} ({f / fe:4.1f}x/{s / se:4.1f}x)")
        print(f"{M:4d} " + " ".join(cells))
    for k, (rf, rs) in ratios.items():
        if rf:
            print(f"  vs {k:8}: factor {min(rf):4.1f}-{max(rf):4.1f}x "
                  f"(median {statistics.median(rf):4.1f}), solve "
                  f"{min(rs):4.1f}-{max(rs):4.1f}x "
                  f"(median {statistics.median(rs):4.1f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input",
                    default="results/quadrotor_endpoint_T30_pardiso.csv")
    ap.add_argument("--output",
                    default="figures/quadrotor_endpoint_T30_baselines.pdf")
    ap.add_argument("--layout", choices=("stacked", "combined"),
                    default="stacked",
                    help="stacked: one panel per phase; combined: both "
                         "phases in one panel (colour = solver, line "
                         "style = phase)")
    ap.add_argument("--tails", type=int, nargs="+",
                    help="scenario counts to keep (default: every one "
                         "in the CSV)")
    ap.add_argument("--horizons", type=int, nargs="+",
                    help="horizons to keep (default: every one in the "
                         "CSV)")
    ap.add_argument("--solvers", nargs="+",
                    default=[s for s, _, _, _ in SERIES],
                    help="solvers to draw, in any order (the fixed series "
                         "order and colours are preserved)")
    args = ap.parse_args()
    table, x_key, fixed = load(args.input, args.tails, args.horizons)
    xlabel, x_name = AXES[x_key]
    selected = set(args.solvers)
    print_table(table, fixed, x_name, selected)
    draw = {'stacked': plot_stacked,
            'combined': plot_combined}[args.layout]
    draw(table, xlabel, args.output, selected)


if __name__ == "__main__":
    main()

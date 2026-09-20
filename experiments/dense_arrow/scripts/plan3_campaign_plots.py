"""Aggregate the plan-3 isolated campaign and produce the section-10.4
artifacts: campaign-A factor/solve speedup heatmaps and tables, scaling
curves for B-E and G, time-per-RHS curves for F, crossover summaries,
and phase breakdowns.  All figures are PDF; all content comes from the
raw records (missing points are skipped, never fabricated).

Usage::

    python scripts/plan3_campaign_plots.py [--input results/plan3_campaign/raw]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

COLORS = {"tree": "#2a78d6", "cudss": "#eb6834"}
LABELS = {"tree": "tree solver (ours)", "cudss": "cuDSS"}


def _style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9,
        "axes.titlesize": 9, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#dddad1",
        "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False,
    })


def load_rows(raw_dir):
    rows = []
    for f in sorted(Path(raw_dir).glob("*.json")):
        r = json.loads(f.read_text())
        pt = r["point"]
        for backend, d in r["backends"].items():
            if d.get("status") == "unavailable":
                continue
            for phase in ("factor", "solve"):
                rows.append({**pt, "backend": backend, "phase": phase,
                             **d[phase],
                             "relative_residual": d["relative_residual"],
                             "status": d["status"],
                             "disagreement":
                                 r.get("tree_cudss_disagreement")})
    return pd.DataFrame(rows)


def med(df, campaign, backend, phase, **filters):
    d = df[(df.campaign == campaign) & (df.backend == backend)
           & (df.phase == phase)]
    for k, v in filters.items():
        d = d[d[k] == v]
    g = d.groupby("seed")["median_ms"].median()
    if not len(g):
        return np.nan, np.nan, np.nan
    return float(g.median()), float(g.min()), float(g.max())


def _save(fig, out_dir, name):
    fig.savefig(out_dir / f"{name}.pdf")
    plt.close(fig)
    print(f"  wrote {out_dir / name}.pdf")


def fig_heatmap_A(df, out_dir):
    """Publication figure (two-column width): speedup heatmaps over the
    application grid.  2 x 3 layout -- rows are factorization and
    triangular solve, columns the three stage block sizes; each column
    shares its x-axis.  LaTeX text."""
    import matplotlib.colors as mcolors
    d = df[df.campaign == "A"]
    Bs = sorted(d["B"].unique())
    Ns = sorted(d["T"].unique())
    nbs = sorted(d["n_b"].unique())
    if not Bs:
        print("  skip fig_A: no data")
        return
    table = []
    grids = {}
    for phase in ("factor", "solve"):
        for nb in nbs:
            M = np.full((len(Bs), len(Ns)), np.nan)
            for i, B in enumerate(Bs):
                for j, N in enumerate(Ns):
                    t, *_ = med(df, "A", "tree", phase, B=B, T=N, n_b=nb)
                    c, *_ = med(df, "A", "cudss", phase, B=B, T=N, n_b=nb)
                    M[i, j] = c / t
                    table.append({"phase": phase, "n_b": nb, "B": B, "N": N,
                                  "tree_ms": t, "cudss_ms": c,
                                  "speedup": c / t})
            grids[(phase, nb)] = M

    with plt.rc_context({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": False}):
        fig = plt.figure(figsize=(7.0, 4.3), layout="constrained")
        # one subfigure per phase so each row carries one shared title
        subfigs = fig.subfigures(2, 1)
        phase_titles = {"factor": "Factorization",
                        "solve": "Triangular solve"}
        for ri, phase in enumerate(("factor", "solve")):
            sf = subfigs[ri]
            sf.suptitle(phase_titles[phase], fontsize=10)
            axs = sf.subplots(1, len(nbs), squeeze=False)[0]
            row = [grids[(phase, nb)] for nb in nbs]
            vmax = max(np.nanmax(M) for M in row)
            vmin = min(np.nanmin(M) for M in row)
            norm = mcolors.TwoSlopeNorm(vcenter=1.0,
                                        vmin=min(0.9, vmin * 0.95),
                                        vmax=max(1.1, vmax))
            for ci, nb in enumerate(nbs):
                ax = axs[ci]
                M = grids[(phase, nb)]
                im = ax.imshow(M, cmap="RdBu_r", norm=norm, aspect="auto",
                               origin="lower")
                for i in range(len(Bs)):
                    for j in range(len(Ns)):
                        v = M[i, j]
                        if not np.isfinite(v):
                            continue
                        # white text only on dark cells (either end of
                        # the diverging map)
                        dark = norm(v) > 0.75 or norm(v) < 0.25
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                fontsize=8,
                                color="white" if dark else "black")
                ax.set_yticks(range(len(Bs)), [str(b) for b in Bs])
                if ri == 1:
                    # columns share the x-axis: labels on the bottom row
                    ax.set_xticks(range(len(Ns)), [str(n) for n in Ns])
                    ax.set_xlabel(r"horizon length $N$")
                else:
                    ax.set_xticks(range(len(Ns)), [])
                    n_rs = sorted(d[d["n_b"] == nb]["n_r"].unique())
                    if len(n_rs) == 1 and n_rs[0] == nb:
                        ax.set_title(rf"$n_b = n_r = {nb}$", fontsize=9)
                    else:
                        ax.set_title(rf"$n_b = {nb}$", fontsize=9)
                if ci == 0:
                    ax.set_ylabel(r"number of scenarios $B$")
                else:
                    ax.tick_params(labelleft=False)
                ax.tick_params(length=0)
                for spine in ax.spines.values():
                    spine.set_visible(False)
            cb = sf.colorbar(im, ax=list(axs), shrink=0.92, pad=0.012,
                             aspect=22)
            cb.set_label("speedup", fontsize=9)
            cb.outline.set_visible(False)
        _save(fig, out_dir, "fig_A_speedup_heatmap")
    pd.DataFrame(table).to_csv(out_dir / "table_A_factor_solve.csv",
                               index=False)
    print(f"  wrote {out_dir / 'table_A_factor_solve.csv'}")


def fig_curves(df, campaign, xkey, xlabel, out_dir, filters=None,
               series_key=None):
    filters = filters or {}
    d = df[df.campaign == campaign]
    if not len(d):
        print(f"  skip campaign {campaign}: no data")
        return
    xs = sorted(d[xkey].unique())
    series_vals = sorted(d[series_key].unique()) if series_key else [None]
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.2), sharex=True)
    for col, phase in enumerate(("factor", "solve")):
        ax_t, ax_s = axes[0][col], axes[1][col]
        for sv in series_vals:
            f = dict(filters)
            if series_key:
                f[series_key] = sv
            for backend in ("tree", "cudss"):
                ys, lo, hi = [], [], []
                for x in xs:
                    m, a, b = med(df, campaign, backend, phase,
                                  **{xkey: x}, **f)
                    ys.append(m); lo.append(a); hi.append(b)
                style = dict(marker="o", markersize=3.5,
                             color=COLORS[backend])
                if series_key and sv == series_vals[-1]:
                    style["linestyle"] = "--"
                label = LABELS[backend] + (
                    f" ({series_key}={sv})" if series_key else "")
                ax_t.plot(xs, ys, label=label, **style)
                ax_t.fill_between(xs, lo, hi, color=COLORS[backend],
                                  alpha=0.12, linewidth=0)
            sp = []
            for x in xs:
                t, *_ = med(df, campaign, "tree", phase, **{xkey: x}, **f)
                c, *_ = med(df, campaign, "cudss", phase, **{xkey: x}, **f)
                sp.append(c / t)
            ax_s.plot(xs, sp, marker="o", markersize=3.5, color="#52514e",
                      linestyle="--" if series_key and sv == series_vals[-1]
                      else "-",
                      label=(f"{series_key}={sv}" if series_key else None))
        ax_t.set_xscale("log", base=2)
        ax_t.set_yscale("log")
        ax_t.set_title(f"{phase} time [ms]")
        ax_s.set_xscale("log", base=2)
        ax_s.axhline(1.0, color="#aaa", linewidth=1)
        ax_s.set_ylabel("speedup (cuDSS/tree)")
        ax_s.set_xlabel(xlabel)
        if col == 0:
            ax_t.legend(fontsize=7)
            if series_key:
                ax_s.legend(fontsize=7)
    fig.suptitle(f"Campaign {campaign}", fontsize=10)
    _save(fig, out_dir, f"fig_{campaign}_curves")


def crossover_summary(df, out_dir):
    lines = ["Crossover summary (first point where tree beats cuDSS):"]
    for campaign, xkey in (("B", "B"), ("C", "T"), ("D", "n_b"),
                           ("E", "n_r")):
        for phase in ("factor", "solve"):
            xs = sorted(df[df.campaign == campaign][xkey].unique())
            first = None
            for x in xs:
                t, *_ = med(df, campaign, "tree", phase, **{xkey: x})
                c, *_ = med(df, campaign, "cudss", phase, **{xkey: x})
                if np.isfinite(t) and np.isfinite(c) and c / t > 1.0:
                    first = x
                    break
            lines.append(f"  campaign {campaign} ({xkey}), {phase}: "
                         + (f"tree faster from {xkey}={first}"
                            if first is not None else "cuDSS faster "
                            "everywhere"))
    text = "\n".join(lines)
    (out_dir / "crossover_summary.txt").write_text(text + "\n")
    print(text)


def residual_check(df, out_dir):
    worst = df.groupby(["precision", "backend"])["relative_residual"].max()
    bad = df[df.status != "ok"]
    with open(out_dir / "residual_summary.txt", "w") as f:
        f.write(worst.to_string() + "\n")
        f.write(f"non-ok rows: {len(bad)}\n")
        d = df.dropna(subset=["disagreement"])
        f.write(f"worst tree/cuDSS disagreement: "
                f"{d['disagreement'].max():.3e}\n")
    print(f"  wrote {out_dir / 'residual_summary.txt'}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="results/plan3_campaign/raw")
    p.add_argument("--output", default="plots")
    args = p.parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    _style()
    df = load_rows(args.input)
    df = df[df.campaign != "smoke"]
    print(f"{len(df)} rows from {df.campaign.nunique()} campaigns")
    df.to_csv(out_dir.parent / "rows.csv", index=False)

    fp64 = df[df.precision == "float64"]
    fig_heatmap_A(fp64, out_dir)
    fig_curves(fp64, "B", "B", "tails B (T=64, n_b=8, n_r=2)", out_dir)
    fig_curves(fp64, "C", "T", "horizon T (B=41, n_b=8, n_r=2)", out_dir)
    fig_curves(fp64, "D", "n_b", "block size n_b (B=41, T=64, n_r=2)",
               out_dir)
    fig_curves(fp64, "E", "n_r", "root dimension n_r (B=41, T=64, n_b=8)",
               out_dir)
    fig_curves(fp64, "G", "B", "tails B (B*T = 4096)", out_dir)
    crossover_summary(fp64, out_dir)
    residual_check(df, out_dir)


if __name__ == "__main__":
    main()

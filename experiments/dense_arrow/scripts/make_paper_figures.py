"""Publication figures for the paper, from the plan-3 isolated campaign
(results/plan3_campaign/raw).  LaTeX-rendered (Computer Modern), PDF:

- fig_paper_scaling.pdf: absolute warm factorization and solve times
  (log-log) along the tail-count and horizon sweeps, both solvers,
  seed spread shaded -- shows the latency floor and asymptotic slopes
  behind the speedups;
- fig_paper_speedup_ecdf.pdf: survival curve of the per-configuration
  speedup over every FP64 configuration (campaigns A-E, G, H; multiple
  RHS excluded), one line per phase -- the one-glance robustness
  summary including the worst case.

The companion coverage figure (speedup heatmap over the application
grid) is plots/fig_A_speedup_heatmap.pdf from
scripts/plan3_campaign_plots.py.

Usage::

    python scripts/make_paper_figures.py [--output plots]
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
RC = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#dddad1",
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
}


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
                             "median_ms": d[phase]["median_ms"]})
    return pd.DataFrame(rows)


def _seed_stats(df, campaign, backend, phase, xkey):
    d = df[(df.campaign == campaign) & (df.backend == backend)
           & (df.phase == phase)]
    xs = sorted(d[xkey].unique())
    med, lo, hi = [], [], []
    for x in xs:
        g = d[d[xkey] == x].groupby("seed")["median_ms"].median()
        med.append(g.median()); lo.append(g.min()); hi.append(g.max())
    return xs, np.array(med), np.array(lo), np.array(hi)


def fig_scaling(df, out_dir):
    panels = [("B", "B", r"scenarios $B$ ($T = 64$)"),
              ("C", "T", r"horizon $T$ ($B = 41$)")]
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.6), sharex="col")
    for ci, (campaign, xkey, xlabel) in enumerate(panels):
        for ri, phase in enumerate(("factor", "solve")):
            ax = axes[ri][ci]
            for backend in ("tree", "cudss"):
                xs, med, lo, hi = _seed_stats(df, campaign, backend,
                                              phase, xkey)
                ax.plot(xs, med, marker="o", markersize=3,
                        linewidth=1.2, color=COLORS[backend],
                        label=LABELS[backend])
                ax.fill_between(xs, lo, hi, color=COLORS[backend],
                                alpha=0.15, linewidth=0)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            if ci == 0:
                title = ("factorization" if phase == "factor"
                         else "triangular solve")
                ax.set_ylabel(rf"{title} [ms]")
            if ri == 1:
                ax.set_xlabel(xlabel)
    axes[0][0].legend(loc="upper left")
    fig.tight_layout(h_pad=1.0)
    fig.savefig(out_dir / "fig_paper_scaling.pdf")
    plt.close(fig)
    print(f"  wrote {out_dir / 'fig_paper_scaling.pdf'}")


def fig_ecdf(df, out_dir):
    # per-configuration speedups: median over seeds, FP64, nrhs = 1
    d = df[(df.precision == "float64") & (df.nrhs == 1)]
    lines = {}
    for phase, style in (("factor", "-"), ("solve", "--")):
        sp = []
        cfg_keys = ["campaign", "B", "T", "n_b", "n_r"]
        for _, g in d[d.phase == phase].groupby(cfg_keys):
            t = g[g.backend == "tree"].groupby("seed")["median_ms"] \
                 .median().median()
            c = g[g.backend == "cudss"].groupby("seed")["median_ms"] \
                 .median().median()
            if np.isfinite(t) and np.isfinite(c):
                sp.append(c / t)
        sp = np.sort(np.asarray(sp))
        lines[phase] = sp
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for phase, style, color in (("factor", "-", "#2a78d6"),
                                ("solve", "--", "#eb6834")):
        sp = lines[phase]
        frac = 1.0 - np.arange(len(sp)) / len(sp)
        label = ("factorization" if phase == "factor"
                 else "triangular solve")
        ax.step(sp, frac, where="post", linestyle=style, color=color,
                linewidth=1.4,
                label=rf"{label} (median ${np.median(sp):.2f}\times$)")
    ax.axvline(1.0, color="#52514e", linewidth=0.8)
    ax.set_xlabel(r"speedup $s = t_{\mathrm{cuDSS}} / t_{\mathrm{tree}}$")
    ax.set_ylabel(r"fraction of configurations $\geq s$")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(left=0.7)
    ax.legend(loc="lower left")
    fig.savefig(out_dir / "fig_paper_speedup_ecdf.pdf")
    plt.close(fig)
    n = len(lines["factor"])
    print(f"  wrote {out_dir / 'fig_paper_speedup_ecdf.pdf'} "
          f"({n} configurations; worst solve "
          f"{lines['solve'].min():.2f}x, worst factor "
          f"{lines['factor'].min():.2f}x)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="results/plan3_campaign/raw")
    p.add_argument("--output", default="plots")
    args = p.parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_rows(args.input)
    df = df[df.campaign != "smoke"]
    print(f"{len(df)} rows")
    with plt.rc_context(RC):
        fig_scaling(df[df.precision == "float64"], out_dir)
        fig_ecdf(df, out_dir)


if __name__ == "__main__":
    main()

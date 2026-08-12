"""Generate the benchmark figures from aggregated result tables.

Reads ``rows.csv`` and ``aggregate.csv`` from ``--input`` (default
``results/summary``) and writes every figure as PDF into
``--output`` (default ``plots``).  All content comes from the result
files; nothing is hand-entered.  Figures whose data is absent are skipped
with a message (never fabricated), and lines are broken -- not connected --
across failed or skipped points.

Usage::

    python scripts/make_plots.py --input results/summary --output plots
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

# Colorblind-safe categorical palette (validated, fixed order; color follows
# the method entity and is never reassigned when a series is filtered out).
METHOD_COLORS = {
    "tree_socu": "#2a78d6",
    "cudss": "#eb6834",
    "cpu_structured": "#1baf7a",
    "tree_graph": "#e34948",
    "cpu_sparse": "#008300",
}
PHASE_GROUPS = {
    "overhead": ["phase_refresh_ms", "phase_solve_stage_ms",
                 "phase_rhs_stage_ms"],
    "tail factor": ["phase_tail_factor_ms", "phase_tail_factor_forward_c_ms"],
    "tail solves": ["phase_tail_solve_coupling_ms", "phase_tail_forward_c_ms",
                    "phase_tail_solve_rhs_ms", "phase_tail_forward_rhs_ms",
                    "phase_tail_backward_rhs_ms"],
    "schur": ["phase_schur_products_ms", "phase_schur_branch_syrk_ms",
              "phase_branch_reduction_ms", "phase_schur_branch_reduction_ms",
              "phase_schur_rhs_ms"],
    "root": ["phase_root_factor_ms", "phase_root_potrf_ms",
             "phase_root_solve_ms", "phase_root_forward_backward_ms"],
    "recovery": ["phase_recovery_ms", "phase_recovery_update_ms"],
}
PHASE_COLORS = ["#c3c2b7", "#2a78d6", "#1baf7a", "#eb6834", "#eda100",
                "#e87ba4"]

SWEEP_PARAMS = {
    "branches": ("num_branches", "number of branches B"),
    "horizon": ("horizon", "horizon N"),
    "block_size": ("block_size", "stage block size n"),
    "separator": ("separator_dim", "separator dimension m"),
}


def _style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "lines.linewidth": 2.0,
        "font.size": 10,
        "legend.frameon": False,
        "savefig.bbox": "tight",
    })


def _save(fig, out_dir, name):
    fig.savefig(out_dir / f"{name}.pdf")
    plt.close(fig)
    print(f"  wrote {name}.pdf")


def _method_stats(df, sweep, param, value_col="warm_total_ms"):
    """Per-method median and p10/p90 across seeds for each param value."""
    d = df[(df["sweep"] == sweep)]
    ok = d[d["status"] == "ok"]
    out = {}
    for method, g in ok.groupby("method"):
        s = g.groupby(param)[value_col]
        out[method] = pd.DataFrame({
            "median": s.median(),
            "lo": s.quantile(0.10),
            "hi": s.quantile(0.90),
        }).sort_index()
    bad = d[d["status"] != "ok"]
    return out, bad


def _plot_sweep_lines(ax, stats, values, ylabel):
    for method, tab in stats.items():
        tab = tab.reindex(values)  # NaN gaps break the line at missing points
        color = METHOD_COLORS.get(method, "#52514e")
        ax.plot(values, tab["median"], marker="o", markersize=4,
                color=color, label=method)
        ax.fill_between(values, tab["lo"], tab["hi"], color=color, alpha=0.15,
                        linewidth=0)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)


def fig_sweep(df, agg, sweep, out_dir):
    if sweep not in set(df["sweep"]):
        print(f"  skip fig_{sweep}: no data")
        return
    param, xlabel = SWEEP_PARAMS[sweep]
    stats, bad = _method_stats(df, sweep, param)
    if not stats:
        print(f"  skip fig_{sweep}: no ok rows")
        return
    values = sorted(set().union(*(set(t.index) for t in stats.values())))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 6), sharex=True,
                                   height_ratios=[2, 1])
    _plot_sweep_lines(ax1, stats, values, "warm factor+solve [ms]")
    if len(bad):
        for _, r in bad.iterrows():
            ax1.plot(r[param], ax1.get_ylim()[0], marker="x", color="#52514e",
                     clip_on=False)
        ax1.plot([], [], marker="x", linestyle="none", color="#52514e",
                 label="failed/skipped")
        ax1.legend(fontsize=8)
    ax1.set_title(f"warm time and speedup vs {xlabel}")

    a = agg[(agg["sweep"] == sweep) & (agg["method"] == "tree_socu")]
    a = a.dropna(subset=["speedup_total"]).sort_values(param)
    if len(a):
        ax2.plot(a[param], a["speedup_total"], marker="o", markersize=4,
                 color=METHOD_COLORS["tree_socu"])
        ax2.axhline(1.0, color="#52514e", linewidth=1, linestyle="--")
        ax2.set_xscale("log", base=2)
        ax2.set_yscale("log")
        ax2.set_ylabel("speedup vs cuDSS")
    ax2.set_xlabel(xlabel)
    _save(fig, out_dir, f"fig_sweep_{sweep}")


def fig_phase_breakdown(df, out_dir):
    d = df[(df["method"] == "tree_socu") & (df["status"] == "ok")
           & df["sweep"].isin(["branches", "total_scaling", "default_point"])]
    if not len(d):
        print("  skip fig_phase_breakdown: no data")
        return
    d = d.sort_values("dimension")
    # representative small / middle / large cases
    rows = d.iloc[[0, len(d) // 2, len(d) - 1]] if len(d) >= 3 else d
    labels = [f"B={int(r.num_branches)}\nN={int(r.horizon)}"
              for r in rows.itertuples()]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    x = np.arange(len(rows))
    stacks = []
    for group, cols in PHASE_GROUPS.items():
        vals = np.zeros(len(rows))
        for c in cols:
            if c in rows:
                vals += rows[c].fillna(0.0).to_numpy()
        stacks.append(vals)
    totals = np.sum(stacks, axis=0)
    bottom = np.zeros(len(rows))
    for vals, (group, _), color in zip(stacks, PHASE_GROUPS.items(),
                                       PHASE_COLORS):
        frac = 100.0 * vals / np.maximum(totals, 1e-30)
        ax.bar(x, frac, bottom=bottom, color=color, label=group, width=0.6,
               edgecolor="white", linewidth=1)
        bottom += frac
    for xi, tot in zip(x, totals):
        ax.text(xi, 101, f"{tot:.2f} ms", ha="center", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 112)
    ax.set_ylabel("share of factor+solve time [%]")
    ax.set_title("phase breakdown (tree_socu)")
    ax.legend(fontsize=8, ncol=2)
    _save(fig, out_dir, "fig_phase_breakdown")


def fig_memory(df, out_dir):
    d = df[(df["status"] == "ok") & df["peak_device_bytes"].notna()
           & (df["peak_device_bytes"] > 0)]
    if not len(d):
        print("  skip fig_memory: no data")
        return
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for method, g in d.groupby("method"):
        if method not in ("tree_socu", "cudss"):
            continue
        g = g.groupby("dimension")["peak_device_bytes"].median()
        ax.plot(g.index, g.to_numpy() / 1e9, marker="o", markersize=4,
                color=METHOD_COLORS.get(method), label=method)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("total dimension"); ax.set_ylabel("peak device memory [GB]")
    ax.set_title("peak GPU memory vs problem size")
    ax.legend(fontsize=8)
    _save(fig, out_dir, "fig_memory")


def fig_accuracy_vs_size(df, out_dir):
    d = df[(df["status"] == "ok")
           & df["sweep"].isin(["branches", "horizon", "total_scaling"])]
    if not len(d):
        print("  skip fig_accuracy_vs_size: no data")
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
    for ax, col, label in ((axes[0], "scaled_residual", "scaled residual"),
                           (axes[1], "forward_error", "forward error")):
        for method, g in d.groupby("method"):
            if method not in ("tree_socu", "cudss", "cpu_structured"):
                continue
            g = g.groupby("dimension")[col].median()
            ax.plot(g.index, g, marker="o", markersize=4,
                    color=METHOD_COLORS.get(method), label=method)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("total dimension"); ax.set_ylabel(label)
    axes[0].legend(fontsize=8)
    fig.suptitle("accuracy vs problem size (FP64)")
    _save(fig, out_dir, "fig_accuracy_vs_size")


def fig_error_vs_condition(df, out_dir):
    d = df[(df["status"] == "ok") & df["kappa_estimate"].notna()
           & df["sweep"].isin(["conditioning", "fp32_subset"])]
    if not len(d):
        print("  skip fig_error_vs_condition: no data")
        return
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for (method, prec), g in d.groupby(["method", "precision"]):
        if method not in ("tree_socu", "cudss"):
            continue
        marker = "o" if prec == "float64" else "^"
        ax.scatter(g["kappa_estimate"], g["forward_error"], s=25,
                   marker=marker, color=METHOD_COLORS.get(method),
                   label=f"{method} ({prec})", alpha=0.8)
    for eps, prec in ((2.2e-16, "float64"), (1.2e-7, "float32")):
        ks = np.array(sorted(d["kappa_estimate"]))
        ax.plot(ks, eps * ks, linestyle=":", color="#52514e", linewidth=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("estimated condition number")
    ax.set_ylabel("forward error")
    ax.set_title("error vs conditioning (dotted: kappa * eps)")
    ax.legend(fontsize=7)
    _save(fig, out_dir, "fig_error_vs_condition")


def fig_ablations(df, out_dir):
    sweeps = [s for s in df["sweep"].unique() if s.startswith("ablation_")]
    sweeps = [s for s in sweeps if len(df[(df["sweep"] == s)
                                          & (df["status"] == "ok")])]
    if not sweeps:
        print("  skip fig_ablations: no data")
        return
    ncols = min(3, len(sweeps))
    nrows = (len(sweeps) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.4 * nrows),
                             squeeze=False)
    for ax, sweep in zip(axes.ravel(), sweeps):
        d = df[(df["sweep"] == sweep) & (df["status"] == "ok")]
        d = d.copy()
        d["case"] = [f"B={int(b)},N={int(h)}" for b, h in
                     zip(d["num_branches"], d["horizon"])]
        if sweep == "ablation_coupling_pattern":
            d["case"] = d["coupling_pattern"]
        elif sweep == "ablation_generator":
            d["case"] = d["generator_mode"]
        tab = d.groupby(["case", "method"])["warm_total_ms"].median().unstack()
        tab = tab.sort_index()
        x = np.arange(len(tab))
        width = 0.8 / max(len(tab.columns), 1)
        for j, method in enumerate(tab.columns):
            ax.bar(x + j * width, tab[method], width * 0.9,
                   color=METHOD_COLORS.get(method, "#52514e"), label=method)
        ax.set_xticks(x + 0.4 - width / 2, tab.index, fontsize=7, rotation=20)
        ax.set_yscale("log")
        ax.set_ylabel("warm total [ms]", fontsize=8)
        ax.set_title(sweep.replace("ablation_", ""), fontsize=9)
        ax.legend(fontsize=6)
    for ax in axes.ravel()[len(sweeps):]:
        ax.axis("off")
    fig.suptitle("ablations")
    fig.tight_layout()
    _save(fig, out_dir, "fig_ablations")


def fig_heatmap(agg, out_dir):
    d = agg[(agg["sweep"] == "grid_BN") & (agg["method"] == "tree_socu")]
    d = d.dropna(subset=["speedup_total"])
    if not len(d):
        print("  skip fig_heatmap: no data")
        return
    tab = d.pivot_table(index="horizon", columns="num_branches",
                        values="speedup_total")
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    data = np.log10(tab.to_numpy(dtype=float))
    lim = np.nanmax(np.abs(data)) or 1.0
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "div", ["#eb6834", "#f0efeb", "#2a78d6"])  # <1 orange, >1 blue
    im = ax.imshow(data, cmap=cmap, vmin=-lim, vmax=lim, origin="lower",
                   aspect="auto")
    ax.set_xticks(range(len(tab.columns)), [int(c) for c in tab.columns])
    ax.set_yticks(range(len(tab.index)), [int(r) for r in tab.index])
    ax.set_xlabel("branches B"); ax.set_ylabel("horizon N")
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = tab.to_numpy()[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="#0b0b0b")
    ax.set_title("speedup of tree_socu over cuDSS (warm total)")
    fig.colorbar(im, ax=ax, label="log10 speedup")
    ax.grid(False)
    _save(fig, out_dir, "fig_heatmap_BN")


def fig_multi_rhs(df, out_dir):
    d = df[(df["sweep"] == "multi_rhs") & (df["status"] == "ok")]
    if not len(d):
        print("  skip fig_multi_rhs: no data")
        return
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for method, g in d.groupby("method"):
        s = g.groupby("num_rhs")[["warm_factor_median", "warm_solve_median"]].median()
        per_rhs = (s["warm_factor_median"] + s["warm_solve_median"]) / s.index
        ax.plot(s.index, per_rhs, marker="o", markersize=4,
                color=METHOD_COLORS.get(method), label=method)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("number of right-hand sides")
    ax.set_ylabel("amortized cost per RHS [ms]")
    ax.set_title("multi-RHS amortization (one factorization)")
    ax.legend(fontsize=8)
    _save(fig, out_dir, "fig_multi_rhs")


def fig_cold_warm(df, out_dir):
    d = df[(df["status"] == "ok") & df["cold_first_factor_s"].notna()]
    # default-sized case per method
    d = d[(d["num_branches"] == 64) & (d["horizon"] == 128)] if len(
        d[(d["num_branches"] == 64) & (d["horizon"] == 128)]) else d
    if not len(d):
        print("  skip fig_cold_warm: no data")
        return
    methods = [m for m in ("tree_socu", "cudss") if m in set(d["method"])]
    parts = ["cold_alloc_s", "cold_transfer_s", "cold_analyze_s",
             "cold_first_factor_s", "cold_first_solve_s"]
    labels = ["alloc", "transfer", "analyze", "first factor", "first solve"]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    x = np.arange(len(methods))
    bottom = np.zeros(len(methods))
    for part, label, color in zip(parts, labels, PHASE_COLORS):
        vals = np.array([d[d["method"] == m][part].median() * 1e3
                         for m in methods])
        vals = np.nan_to_num(vals)
        ax.bar(x - 0.2, vals, 0.35, bottom=bottom, color=color, label=label,
               edgecolor="white", linewidth=1)
        bottom += vals
    warm = np.array([d[d["method"] == m]["warm_total_ms"].median()
                     for m in methods])
    ax.bar(x + 0.2, warm, 0.35, color="#52514e", label="warm total")
    ax.set_xticks(x, methods)
    ax.set_yscale("log")
    ax.set_ylabel("time [ms]")
    ax.set_title("cold (incl. JIT/analysis) vs warm execution")
    ax.legend(fontsize=8)
    _save(fig, out_dir, "fig_cold_warm")


def fig_crossover(agg, out_dir):
    rows = []
    for sweep, (param, label) in SWEEP_PARAMS.items():
        d = agg[(agg["sweep"] == sweep) & (agg["method"] == "tree_socu")]
        d = d.dropna(subset=["speedup_total"]).sort_values(param)
        if not len(d):
            continue
        win = d[d["speedup_total"] >= 1.0]
        rows.append({
            "sweep": sweep, "parameter": label,
            "first_winning_value": (int(win.iloc[0][param]) if len(win)
                                    else "none"),
            "max_speedup": round(float(d["speedup_total"].max()), 2),
        })
    if not rows:
        print("  skip fig_crossover: no data")
        return
    tab = pd.DataFrame(rows)
    tab.to_csv(out_dir / "crossover_table.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 0.5 + 0.4 * len(tab)))
    ax.axis("off")
    table = ax.table(cellText=tab.values, colLabels=tab.columns,
                     loc="center", cellLoc="center")
    table.scale(1, 1.4)
    ax.set_title("crossover points: tree_socu first beats cuDSS")
    _save(fig, out_dir, "fig_crossover_table")


def fig_target_points(df, out_dir):
    """Stacked bars (factorization + solve, measured in separate
    repetition loops) at the application target points (B, N, n), FP64
    and FP32 panels."""
    points = [(11, 32, 8), (21, 64, 8), (41, 96, 8), (81, 128, 8)]
    labels = [f"({b},{n},{k})" for b, n, k in points]
    panels = [("target_points", "FP64", ["tree_socu", "tree_graph",
                                         "cudss"]),
              ("target_points_fp32", "FP32", ["tree_socu", "tree_graph",
                                              "cudss"])]
    if not (df["sweep"] == "target_points").any():
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=False)
    for ax, (sweep, title, methods) in zip(axes, panels):
        ok = df[(df["sweep"] == sweep) & (df["status"] == "ok")]
        width = 0.8 / len(methods)
        for mi, method in enumerate(methods):
            g = ok[ok["method"] == method]
            fac, sol = [], []
            for b, n, k in points:
                p = g[(g["num_branches"] == b) & (g["horizon"] == n)
                      & (g["block_size"] == k)]
                fac.append(p["warm_factor_median"].median())
                sol.append(p["warm_solve_median"].median())
            x = [i + (mi - (len(methods) - 1) / 2) * width
                 for i in range(len(points))]
            color = METHOD_COLORS.get(method, "#52514e")
            ax.bar(x, fac, width=width, color=color,
                   label=method if ax is axes[0] or True else None)
            ax.bar(x, sol, width=width, bottom=fac, color=color, alpha=0.45,
                   hatch="//", linewidth=0)
        ax.set_xticks(range(len(points)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel("(B, N, n), m = 64")
        ax.set_ylabel("warm time [ms]")
        ax.set_title(f"{title} (solid: factorization, hatched: solve)",
                     fontsize=9)
        ax.legend(fontsize=8)
    fig.suptitle("Application target points: structured solver vs cuDSS",
                 fontsize=11)
    _save(fig, out_dir, "fig_target_points")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="results/summary")
    p.add_argument("--output", default="plots")
    args = p.parse_args()
    in_dir, out_dir = Path(args.input), Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    _style()

    df = pd.read_csv(in_dir / "rows.csv")
    agg = pd.read_csv(in_dir / "aggregate.csv")
    print(f"plotting from {len(df)} rows")

    for sweep in SWEEP_PARAMS:
        fig_sweep(df, agg, sweep, out_dir)
    fig_phase_breakdown(df, out_dir)
    fig_memory(df, out_dir)
    fig_accuracy_vs_size(df, out_dir)
    fig_error_vs_condition(df, out_dir)
    fig_ablations(df, out_dir)
    fig_heatmap(agg, out_dir)
    fig_multi_rhs(df, out_dir)
    fig_cold_warm(df, out_dir)
    fig_crossover(agg, out_dir)
    fig_target_points(df, out_dir)
    print(f"figures in {out_dir}")


if __name__ == "__main__":
    main()

"""Plot the end-to-end ADMM benchmark (tree vs cuDSS vs OSQP).

Reads the JSON written by ``benchmarks/admm.py --json`` and produces
three PDF panels comparing the two GPU linear-system implementations
and the OSQP CPU baseline across the tail-count sweep:

1. isolated linear-system kernel medians (factorize and prepared solve);
2. warm ADMM time per iteration (state-restored fixed-iteration median;
   the captured-graph path when the linear system supports capture);
3. MPC-style step time (matrix-value update + warm solve), when the
   benchmark was run with ``--mpc``.

With ``--campaign`` pointing at the directory written by
``scripts/admm2_campaign.py`` it additionally renders the warm-phase
breakdown, the convergence-check period cost, and the over-relaxation
comparison.

Usage::

    python scripts/make_admm_plots.py [--input results/admm/admm_bench.json]
                                      [--campaign results/admm2]
                                      [--output plots/admm]
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"tree": "#2a78d6", "cudss": "#eb6834", "osqp": "#6a6a6a"}
LABELS = {"tree": "tree solver (ours)", "cudss": "cuDSS",
          "osqp": "OSQP (CPU)"}


def _style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9,
        "axes.titlesize": 9, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#dddad1",
        "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False,
    })


def _series(records, impl, key):
    pts = [(r["B"], r[key]) for r in records
           if r["impl"] == impl and key in r and r[key] is not None]
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]


def _save(fig, out_dir, name):
    path = out_dir / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


def fig_linsys_kernels(records, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
    for ax, key, title in ((axes[0], "factorize_ms", "factorization"),
                           (axes[1], "linear_solve_ms",
                            "prepared triangular solve")):
        for impl in ("tree", "cudss"):
            B, y = _series(records, impl, key)
            if B:
                ax.plot(B, y, "o-", ms=4, lw=1.4, color=COLORS[impl],
                        label=LABELS[impl])
        ax.set_title(title)
        ax.set_xlabel("tails B")
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted({r["B"] for r in records}))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axes[0].set_ylabel("median time [ms]")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(loc="upper left")
    fig.suptitle("Linear-system kernels inside ADMM (FP64, medians)", y=1.03)
    _save(fig, out_dir, "admm_linsys_kernels")


def _warm_iter_ms(rec):
    """Warm fixed-iteration median in ms (graph path when captured)."""
    ph = rec.get("phases")
    if not ph:
        return None
    us = ph.get("graph_us_per_iter") or ph.get("total_us_per_iter")
    return us * 1e-3 if us is not None else None


def fig_warm_iteration(records, out_dir):
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for impl in ("tree", "cudss"):
        rows = sorted((r["B"], _warm_iter_ms(r))
                      for r in records if r["impl"] == impl
                      and _warm_iter_ms(r) is not None)
        if rows:
            ax.plot([b for b, _ in rows], [v for _, v in rows], "o-",
                    ms=4, lw=1.4, color=COLORS[impl], label=LABELS[impl])
    B, y = _series(records, "osqp", "per_iter_ms")
    if B:
        ax.plot(B, y, "s--", ms=4, lw=1.2, color=COLORS["osqp"],
                label=LABELS["osqp"])
    ax.set_xlabel("tails B")
    ax.set_ylabel("time per ADMM iteration [ms]")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({r["B"] for r in records}))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    ax.set_title("Warm ADMM iteration time (fixed-iteration median)")
    _save(fig, out_dir, "admm_warm_iteration")


def fig_mpc_step(records, out_dir):
    rows = [r for r in records if "mpc" in r]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for impl in ("tree", "cudss"):
        pts = sorted((r["B"], r["mpc"]["step_median_s"] * 1e3)
                     for r in rows if r["impl"] == impl)
        if pts:
            ax.plot([b for b, _ in pts], [v for _, v in pts], "o-",
                    ms=4, lw=1.4, color=COLORS[impl], label=LABELS[impl])
    ax.set_xlabel("tails B")
    ax.set_ylabel("MPC step (update + solve) [ms]")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({r["B"] for r in rows}))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    ax.set_title("MPC-style step time (median)")
    _save(fig, out_dir, "admm_mpc_step")


def fig_phase_breakdown(matrix_records, out_dir):
    """Stacked warm-iteration phase medians of the tree path across the
    tail-count sweep.  Phases are timed with individual synchronization,
    so the stack shows relative weight; the pipelined per-iteration wall
    time (the marker) is lower than the stack height by construction."""
    rows = sorted((r["B"], r["phases"]) for r in matrix_records
                  if r.get("impl") == "tree" and r.get("phases")
                  and r.get("T") == 64 and r.get("nx") == 8)
    if not rows:
        return
    B = [b for b, _ in rows]
    xs = range(len(B))
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    phase_keys = (("rhs_ms", "RHS build", "#2a78d6"),
                  ("solve_ms", "tree solve", "#7db4ee"),
                  ("project_ms", "project + dual", "#eb6834"))
    bottom = [0.0] * len(B)
    for key, label, color in phase_keys:
        vals = [ph[key] * 1e3 for _, ph in rows]
        ax.bar(xs, vals, 0.55, bottom=bottom, label=label, color=color,
               edgecolor="white", linewidth=1)
        bottom = [b + v for b, v in zip(bottom, vals)]
    warm = [(ph.get("graph_us_per_iter") or ph["total_us_per_iter"])
            for _, ph in rows]
    ax.plot(xs, warm, "k_", ms=16, label="pipelined iteration")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([str(b) for b in B])
    ax.set_xlabel("tails B")
    ax.set_ylabel("time [us]")
    ax.set_title("Warm iteration phases (tree, T=64)")
    ax.legend(loc="upper left", fontsize=7)
    _save(fig, out_dir, "admm_phase_breakdown")


def fig_check_every(check_records, out_dir):
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    xs = range(len(check_records))
    ax.bar(xs, [r["us_per_iter"] for r in check_records], 0.5,
           color="#2a78d6", label="with checks")
    ax.bar(xs, [r["uncheckd_us_per_iter"] for r in check_records], 0.5,
           color="#7db4ee", label="iterations only")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([str(r["check_every"]) for r in check_records])
    ax.set_xlabel("check_every")
    ax.set_ylabel("amortized time per iteration [us]")
    ax.set_title("Convergence-check period cost")
    ax.legend(loc="lower left", fontsize=7)
    _save(fig, out_dir, "admm_check_every")


def fig_alpha(alpha_records, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.5))
    xs = [r["alpha"] for r in alpha_records]
    axes[0].plot(xs, [r["iterations"] for r in alpha_records], "o-",
                 color="#2a78d6")
    axes[0].set_ylabel("iterations to tolerance")
    axes[1].plot(xs, [r["warm_time_to_solution_ms"] for r in alpha_records],
                 "o-", color="#eb6834")
    axes[1].set_ylabel("warm time to solution [ms]")
    for ax in axes:
        ax.set_xlabel("over-relaxation alpha")
        ax.set_ylim(bottom=0)
    fig.suptitle("Over-relaxation on the headline case", y=1.03)
    _save(fig, out_dir, "admm_alpha")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="results/admm/admm_bench.json")
    p.add_argument("--campaign", default=None)
    p.add_argument("--output", default="plots/admm")
    args = p.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    _style()
    if Path(args.input).exists():
        records = json.loads(Path(args.input).read_text())
        fig_linsys_kernels(records, out_dir)
        fig_warm_iteration(records, out_dir)
        fig_mpc_step(records, out_dir)
    if args.campaign:
        cdir = Path(args.campaign)
        if (cdir / "matrix.json").exists():
            fig_phase_breakdown(
                json.loads((cdir / "matrix.json").read_text()), out_dir)
        if (cdir / "check_every.json").exists():
            fig_check_every(
                json.loads((cdir / "check_every.json").read_text()),
                out_dir)
        if (cdir / "alpha.json").exists():
            fig_alpha(json.loads((cdir / "alpha.json").read_text()),
                      out_dir)


if __name__ == "__main__":
    main()

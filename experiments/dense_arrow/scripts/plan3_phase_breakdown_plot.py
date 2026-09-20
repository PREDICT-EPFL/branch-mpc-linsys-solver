"""Publication phase-breakdown figure: stacked stage times of the tree
factorization and triangular solve at the B=50, N=128 heatmap configs
(n_b = n_r in {8, 16, 32}), with the matching cuDSS median time as a
reference marker.  Reads results/plan3_campaign/phase_breakdown.json
(written by scripts/plan3_phase_breakdown.py) and the campaign raw
records for the cuDSS references.  LaTeX text, PDF output."""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FACTOR_STAGES = [
    ("refresh", "refresh (copy)"),
    ("tail_factor", "tail POTRF + TRSM"),
    ("root_diagonal_update", "root SYRK"),
    ("root_diagonal_reduce", "root reduction"),
    ("root_factor", "root POTRF"),
]
SOLVE_STAGES = [
    ("tail_forward", "tail TRSV (forward)"),
    ("root_rhs_update", "root GEMV"),
    ("root_solve", "root TRSV"),
    ("tail_rhs_correction", "tail GEMV"),
    ("tail_backward", "tail TRSV (backward)"),
]
STAGE_COLORS = ["#c3c2b7", "#2a78d6", "#eb6834", "#eda100", "#1baf7a"]

RC = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.grid": True,
    "axes.grid.axis": "x",
    "grid.color": "#dddad1",
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
}


def cudss_reference(raw_dir, key, phase):
    """Median cuDSS time (us) over seeds for one breakdown config,
    from the campaign-A raw records."""
    vals = []
    for f in Path(raw_dir).glob(f"A_{key}_*.json"):
        r = json.loads(f.read_text())
        d = r["backends"].get("cudss", {})
        if phase in d:
            vals.append(d[phase]["median_ms"] * 1000.0)
    return float(np.median(vals)) if vals else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",
                   default="results/plan3_campaign/phase_breakdown.json")
    p.add_argument("--raw", default="results/plan3_campaign/raw")
    p.add_argument("--output", default="plots")
    args = p.parse_args()
    data = json.loads(Path(args.input).read_text())
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = list(data.keys())          # e.g. B50_T128_n8_m8_q1

    def ylabel(key):
        nb = key.split("_n")[1].split("_")[0]
        return rf"$n_b = n_r = {nb}$"

    def raw_key(key):
        return key.rsplit("_q", 1)[0]  # A_<raw_key>_* matches records

    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, 1, figsize=(6.6, 3.6))
        for ax, stages, phase, title in (
                (axes[0], FACTOR_STAGES, "factor", "Factorization"),
                (axes[1], SOLVE_STAGES, "solve", "Triangular solve")):
            totals, refs = {}, {}
            for yi, key in enumerate(keys):
                left = 0.0
                for si, (stage, label) in enumerate(stages):
                    v = data[key].get(stage, 0.0)
                    ax.barh(yi, v, left=left, height=0.62,
                            color=STAGE_COLORS[si % len(STAGE_COLORS)],
                            label=label if yi == 0 else None)
                    left += v
                totals[yi] = left
                ref = cudss_reference(args.raw, raw_key(key), phase)
                refs[yi] = ref
                if ref is not None:
                    ax.plot([ref], [yi], marker="D", markersize=5,
                            color="black", zorder=5,
                            label="cuDSS total" if yi == 0 else None)
            # total labels: right of the bar, or above its end when the
            # cuDSS marker sits nearby
            span = max(max(totals.values()),
                       max(r for r in refs.values() if r is not None))
            for yi, left in totals.items():
                ref = refs[yi]
                near = ref is not None and abs(ref - left) < 0.13 * span
                if near:
                    ax.text(left, yi + 0.40, rf"{left:.0f}\,$\mu$s",
                            ha="center", va="bottom", fontsize=8,
                            color="#52514e")
                else:
                    ax.text(left + 0.012 * span, yi,
                            rf"{left:.0f}\,$\mu$s", ha="left",
                            va="center", fontsize=8, color="#52514e")
            ax.set_yticks(range(len(keys)), [ylabel(k) for k in keys])
            ax.set_xlim(0, None)
            ax.set_title(title, fontsize=10, loc="left")
            ax.tick_params(length=0)
            if ax is axes[0]:
                ax.legend(ncol=3, loc="lower right",
                          bbox_to_anchor=(1.0, 0.98), borderaxespad=0.0)
            else:
                ax.legend(ncol=3, loc="lower right",
                          bbox_to_anchor=(1.0, 0.98), borderaxespad=0.0)
                ax.set_xlabel(r"stage time [$\mu$s]")
        fig.tight_layout(h_pad=2.2)
        fig.savefig(out_dir / "fig_phase_breakdown.pdf")
    print(f"wrote {out_dir / 'fig_phase_breakdown.pdf'}")


if __name__ == "__main__":
    main()

"""Plan-4 paired tree-versus-cuDSS evidence at the five frozen
representative points (three seeds each), using the campaign
measurement protocol (identical device values, cuDSS analysis outside
timing, separate factor/solve loops, batched CUDA events).

Writes one JSON with every record to results/plan4/cudss_pairs.json.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan3_campaign import run_point  # noqa: E402

POINTS = [(11, 16, 8, 2, 1), (41, 64, 8, 2, 1), (81, 128, 8, 2, 1),
          (64, 128, 16, 64, 1), (41, 64, 8, 2, 16)]
SEEDS = [0, 1, 2]


def main():
    import warp as wp
    wp.init()
    records = []
    for B, T, n_b, n_r, nrhs in POINTS:
        point = {"B": B, "T": T, "n_b": n_b, "n_r": n_r, "nrhs": nrhs,
                 "precision": "float64"}
        for seed in SEEDS:
            rec = run_point("plan4", point, seed)
            records.append(rec)
            t = rec["backends"]["tree"]
            c = rec["backends"]["cudss"]
            print(f"B{B} T{T} n{n_b} m{n_r} q{nrhs} s{seed}: "
                  f"tree f={t['factor']['median_ms']:.4f} "
                  f"s={t['solve']['median_ms']:.4f} | cudss "
                  f"f={c['factor']['median_ms']:.4f} "
                  f"s={c['solve']['median_ms']:.4f} | "
                  f"res tree={t['relative_residual']:.2e} "
                  f"cudss={c['relative_residual']:.2e}")
    out = Path("results/plan4/cudss_pairs.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=1))
    print(f"wrote {out} ({len(records)} records)")


if __name__ == "__main__":
    main()

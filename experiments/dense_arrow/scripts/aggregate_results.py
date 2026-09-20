"""Aggregate raw benchmark records into flat tables.

Reads every ``*.json`` case record under ``--input`` (default
``results/raw``) and writes to ``--output`` (default ``results/summary``):

- ``rows.jsonl``   : one flattened JSON object per case (raw repetition
  samples preserved);
- ``rows.csv``     : the same rows as a flat CSV (samples excluded);
- ``aggregate.csv``: medians across seeds per (sweep, method, point), with
  speedup over cuDSS where the cuDSS twin ran on the same instances;
- ``aggregate_log.txt`` : counts of ok/failed/skipped inputs.

Usage::

    python scripts/aggregate_results.py --input results/raw --output results/summary
"""

import argparse
import json
from pathlib import Path

import pandas as pd

_POINT_KEYS = ["num_tails", "horizon", "block_size", "root_dim",
               "num_rhs", "precision", "coupling_pattern", "condition_target",
               "rho_tail", "rho_sep", "generator_mode", "band_width"]


def flatten_record(rec: dict) -> dict:
    row = {
        "case_id": rec.get("case_id"),
        "sweep": rec.get("sweep"),
        "method": rec.get("method"),
        "seed": rec.get("seed"),
        "status": rec.get("status"),
        "skip_reason": rec.get("skip_reason") or rec.get("error"),
        "timestamp": rec.get("timestamp"),
        "git_commit": rec.get("git_commit"),
        "git_dirty": rec.get("git_dirty"),
        "dimension": rec.get("dimension"),
        "nnz_lower": rec.get("nnz_lower", rec.get("nnz_lower_structural")),
        "generation_s": rec.get("generation_s"),
        "gamma": rec.get("gamma"),
        "kappa_estimate": rec.get("kappa_estimate"),
        "kappa_method": rec.get("kappa_method"),
        "warm_total_ms": rec.get("warm_total_ms"),
        "warm_overhead_ms": rec.get("warm_overhead_ms"),
        "jit_s": rec.get("jit_s"),
        "peak_device_bytes": rec.get("peak_device_bytes"),
        "workspace_bytes": rec.get("workspace_bytes"),
        "min_pivot": rec.get("min_pivot"),
        "scaled_residual": rec.get("scaled_residual"),
        "rhs_relative_residual": rec.get("rhs_relative_residual"),
        "forward_error": rec.get("forward_error"),
        "max_componentwise_backward_error":
            rec.get("max_componentwise_backward_error"),
        "nan_or_inf": rec.get("nan_or_inf"),
        "raw_factor_ms": rec.get("raw_factor_ms"),
        "raw_solve_ms": rec.get("raw_solve_ms"),
    }
    for k in _POINT_KEYS:
        row[k] = (rec.get("spec") or {}).get(k)
    for group in ("warm_factor", "warm_solve"):
        for stat, v in (rec.get(group) or {}).items():
            row[f"{group}_{stat}"] = v
    for phase, v in (rec.get("phase_ms") or {}).items():
        row[f"phase_{phase}_ms"] = v
    for k, v in rec.items():
        if k.startswith("cold_") or k in ("host_assembly_s", "cpu_solve_s"):
            row[k] = v
    return row


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Medians across seeds per (sweep, method, point) plus speedups."""
    ok = df[df["status"] == "ok"].copy()
    keys = ["sweep", "method"] + _POINT_KEYS
    num_cols = ["warm_total_ms", "warm_factor_median", "warm_solve_median",
                "peak_device_bytes", "scaled_residual", "forward_error",
                "rhs_relative_residual", "jit_s", "kappa_estimate", "gamma",
                "dimension", "nnz_lower"]
    num_cols = [c for c in num_cols if c in ok.columns]
    agg = (ok.groupby(keys, dropna=False)[num_cols]
             .median(numeric_only=True).reset_index())
    counts = (ok.groupby(keys, dropna=False)["seed"].count()
                .rename("num_seeds").reset_index())
    agg = agg.merge(counts, on=keys)

    # speedup over cuDSS on identical instances (matched per seed)
    point_keys = ["sweep", "seed"] + _POINT_KEYS
    base = df[(df["method"] == "cudss") & (df["status"] == "ok")]
    base = base[point_keys + ["warm_total_ms", "warm_factor_median",
                              "warm_solve_median"]]
    base = base.rename(columns={
        "warm_total_ms": "cudss_total_ms",
        "warm_factor_median": "cudss_factor_ms",
        "warm_solve_median": "cudss_solve_ms"})
    joined = df[df["status"] == "ok"].merge(base, on=point_keys, how="left")
    for tgt, src in (("speedup_total", "cudss_total_ms"),
                     ("speedup_factor", "cudss_factor_ms"),
                     ("speedup_solve", "cudss_solve_ms")):
        col = {"speedup_total": "warm_total_ms",
               "speedup_factor": "warm_factor_median",
               "speedup_solve": "warm_solve_median"}[tgt]
        joined[tgt] = joined[src] / joined[col]
    sp = (joined.groupby(["sweep", "method"] + _POINT_KEYS, dropna=False)
          [["speedup_total", "speedup_factor", "speedup_solve"]]
          .median().reset_index())
    return agg.merge(sp, on=keys, how="left")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="results/raw")
    p.add_argument("--output", default="results/summary")
    args = p.parse_args()

    in_dir, out_dir = Path(args.input), Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, log = [], []
    for f in sorted(in_dir.glob("*.json")):
        if f.name == "run_metadata.json":
            continue
        try:
            rows.append(flatten_record(json.loads(f.read_text())))
        except Exception as exc:  # noqa: BLE001
            log.append(f"unreadable {f.name}: {exc}")
    if not rows:
        raise SystemExit(f"no case records found in {in_dir}")

    with open(out_dir / "rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")

    df = pd.DataFrame(rows)
    df.drop(columns=["raw_factor_ms", "raw_solve_ms"]).to_csv(
        out_dir / "rows.csv", index=False)
    agg = aggregate(df)
    agg.to_csv(out_dir / "aggregate.csv", index=False)

    for status, cnt in df["status"].value_counts().items():
        log.append(f"{status}: {cnt}")
    (out_dir / "aggregate_log.txt").write_text("\n".join(log) + "\n")
    print(f"aggregated {len(df)} rows -> {out_dir}")
    print("\n".join(log))


if __name__ == "__main__":
    main()

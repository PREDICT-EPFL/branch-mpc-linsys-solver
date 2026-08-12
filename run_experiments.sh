#!/usr/bin/env bash
#
# Run a benchmark configuration end to end: benchmarks -> aggregation ->
# figures (PDF). Resumable: re-running skips completed cases.
#
# Usage:
#   ./run_experiments.sh                 # smoke config (a few minutes)
#   ./run_experiments.sh paper           # full paper sweep (long)
#   ./run_experiments.sh correctness     # runner-level correctness suite
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# ---- conda env -------------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-socu}"
# shellcheck disable=SC1091
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || \
  source /home/fenglong/miniforge3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

CONFIG="${1:-smoke}"
DEVICE="${DEVICE:-cuda:0}"

echo "=== benchmarks (experiments/${CONFIG}.yaml, device ${DEVICE}) ==="
python scripts/run_benchmarks.py --config "experiments/${CONFIG}.yaml" \
  --device "$DEVICE" --resume

echo "=== aggregation ==="
python scripts/aggregate_results.py --input results/raw --output results/summary

# ---- figures (always last; saved as PDF) -----------------------------------
echo "=== figures ==="
python scripts/make_plots.py --input results/summary --output plots

echo "Done. Raw records in results/raw, tables in results/summary, figures in plots/."

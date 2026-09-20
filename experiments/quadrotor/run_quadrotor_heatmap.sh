#!/usr/bin/env bash
# Quadrotor endpoint-solver heatmap: proposed vs cuDSS, CHOLMOD, PARDISO.
#
# CPU baselines run on the 8 performance cores only (CPUs 0-7 on the
# Core Ultra 9 285K; the 16 efficiency cores are CPUs 8-23), with the
# MKL, OpenMP and OpenBLAS thread counts all set to 8 so nothing
# oversubscribes the bound CPUs.  The GPU solvers are unaffected by the
# binding.  Pass extra benchmark arguments through, e.g.
#   ./run_quadrotor_heatmap.sh --tails 20 100 200 --horizons 30 --output results/slice.csv
# The benchmark draws the speedup heat maps (PDF) at the end unless
# --no-plots is given.
set -euo pipefail
cd "$(dirname "$0")"

PCORES="${PCORES:-0-7}"
THREADS="${THREADS:-8}"
PY="${PY:-$HOME/miniforge3/envs/socu/bin/python}"

export MKL_THREADING_LAYER=GNU
export MKL_NUM_THREADS="$THREADS" OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
# keep the runtimes from silently lowering the thread count mid-run
export MKL_DYNAMIC=FALSE OMP_DYNAMIC=FALSE
export JAX_PLATFORMS=cpu MPLBACKEND=Agg

echo "binding CPU work to cores $PCORES with $THREADS threads"
exec taskset -c "$PCORES" "$PY" quadrotor_endpoint_benchmark.py "$@"

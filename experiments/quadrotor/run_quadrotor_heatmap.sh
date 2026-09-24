#!/usr/bin/env bash
# Quadrotor endpoint-solver heatmap: proposed vs cuDSS, CHOLMOD, PARDISO.
#
# CPU baselines run on the 8 performance cores only (CPUs 0-7 on the
# Core Ultra 9 285K; the 16 efficiency cores are CPUs 8-23), with the
# MKL and OpenMP thread counts set to 8 and CHOLMOD's BLAS single
# threaded, so nothing oversubscribes the bound CPUs.  The GPU solvers
# are unaffected by the binding.  Pass extra benchmark arguments
# through, e.g.
#   ./run_quadrotor_heatmap.sh --tails 20 100 200 --horizons 30 --output results/slice.csv
# The benchmark draws the speedup heat maps (PDF) at the end unless
# --no-plots is given.  Thermal protocol (burn-in, per-point re-warm,
# blocked interleaving, random grid order, telemetry): see the module
# docstring of quadrotor_endpoint_benchmark.py.
set -euo pipefail
cd "$(dirname "$0")"

PCORES="${PCORES:-0-7}"
THREADS="${THREADS:-8}"
PY="${PY:-$HOME/miniforge3/envs/socu/bin/python}"

export MKL_THREADING_LAYER=GNU
export MKL_NUM_THREADS="$THREADS" OMP_NUM_THREADS="$THREADS"
# CHOLMOD is only the untimed reference solution, but it is loaded in
# the same process: with a threaded BLAS its idle OpenBLAS workers
# would spin on the same cores while PARDISO is being timed and inflate
# PARDISO's factorization several-fold (measured), so its BLAS is
# single threaded.
export OPENBLAS_NUM_THREADS=1
# keep the runtimes from silently lowering the thread count mid-run
export MKL_DYNAMIC=FALSE
export OMP_DYNAMIC=FALSE
export OMP_WAIT_POLICY=ACTIVE
export JAX_PLATFORMS=cpu MPLBACKEND=Agg

echo "binding CPU work to cores $PCORES with $THREADS threads"
exec taskset -c "$PCORES" "$PY" quadrotor_endpoint_benchmark.py "$@"

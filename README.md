# tree-socu: GPU One-Level Scenario-Tree Cholesky Solver

A Python/Warp research prototype and benchmark suite for solving symmetric
positive-definite (SPD) linear systems with one-level scenario-tree
structure on the GPU: `B` independent block-tridiagonal branches (`N`
stage blocks of size `n`) coupled to a shared separator of dimension `m`.

The structured solver factors all branch chains in parallel with the
upstream [SOCU](https://github.com/PREDICT-EPFL/socu) batched
block-tridiagonal Cholesky: the
coupling is transformed once with the forward substitution only
(`G = L^-1 C^T`, fused with the factorization) and the dense `m x m`
separator Schur complement is built as a lower-triangle-only symmetric
rank update with custom Warp tile kernels (the dense root factorization
itself goes through SOCU treated as a one-stage chain). The global sparse
matrix is never assembled. The whole GPU stack is Warp-only --
no CuPy -- with persistent preallocated buffers and no allocations inside
timed iterations. It is benchmarked against NVIDIA cuDSS (general sparse
direct Cholesky, driven through the raw `nvmath.bindings.cudss` interface
on Warp device buffers) on identical instances. See `doc/math.md` for the
derivation and `plans/plan_1.md` for the experiment plan this implements;
`plans/reviews/` contains the implementation review this version
addresses.

## Layout

    src/                       the solver package (import name: src)
      problem.py               TreeShape / TreeMatrix (matvec, to_csr_lower,
                               from_C) / TreeVector / structural_matvec
      solver.py                TreeSolver and PreparedSolve
      workspace.py             matrix-side device buffers, capability probes
      _utils.py                array validation/transfer, layout helpers
      runtime.py               idempotent Warp initialization
      validation.py            CPU chain Cholesky, error metrics
      socu_adapter.py          thin adapter over the upstream SOCU package
      socu_patch.py            upstream-candidate fused factor+forward launch
      kernels/                 tree-level Warp kernels
        schur.py               Schur matrix/RHS products, recovery update
        reduction.py           deterministic/fused cross-branch reductions
        root.py                single-tile root Cholesky fallback
        diagnostics.py         pivot checks, minimum-pivot reduction
    benchmarks/                benchmark engine (never imported by src/)
      problems.py              ProblemSpec / GeneratedProblem / generator
      config.py, runners.py, memory.py, metadata.py, storage.py, timing.py
    baselines/                 comparison solvers (never imported by src/)
      scipy_reference.py       CPU references (dense, sparse, structured)
      cudss.py                 cuDSS SPD Cholesky adapter (nvmath-python)
    experiments/               smoke.yaml, correctness.yaml, paper.yaml
    scripts/                   run_benchmarks.py, aggregate_results.py, make_plots.py
    tests/                     pytest suite (CPU tests run without a GPU)
    results/raw, results/summary, plots/   benchmark outputs

## Environment

Developed and measured with conda env `socu` (see `environment.yml`):
Python 3.12.12, warp-lang 1.16.0, nvmath-python 0.7.0 (cuDSS 0.7.1
bindings only), numpy 2.2.6, scipy 1.16.3, pandas 2.3.3,
matplotlib 3.10.7, PyYAML 6.0.3, pytest 9.0.2, and SOCU installed from
the local checkout `~/Documents/gpu_code/socu`, branch
`feature/sequential` (base commit `2ff9dc0`, which adds the public
forward/backward substitution launches and the sequential algorithm):

    conda env create -f environment.yml
    conda activate socu
    pip install ~/Documents/gpu_code/socu

GPU used for the reported experiments: NVIDIA GeForce RTX 5090 (32 GB),
driver 580.159.03.

## Commands

    # tests (CPU tests work without a GPU; GPU tests skip automatically)
    python -m pytest -q

    # quick smoke benchmark (a few minutes)
    python scripts/run_benchmarks.py --config experiments/smoke.yaml

    # full paper sweep; resumable after interruption (re-run the command)
    python scripts/run_benchmarks.py --config experiments/paper.yaml \
        --device cuda:0 --resume

    # print the expanded case list without running anything
    python scripts/run_benchmarks.py --config experiments/paper.yaml --dry-run

    # restrict to specific sweeps/methods/seeds
    python scripts/run_benchmarks.py --config experiments/paper.yaml \
        --sweeps branches horizon --methods tree_socu cudss --seed 10

    # aggregate raw records and regenerate every figure
    python scripts/aggregate_results.py --input results/raw --output results/summary
    python scripts/make_plots.py --input results/summary --output plots

    # everything end to end (benchmarks -> aggregation -> PDF figures)
    ./run_experiments.sh smoke        # or: paper, correctness

Raw records are one JSON file per case in `results/raw/` (atomic writes;
existing cases are skipped, pass `--overwrite` to re-run). Aggregates land
in `results/summary/` (`rows.jsonl`, `rows.csv`, `aggregate.csv`,
`crossover_table.csv`), figures in `plots/` as PDF. Detailed
findings are in `results/summary/findings.md`; the top-level takeaways
(target-point comparison, tuning, recommendation) in `conclusions.md`.

## Solver usage

```python
from src import TreeSolver
from benchmarks.problems import ProblemSpec, generate_problem

spec = ProblemSpec(num_branches=64, horizon=128, block_size=16,
                   separator_dim=64, seed=10)
problem = generate_problem(spec)

solver = TreeSolver(problem.shape, device="cuda:0")
solver.stage_matrix(problem.matrix)      # explicit host -> device
solver.factorize(check=True)             # all matrix-dependent work
solution = solver.solve(problem.rhs)     # owned device TreeVector
host = solution.numpy()                  # host TreeVector
w, y = host.branch, host.separator

# allocation-free warm loop with CUDA graphs (the benchmark path)
solver.prepare(use_cuda_graph=True)
rhs_dev = solver.upload_rhs(problem.rhs)
prepared = solver.prepare_solve(problem.rhs.nrhs)
solver.factorize()
prepared.solve_into(rhs_dev)             # result in prepared.out
```

The full API contract (layouts, ownership, synchronization, and
allocation behavior of every method) is documented in `doc/api.md`.

## Deviations from the plan

- CuPy is not used at all (user decision): the Schur/root work that plan
  6.2 allowed to route through CuPy runs in custom Warp tile kernels and
  through SOCU's own blocked Cholesky (the root treated as a one-stage
  chain), so the Warp-vs-CuPy ablation (plan 11.6) does not apply.  The
  cuDSS baseline uses the plan's priority-2 path (a minimal adapter over
  the installed cuDSS library via `nvmath.bindings.cudss`).
- `seaborn` and `pynvml` are not used: figures are plain matplotlib with a
  colorblind-safe palette, and GPU metadata comes from `nvidia-smi`
  parsing (both alternatives the plan permits).
- Stage block sizes `n` are restricted to SOCU-aligned values instead of
  padding through the adapter (permitted by plan 6.1); all required sweep
  sizes are aligned, so the padding ablation (plan 11.7) reduces to
  recording `logical == padded` block sizes.  Separator sizes `m` are
  unrestricted (tiled kernels; SOCU-unaligned small `m` uses a
  single-tile root fallback).
- The workspace-reuse ablation (plan 11.10) is covered by the separately
  reported cold-phase allocation times rather than a dedicated
  per-call-allocation solver mode; the solver always preallocates and the
  timed paths perform no allocations.
- The forward/backward-only launch builders were adopted upstream on the
  `feature/sequential` branch (`create_forward_substitution_launch` /
  `create_backward_substitution_launch`); `tree_socu/socu_patch.py` now
  only aliases them and keeps the fused factor-plus-forward builder as a
  remaining upstream candidate.
- The rectangular lower-SYRK of plan 2 section 5.1 is realized as the
  lower-tile-triangle mode of the project's tiled Warp Schur kernels
  rather than a SOCU-internal `*_blocked_func` extension; behavior and
  flop savings match the plan's specification.
- SOCU's blocked multi-stream path (`n >= 32`) is not run-to-run bitwise
  deterministic (upstream behavior, epsilon-level); the deterministic
  Schur reduction guarantee covers the cross-branch reduction.

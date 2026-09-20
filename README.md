# socu_tree: GPU direct solvers for root-coupled block-tridiagonal systems

Python/Warp research code for solving symmetric positive-definite (SPD)
linear systems in which many block-tridiagonal scenario tails are
coupled through one shared root block.  The current work is the
**endpoint-coupled** solver in `endpoint_tree/`: each scenario tail
couples to the root through its final (root-facing) block only, which
is the structure produced by immediate-branching Branch MPC and by
delayed-branching contingency MPC after prefix condensation.

Each tail is stored and eliminated in leaf-to-root order.  Because only
the last block of a tail touches the root, the coupling is one
`(B, n_b, n_r)` block per tail, so all root-related storage and work is
independent of the horizon.  The tail prefixes are factored in parallel
with the upstream [SOCU](https://github.com/PREDICT-EPFL/socu) batched
block-tridiagonal Cholesky, the transformed connector is confined to a
logarithmic-depth ancestor path in the cyclic-reduction elimination
tree, and the root Schur complement is accumulated by parallel
reduction.  The global sparse matrix is never assembled.  The GPU stack
is Warp-only (no CuPy), with persistent preallocated buffers, no
allocations inside timed iterations, and separate CUDA graphs for the
warm factorization and solve pipelines.

## Layout

    src/                       the two solver packages
      general_arrow/           general block-arrow solver: every stage of
                               a tail may couple to the shared root
        problem.py             TreeShape / TreeMatrix / TreeVector
        solver.py              Solver (update/factorize/solve, CUDA graphs)
        socu.py, _utils.py     SOCU adapter, device/dtype helpers
        kernels/               coupling and dense-root Warp kernels
      endpoint_tree/           endpoint-coupled solver (current work):
                               only the root-facing block of each tail
                               couples to the root
        problem.py             EndpointTreeShape / EndpointTreeMatrix
                               (matvec, to_csr_lower) / EndpointTreeVector
        solver.py              EndpointTreeSolver (update/factorize/solve,
                               CUDA graphs, phase timing)
        kernels/               connector-path, endpoint and root kernels
        _reuse.py              the single place that imports shared
                               primitives from general_arrow (one-way)
        concept/               sparsity-pattern figures explaining the
                               ordering (drawing programs, not timings)
    experiments/               runnable experiments and their outputs;
                               never imported by src/
      quadrotor/               full-attitude quadrotor branch-MPC
                               benchmark (the paper's application), its
                               results/ and figures/
      endpoint_tree/           figures/ written by endpoint_tree/concept
      general_arrow/           benchmark engine, campaign scripts and
                               outputs for the general block-arrow solver
        benchmarks/, scripts/, results/, plots/
    admm/                      sparse-QP ADMM solver (OSQP-style API;
                               tree or cuDSS normal-matrix backend)
    baselines/                 comparison solvers shared by every
                               experiment: cuDSS, CHOLMOD, PARDISO,
                               QDLDL, SciPy references
    paper_draft/               ACC paper source (main.tex)
    tests/                     pytest suites: general_arrow/ and
                               endpoint_tree/ (CPU tests run without a GPU)
    plans/, doc/               plan documents, reviews, math and API notes
    RiskAverseTrajOpt/         vendored risk-averse trajectory-optimization
                               code (Lew et al.); the drone and hopper
                               endpoint benchmarks live here

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

## Solver usage

```python
from src.endpoint_tree import EndpointTreeShape, EndpointTreeSolver
# a synthetic SPD instance with this structure (test fixture generator)
from tests.endpoint_tree.problems import generate_endpoint_problem

# n_r = n_b is the regime of interest
shape = EndpointTreeShape(num_tails=64, num_stages=32,
                          tail_block_dim=8, root_dim=8)
problem = generate_endpoint_problem(shape, seed=0)

solver = EndpointTreeSolver(shape, device="cuda:0")
solver.update(problem.matrix)      # values only (explicit transfer)
solver.factorize()                 # complete permuted Cholesky factor
solution = solver.solve(problem.rhs)   # owned device EndpointTreeVector
```

`num_stages` counts uniform algebraic blocks per scenario in
**leaf-to-root** storage order: `tail[:, 0]` is the leaf and
`tail[:, T-1]` is the root-facing endpoint that carries `G_T`.  The
application builder maps physical variables into uniform blocks.
Binding device `rhs`/`out` once and calling `solve(rhs, out=out)`
replays the captured graph with no allocation.

## Benchmarks

    # the paper's application benchmark on real quadrotor matrices
    # (CPU baselines bound to the 8 performance cores)
    cd experiments/quadrotor && ./run_quadrotor_heatmap.sh
    # redraw its figures from an existing CSV, without re-measuring
    python quadrotor_endpoint_benchmark.py --replot results/quadrotor_endpoint_heatmap.csv

The ordering figures used in the paper are drawn (not timed) by

    python -m src.endpoint_tree.concept.ordering_merged
    python -m src.endpoint_tree.concept.ordering_idea
    python -m src.endpoint_tree.concept.ordering_fill

which write into `experiments/endpoint_tree/figures/`.

Baselines are cuDSS (general GPU sparse direct Cholesky, through the
raw `nvmath.bindings.cudss` interface on Warp device buffers) and, on
the CPU, Intel MKL PARDISO and CHOLMOD -- all receiving the identical
matrix, precision and right-hand side.  Reported quantities are CUDA-event medians with the
interquartile range, factorization and solve separately, plus residual,
relative error, persistent workspace bytes, connector path length and a
per-phase breakdown.

## Sparse-QP ADMM solver

`admm/` solves `min 0.5 x'Px + q'x  s.t.  l <= Ax <= u` from ordinary
CSC input with scaled ADMM at fixed `rho`; the normal matrix
`K = P + rho A'A` is factored either by the structured tree backend
(selected automatically when the conservative analyzer verifies the
block-arrow pattern) or by cuDSS -- one ADMM loop, one sparse frontend,
identical stopping rule (see `plans/plan_admm.md`):

```python
from admm import Solver, Settings

solver = Solver()
solver.setup(P, q, A, l, u, settings=Settings(rho=1.0,
                                              linear_solver="auto"))
result = solver.solve()          # result.info.linear_solver: tree/cudss
solver.update(q=q_new)           # values only; sparsity is immutable
```

## Tests

    python -m pytest -q               # CPU tests run without a GPU

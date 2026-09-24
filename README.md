# socu_tree: a GPU direct solver for Branch MPC linear systems

Python/Warp code accompanying the paper on a two-level parallel direct
solver for the symmetric positive-definite (SPD) linear systems of
Branch MPC: many block-tridiagonal scenario tails coupled to one shared
root block, each tail through its root-facing (endpoint) block only.
This is the structure of immediate-branching scenario MPC, where all
scenarios share the first decision and evolve independently afterwards.

The solver in `src/endpoint_tree/` exploits two levels of parallelism
in one Cholesky factorization and triangular solve: the scenario tails
are processed concurrently, and inside every tail the stages are
processed in the logarithmic-depth levels of a cyclic-reduction
ordering.  The tail prefixes are factored with the upstream
[SOCU](https://github.com/PREDICT-EPFL/socu) batched block-tridiagonal
Cholesky, the endpoint coupling is confined to a logarithmic-depth
ancestor path of the elimination tree, and the root Schur complement is
accumulated by parallel reduction.  The global sparse matrix is never
assembled.  The GPU stack is Warp only, with persistent preallocated
buffers, no allocations inside timed iterations, and one CUDA graph
each for the factorization and the solve.

## Layout

    src/
      endpoint_tree/           the solver of the paper
        problem.py             EndpointTreeShape / EndpointTreeMatrix
                               (matvec, to_csr_lower) / EndpointTreeVector
        solver.py              EndpointTreeSolver (update / factorize /
                               solve, CUDA graphs, phase timing)
        kernels/               tail, connector-path and root Warp kernels
        reference.py           NumPy reference factorization and solve
        _reuse.py              the single place that imports shared
                               primitives from dense_arrow (one-way)
        concept/               the sparsity-pattern figure explaining
                               the ordering (a drawing, not a timing)
      dense_arrow/             general block-arrow solver (every stage
                               of a tail may couple to the root); provides
                               the SOCU adapter and root kernels the
                               endpoint solver reuses
    baselines/                 cuDSS, CHOLMOD and PARDISO wrappers and
                               the CPU validation references
    experiments/quadrotor/     the paper's benchmark on full-attitude
                               quadrotor Branch-MPC matrices, its thermal
                               protocol and figure scripts
    tests/                     pytest suites (CPU tests run without a GPU)

## Installation

Developed and measured in the conda environment of `environment.yml`
(Python 3.12, warp-lang 1.16.0, numpy 2.2.6, scipy 1.16.3,
nvmath-python 0.7.0 for the cuDSS bindings) on an NVIDIA GeForce
RTX 5090 with driver 580:

    conda env create -f environment.yml
    conda activate socu

SOCU is installed from its public repository by the environment file;
the solver needs its public forward and backward substitution launches
(upstream `main`).  Intel MKL PARDISO (through `mkl` and its Python
bindings) and scikit-sparse are optional and only used by the CPU
baselines.

## Solver usage

```python
from src.endpoint_tree import EndpointTreeShape, EndpointTreeSolver
# a synthetic SPD instance with this structure (test fixture generator)
from tests.endpoint_tree.problems import generate_endpoint_problem

shape = EndpointTreeShape(num_tails=64, num_stages=32,
                          tail_block_dim=8, root_dim=8)
problem = generate_endpoint_problem(shape, seed=0)

solver = EndpointTreeSolver(shape, device="cuda:0")
solver.update(problem.matrix)          # values only (explicit transfer)
solver.factorize()                     # permuted Cholesky factor
solution = solver.solve(problem.rhs)   # device EndpointTreeVector
```

`num_stages` counts uniform blocks per scenario in **leaf-to-root**
storage order: `tail[:, 0]` is the leaf and `tail[:, T-1]` the
root-facing endpoint that carries the coupling block.  Binding device
`rhs` / `out` once and calling `solve(rhs, out=out)` replays the
captured graph with no allocation; `update_device(D, E, G_T, R)` takes
new values that already live on the device.  The block size must be
aligned to the SOCU storage rules (even, for float64).

## Reproducing the paper's experiments

    # speedup heat maps against cuDSS and eight-core PARDISO
    cd experiments/quadrotor && ./run_quadrotor_heatmap.sh
    # redraw the figures from an existing CSV without re-measuring
    python quadrotor_endpoint_benchmark.py --replot results/quadrotor_endpoint_heatmap.csv
    # absolute latency along one axis (fixed N or fixed M)
    ./run_quadrotor_heatmap.sh --tails 20 40 80 100 200 --horizons 30 --output results/T30.csv
    python quadrotor_cpu_baselines_plot.py --input results/T30.csv --layout combined

The benchmark assembles the KKT matrices of an immediate-branching
scenario MPC for a 12-state quadrotor for every (scenarios, horizon)
grid point and times the proposed solver, cuDSS and PARDISO on the
identical matrix, precision and right-hand side; CHOLMOD provides the
untimed reference solution.  Reported quantities are medians with the
interquartile range over the timed executions, factorization and solve
separately (CUDA events on the GPU, wall clock around the bare MKL call
on the CPU).  The thermal protocol (burn-in, randomized grid order,
phase-specific re-warm, block-alternating interleaving of the GPU
solvers, telemetry and a raw per-sample sidecar) is documented in the
module docstring of `quadrotor_endpoint_benchmark.py`; the shell script
binds the CPU baselines to the performance cores.

The ordering figure of the paper is drawn (not timed) by

    python -m src.endpoint_tree.concept.ordering_merged

which writes `endpoint_orderings_2x2.pdf` into `src/endpoint_tree/concept/`.
The scripts `quadrotor_smpc_*.py` and `quadrotor_nlp_*.py` draw the
open- and closed-loop maneuvers the benchmark matrices come from (SciPy
and CasADi/IPOPT respectively).

## Tests

    python -m pytest -q        # GPU tests are skipped without a CUDA device

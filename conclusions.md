# Conclusions

State of the structured scenario-tree solver as of 2026-08-12: after
the Schur tile tuning (TILE_M = 16), the stage-chunked kernels, the
API/organization refactor, and the removal of the three losing solver
options (inverse action, sequential tails, per-branch dispatch).  All
numbers below come from one fresh paper campaign on the final source
(single source hash, paired factor/solve measurements).  Detailed
evidence and methodology: `results/summary/findings.md`; figures:
`plots/` (target points: `plots/fig_target_points.pdf`).

Setup: NVIDIA GeForce RTX 5090 (32 GB, driver 580.159.03), conda env
`socu` (Warp 1.16.0, cuDSS 0.7.1 via nvmath 0.7.0, SOCU branch
`feature/sequential`, base `2ff9dc0`).  Numbers are CUDA-event medians
of warm factorize+solve iterations over 3 seeds.

## 1. Application target points, factorization and solve split

At the four operating points of interest, (B, N, n) with m = 64.
Factorization and solve are measured in separate repetition loops
(stacked in `plots/fig_target_points.pdf`); ms, speedup = cuDSS over
src.

FP64 (tree_socu / cuDSS, speedup; graph = tree_socu with CUDA graphs):

| (B, N, n) | factor | solve | total | total w/ graphs |
|---|---|---|---|---|
| (11, 32, 8)  | 0.248 / 0.349 (**1.41x**) | 0.152 / 0.190 (**1.25x**) | 0.399 / 0.539 (**1.35x**) | 0.343 (**1.57x**) |
| (21, 64, 8)  | 0.315 / 0.528 (**1.68x**) | 0.184 / 0.257 (**1.40x**) | 0.499 / 0.785 (**1.57x**) | 0.440 (**1.78x**) |
| (41, 96, 8)  | 0.524 / 0.778 (**1.48x**) | 0.236 / 0.271 (**1.15x**) | 0.759 / 1.048 (**1.38x**) | 0.704 (**1.49x**) |
| (81, 128, 8) | 1.044 / 1.645 (**1.58x**) | 0.389 / 0.388 (1.00x) | 1.433 / 2.035 (**1.42x**) | 1.372 (**1.48x**) |

FP32:

| (B, N, n) | factor | solve | total | total w/ graphs |
|---|---|---|---|---|
| (11, 32, 8)  | 0.133 / 0.184 (**1.38x**) | 0.146 / 0.084 (0.58x) | 0.279 / 0.268 (0.96x) | 0.144 (**1.86x**) |
| (21, 64, 8)  | 0.152 / 0.290 (**1.91x**) | 0.172 / 0.128 (0.74x) | 0.325 / 0.419 (**1.29x**) | 0.178 (**2.35x**) |
| (41, 96, 8)  | 0.166 / 0.427 (**2.57x**) | 0.171 / 0.132 (0.77x) | 0.338 / 0.559 (**1.65x**) | 0.232 (**2.41x**) |
| (81, 128, 8) | 0.260 / 0.979 (**3.77x**) | 0.166 / 0.196 (**1.18x**) | 0.427 / 1.176 (**2.75x**) | 0.365 (**3.22x**) |

The split matters for the use case:

- **Refactorize-every-iteration workloads (MPC / interior point): the
  structured solver wins all eight configurations on the total**
  (1.35-1.57x FP64, 0.96-2.75x FP32 plain; with CUDA graphs
  1.48-1.86x FP64 and 1.86-3.22x FP32).  Factorization -- the
  dominant cost -- is where the structure pays: 1.41-1.68x FP64 and
  up to 3.77x FP32.
- **Factor-once-solve-many workloads: with CUDA graphs the
  structured solve wins at every point too** (FP64 1.10-1.63x, FP32
  1.14-1.45x on the solve alone).  Without graphs, plain tree_socu
  solve wins or ties everywhere in FP64, and in FP32 loses below
  B = 81 (0.57-0.78x): the structured solve does
  more per call (forward + Schur-RHS + root + recovery + backward; 19
  launches) than two sparse triangular sweeps, and at sub-0.5 ms
  scales the launch overhead only disappears under graphs.  With many
  simultaneous right-hand sides the advantage grows (2.73x at 64
  RHS).
- `tree_graph` (CUDA graphs) is the recommended warm-loop
  configuration everywhere; graphs buy 4-48%, most at small sizes.
- Accuracy: forward errors ~1e-14 in FP64 and <= 8.3e-6 in FP32 on
  well-conditioned instances (kappa ~ 1e4).
- The sequential tail algorithm loses at all four points; even 11
  branches prefer cyclic reduction's log-depth chains.

## 2. Kernel tuning mattered

Sweeping the Schur tile width TILE_M over {8, 16, 32, 64} showed the
original 32 was suboptimal everywhere: 16 cuts the branch SYRK phase by
24-43% and warm totals by ~10-32% depending on the point, with the
largest gains at small separators (m = 8: 2.03 -> 1.38 ms total).
TILE_M = 16 is now the default (overridable via `TREE_SOCU_TILE_M`);
thread-block size stays 128 after a mixed 64-vs-128 result.

A second, profiling-driven kernel change followed: the Schur SYRK,
Schur-RHS, and recovery kernels now process each branch's G as one
contiguous (B, T*n_b, n_y) matrix in 32-row chunks instead of looping
stages with tiny (n_b x 16) tiles (per-stage kernels retained for
restricted coupling stage ranges).  Measured at (81, 128, 8) FP64:
SYRK 0.51 -> 0.43 ms, Schur-RHS 0.14 -> 0.09 ms.  All recorded
tree-method results were re-measured after each change (archives:
`results/raw_tile32_backup/`, `results/raw_perstage_backup/`).

## 3. Broader comparison with cuDSS (FP64)

- Default study point (B=64, N=128, n=16, m=64): 1.24x, up to 1.91x
  at N=16.
- The branch crossover sits near B ~ 12; cuDSS keeps only the
  few-branch long-horizon regime (0.88-0.99x at B <= 8, N = 128).
- The whole separator sweep wins (1.12-1.53x, small separators
  included).
- Multi-RHS: 2.20x at 16 RHS, 2.73x at 64 RHS.  FP32: up to ~3.7x
  (factorization at the largest target point).
- The benchmarked comparison set is deliberately lean (user decision):
  cuDSS runs its default configuration only, and the sequential tail
  mode is no longer benchmarked (it remains a supported, tested solver
  option; its measured 2.9-3.0x loss at these shapes is archived in
  `results/raw_archive_20260812_prepaired/`).

## 4. Algorithmic conclusions (unchanged by the tuning)

- The factor-space algorithm (G = L^-1 C^T forward-only, then the
  lower-triangle Schur SYRK S = R - sum G^T G) is now the solver's
  only formulation; the explicit inverse action X = K^-1 C^T it
  replaced measured ~20% slower and was removed (archived ablation).
- Batching all branches into single SOCU launches is essential and
  hardcoded (the removed per-branch ablation measured 5-15x slower).
  The fused factor+forward launch (worth 5-10% over separate launches)
  and the deterministic pairwise reduction (measured equal to atomic
  accumulation, and bitwise reproducible) are likewise hardcoded; the
  losing variants are archived, not options.
- Sequential (block-Thomas) tails were latency-bound on this GPU (flat
  in B, ~3x slower at the default point) and were removed from both
  the solver options and the benchmark; SOCU's parallel cyclic
  reduction is the only tail algorithm (archived ablation).
- Beyond warm speed, the structured solver wins cold start (~0.3 s
  constant analyze vs ~1.6 s cuDSS analysis+assembly at dimension
  131k), memory reach (no fill-in), and robustness (it detects the
  singular strong-coupling instances that cuDSS silently mis-solves;
  its FP64 pivot checks are explicit and optional).

## 5. Profile of the (81, 128, 8) FP64 point (Nsight Systems)

Factorization and solve profiled in separate 50-iteration loops.

**Factorization-only: 1.20 ms/iter, GPU 87.3% busy, ~18 launches.**
Compute-bound; two kernels are 92% of its GPU time:

| kernel(s) | us/iter | share | note |
|---|---|---|---|
| Schur SYRK tile kernel (1 launch)      | 510 | 49% | ~0.84 of ~1.64 TFLOP/s FP64 peak; only ~410 GB/s of traffic, so compute-bound |
| SOCU fused factor+forward (8 launches) | 457 | 44% | cyclic-reduction levels; late levels occupancy-starved (81 blocks / 170 SMs) |
| root POTRF (SOCU blocked, m=64)        |  71 |  7% | single-tile kernel measured faster |
| pair-reduce + finalize (8 launches)    |  10 |  1% | |

**Solve-only: 0.53 ms/iter, GPU only 73.7% busy, ~19 launches.**
Unlike factorization, the solve is partly launch/latency-bound
(~0.14 ms idle) and its kernels are memory-latency-bound at nrhs = 1:

| kernel(s) | us/iter | share | note |
|---|---|---|---|
| Schur-RHS reduction (1 launch)          | 131 | 33% | reads all of G (42.5 MB) at only ~325 GB/s -- tiny (8 x 16)^T (8 x 1) tile ops |
| recovery update (1 launch)              | 120 | 31% | same pattern, reads G again |
| SOCU fwd+bwd substitution (16 launches) |  96 | 25% | 8 levels x 2 directions, ~6 us each: level-chain latency |
| root solves + reductions (9 launches)   |  42 | 11% | |

This explains the solve-only loss to cuDSS (0.434 vs 0.366 ms): the
two big solve kernels run at ~1/5 of memory bandwidth, and a quarter of
the wall time is launch gaps.  A full read of G costs ~24 us at peak
bandwidth, so the algorithmic floor of Schur-RHS + recovery is ~50 us
against the current 250 us.

Improvement directions, largest first (prototype-validated where noted):

1. **Stage-chunked Schur kernels -- IMPLEMENTED** (see section 2).
   Measured outcome at (81, 128, 8) FP64: SYRK 510 -> 427 us,
   Schur-RHS 131 -> 94 us; totals at the target points improved
   5-21%, and with CUDA graphs the solve-only path now beats cuDSS at
   every target point in both precisions.  Remaining solve-side
   headroom is the SOCU substitution level-chain latency (item 2) and
   the recovery kernel (already near its floor at this shape).
2. **Upstream SOCU: level-chain latency.**  Factor side: the last
   cyclic-reduction levels run 81 blocks on 170 SMs (~0.1 ms
   opportunity).  Solve side: 16 substitution launches of ~6 us each
   are almost pure level-chain latency at nrhs = 1; merging levels or
   a short-chain cutover is the upstream fix.
3. **FP32** remains the big lever where tolerable: the full iteration
   runs 2.9x faster (0.54 ms) on consumer Blackwell's 64:1 FP32:FP64
   throughput.
4. Minor: route the m <= 64 root through the single-tile kernel even
   when SOCU-aligned (~17 us), and keep CUDA graphs on (covers the
   idle fraction: 12.7% of factorization, 26.3% of the solve).

## 6. Recommendation

For the target workloads: `tree_socu` with factor space, SOCU's
parallel algorithm, TILE_M = 16, batched mode, and CUDA graphs for warm
loops -- in FP32 whenever the application tolerates ~1e-6 errors (it is
another ~2-3x on top of FP64).  cuDSS remains preferable only for FP64
warm loops with B <= 8 at long horizons, or for strict
factor-once-solve-many single-RHS workloads at the larger points until
the solve-side kernel work of section 5 lands.

"""Baseline solvers used for benchmarking and validation.

- :mod:`baselines.scipy_reference` -- CPU references (NumPy dense
  Cholesky, SciPy sparse direct, and a CPU structured root-update solve).
- :mod:`baselines.cudss` -- NVIDIA cuDSS sparse direct Cholesky
  (the primary GPU baseline); reports an explicit unavailable status when
  cuDSS cannot be loaded.
"""

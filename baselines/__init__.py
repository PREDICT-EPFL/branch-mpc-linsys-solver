"""Comparison solvers and validation references.

- :mod:`baselines.cudss` -- NVIDIA cuDSS sparse direct Cholesky on Warp
  device buffers (the GPU baseline); reports an explicit unavailable
  status when cuDSS cannot be loaded.
- :mod:`baselines.cpu_direct` -- CHOLMOD (scikit-sparse) and Intel MKL
  PARDISO on the CPU, receiving the identical matrix.
- :mod:`baselines.reference` -- validation metrics (residuals, errors
  against the known solution of a generated problem).
- :mod:`baselines.scipy_reference` -- CPU references (NumPy dense
  Cholesky, SciPy sparse direct, and a CPU structured root-update solve).
"""

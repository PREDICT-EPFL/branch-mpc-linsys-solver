"""Structured GPU direct solvers for root-coupled block-tridiagonal
SPD systems.

Two solver packages share the SOCU adapter and the dense root
kernels:

- :mod:`src.dense_arrow` -- the general block-arrow solver, in
  which every stage of a tail may couple to the shared root.
- :mod:`src.endpoint_tree` -- the endpoint-coupled solver, in which
  only the root-facing block of each tail couples to the root.

The dependency direction is one-way: ``endpoint_tree`` reuses
selected primitives from ``dense_arrow`` (see
:mod:`src.endpoint_tree._reuse`); nothing in ``dense_arrow`` may
import ``endpoint_tree``.
"""

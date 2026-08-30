"""Two-level permuted structured Cholesky solver (Warp-only).

:class:`Solver` is a direct Cholesky solver for the globally permuted
scenario-tree matrix ``Pi Phi Pi^T = L_hat L_hat^T``: SOCU's recursive
odd-even permutation inside every tail tail, identity on the root.
``factorize()`` constructs the complete lower factor (permuted tail
factors, root coupling columns ``M_i^T = L_hat_i^{-1} C_hat_i^T``, root
diagonal ``L_R L_R^T = R - sum_i M_i M_i^T``); ``solve()`` applies
``Pi^T L_hat^-T L_hat^-1 Pi`` with input and output in physical/user
ordering.  Both pipelines run as replayed CUDA graphs; warm calls
allocate nothing and never synchronize the host.

Lifecycle::

    solver = Solver(shape, device="cuda:0")
    solver.update(matrix)     # values only; invalidates the factor
    solver.factorize()
    x = solver.solve(rhs, out=x_workspace)   # allocation-free path

``solve`` consumes its RHS workspace: with device-resident ``rhs`` and
``out`` the caller buffers are bound directly into the solve graph (the
forward substitution overwrites ``rhs``); host-array RHS fields are
staged through internal buffers instead.  The solve graph is complete
(including the root output store), so a warm solve is exactly one graph
replay.  One binding is kept; it is rebuilt only when ``nrhs`` or any
RHS/output pointer changes.
"""

import warp as wp
from socu.block_tridiag_solver import (
    create_cholesky_factor_launch,
    create_cholesky_solve_launch,
)

from src._utils import copy_into, require_cuda_device, wp_dtype
from src.kernels import BLOCK_DIM, num_root_tiles
from src.kernels.coupling import (
    ROOT_RHS_VECTOR_PARTITIONS,
    SMALL_ROOT_CHUNKS,
    create_atomic_root_rhs_kernel,
    create_atomic_root_update_kernel,
    create_chunked_recover_kernel,
    create_chunked_root_update_kernel,
    create_chunked_root_rhs_kernel,
    create_root_rhs_vector_partial_kernel,
    create_root_rhs_vector_reduce_kernel,
    create_small_root_correction_kernel,
    create_small_root_reduce_kernel,
    create_small_root_rhs_partial_kernel,
    create_small_root_update_partial_kernel,
    create_tail_root_vector_correction_kernel,
    num_row_chunks,
    small_root_segment,
    use_small_root,
)
from src.kernels.root import (
    create_root_factor_kernel,
    create_root_solve_kernel,
)
from src.problem import TreeMatrix, TreeShape, TreeVector
from src.socu import TailEngine, is_block_size_aligned


def _device_ptr(array):
    """Pointer identity of a Warp array, or None for host arrays."""
    return array.ptr if isinstance(array, wp.array) else None


def _is_solver_array(array, solver):
    """True when ``array`` is a Warp array on the solver's device with
    the solver's dtype (the zero-copy requirement for every field)."""
    return (isinstance(array, wp.array) and array.device == solver._device
            and array.dtype == solver._dtype)


def _binding_key(rhs, out):
    """Complete pointer key of one solve binding: ``nrhs`` plus every
    captured RHS and output pointer."""
    return (rhs.nrhs, _device_ptr(rhs.tail), _device_ptr(rhs.root),
            None if out is None else _device_ptr(out.tail),
            None if out is None else _device_ptr(out.root))

    if not solver._root_via_socu:
        key = (devkey, "root_solve", n_r, nrhs, solver._shape.precision)
        if key not in _PROBED_SOLVE:
            # 2I is its own Cholesky factor of 4I: solving 4.0 gives 1.0
            L = wp.array(2.0 * np.eye(n_r, dtype=solver._shape.np_dtype),
                         dtype=dt, device=dev)
            x = wp.array(np.full((n_r, nrhs), 4.0,
                                 dtype=solver._shape.np_dtype),
                         dtype=dt, device=dev)
            wp.launch_tiled(create_root_solve_kernel(n_r, nrhs, dt),
                            dim=[1], inputs=[L, x],
                            block_dim=BLOCK_DIM, device=dev)
            wp.synchronize_device(dev)
            if abs(float(x.numpy()[n_r - 1, nrhs - 1]) - 1.0) > 1e-3:
                raise fail(f"root solve tile kernel (n_r={n_r}, "
                           f"nrhs={nrhs})")
            _PROBED_SOLVE.add(key)


class _SolveBinding:
    """One pointer-bound solve pipeline: buffers, launches, and its CUDA
    graph.  Rebuilt by :meth:`Solver.solve` when ``nrhs`` or the bound
    pointers change."""

    def __init__(self, solver, rhs, out):
        B, T, n_b, n_r = solver._dims
        dt, dev = solver._dtype, solver._device
        nrhs = rhs.nrhs
        self.key = _binding_key(rhs, out)
        # RHS storage: caller arrays when the caller passed an explicit
        # output vector AND every RHS field is a Warp array on the
        # solver device with the solver dtype (zero-copy; the solve
        # consumes them).  With ``out=None`` (convenience path) or any
        # host-array/mismatched RHS field the binding owns staging
        # buffers, so the caller's RHS is never overwritten.
        self.zero_copy = (out is not None
                          and _is_solver_array(rhs.tail, solver)
                          and _is_solver_array(rhs.root, solver))
        if self.zero_copy:
            self.rhs_tail = rhs.tail
            self.root_rhs = rhs.root
        else:
            self.rhs_tail = wp.zeros((B, T, n_b, nrhs), dtype=dt,
                                       device=dev)
            self.root_rhs = wp.zeros((n_r, nrhs), dtype=dt, device=dev)
        self.root_rhs4 = self.root_rhs.reshape((1, 1, n_r, nrhs))
        # output: caller vector or an owned one (convenience path)
        self.owns_out = out is None
        self.out = out if out is not None else TreeVector(
            solver.shape,
            wp.zeros((B, T, n_b, nrhs), dtype=dt, device=dev),
            wp.zeros((n_r, nrhs), dtype=dt, device=dev))
        self.rhs3 = self.rhs_tail.reshape((B, T * n_b, nrhs))
        self.out3 = self.out.tail.reshape((B, T * n_b, nrhs))
        self.contrib = wp.zeros((B, n_r, nrhs), dtype=dt, device=dev)
        self.vector_rhs_partial = (
            wp.zeros((B, ROOT_RHS_VECTOR_PARTITIONS, n_r), dtype=dt,
                     device=dev) if not solver._small_root and nrhs == 1
            else None)
        self.rhs_partial = (
            wp.zeros((B, SMALL_ROOT_CHUNKS, n_r, nrhs), dtype=dt, device=dev)
            if solver._small_root else None)
        self.forward = solver._socu.build_forward_launch(self.rhs_tail)
        self.backward = solver._socu.build_backward_launch(self.out.tail)
        # A single-tile solve wins for one RHS through n_r=64. Larger
        # roots retain SOCU because the tile exceeds device shared memory.
        tile_root_solve = (not solver._root_via_socu
                           or (nrhs == 1 and n_r <= 64))
        if not tile_root_solve:
            self.root_solve = create_cholesky_solve_launch(
                solver._root_diagonal4, solver._root_E, self.root_rhs4,
                dtype=dt, device=dev)
        else:
            kernel = create_root_solve_kernel(n_r, nrhs, dt)
            root_diagonal, s = solver._root_diagonal, self.root_rhs

            def tile_root_solve():
                wp.launch_tiled(kernel, dim=[1], inputs=[root_diagonal, s],
                                block_dim=BLOCK_DIM, device=dev)
            self.root_solve = tile_root_solve
        self.graph = None


class Solver:
    """Permuted structured Cholesky solver for one-level scenario-tree
    SPD systems (see the module docstring for the lifecycle).

    Shape, dtype, and device are fixed at construction; ``update()``
    changes numerical values only.  The tail block dimension must be
    SOCU-aligned; root dimensions below ``TILE_M`` use the scalar
    small-root kernels and the single-tile root factor, larger roots the
    tiled kernels with SOCU factoring the root (which must then be
    SOCU-aligned as well).
    """

    def __init__(self, shape: TreeShape, device: str = "cuda:0"):
        if not isinstance(shape, TreeShape):
            raise TypeError("shape must be a TreeShape")
        self._shape = shape
        self._device = require_cuda_device(device)
        self._dtype = wp_dtype(shape.precision)
        B, T, n_b, n_r = shape.dims()
        self._dims = (B, T, n_b, n_r)
        self._mt = num_root_tiles(n_r)
        self._small_root = use_small_root(n_r)
        self._root_via_socu = not self._small_root
        if self._root_via_socu and not is_block_size_aligned(n_r,
                                                             self._dtype):
            raise ValueError(
                f"root dimension {n_r} is not SOCU-aligned; roots of at "
                f"least TILE_M must be aligned to SOCU's padding multiples")

        dt, dev = self._dtype, self._device
        self._socu = TailEngine(B, T, n_b, dt, dev)
        # persistent factor blocks and workspace.  root_coupling_T holds
        # M_i^T (rows in physical stage order); root_diagonal the
        # persistent root factor L_R (4D alias so SOCU can factor it as a
        # one-stage chain)
        self._C_T = wp.zeros((B, T, n_b, n_r), dtype=dt, device=dev)
        self._root_coupling_T = wp.zeros((B, T, n_b, n_r), dtype=dt,
                                         device=dev)
        self._root_coupling_T3 = self._root_coupling_T.reshape(
            (B, T * n_b, n_r))
        self._R = wp.zeros((n_r, n_r), dtype=dt, device=dev)
        self._root_diagonal4 = wp.zeros((1, 1, n_r, n_r), dtype=dt,
                                        device=dev)
        self._root_diagonal = self._root_diagonal4.reshape((n_r, n_r))
        self._contrib = wp.zeros((B, n_r, n_r), dtype=dt, device=dev)
        self._root_partial = (
            wp.zeros((B, SMALL_ROOT_CHUNKS, n_r, n_r), dtype=dt, device=dev)
            if self._small_root else None)
        if self._root_via_socu:
            from socu.block_tridiag_solver import (
                calculate_off_diag_storage_len)
            self._root_E = wp.zeros(
                (1, calculate_off_diag_storage_len(1), n_r, n_r),
                dtype=dt, device=dev)
            self._root_factor_launch = create_cholesky_factor_launch(
                self._root_diagonal4, self._root_E, dtype=dt, device=dev)
        # fused tail factorization + coupling transform (one launch)
        self._tail_factor_coupling = self._socu.build_factor_forward_launch(
            self._root_coupling_T)
        self._factor_graph = None
        self._binding = None
        self._has_values = False
        self._has_factor = False

    # -------------------------------------------------------------- public
    @property
    def shape(self) -> TreeShape:
        """The problem dimensions this solver was built for."""
        return self._shape

    def update(self, matrix: TreeMatrix) -> None:
        """Copy new matrix values into the persistent buffers and
        invalidate the factor (no implicit factorization).  Shapes are
        fixed at construction; buffer pointers stay stable, so captured
        graphs remain valid."""
        if matrix.shape != self._shape:
            raise ValueError(f"matrix shape {matrix.shape} does not match "
                             f"solver shape {self._shape}")
        self._socu.stage(matrix.D, matrix.E)
        copy_into(self._C_T, matrix.C_T, "C_T")
        copy_into(self._R, matrix.R, "R")
        self._has_values = True
        self._has_factor = False

    def factorize(self) -> None:
        """Construct the complete permuted Cholesky factor (captured in
        a CUDA graph on first use, replayed afterwards)."""
        if not self._has_values:
            raise RuntimeError("factorize() called before update()")
        if self._factor_graph is None:
            self._factorize_numeric(None)  # warm compile + real compute
            with wp.ScopedCapture(device=self._device) as cap:
                self._factorize_numeric(None)
            self._factor_graph = cap.graph
        else:
            wp.capture_launch(self._factor_graph)
        self._has_factor = True

    def solve(self, rhs: TreeVector,
              out: TreeVector | None = None) -> TreeVector:
        """Apply ``Pi^T L_hat^-T L_hat^-1 Pi`` to ``rhs``.

        With device-resident ``rhs`` and ``out`` the caller buffers are
        bound into the solve graph and reused across calls with the same
        pointers: no staging copies, no allocation; the solve consumes
        (overwrites) the bound RHS storage.  With ``out=None`` or any
        host-array RHS field the call stages through internal buffers
        (the caller's RHS is left untouched) and may allocate
        (convenience path).  Requires a valid factor; never factorizes
        implicitly.
        """
        if not self._has_factor:
            raise RuntimeError("solve() requires a valid factor; call "
                               "factorize() after update()")
        if rhs.shape != self._shape:
            raise ValueError(f"rhs shape {rhs.shape} does not match "
                             f"solver shape {self._shape}")
        if out is not None:
            if out.nrhs != rhs.nrhs:
                raise ValueError(f"out has {out.nrhs} columns, rhs has "
                                 f"{rhs.nrhs}")
            if not (_is_solver_array(out.tail, self)
                    and _is_solver_array(out.root, self)):
                raise TypeError("out must hold Warp arrays on the solver "
                                "device with the solver dtype")
        b = self._binding
        key = _binding_key(rhs, out)
        if b is None or b.key != key:
            b = self._binding = _SolveBinding(self, rhs, out)
        if not b.zero_copy:
            copy_into(b.rhs_tail, rhs.tail, "rhs.tail")
            copy_into(b.root_rhs, rhs.root, "rhs.root")
        if b.graph is None:
            self._solve_numeric(b, None)  # warm compile + real compute
            with wp.ScopedCapture(device=self._device) as cap:
                self._solve_numeric(b, None)
            b.graph = cap.graph
        else:
            wp.capture_launch(b.graph)
        if not b.owns_out:
            return b.out
        return TreeVector(self._shape, wp.clone(b.out.tail),
                          wp.clone(b.out.root))

    # ---------------------------------------------------- numerical stages
    def _factorize_numeric(self, timer):
        B, T, n_b, n_r = self._dims
        dt, dev = self._dtype, self._device
        self._socu.refresh()
        wp.copy(self._root_coupling_T, self._C_T)
        if timer: timer.mark("refresh")
        # factor every permuted tail in place and transform the coupling
        # to M^T = L_hat^-1 C_hat^T, one fused launch
        self._tail_factor_coupling()
        if timer: timer.mark("tail_factor")
        # root diagonal update R - sum_i M_i M_i^T: per-tail
        # contributions, then cross-tail accumulation into the
        # initialized POTRF input
        M_T3 = self._root_coupling_T3
        if self._small_root:
            rows = T * n_b
            wp.launch(create_small_root_update_partial_kernel(dt),
                      dim=[B, SMALL_ROOT_CHUNKS, n_r, n_r],
                      inputs=[small_root_segment(rows), rows, M_T3,
                              self._root_partial], device=dev)
            wp.launch(create_small_root_reduce_kernel(dt), dim=[B, n_r, n_r],
                      inputs=[self._root_partial, self._contrib], device=dev)
        else:
            wp.launch_tiled(create_chunked_root_update_kernel(dt),
                            dim=[B, self._mt, self._mt],
                            inputs=[num_row_chunks(T * n_b), M_T3, M_T3,
                                    self._contrib],
                            block_dim=BLOCK_DIM, device=dev)
        if timer: timer.mark("root_diagonal_update")
        # initialize the POTRF input from R, then accumulate every
        # per-tail contribution atomically (separate launches, so the
        # initialization never races the updates)
        wp.copy(self._root_diagonal, self._R)
        wp.launch(create_atomic_root_update_kernel(dt), dim=[B, n_r, n_r],
                  inputs=[self._contrib, self._root_diagonal], device=dev)
        if timer: timer.mark("root_diagonal_reduce")
        if self._root_via_socu:
            self._root_factor_launch()
        else:
            wp.launch_tiled(create_root_factor_kernel(n_r, dt), dim=[1],
                            inputs=[self._root_diagonal],
                            block_dim=BLOCK_DIM, device=dev)
        if timer: timer.mark("root_factor")

    def _solve_numeric(self, b, timer):
        B, T, n_b, n_r = self._dims
        dt, dev = self._dtype, self._device
        nrhs = b.contrib.shape[2]
        M_T3 = self._root_coupling_T3
        # forward tail solve (in place; consumes the RHS storage)
        b.forward()
        if timer: timer.mark("tail_forward")
        # root RHS update q - sum_i M_i y_i
        if self._small_root:
            rows = T * n_b
            wp.launch(create_small_root_rhs_partial_kernel(dt),
                      dim=[B, SMALL_ROOT_CHUNKS, n_r, nrhs],
                      inputs=[small_root_segment(rows), rows, M_T3, b.rhs3,
                              b.rhs_partial], device=dev)
            wp.launch(create_small_root_reduce_kernel(dt), dim=[B, n_r, nrhs],
                      inputs=[b.rhs_partial, b.contrib], device=dev)
        elif nrhs == 1:
            partitions = ROOT_RHS_VECTOR_PARTITIONS
            rows = T * n_b
            wp.launch(create_root_rhs_vector_partial_kernel(
                          partitions, n_r, dt),
                      dim=[B, partitions, n_r],
                      inputs=[(rows + partitions - 1) // partitions,
                              rows, M_T3, b.rhs3, b.vector_rhs_partial],
                      device=dev)
            wp.launch(create_root_rhs_vector_reduce_kernel(
                          partitions, n_r, dt), dim=[B, n_r],
                      inputs=[b.vector_rhs_partial, b.contrib], device=dev)
        else:
            wp.launch_tiled(create_chunked_root_rhs_kernel(nrhs, dt),
                            dim=[B, self._mt],
                            inputs=[num_row_chunks(T * n_b), M_T3, b.rhs3,
                                    b.contrib],
                            block_dim=BLOCK_DIM, device=dev)
        # the RHS root buffer already holds q; accumulate atomically
        wp.launch(create_atomic_root_rhs_kernel(dt), dim=[B, n_r, nrhs],
                  inputs=[b.contrib, b.root_rhs], device=dev)
        if timer: timer.mark("root_rhs_update")
        b.root_solve()
        if timer: timer.mark("root_solve")
        # tail correction y_i - M_i^T x_r, straight into the output
        if self._small_root:
            wp.launch(create_small_root_correction_kernel(nrhs, dt),
                      dim=[B, T * n_b, nrhs],
                      inputs=[n_r, M_T3, b.rhs3, b.root_rhs, b.out3], device=dev)
        elif nrhs == 1:
            wp.launch(create_tail_root_vector_correction_kernel(n_r, dt),
                      dim=[B, T * n_b],
                      inputs=[n_r, M_T3, b.rhs3, b.root_rhs, b.out3],
                      device=dev)
        else:
            wp.launch_tiled(create_chunked_recover_kernel(nrhs, dt),
                            dim=[B, num_row_chunks(T * n_b)],
                            inputs=[self._mt, M_T3, b.rhs3, b.root_rhs, b.out3],
                            block_dim=BLOCK_DIM, device=dev)
        if timer: timer.mark("tail_rhs_correction")
        # backward tail solve, in place in the output
        b.backward()
        if timer: timer.mark("tail_backward")
        # root result store, the last graph node: the root solve ran in
        # the RHS root buffer, which the tail correction also reads.
        # Skipped when the output aliases that buffer.
        if b.out.root.ptr != b.root_rhs.ptr:
            wp.copy(b.out.root, b.root_rhs)
        if timer: timer.mark("root_store")

#: Backward-compatible export name (the historical class name).
TreeSolver = Solver

"""SOCU-prefix hybrid GPU solver for endpoint-coupled scenario trees.

Three-way partition per scenario (leaf-to-root storage): SOCU factors
the block-tridiagonal prefix (blocks ``0..T-2``) with its batched
odd-even cyclic reduction; the root-facing final node (block ``T-1``)
and the coupling ``G_T`` stay OUTSIDE SOCU's permutation and are
handled by fused boundary kernels; the shared root is a dense
Cholesky.  The prefix-to-final-node connector is transformed and
stored only on the SOCU elimination-tree ancestor path of the prefix
endpoint (``O(log T)`` blocks) -- there is no ``(B, T, n_b, n_r)``
coupling tensor and no full-prefix connector anywhere.

Lifecycle::

    solver = EndpointTreeSolver(shape, device="cuda:0")
    solver.update(matrix)     # copy + pad values, invalidate factor
    solver.factorize()        # one captured CUDA graph
    x = solver.solve(rhs, out=workspace)   # one captured CUDA graph

Warm ``factorize()`` and ``solve()`` replay captured CUDA graphs and
allocate nothing.  ``solve`` does NOT consume the RHS: the bound
device buffers are read by a pack kernel inside the graph.  Host
inputs are staged as a convenience (never timed).  Odd block sizes
that violate SOCU's padding rules are padded internally (identity
dummy diagonals, zero couplings); the public API stays logical and the
padded size is reported by :attr:`padded_block_dim`.
"""

import numpy as np
import warp as wp

from src.endpoint_tree._reuse import (
    TailEngine,
    copy_into,
    create_cholesky_factor_launch,
    create_root_factor_kernel,
    create_root_solve_kernel,
    is_block_size_aligned,
    require_cuda_device,
    wp_dtype,
)
from src.endpoint_tree.kernels import BLOCK_DIM
from src.endpoint_tree.kernels.boundary import (
    FUSED_ROOT_MAX,
    FUSED_ROOT_MAX_TAILS,
    create_boundary_factor_kernel,
    create_final_node_recover_kernel,
    create_final_node_rhs_kernel,
    create_fused_root_factor_kernel,
    create_fused_root_solve_kernel,
    create_root_rhs_kernel,
    create_root_schur_kernel,
)
from src.endpoint_tree.kernels.tail import (
    build_forward_path_program,
    create_pack_rhs_kernel,
    create_path_transform_kernel,
    create_unpack_solution_kernel,
    pad_blocks,
)
from src.endpoint_tree.problem import (
    EndpointTreeMatrix,
    EndpointTreeShape,
    EndpointTreeVector,
)

__all__ = ["EndpointTreeSolver"]


def _ptr(array):
    return array.ptr if isinstance(array, wp.array) else id(array)


def padded_dim(n_b: int, dtype) -> int:
    """Smallest SOCU-aligned storage block size >= ``n_b``."""
    n = n_b
    while not is_block_size_aligned(n, dtype):
        n += 1
    return n


class _SolveBinding:
    def __init__(self, key):
        self.key = key
        self.graph = None


class EndpointTreeSolver:
    """Direct solver for one fixed :class:`EndpointTreeShape` (see the
    module docstring for the algorithm and lifecycle)."""

    def __init__(self, shape: EndpointTreeShape, device="cuda:0"):
        if not isinstance(shape, EndpointTreeShape):
            raise TypeError("shape must be an EndpointTreeShape")
        self._shape = shape
        self._device = require_cuda_device(device)
        self._dtype = wp_dtype(shape.precision)
        B, T, n_b, n_r = shape.dims()
        self._P = P = T - 1
        self._n_p = n_p = padded_dim(n_b, self._dtype)
        dt, dev = self._dtype, self._device

        # ancestor path of the prefix endpoint (host precomputed)
        self._path, ops = build_forward_path_program(P)
        Lp = len(self._path)
        self._Lp = Lp
        if P > 0:
            arr = np.asarray(ops, dtype=np.int32).reshape(-1, 5)
            self._op_kind = wp.array(arr[:, 0], dtype=wp.int32, device=dev)
            self._op_src = wp.array(arr[:, 1], dtype=wp.int32, device=dev)
            self._op_dst = wp.array(arr[:, 2], dtype=wp.int32, device=dev)
            self._op_od = wp.array(arr[:, 3], dtype=wp.int32, device=dev)
            self._op_lslot = wp.array(arr[:, 4], dtype=wp.int32,
                                      device=dev)
            self._n_ops = arr.shape[0]
            self._slots = wp.array(np.asarray(self._path, dtype=np.int32),
                                   dtype=wp.int32, device=dev)
            # SOCU prefix engine (pristine staging + factor buffers)
            self._engine = TailEngine(B, P, n_p, dt, dev)
            self._prefix_factor = create_cholesky_factor_launch(
                self._engine.diag_factor, self._engine.offdiag_factor,
                dtype=dt, device=dev)
            self._v = wp.zeros((B, P, n_p, 1), dtype=dt, device=dev)
            self._prefix_forward = self._engine.build_forward_launch(
                self._v)
            self._prefix_backward = self._engine.build_backward_launch(
                self._v)
            self._H = wp.zeros((B, Lp, n_p, n_p), dtype=dt, device=dev)
            self._H_zero = wp.zeros_like(self._H)
            self._Bc = wp.zeros((B, n_p, n_p), dtype=dt, device=dev)
        else:
            self._engine = None
            self._v = wp.zeros((B, 1, n_p, 1), dtype=dt, device=dev)
            self._H = wp.zeros((B, 0, n_p, n_p), dtype=dt, device=dev)
            self._slots = wp.zeros(1, dtype=wp.int32, device=dev)

        # final node and root buffers
        self._D0f = wp.zeros((B, n_p, n_p), dtype=dt, device=dev)
        self._G = wp.zeros((B, n_p, n_r), dtype=dt, device=dev)
        self._R = wp.zeros((n_r, n_r), dtype=dt, device=dev)
        self._J = wp.zeros((B, n_p, n_p), dtype=dt, device=dev)
        self._M = wp.zeros((B, n_p, n_r), dtype=dt, device=dev)
        self._S = wp.zeros((n_r, n_r), dtype=dt, device=dev)
        self._r0 = wp.zeros((B, n_p), dtype=dt, device=dev)
        self._v0 = wp.zeros((B, n_p), dtype=dt, device=dev)
        self._w0 = wp.zeros((B, n_p), dtype=dt, device=dev)
        self._q = wp.zeros((n_r, 1), dtype=dt, device=dev)
        self._y = wp.zeros((n_r, 1), dtype=dt, device=dev)
        # host-input staging (convenience path only)
        self._stage_tail = wp.zeros((B, T, n_b), dtype=dt, device=dev)
        self._stage_root = wp.zeros((n_r,), dtype=dt, device=dev)
        self._stage_out = None

        # kernels (compile-time specialization on padded sizes)
        self._k_path = create_path_transform_kernel(n_p, dt)
        self._k_boundary = create_boundary_factor_kernel(n_p, n_r, Lp, dt)
        self._k_pack = create_pack_rhs_kernel(dt)
        self._k_unpack = create_unpack_solution_kernel(dt)
        self._k_fnode_rhs = create_final_node_rhs_kernel(n_p, Lp, dt)
        self._k_fnode_rec = create_final_node_recover_kernel(n_p, n_r,
                                                             Lp, dt)
        self._fused_root = (n_r <= FUSED_ROOT_MAX
                            and B <= FUSED_ROOT_MAX_TAILS)
        if self._fused_root:
            self._k_root_factor = create_fused_root_factor_kernel(
                n_p, n_r, dt)
            self._k_root_solve = create_fused_root_solve_kernel(
                n_p, n_r, dt)
        else:
            self._k_schur = create_root_schur_kernel(dt)
            self._k_root_rhs = create_root_rhs_kernel(dt)
            self._k_root_factor = create_root_factor_kernel(n_r, dt)
            self._k_root_solve = create_root_solve_kernel(n_r, 1, dt)

        self._has_values = False
        self._has_factor = False
        self._factor_graph = None
        self._binding = None
        self.factor_graph_nodes = None
        self.solve_graph_nodes = None

    # ------------------------------------------------------------- public
    @property
    def shape(self) -> EndpointTreeShape:
        """The fixed logical problem shape."""
        return self._shape

    @property
    def padded_block_dim(self) -> int:
        """Internal SOCU-aligned storage block size (equals the logical
        ``tail_block_dim`` when that is already aligned)."""
        return self._n_p

    @property
    def connector_path(self) -> tuple:
        """Physical prefix slots of the stored connector (the SOCU
        elimination-tree ancestor path of the prefix endpoint)."""
        return tuple(self._path)

    @property
    def boundary_coupling_entries(self) -> int:
        """Stored root-coupling entries: ``B * n_p * n_r``."""
        return int(np.prod(self._G.shape))

    @property
    def persistent_buffer_shapes(self) -> dict:
        """Shapes of every persistent device buffer (the endpoint and
        path-sparsity guarantees are testable against these)."""
        out = {"D0_final": tuple(self._D0f.shape),
               "G_values": tuple(self._G.shape),
               "R_values": tuple(self._R.shape),
               "J_factor": tuple(self._J.shape),
               "M_coupling": tuple(self._M.shape),
               "S_root": tuple(self._S.shape),
               "H_connector": tuple(self._H.shape),
               "v_prefix": tuple(self._v.shape)}
        if self._engine is not None:
            out["socu_diag"] = tuple(self._engine.diag_factor.shape)
            out["socu_offdiag"] = tuple(self._engine.offdiag_factor.shape)
        return out

    def factor_diagnostics(self) -> dict:
        """Host copies of the condensed final-node factor, the root
        factor, and the connector path (test/metadata use)."""
        if not self._has_factor:
            raise RuntimeError("factor_diagnostics() requires "
                               "factorize()")
        return {"J": self._J.numpy(), "S_factor": self._S.numpy(),
                "H_path": self._H.numpy(), "path": tuple(self._path)}

    def update(self, matrix: EndpointTreeMatrix) -> None:
        """Copy (and pad, if needed) values into the persistent buffers
        and invalidate the factor."""
        if not isinstance(matrix, EndpointTreeMatrix):
            raise TypeError(
                "update() requires an EndpointTreeMatrix (convert general "
                "matrices explicitly with "
                "EndpointTreeMatrix.from_general_tree_matrix)")
        if matrix.shape != self._shape:
            raise ValueError(f"matrix shape {matrix.shape} does not match "
                             f"solver shape {self._shape}")
        B, T, n_b, n_r = self._shape.dims()
        P, n_p = self._P, self._n_p
        D, E, G = pad_blocks(np.asarray(matrix.D), np.asarray(matrix.E),
                             np.asarray(matrix.G_T), n_p)
        if P > 0:
            self._engine.stage(D[:, :P], E[:, :P - 1])
            # connector B = (prefix rows, final cols) = E[T-2]^T
            copy_into(self._Bc,
                      np.ascontiguousarray(
                          np.swapaxes(E[:, P - 1], -1, -2)), "B_c")
        copy_into(self._D0f, D[:, T - 1], "D0")
        copy_into(self._G, G, "G_T")
        copy_into(self._R, np.asarray(matrix.R), "R")
        self._has_values = True
        self._has_factor = False

    def factorize(self) -> None:
        """Numeric factorization: SOCU prefix, path-sparse connector
        transform, fused final node, root (one captured CUDA graph)."""
        if not self._has_values:
            raise RuntimeError("factorize() requires update() first")
        if self._factor_graph is None:
            self._factor_numeric()  # warm compile + real compute
            self.factor_graph_nodes = self._count_launches(
                self._factor_numeric)
            with wp.ScopedCapture(device=self._device) as cap:
                self._factor_numeric()
            self._factor_graph = cap.graph
        wp.capture_launch(self._factor_graph)
        self._has_factor = True

    def solve(self, rhs: EndpointTreeVector,
              out: "EndpointTreeVector | None" = None, *,
              graph: bool = True) -> EndpointTreeVector:
        """Solve ``K x = rhs`` for one vector (see the module docstring
        for the RHS ownership contract; ``graph=False`` enqueues the
        same launches eagerly)."""
        if not self._has_factor:
            raise RuntimeError("solve() requires factorize()")
        if not isinstance(rhs, EndpointTreeVector):
            raise TypeError("rhs must be an EndpointTreeVector")
        if out is not None and not isinstance(out, EndpointTreeVector):
            raise TypeError("out must be an EndpointTreeVector")
        B, T, n_b, n_r = self._shape.dims()

        rhs_dev = (isinstance(rhs.tail, wp.array)
                   and isinstance(rhs.root, wp.array))
        if not rhs_dev:
            copy_into(self._stage_tail, np.asarray(rhs.tail), "rhs.tail")
            copy_into(self._stage_root, np.asarray(rhs.root), "rhs.root")
            rhs = EndpointTreeVector(self._shape, self._stage_tail,
                                     self._stage_root)
        else:
            self._check_device_array(rhs.tail, (B, T, n_b), "rhs.tail")
            self._check_device_array(rhs.root, (n_r,), "rhs.root")
        if out is None:
            if self._stage_out is None:
                self._stage_out = EndpointTreeVector(
                    self._shape,
                    wp.zeros((B, T, n_b), dtype=self._dtype,
                             device=self._device),
                    wp.zeros((n_r,), dtype=self._dtype,
                             device=self._device))
            out = self._stage_out
        else:
            self._check_device_array(out.tail, (B, T, n_b), "out.tail")
            self._check_device_array(out.root, (n_r,), "out.root")

        if not graph:
            self._solve_numeric(rhs, out)
            return out
        key = (_ptr(rhs.tail), _ptr(rhs.root), _ptr(out.tail),
               _ptr(out.root))
        b = self._binding
        if b is None or b.key != key:
            b = self._binding = _SolveBinding(key)
            b.rhs, b.out = rhs, out
        if b.graph is None:
            self._solve_numeric(b.rhs, b.out)  # warm compile
            self.solve_graph_nodes = self._count_launches(
                lambda: self._solve_numeric(b.rhs, b.out))
            with wp.ScopedCapture(device=self._device) as cap:
                self._solve_numeric(b.rhs, b.out)
            b.graph = cap.graph
        wp.capture_launch(b.graph)
        return out

    def time_phases(self, repeats: int = 50) -> dict:
        """CUDA-event medians (ms) of the individually synchronized
        pipeline phases (instrumentation; never part of the solve
        path; solver state is restored by re-deriving the factor).

        Factor phases: ``socu_prefix_ms`` (refresh + SOCU prefix
        factorization), ``connector_ms`` (path-sparse transform),
        ``final_node_ms`` (fused condense/factor/coupling),
        ``root_ms``.  Solve phases: ``prefix_forward_ms``,
        ``final_node_solve_ms``, ``root_solve_ms``,
        ``recover_backward_ms`` (recovery + path correction + SOCU
        backward).
        """
        if not self._has_factor:
            raise RuntimeError("time_phases() requires factorize()")
        dev = self._device
        e0 = wp.Event(dev, enable_timing=True)
        e1 = wp.Event(dev, enable_timing=True)

        def med(fn):
            for _ in range(5):
                fn()
            wp.synchronize_device(dev)
            samples = []
            for _ in range(repeats):
                wp.record_event(e0)
                fn()
                wp.record_event(e1)
                wp.synchronize_event(e1)
                samples.append(wp.get_event_elapsed_time(
                    e0, e1, synchronize=False))
            return float(np.median(samples))

        out = {}
        if self._P > 0:
            def socu_prefix():
                self._engine.refresh()
                self._prefix_factor()

            out["socu_prefix_ms"] = med(socu_prefix)
            out["connector_ms"] = med(self._connector_transform)
            out["prefix_forward_ms"] = med(self._prefix_forward)
            out["recover_backward_ms"] = med(lambda: (
                self._launch_fnode_recover(), self._prefix_backward()))
        else:
            out["socu_prefix_ms"] = 0.0
            out["connector_ms"] = 0.0
            out["prefix_forward_ms"] = 0.0
            out["recover_backward_ms"] = med(self._launch_fnode_recover)
        out["final_node_ms"] = med(self._launch_boundary_factor)
        out["root_ms"] = med(self._launch_root_factor)
        out["final_node_solve_ms"] = med(self._launch_fnode_rhs)
        out["root_solve_ms"] = med(self._launch_root_solve)
        self._factor_numeric()  # leave a consistent factor behind
        wp.synchronize_device(dev)
        return out

    # ----------------------------------------------------------- internal
    def _check_device_array(self, arr, shape, name):
        if not isinstance(arr, wp.array):
            raise TypeError(f"{name} must be a Warp device array")
        if arr.dtype != self._dtype:
            raise TypeError(f"{name}: expected dtype {self._dtype}")
        if tuple(arr.shape) != tuple(shape):
            raise ValueError(f"{name}: expected shape {shape}, got "
                             f"{tuple(arr.shape)}")
        if arr.device != self._device:
            raise ValueError(f"{name} must live on {self._device}")

    def _count_launches(self, fn):
        """Count host enqueue operations (kernel launches, recorded
        launches, copies) of one eager pipeline run (setup-time
        instrumentation only)."""
        count = [0]
        saved = (wp.launch, wp.launch_tiled, wp.copy)
        launch_cls = type(wp.launch_tiled(
            self._k_pack, dim=[1, 1, 1],
            inputs=[1, 1, self._stage_tail, self._v, self._r0],
            block_dim=32, device=self._device, record_cmd=True))
        saved_launch = launch_cls.launch

        def wrap(f):
            def inner(*a, **k):
                count[0] += 1
                return f(*a, **k)
            return inner

        wp.launch, wp.launch_tiled, wp.copy = map(wrap, saved)
        launch_cls.launch = wrap(saved_launch)
        try:
            fn()
        finally:
            wp.launch, wp.launch_tiled, wp.copy = saved
            launch_cls.launch = saved_launch
        return count[0]

    # ------------------------------------------------ pipeline pieces
    def _connector_transform(self):
        wp.copy(self._H, self._H_zero)
        wp.copy(self._H[:, self._path.index(self._P - 1)], self._Bc)
        wp.launch_tiled(self._k_path, dim=[self._H.shape[0]],
                        inputs=[self._n_ops, self._op_kind, self._op_src,
                                self._op_dst, self._op_od,
                                self._op_lslot,
                                self._engine.diag_factor,
                                self._engine.offdiag_factor, self._H],
                        block_dim=BLOCK_DIM, device=self._device)

    def _launch_boundary_factor(self):
        B = self._D0f.shape[0]
        wp.copy(self._M, self._G)   # transformed in place to M = J^-1 G
        wp.launch_tiled(self._k_boundary, dim=[B],
                        inputs=[self._D0f, self._H, self._J, self._M],
                        block_dim=BLOCK_DIM, device=self._device)

    def _launch_root_factor(self):
        B, n_r = self._M.shape[0], self._R.shape[0]
        if self._fused_root:
            wp.launch_tiled(self._k_root_factor, dim=[1],
                            inputs=[B, self._R, self._M, self._S],
                            block_dim=BLOCK_DIM, device=self._device)
        else:
            wp.copy(self._S, self._R)
            wp.launch(self._k_schur, dim=[B, n_r, n_r],
                      inputs=[self._M, self._S], device=self._device)
            wp.launch_tiled(self._k_root_factor, dim=[1],
                            inputs=[self._S], block_dim=BLOCK_DIM,
                            device=self._device)

    def _launch_fnode_rhs(self):
        B = self._D0f.shape[0]
        wp.launch_tiled(self._k_fnode_rhs, dim=[B],
                        inputs=[self._slots, self._r0, self._H, self._v,
                                self._J, self._v0],
                        block_dim=BLOCK_DIM, device=self._device)

    def _launch_root_solve(self):
        B, n_r = self._M.shape[0], self._R.shape[0]
        if self._fused_root:
            wp.launch_tiled(self._k_root_solve, dim=[1],
                            inputs=[B, self._q, self._M, self._v0,
                                    self._S, self._y],
                            block_dim=BLOCK_DIM, device=self._device)
        else:
            wp.copy(self._y, self._q)
            wp.launch(self._k_root_rhs, dim=[B, n_r],
                      inputs=[self._M, self._v0, self._y],
                      device=self._device)
            wp.launch_tiled(self._k_root_solve, dim=[1],
                            inputs=[self._S, self._y],
                            block_dim=BLOCK_DIM, device=self._device)

    def _launch_fnode_recover(self):
        B = self._D0f.shape[0]
        wp.launch_tiled(self._k_fnode_rec, dim=[B],
                        inputs=[self._slots, self._M, self._y, self._J,
                                self._H, self._v0, self._w0, self._v],
                        block_dim=BLOCK_DIM, device=self._device)

    def _factor_numeric(self):
        if self._P > 0:
            self._engine.refresh()
            self._prefix_factor()
            self._connector_transform()
        self._launch_boundary_factor()
        self._launch_root_factor()

    def _solve_numeric(self, rhs, out):
        B, T, n_b, n_r = self._shape.dims()
        dev = self._device
        wp.launch(self._k_pack, dim=[B, T, self._n_p],
                  inputs=[n_b, self._P, rhs.tail, self._v, self._r0],
                  device=dev)
        wp.copy(self._q, rhs.root.reshape((n_r, 1)))
        if self._P > 0:
            self._prefix_forward()
        self._launch_fnode_rhs()
        self._launch_root_solve()
        self._launch_fnode_recover()
        if self._P > 0:
            self._prefix_backward()
        wp.launch(self._k_unpack, dim=[B, T, n_b],
                  inputs=[self._P, self._v, self._w0, out.tail],
                  device=dev)
        wp.copy(out.root, self._y.reshape((n_r,)))

"""Narrow SOCU API patch: partial-solve launch builders.

UPSTREAM STATUS.  The forward-only and backward-only solve builders were
adopted upstream (socu commit 2ff9dc0, tail feature/sequential) as
``create_cholesky_forward_substitution_launch`` / ``create_cholesky_backward_substitution_launch``;
this module re-exports them under the names used throughout this project.
The fused factor-plus-forward builder below is still local-only and is
written to be moved verbatim into ``socu/block_tridiag_solver.py`` when
upstreamed.  It imports SOCU's own public kernel factories and adds only
launch orchestration; the stride/offset schedule and stream dependencies
mirror the upstream factor-and-solve builder exactly.

Math: SOCU's forward operator ``F`` satisfies ``F^T F = K^{-1}`` (it is
the inverse of a triangular square-root factor of a symmetric permutation
of ``K``), and forward followed by backward equals the full solve --
the identities the factor-space formulation relies on.
"""

import math

import warp as wp
import socu.block_tridiag_solver as _upstream
from socu.block_tridiag_solver import (
    calculate_off_diag_storage_len,
    calculate_recursive_iterations,
    create_cholesky_backward_substitution_launch,
    create_cholesky_factor_forward_substituition_iteration_kernel,
    create_cholesky_factor_gemm_nn_blocked_kernel,
    create_cholesky_factor_potrf_l_blocked_kernel,
    create_cholesky_factor_syrk_ln_blocked_kernel,
    create_cholesky_factor_syrk_lt_blocked_kernel,
    create_cholesky_factor_trsm_llnn_blocked_kernel,
    create_cholesky_factor_trsm_rltn_blocked_kernel,
    create_forward_substitution_gemm_nn_blocked_kernel,
    create_forward_substitution_gemm_tn_blocked_kernel,
    create_cholesky_forward_substitution_launch,
    create_forward_substitution_trsm_llnn_blocked_kernel,
    optimal_problem_settings,
)
from socu.utils import create_cuda_graph_callback

#: Upstream builders under this project's historical names.
create_cholesky_forward_solve_launch = create_cholesky_forward_substitution_launch
create_cholesky_backward_solve_launch = create_cholesky_backward_substitution_launch


def _solve_launch_preamble(L, E, x, block_dim, block_size, dtype, phase):
    """Shared shape validation/normalization (mirrors upstream)."""
    assert E.ndim == L.ndim and x.ndim == L.ndim, \
        "L, E, and x must have the same number of dimensions"

    if L.ndim == 4:
        batch_dim = L.shape[0]
    elif L.ndim == 3:
        batch_dim = 1
        L = L.reshape((1, *L.shape))
        E = E.reshape((1, *E.shape))
        x = x.reshape((1, *x.shape))
    else:
        raise ValueError("L must have 4 or 3 dimensions")

    assert E.shape[0] == batch_dim and x.shape[0] == batch_dim, \
        "L, E, and x must have same batch dimension"

    horizon = L.shape[1]
    n = L.shape[2]
    assert L.shape[3] == n, "The last two dimensions of L must be the same"
    assert E.shape[2] == n and E.shape[3] == n, \
        "L and E must have same block size"
    n_E = calculate_off_diag_storage_len(horizon)
    assert E.shape[1] == n_E, \
        "E has incorrect size, use calculate_off_diag_storage_len"

    n_rhs = x.shape[3]
    assert x.shape[1] == horizon and x.shape[2] == n, \
        "x has incorrect dimensions"

    iterations = calculate_recursive_iterations(horizon)

    opt_settings = optimal_problem_settings(n, dtype)
    if block_dim is None:
        block_dim = opt_settings["block_dim"][phase]
    if block_size is None:
        block_size = opt_settings["block_size"][phase]

    return (L, E, x, batch_dim, horizon, n, n_rhs, iterations,
            block_dim, block_size)


def _local_factor_and_forward_solve_launch(L: wp.array,
                                           E: wp.array,
                                           x: wp.array,
                                           block_dim=None,
                                           block_size=None,
                                           dtype=wp.float64,
                                           device=None,
                                           stream=None,
                                           use_cuda_graph=False):
    """Fused factorization plus forward half (see module docs)."""
    (L, E, x, batch_dim, horizon, n, n_rhs, iterations,
     block_dim, block_size) = _solve_launch_preamble(
        L, E, x, block_dim, block_size, dtype, "factor")

    if n < block_size:
        fac_fwd_launch = wp.launch_tiled(
            create_cholesky_factor_forward_substituition_iteration_kernel(
                n, n_rhs, dtype),
            record_cmd=True, device=device, stream=stream,
            dim=[batch_dim, 1], inputs=[0, 0, 0, 0, horizon, L, E, x],
            block_dim=block_dim)

        def callback():
            stride = 1
            prev_off_diag_offset = 0
            curr_off_diag_offset = 0
            next_off_diag_offset = horizon - 1
            for _ in range(iterations):
                dim = (horizon + stride) // (2 * stride)
                fac_fwd_launch.set_dim([batch_dim, dim, block_dim])
                fac_fwd_launch.set_params([
                    stride, prev_off_diag_offset, curr_off_diag_offset,
                    next_off_diag_offset])
                fac_fwd_launch.launch()
                stride *= 2
                prev_off_diag_offset = curr_off_diag_offset
                curr_off_diag_offset = next_off_diag_offset
                next_off_diag_offset += horizon // stride - 1

    else:
        if stream is None:
            device = wp.get_device(device)
            stream = device.stream
        else:
            device = stream.device

        stream2 = wp.Stream(device)
        stream3 = wp.Stream(device)
        stream4 = wp.Stream(device)
        stream5 = wp.Stream(device)

        fac_potrf_launch = wp.launch_tiled(
            create_cholesky_factor_potrf_l_blocked_kernel(n, block_size, dtype),
            record_cmd=True, device=device, dim=[batch_dim, 1],
            inputs=[0, L], block_dim=block_dim)
        fac_trsm_rltn_launch = wp.launch_tiled(
            create_cholesky_factor_trsm_rltn_blocked_kernel(n, block_size, dtype),
            record_cmd=True, device=device, dim=[batch_dim, 1],
            inputs=[0, 0, horizon, L, E], block_dim=block_dim)
        fac_trsm_llnn_launch = wp.launch_tiled(
            create_cholesky_factor_trsm_llnn_blocked_kernel(n, block_size, dtype),
            record_cmd=True, device=device, dim=[batch_dim, 1],
            inputs=[0, 0, L, E], block_dim=block_dim)
        fac_syrk_ln_launch = wp.launch_tiled(
            create_cholesky_factor_syrk_ln_blocked_kernel(n, block_size, dtype),
            record_cmd=True, device=device, dim=[batch_dim, 1, 1],
            inputs=[0, 0, horizon, L, E], block_dim=block_dim)
        fac_syrk_lt_launch = wp.launch_tiled(
            create_cholesky_factor_syrk_lt_blocked_kernel(n, block_size, dtype),
            record_cmd=True, device=device, dim=[batch_dim, 1, 1],
            inputs=[0, 0, L, E], block_dim=block_dim)
        fac_gemm_nn_launch = wp.launch_tiled(
            create_cholesky_factor_gemm_nn_blocked_kernel(n, block_size, dtype),
            record_cmd=True, device=device, dim=[batch_dim, 1, 1],
            inputs=[0, 0, 0, horizon, E], block_dim=block_dim)

        forward_trsm_llnn_launch = wp.launch_tiled(
            create_forward_substitution_trsm_llnn_blocked_kernel(
                n, n_rhs, block_size, dtype),
            record_cmd=True, device=device, dim=[batch_dim, 1],
            inputs=[0, L, x], block_dim=block_dim)
        forward_gemm_nn_launch = wp.launch_tiled(
            create_forward_substitution_gemm_nn_blocked_kernel(
                n, n_rhs, block_size, dtype),
            record_cmd=True, device=device, dim=[batch_dim, 1, 1],
            inputs=[0, 0, horizon, E, x], block_dim=block_dim)
        forward_gemm_tn_launch = wp.launch_tiled(
            create_forward_substitution_gemm_tn_blocked_kernel(
                n, n_rhs, block_size, dtype),
            record_cmd=True, device=device, dim=[batch_dim, 1, 1],
            inputs=[0, 0, E, x], block_dim=block_dim)

        def callback():
            num_blocks_per_row = math.ceil(n / block_size)
            num_tril_blocks = num_blocks_per_row * (num_blocks_per_row + 1) // 2
            num_fac_gemm_blocks = num_blocks_per_row * num_blocks_per_row

            num_blocks_per_rhs_col = math.ceil(n_rhs / block_size)
            num_solve_gemm_blocks = num_blocks_per_row * num_blocks_per_rhs_col

            stride = 1
            curr_off_diag_offset = 0
            next_off_diag_offset = horizon - 1
            for i in range(iterations):
                dim = (horizon + stride) // (2 * stride)

                if i > 0:
                    stream.wait_stream(stream2)

                fac_potrf_launch.set_dim([batch_dim, dim, block_dim])
                fac_potrf_launch.set_params([stride])
                fac_potrf_launch.launch(stream=stream)

                stream4.wait_stream(stream)
                if i > 0:
                    stream4.wait_stream(stream5)

                forward_trsm_llnn_launch.set_dim([batch_dim, dim, block_dim])
                forward_trsm_llnn_launch.set_params([stride])
                forward_trsm_llnn_launch.launch(stream=stream4)

                if i < iterations - 1:
                    stream2.wait_stream(stream)
                    stream5.wait_stream(stream4)

                    if i > 0:
                        stream.wait_stream(stream3)
                        stream2.wait_stream(stream3)

                    fac_trsm_rltn_launch.set_dim([batch_dim, dim, block_dim])
                    fac_trsm_rltn_launch.set_params(
                        [stride, curr_off_diag_offset])
                    fac_trsm_rltn_launch.launch(stream=stream)

                    fac_trsm_llnn_launch.set_dim([batch_dim, dim, block_dim])
                    fac_trsm_llnn_launch.set_params(
                        [stride, curr_off_diag_offset])
                    fac_trsm_llnn_launch.launch(stream=stream2)

                    stream4.wait_stream(stream)
                    stream5.wait_stream(stream2)

                    forward_gemm_nn_launch.set_dim(
                        [batch_dim, num_solve_gemm_blocks, dim, block_dim])
                    forward_gemm_nn_launch.set_params(
                        [stride, curr_off_diag_offset])
                    forward_gemm_nn_launch.launch(stream=stream4)

                    forward_gemm_tn_launch.set_dim(
                        [batch_dim, num_solve_gemm_blocks, dim, block_dim])
                    forward_gemm_tn_launch.set_params(
                        [stride, curr_off_diag_offset])
                    forward_gemm_tn_launch.launch(stream=stream5)

                    stream3.wait_stream(stream)
                    stream3.wait_stream(stream2)

                    fac_syrk_ln_launch.set_dim(
                        [batch_dim, num_tril_blocks, dim, block_dim])
                    fac_syrk_ln_launch.set_params(
                        [stride, curr_off_diag_offset])
                    fac_syrk_ln_launch.launch(stream=stream)

                    fac_syrk_lt_launch.set_dim(
                        [batch_dim, num_tril_blocks, dim, block_dim])
                    fac_syrk_lt_launch.set_params(
                        [stride, curr_off_diag_offset])
                    fac_syrk_lt_launch.launch(stream=stream2)

                    fac_gemm_nn_launch.set_dim(
                        [batch_dim, num_fac_gemm_blocks, dim, block_dim])
                    fac_gemm_nn_launch.set_params(
                        [stride, curr_off_diag_offset, next_off_diag_offset])
                    fac_gemm_nn_launch.launch(stream=stream3)

                stride *= 2
                curr_off_diag_offset = next_off_diag_offset
                next_off_diag_offset += horizon // stride - 1

            stream.wait_stream(stream4)
            if iterations > 1:
                stream.wait_stream(stream2)
                stream.wait_stream(stream3)
                stream.wait_stream(stream5)

    if use_cuda_graph:
        return create_cuda_graph_callback(callback, device, stream)
    return callback


#: Capability-driven selection: use the upstream fused builder as soon as
#: SOCU adopts it; fall back to the verbatim local implementation above.
create_cholesky_factor_and_forward_solve_launch = getattr(
    _upstream, "create_cholesky_factor_and_forward_solve_launch",
    _local_factor_and_forward_solve_launch)

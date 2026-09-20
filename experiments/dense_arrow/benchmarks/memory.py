"""Device and host memory guards: predict a case's footprint before any
allocation so oversized cases are skipped with an explicit record instead
of crashing the run (or the machine)."""

from experiments.general_arrow.benchmarks.problems import ProblemSpec


def free_device_bytes(device="cuda:0") -> int:
    """Currently free device memory."""
    import warp as wp
    wp.init()
    return int(wp.get_device(device).free_memory)


def available_host_bytes() -> int:
    """Available host memory (Linux); a huge sentinel when unknown so the
    guard never blocks on missing information."""
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 1 << 62  # unknown: do not guard


def estimate_host_bytes(spec: ProblemSpec, method_def: dict) -> int:
    """Rough peak host memory of generation plus method-side assembly.

    cuDSS host CSR assembly builds int64 row/col index lists, their
    concatenated copies, and the COO-to-CSR conversion buffers -- about
    150 bytes per stored nonzero at peak; the generator itself holds the
    coupling block and one same-sized workspace.
    """
    B, T, n_b, n_r = spec.shape.dims()
    gen = (2 * B * T * n_b * n_r + 4 * B * T * n_b * n_b) * 8
    if method_def.get("kind") == "cudss":
        return gen + spec.nnz_lower * 150
    return gen


def estimate_method_bytes(spec: ProblemSpec, method_def: dict) -> int:
    """Predicted device memory of one method on one problem."""
    kind = method_def.get("kind", "tree")
    itemsize = 8 if spec.precision == "float64" else 4
    if kind == "tree":
        estimate_solver_bytes = _estimate_solver_bytes
        return estimate_solver_bytes(spec.shape, spec.num_rhs)
    if kind == "cudss":
        nnz = spec.nnz_lower
        dim = spec.total_dimension
        # values + indices + rhs/solution, plus a generous 3x factor for
        # the cuDSS factor and workspace (checked again by cuDSS's own
        # estimate)
        return int((nnz * (itemsize + 4) + dim * 2 * itemsize) * 4)
    return 0


def _estimate_solver_bytes(shape, num_rhs=1) -> int:
    """Predicted persistent device memory of a tree solver (moved here
    from the former src.workspace; used only by the memory guard)."""
    from socu.block_tridiag_solver import calculate_off_diag_storage_len
    B, T, n_b, n_r = shape.dims()
    itemsize = 8 if shape.precision == "float64" else 4
    n_off = calculate_off_diag_storage_len(T)
    blocks = 2 * B * T * n_b * n_b + 2 * B * n_off * n_b * n_b
    blocks += 2 * B * T * n_b * n_r                      # C_T and M^T
    blocks += 4 * n_r * n_r + B * n_r * n_r              # R, update, L_R, contrib
    blocks += 2 * B * T * n_b * num_rhs + n_r * num_rhs + B * n_r * num_rhs
    if n_r < 16:
        blocks += 64 * B * n_r * (n_r + num_rhs)         # scalar partials
    return blocks * itemsize

"""Device and host memory guards: predict a case's footprint before any
allocation so oversized cases are skipped with an explicit record instead
of crashing the run (or the machine)."""

from benchmarks.problems import ProblemSpec


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
    B, T, n_b, n_y = spec.shape.dims()
    gen = (2 * B * T * n_b * n_y + 4 * B * T * n_b * n_b) * 8
    if method_def.get("kind") == "cudss":
        return gen + spec.nnz_lower * 150
    return gen


def estimate_method_bytes(spec: ProblemSpec, method_def: dict) -> int:
    """Predicted device memory of one method on one problem."""
    kind = method_def.get("kind", "tree")
    itemsize = 8 if spec.precision == "float64" else 4
    if kind == "tree":
        from src.workspace import estimate_solver_bytes
        return estimate_solver_bytes(spec.shape, spec.num_rhs)
    if kind == "cudss":
        nnz = spec.nnz_lower
        dim = spec.total_dimension
        # values + indices + rhs/solution, plus a generous 3x factor for
        # the cuDSS factor and workspace (checked again by cuDSS's own
        # estimate)
        return int((nnz * (itemsize + 4) + dim * 2 * itemsize) * 4)
    return 0

"""Every import from the general solver package in one place.

The dependency direction is one-way: ``endpoint_tree`` reuses selected
primitives from ``src``; nothing in ``src`` may import this package.
Keeping the imports centralized makes the shared surface explicit and
prevents the specialized solver from accidentally depending on (or
mutating) the general implementation.
"""

# device/dtype/copy helpers (unchanged semantics)
from src.dense_arrow._utils import (  # noqa: F401
    Launch,
    copy_into,
    init_warp,
    require_cuda_device,
    wp_dtype,
)

# dense root Cholesky/solve tile kernels (work for arbitrary n_r)
from src.dense_arrow.kernels.root import (  # noqa: F401
    create_root_factor_kernel,
    create_root_solve_kernel,
)

# the SOCU adapter (batched prefix factorization and substitutions)
# and the upstream factor-launch builder it does not wrap
from socu.block_tridiag_solver import (  # noqa: F401
    calculate_off_diag_storage_len,
    create_cholesky_factor_launch,
)
from src.dense_arrow.socu import TailEngine, is_block_size_aligned  # noqa: F401

# the general containers, used ONLY by the conversion/validation
# adapters (to_general_tree_matrix / from_general_tree_matrix) and the
# benchmark comparison -- never by the specialized solve path
from src.dense_arrow.problem import TreeMatrix, TreeShape, TreeVector  # noqa: F401

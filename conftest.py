"""Pytest configuration for the tree_socu test suite.

Placing this conftest at the project root makes pytest prepend this
directory to ``sys.path`` so the ``tree_socu`` package is importable.

CPU tests run without a GPU.  Tests marked ``gpu`` are skipped
automatically when Warp cannot find a CUDA device.
"""

import pytest


def _cuda_available() -> bool:
    try:
        import warp as wp
        import logging
        wp.config.log_level = logging.WARNING
        wp.init()
        return wp.get_cuda_device_count() > 0
    except Exception:
        return False


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires a CUDA device")


def pytest_collection_modifyitems(config, items):
    if _cuda_available():
        return
    skip = pytest.mark.skip(reason="no CUDA device available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)

"""Provenance metadata recorded beside every benchmark run: software
versions, GPU state, git status, and a source hash of the exact code."""

import hashlib
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def _run_cmd(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def source_hash() -> str:
    """SHA-256 over everything that determines a measurement: solver and
    benchmark sources, experiment configurations, and the environment
    pins.  Hashed by repository-relative path, so results remain
    attributable to the exact code even when the worktree is not
    committed (recorded beside the git commit and dirty flag)."""
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for pattern in ("src/**/*.py", "benchmarks/*.py", "baselines/*.py",
                    "scripts/*.py", "experiments/*.yaml",
                    "environment.yml", "pyproject.toml"):
        for f in sorted(root.glob(pattern)):
            h.update(str(f.relative_to(root)).encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def provenance_block(metadata: dict, config_path) -> dict:
    """The immutable provenance recorded inside every raw record
    (including skipped and failed ones), so a record is interpretable
    without ``run_metadata.json``."""
    keys = ("source_hash", "git_commit", "git_dirty", "warp", "numpy",
            "python", "socu_version", "socu_commit", "cudss_version",
            "gpu_name", "driver_version")
    block = {k: metadata.get(k) for k in keys if k in metadata}
    block["config"] = str(config_path)
    return block


def collect_metadata(device="cuda:0") -> dict:
    """Software/hardware metadata recorded beside every run."""
    import warp as wp
    md = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": platform.processor() or _run_cmd(
            ["sh", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"]),
        "hostname": platform.node(),
        "numpy": np.__version__,
        "warp": wp.__version__,
        "blas_threads": {k: os.environ.get(k, "") for k in
                         ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                          "MKL_NUM_THREADS")},
        "git_commit": _run_cmd(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_run_cmd(["git", "status", "--porcelain"])),
    }
    try:
        import scipy
        md["scipy"] = scipy.__version__
    except ImportError:
        pass
    try:
        from src.socu_adapter import socu_commit, socu_version
        md["socu_version"] = socu_version()
        md["socu_commit"] = socu_commit()
    except Exception:  # noqa: BLE001
        pass
    try:
        from baselines.cudss import cudss_version
        md["cudss_version"] = cudss_version()
    except Exception:  # noqa: BLE001
        pass
    smi = _run_cmd([
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,temperature.gpu,clocks.sm,"
        "power.draw,utilization.gpu,persistence_mode,mig.mode.current",
        "--format=csv,noheader"])
    if smi:
        fields = [f.strip() for f in smi.splitlines()[0].split(",")]
        keys = ["gpu_name", "gpu_uuid", "driver_version", "gpu_temperature_c",
                "gpu_sm_clock", "gpu_power_draw", "gpu_utilization",
                "persistence_mode", "mig_mode"]
        md.update(dict(zip(keys, fields)))
    md["device"] = device
    md["source_hash"] = source_hash()
    md["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return md


def gpu_snapshot() -> dict:
    """Small per-case GPU state snapshot (thermal/clock drift tracking)."""
    smi = _run_cmd(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,"
                    "power.draw,utilization.gpu", "--format=csv,noheader"])
    if not smi:
        return {}
    f = [x.strip() for x in smi.splitlines()[0].split(",")]
    return dict(zip(["gpu_temperature_c", "gpu_sm_clock", "gpu_power_draw",
                     "gpu_utilization"], f))

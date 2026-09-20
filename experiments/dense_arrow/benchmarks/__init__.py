"""Benchmark engine: case expansion, timed method runs, resumable output.

This package is the importable core behind ``scripts/run_benchmarks.py``.
A YAML experiment configuration defines problem defaults, method variants,
sweeps, and repetition rules; the engine expands them into cases, runs
every method on identical generated instances, and writes one atomically
renamed JSON record per (point, seed, method) into ``results/raw/`` so
interrupted runs resume by skipping existing records.

Submodules: :mod:`~benchmarks.config` (YAML parsing and case
expansion), :mod:`~benchmarks.runners` (timed tree/cuDSS/CPU
runs), :mod:`~benchmarks.memory` (pre-allocation memory guards),
:mod:`~benchmarks.metadata` (provenance), and
:mod:`~benchmarks.storage` (atomic record output).
"""

import json
import sys
import time
from pathlib import Path

import yaml

from experiments.general_arrow.benchmarks.config import case_id, expand_cases, spec_from_params
from experiments.general_arrow.benchmarks.memory import (
    available_host_bytes,
    estimate_host_bytes,
    estimate_method_bytes,
    free_device_bytes,
)
from experiments.general_arrow.benchmarks.metadata import (
    collect_metadata,
    gpu_snapshot,
    provenance_block,
    source_hash,
)
from experiments.general_arrow.benchmarks.runners import RUNNERS
from experiments.general_arrow.benchmarks.storage import atomic_write_json
from experiments.general_arrow.benchmarks.problems import generate_problem

__all__ = [
    "run_config", "expand_cases", "case_id", "spec_from_params",
    "collect_metadata", "source_hash", "atomic_write_json",
]


def run_config(config_path, device="cuda:0", output="results/raw",
               resume=True, overwrite=False, dry_run=False,
               select_sweeps=None, select_methods=None, seed_override=None,
               log=print):
    """Run (or list, with ``dry_run=True``) every case of a YAML config."""
    cfg = yaml.safe_load(Path(config_path).read_text())
    methods = cfg.get("methods", {})
    rules = cfg.get("repetitions", {})
    mem_fraction = float(cfg.get("memory_guard_fraction", 0.8))
    estimate_cond = cfg.get("estimate_condition", True)

    cases = expand_cases(cfg, select_sweeps, select_methods, seed_override)
    if dry_run:
        log(f"config {config_path}: {len(cases)} cases")
        for c in cases:
            log(f"  {case_id(c['sweep'], c['point'], c['seed'], c['method'])}")
        return []

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        # release freed Warp device memory eagerly so the per-case memory
        # guard sees the true free amount between cases
        import warp as wp
        wp.init()
        wp.set_mempool_release_threshold(device, 0)
    except Exception:  # noqa: BLE001
        pass
    metadata = collect_metadata(device)
    provenance = provenance_block(metadata, config_path)
    (out_dir / "run_metadata.json").write_text(
        json.dumps({**metadata, "config": cfg,
                    "command_line": " ".join(sys.argv)}, indent=1))

    written = []
    problem_cache = {}  # (point_json, seed, est) -> GeneratedProblem (last only)
    for i, case in enumerate(cases):
        sweep, point, seed, mname = (case["sweep"], case["point"],
                                     case["seed"], case["method"])
        est_cond = case["estimate_condition"]
        if est_cond is None:
            est_cond = bool(estimate_cond)
        cid = case_id(sweep, point, seed, mname)
        path = out_dir / f"{cid}.json"
        if path.exists() and not overwrite:
            if resume:
                log(f"[{i+1}/{len(cases)}] skip existing {cid}")
                continue
            raise FileExistsError(
                f"{path} exists; pass --overwrite to replace it or --resume "
                f"to skip completed cases")

        spec = spec_from_params({**point, "seed": seed})
        method_def = dict(methods[mname])
        kind = method_def.get("kind", "tree")

        record = {
            "case_id": cid, "sweep": sweep, "method": mname,
            "method_def": method_def, "seed": seed,
            "spec": spec.to_dict(),
            "dimension": spec.total_dimension,
            "nnz_lower_structural": spec.nnz_lower,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_commit": metadata.get("git_commit", ""),
            "git_dirty": metadata.get("git_dirty", False),
            "provenance": provenance,
            "gpu_state": gpu_snapshot(),
        }

        # memory guard before any allocation
        if kind in ("tree", "cudss"):
            need = estimate_method_bytes(spec, method_def)
            free = free_device_bytes()
            record["estimated_bytes"] = need
            record["free_bytes_before"] = free
            need_host = estimate_host_bytes(spec, method_def)
            free_host = available_host_bytes()
            record["estimated_host_bytes"] = need_host
            skip = None
            if need > mem_fraction * free:
                skip = (f"estimated {need/1e9:.2f} GB device exceeds "
                        f"{mem_fraction:.0%} of free {free/1e9:.2f} GB")
            elif need_host > 0.6 * free_host:
                skip = (f"estimated {need_host/1e9:.2f} GB host exceeds "
                        f"60% of available {free_host/1e9:.2f} GB")
            if skip:
                record["status"] = "skipped"
                record["skip_reason"] = skip
                atomic_write_json(path, record)
                log(f"[{i+1}/{len(cases)}] SKIP (memory) {cid}")
                written.append(path)
                continue

        key = (json.dumps(point, sort_keys=True, default=str), seed, est_cond)
        if key not in problem_cache:
            problem_cache.clear()  # keep at most one system in host memory
            problem_cache[key] = generate_problem(
                spec, estimate_condition=est_cond)
        problem = problem_cache[key]
        record["generation_s"] = problem.generation_seconds
        record["gamma"] = problem.gamma
        record["kappa_estimate"] = problem.kappa_estimate
        record["kappa_method"] = problem.kappa_method

        log(f"[{i+1}/{len(cases)}] run {cid}")
        try:
            result = RUNNERS[kind](problem, method_def, rules, device)
        except Exception as exc:  # noqa: BLE001 - recorded, not hidden
            result = {"status": "failed",
                      "error": f"{type(exc).__name__}: {exc}"}
        record.update(result)
        atomic_write_json(path, record)
        written.append(path)
        status = record.get("status")
        if status != "ok":
            log(f"    -> {status}: {record.get('skip_reason') or record.get('error')}")
    return written

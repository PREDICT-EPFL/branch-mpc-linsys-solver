"""Benchmark configuration: YAML parsing, case expansion, case identity.

The serialized parameter names (``num_branches``, ``horizon``,
``block_size``, ``separator_dim``, ``num_rhs``) are part of the stable
raw-record schema and of the case-identifier hash; they intentionally
match :class:`~src.problem.ProblemSpec` rather than the canonical
code names, so existing records stay resumable and comparable.
"""

import dataclasses
import hashlib
import json
from itertools import product

from benchmarks.problems import ProblemSpec

_SPEC_KEYS = {f.name for f in dataclasses.fields(ProblemSpec)}

# YAML sweep parameter aliases -> ProblemSpec field names
_PARAM_ALIASES = {
    "B": "num_branches", "N": "horizon", "n": "block_size",
    "m": "separator_dim", "nrhs": "num_rhs",
}

_FLOAT_KEYS = {"rho_tail", "rho_sep", "condition_target"}
_INT_KEYS = {"num_branches", "horizon", "block_size", "separator_dim",
             "num_rhs", "band_width", "seed"}


def spec_from_params(params: dict) -> ProblemSpec:
    """Build a :class:`ProblemSpec` from a YAML parameter dict (aliases
    resolved, numeric strings coerced, unknown keys ignored)."""
    kv = {}
    for k, v in params.items():
        key = _PARAM_ALIASES.get(k, k)
        if key not in _SPEC_KEYS:
            continue
        # YAML 1.1 parses "1.0e4" as a string; coerce numeric fields
        if key in _FLOAT_KEYS and v is not None:
            v = float(v)
        elif key in _INT_KEYS:
            v = int(v)
        kv[key] = v
    return ProblemSpec(**kv)


def case_id(sweep: str, point: dict, seed: int, method_name: str) -> str:
    """Stable case identifier derived from the sweep name and all
    problem/method parameters (each sweep is self-contained, so shared
    points are re-measured per sweep rather than cross-referenced)."""
    payload = json.dumps({"sweep": sweep, "point": point, "seed": seed,
                          "method": method_name}, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
    spec = spec_from_params(point)
    tag = (f"B{spec.num_branches}_N{spec.horizon}_n{spec.block_size}_"
           f"m{spec.separator_dim}_q{spec.num_rhs}_"
           f"{spec.precision.replace('float', 'f')}_s{seed}")
    return f"{sweep}_{method_name}_{tag}_{digest}"


def expand_cases(cfg: dict, select_sweeps=None, select_methods=None,
                 seed_override=None):
    """Expand a config into case dicts with keys ``sweep``, ``point``,
    ``seed``, ``method``, ``estimate_condition``.

    Every point dict is the full parameter set (defaults + sweep
    override).  A sweep may set ``estimate_condition: true/false`` to
    override the config-level default (condition estimation is expensive
    for very large systems and only needed for
    accuracy-versus-conditioning plots).
    """
    defaults = dict(cfg.get("defaults", {}))
    methods = cfg.get("methods", {})
    seeds = cfg.get("seeds", [0])
    cases = []
    for sweep in cfg.get("sweeps", []):
        name = sweep["name"]
        if select_sweeps and name not in select_sweeps:
            continue
        sweep_methods = sweep.get("methods", list(methods))
        sweep_seeds = seed_override or sweep.get("seeds", seeds)

        points = []
        if "param" in sweep:
            for v in sweep["values"]:
                points.append({**defaults, **sweep.get("overrides", {}),
                               sweep["param"]: v})
        elif "grid" in sweep:
            keys = list(sweep["grid"])
            for combo in product(*(sweep["grid"][k] for k in keys)):
                points.append({**defaults, **sweep.get("overrides", {}),
                               **dict(zip(keys, combo))})
        elif "points" in sweep:
            for p in sweep["points"]:
                points.append({**defaults, **sweep.get("overrides", {}), **p})
        else:
            points.append({**defaults, **sweep.get("overrides", {})})

        for idx, point in enumerate(points):
            for seed in sweep_seeds:
                # rotate method order across points to reduce thermal bias
                order = list(sweep_methods)
                rot = idx % max(len(order), 1)
                order = order[rot:] + order[:rot]
                for mname in order:
                    if select_methods and mname not in select_methods:
                        continue
                    cases.append({
                        "sweep": name, "point": point, "seed": int(seed),
                        "method": mname,
                        "estimate_condition": sweep.get("estimate_condition"),
                    })
    return cases

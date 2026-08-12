"""Raw-record storage: one atomically renamed JSON file per case, so
interrupted runs resume by skipping existing records and a killed writer
never leaves a truncated record behind."""

import json
from pathlib import Path

import numpy as np


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def atomic_write_json(path: Path, payload: dict):
    """Write ``payload`` as JSON via a temp file and atomic rename."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=_json_default))
    tmp.replace(path)

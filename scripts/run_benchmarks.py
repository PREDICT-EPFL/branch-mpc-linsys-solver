"""Run benchmark cases defined by a YAML experiment configuration.

Examples::

    python scripts/run_benchmarks.py --config experiments/smoke.yaml
    python scripts/run_benchmarks.py --config experiments/paper.yaml \
        --device cuda:0 --resume
    python scripts/run_benchmarks.py --config experiments/paper.yaml --dry-run
    python scripts/run_benchmarks.py --config experiments/paper.yaml \
        --sweeps branches horizon --methods tree_socu cudss

Existing raw records are skipped (resume) by default and never silently
overwritten; pass ``--overwrite`` to replace them.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import run_config  # noqa: E402


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="YAML experiment file")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", default="results/raw")
    p.add_argument("--resume", action="store_true", default=True,
                   help="skip cases whose raw record already exists (default)")
    p.add_argument("--overwrite", action="store_true",
                   help="re-run and replace existing raw records")
    p.add_argument("--dry-run", action="store_true",
                   help="print the expanded case list without running")
    p.add_argument("--sweeps", nargs="+", default=None,
                   help="restrict to these sweep names")
    p.add_argument("--methods", nargs="+", default=None,
                   help="restrict to these method names")
    p.add_argument("--seed", type=int, nargs="+", default=None,
                   help="override the seed list")
    args = p.parse_args()

    written = run_config(
        args.config, device=args.device, output=args.output,
        resume=args.resume, overwrite=args.overwrite, dry_run=args.dry_run,
        select_sweeps=args.sweeps, select_methods=args.methods,
        seed_override=args.seed)
    if not args.dry_run:
        print(f"wrote {len(written)} record(s) to {args.output}")


if __name__ == "__main__":
    main()

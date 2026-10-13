#!/usr/bin/env python3
"""Run and aggregate the frozen P7 experiment suite."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from indoor_fire_world.experiments import load_experiment_suite, run_experiment_suite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "experiments" / "p7_suite.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "p7",
    )
    args = parser.parse_args()
    suite = load_experiment_suite(args.config)
    results = run_experiment_suite(suite, args.output)
    print(f"Wrote {len(results)} P7 runs and aggregates to {args.output.resolve()}")


if __name__ == "__main__":
    main()

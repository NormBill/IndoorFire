#!/usr/bin/env python3
"""Run one reproducible IndoorFireWorld experiment from a P7 config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from indoor_fire_world.experiments import load_experiment_suite, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scenario-id")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "p7",
    )
    args = parser.parse_args()
    suite = load_experiment_suite(args.config)
    if args.scenario_id is None:
        if len(suite.experiments) != 1:
            raise ValueError("--scenario-id is required for a multi-run suite config")
        spec = suite.experiments[0]
    else:
        matches = [
            experiment
            for experiment in suite.experiments
            if experiment.scenario_id == args.scenario_id
        ]
        if len(matches) != 1:
            raise ValueError(f"Unknown scenario ID: {args.scenario_id}")
        spec = matches[0]
    result = run_experiment(suite, spec, args.output)
    print(
        json.dumps(
            {
                "scenario_id": spec.scenario_id,
                "artifact_directory": str(
                    (args.output / "runs" / result.artifact_directory).resolve()
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

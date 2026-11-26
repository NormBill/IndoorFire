#!/usr/bin/env python3
"""Run the simulation-only P9 post-arrival intervention experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from indoor_fire_world.experiments.p9_runner import run_p9_suite, write_p9_artifacts


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "p9_postarrival_intervention.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "p9_postarrival_intervention",
    )
    args = parser.parse_args()
    output = run_p9_suite(args.config)
    paths = write_p9_artifacts(output, args.config, args.output)
    print(f"Wrote {output.summary['matrix_shape']['action_branches']} P9 action branches")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()


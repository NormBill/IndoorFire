#!/usr/bin/env python3
"""Run P8 snapshot experiments and write deterministic artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from indoor_fire_world.experiments.p8_runner import run_p8_suite, write_p8_artifacts


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "p8_registered_closed_loop.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "p8_registered_closed_loop",
    )
    args = parser.parse_args()
    output = run_p8_suite(args.config)
    paths = write_p8_artifacts(output, args.config, args.output)
    print(f"Wrote {output.summary['matrix_shape']['normal_cases']} P8 normal cases")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

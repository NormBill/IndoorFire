#!/usr/bin/env python3
"""Run the deterministic P6 closed-loop replanning experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from indoor_fire_world.simulation import run_p6_closed_loop_demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "p6_closed_loop.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "p6" / "summary.json",
    )
    args = parser.parse_args()

    result = run_p6_closed_loop_demo(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote P6 closed-loop summary to {args.output.resolve()}")
    if not result.summary["acceptance"]["controlled_scenario_pass"]:
        failed = [
            name
            for name, passed in result.summary["acceptance"].items()
            if name != "controlled_scenario_pass" and not passed
        ]
        raise RuntimeError(f"P6 controlled scenario failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()

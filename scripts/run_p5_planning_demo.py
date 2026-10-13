#!/usr/bin/env python3
"""Run deterministic risk-aware entry and retreat planning on the P4 posterior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from indoor_fire_world.simulation import run_p5_demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=REPO_ROOT / "configs" / "scene_w1_4f.yaml")
    parser.add_argument("--truth-config", type=Path, default=REPO_ROOT / "configs" / "fire_truth.yaml")
    parser.add_argument("--prior-config", type=Path, default=REPO_ROOT / "configs" / "fire_prior.yaml")
    parser.add_argument("--p3-config", type=Path, default=REPO_ROOT / "configs" / "p3_assimilation.yaml")
    parser.add_argument("--hazard-config", type=Path, default=REPO_ROOT / "configs" / "hazard.yaml")
    parser.add_argument("--planning-config", type=Path, default=REPO_ROOT / "configs" / "planning_cost.yaml")
    parser.add_argument("--mission-config", type=Path, default=REPO_ROOT / "configs" / "planning_mission.yaml")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "artifacts" / "p5_planning" / "mission.json")
    args = parser.parse_args()

    result = run_p5_demo(
        args.scene,
        args.truth_config,
        args.prior_config,
        args.p3_config,
        args.hazard_config,
        args.planning_config,
        args.mission_config,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote P5 mission plan to {args.output.resolve()}")
    if not result.summary["mission"]["feasible"]:
        raise RuntimeError("P5 mission is infeasible; inspect rejection_reasons in output")


if __name__ == "__main__":
    main()

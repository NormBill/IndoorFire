#!/usr/bin/env python3
"""Report P5 routes for configured hazard/uncertainty weight combinations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from indoor_fire_world.simulation import run_p5_weight_sensitivity


CSV_FIELDS = (
    "hazard_weight",
    "uncertainty_weight",
    "mission_feasible",
    "path_identity",
    "entry_path_hash",
    "retreat_path_hash",
    "selected_exit",
    "total_cost",
    "geometric_length_m",
    "integrated_hazard",
    "integrated_uncertainty",
    "max_hazard",
    "rejection_reasons",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=REPO_ROOT / "configs" / "scene_w1_4f.yaml")
    parser.add_argument("--truth-config", type=Path, default=REPO_ROOT / "configs" / "fire_truth.yaml")
    parser.add_argument("--prior-config", type=Path, default=REPO_ROOT / "configs" / "fire_prior.yaml")
    parser.add_argument("--p3-config", type=Path, default=REPO_ROOT / "configs" / "p3_assimilation.yaml")
    parser.add_argument("--hazard-config", type=Path, default=REPO_ROOT / "configs" / "hazard.yaml")
    parser.add_argument("--planning-config", type=Path, default=REPO_ROOT / "configs" / "planning_cost.yaml")
    parser.add_argument("--mission-config", type=Path, default=REPO_ROOT / "configs" / "planning_mission.yaml")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "artifacts" / "p5_planning")
    args = parser.parse_args()

    records = run_p5_weight_sensitivity(
        args.scene,
        args.truth_config,
        args.prior_config,
        args.p3_config,
        args.hazard_config,
        args.planning_config,
        args.mission_config,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "weight_sensitivity.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output / "weight_sensitivity.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["rejection_reasons"] = json.dumps(row["rejection_reasons"])
            writer.writerow(row)
    print(f"Wrote {len(records)} weight combinations to {args.output.resolve()}")


if __name__ == "__main__":
    main()

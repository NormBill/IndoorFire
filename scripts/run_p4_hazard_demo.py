#!/usr/bin/env python3
"""Generate before/after P4 hazard, uncertainty, and planning-cost artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from indoor_fire_world.simulation import run_p4_demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=REPO_ROOT / "configs" / "scene_w1_4f.yaml")
    parser.add_argument("--truth-config", type=Path, default=REPO_ROOT / "configs" / "fire_truth.yaml")
    parser.add_argument("--prior-config", type=Path, default=REPO_ROOT / "configs" / "fire_prior.yaml")
    parser.add_argument("--p3-config", type=Path, default=REPO_ROOT / "configs" / "p3_assimilation.yaml")
    parser.add_argument("--hazard-config", type=Path, default=REPO_ROOT / "configs" / "hazard.yaml")
    parser.add_argument("--planning-config", type=Path, default=REPO_ROOT / "configs" / "planning_cost.yaml")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "artifacts" / "p4_hazard")
    args = parser.parse_args()

    result = run_p4_demo(
        args.scene,
        args.truth_config,
        args.prior_config,
        args.p3_config,
        args.hazard_config,
        args.planning_config,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    for name, array in result.arrays.items():
        filename = f"{name}.npy"
        np.save(args.output / filename, array, allow_pickle=False)
        files.append(filename)
    summary = dict(result.summary)
    summary["files"] = files
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not summary["observation_uncertainty_reduced"]:
        raise RuntimeError("P4 acceptance failed: local normalized uncertainty did not decrease")
    print(f"Wrote {len(files)} P4 arrays and summary to {args.output.resolve()}")


if __name__ == "__main__":
    main()

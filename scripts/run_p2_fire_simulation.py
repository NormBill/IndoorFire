#!/usr/bin/env python3
"""Generate deterministic P2 truth/prior numerical snapshots at 0, 2, 4, and 6 s."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from indoor_fire_world.fire import GroundTruthSimulator
from indoor_fire_world.scene import load_scene
from indoor_fire_world.world_model import PhysicsPrior


SNAPSHOT_TIMES_S = (0.0, 2.0, 4.0, 6.0)


def collect_snapshots(simulator: GroundTruthSimulator | PhysicsPrior) -> dict[float, object]:
    snapshots: dict[float, object] = {0.0: simulator.current_state}
    for target in SNAPSHOT_TIMES_S[1:]:
        while simulator.current_state.timestamp < target - 1e-12:
            remaining = target - simulator.current_state.timestamp
            simulator.step(min(simulator.parameters.time_step_s, remaining))
        snapshots[target] = simulator.current_state
    return snapshots


def save_snapshots(label: str, snapshots: dict[float, object], output: Path) -> list[str]:
    files: list[str] = []
    for timestamp, state in snapshots.items():
        time_label = f"{timestamp:04.1f}"
        for field_name in ("temperature", "co", "smoke", "visibility"):
            filename = f"{label}_t{time_label}_{field_name}.npy"
            np.save(output / filename, getattr(state, field_name), allow_pickle=False)
            files.append(filename)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=REPO_ROOT / "configs" / "scene_w1_4f.yaml")
    parser.add_argument("--truth-config", type=Path, default=REPO_ROOT / "configs" / "fire_truth.yaml")
    parser.add_argument("--prior-config", type=Path, default=REPO_ROOT / "configs" / "fire_prior.yaml")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "artifacts" / "p2_fire")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    scene = load_scene(args.scene)
    truth = GroundTruthSimulator(scene, args.truth_config)
    prior = PhysicsPrior(scene, args.prior_config)
    truth_snapshots = collect_snapshots(truth)
    prior_snapshots = collect_snapshots(prior)
    files = save_snapshots("ground_truth", truth_snapshots, args.output)
    files.extend(save_snapshots("physics_prior", prior_snapshots, args.output))
    manifest = {
        "scene_id": scene.scene_id,
        "snapshot_times_s": list(SNAPSHOT_TIMES_S),
        "ground_truth_parameters": asdict(truth.parameters),
        "physics_prior_parameters": asdict(prior.parameters),
        "array_axis_order": ["row_y", "column_x"],
        "files": files,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(files)} arrays and manifest to {args.output.resolve()}")


if __name__ == "__main__":
    main()

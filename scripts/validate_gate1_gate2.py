#!/usr/bin/env python3
"""Validate registered-map anchors and export the staged electrical-fire profile."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from indoor_fire_world.fire import GroundTruthSimulator, load_fire_parameters
from indoor_fire_world.scene import load_scene
from indoor_fire_world.visualization import render_registration_overlay_svg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate1_gate2",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    scene = load_scene(REPO_ROOT / "configs" / "scene_w1_4f_registered_v1.yaml")
    truth = load_fire_parameters(
        REPO_ROOT / "configs" / "fire_microwave_electrical_truth_v1.yaml"
    )
    prior = load_fire_parameters(
        REPO_ROOT / "configs" / "fire_microwave_electrical_prior_v1.yaml"
    )
    truth_source = GroundTruthSimulator(scene, truth).source
    prior_source = GroundTruthSimulator(scene, prior).source
    sample_times = (0.0, 30.0, 60.0, 90.0, 100.0, 180.0, 240.0, 300.0)
    registration = scene.map_registration
    assert registration is not None
    profile = {
        "scene_id": scene.scene_id,
        "geometry_status": scene.geometry_status,
        "calibration_status": registration.calibration_status,
        "scale_reference": registration.scale_reference,
        "exits": [item.id for item in scene.semantic.by_kind("exit")],
        "truth_phases": [asdict(phase) for phase in truth.source_phases],
        "prior_phases": [asdict(phase) for phase in prior.source_phases],
        "hrr_samples_kw": [
            {
                "time_s": timestamp,
                "truth": truth_source.heat_release_rate_kw(timestamp),
                "prior": prior_source.heat_release_rate_kw(timestamp),
            }
            for timestamp in sample_times
        ],
    }
    (args.output / "fire_scenario_profile.json").write_text(
        json.dumps(profile, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    overlay = render_registration_overlay_svg(
        scene,
        args.output / "w1_4f_registration_overlay.svg",
    )
    print(f"Wrote Gate 1 registration overlay to {overlay}")
    print(f"Wrote Gate 2 source profile to {(args.output / 'fire_scenario_profile.json').resolve()}")


if __name__ == "__main__":
    main()

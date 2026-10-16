#!/usr/bin/env python3
"""Validate converted FDS planes and write the portable P8 v2 manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from indoor_fire_world.fire import sha256_file


TIMES = (60.0, 100.0, 180.0, 300.0, 360.0, 420.0, 480.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--fds-version", required=True)
    parser.add_argument("--fds-input", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.arrays, allow_pickle=False) as arrays:
        shape = None
        for index, _ in enumerate(TIMES):
            for modality in ("temperature", "co", "smoke"):
                key = f"{modality}_t{index}"
                if key not in arrays:
                    raise ValueError(f"Missing converted FDS array {key}")
                value = np.asarray(arrays[key])
                if value.ndim != 2 or not np.all(np.isfinite(value)):
                    raise ValueError(f"Converted FDS array {key} must be finite and two-dimensional")
                shape = value.shape if shape is None else shape
                if value.shape != shape:
                    raise ValueError("Converted FDS arrays must share one registered-grid shape")
    payload = {
        "schema_version": 1,
        "scenario_id": args.scenario_id,
        "reference_semantics": "high_fidelity_synthetic_reference",
        "assumption_status": "simulation_assumed",
        "fds_version": args.fds_version,
        "fds_input_sha256": sha256_file(args.fds_input),
        "arrays_file": args.arrays.name,
        "arrays_sha256": sha256_file(args.arrays),
        "times_s": list(TIMES),
        "shape": list(shape or ()),
        "clear_air_visibility_m": 30.0,
        "smoke_extinction_coefficient": 3.0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()

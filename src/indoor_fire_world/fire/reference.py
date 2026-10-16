"""Portable snapshot contract for offline FDS synthetic references."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .state import FireField


@dataclass(frozen=True)
class FDSSnapshotManifest:
    path: Path
    scenario_id: str
    fds_version: str
    fds_input_sha256: str
    arrays_sha256: str
    times_s: tuple[float, ...]
    shape: tuple[int, int]
    assumption_status: str


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_fds_snapshot_manifest(path: str | Path) -> FDSSnapshotManifest:
    manifest_path = Path(path).resolve()
    data: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported FDS snapshot manifest schema")
    arrays_path = (manifest_path.parent / data["arrays_file"]).resolve()
    if sha256_file(arrays_path) != data["arrays_sha256"]:
        raise ValueError("FDS snapshot array hash mismatch")
    if data.get("reference_semantics") != "high_fidelity_synthetic_reference":
        raise ValueError("FDS snapshots must be labelled as synthetic reference data")
    return FDSSnapshotManifest(
        manifest_path,
        str(data["scenario_id"]),
        str(data["fds_version"]),
        str(data["fds_input_sha256"]),
        str(data["arrays_sha256"]),
        tuple(float(value) for value in data["times_s"]),
        (int(data["shape"][0]), int(data["shape"][1])),
        str(data["assumption_status"]),
    )


def load_fds_snapshots(manifest: FDSSnapshotManifest) -> dict[float, FireField]:
    data: dict[str, Any] = json.loads(manifest.path.read_text(encoding="utf-8"))
    arrays_path = (manifest.path.parent / data["arrays_file"]).resolve()
    result: dict[float, FireField] = {}
    with np.load(arrays_path, allow_pickle=False) as arrays:
        for index, time_s in enumerate(manifest.times_s):
            fields = {
                name: np.asarray(arrays[f"{name}_t{index}"], dtype=np.float64)
                for name in ("temperature", "co", "smoke")
            }
            if any(value.shape != manifest.shape for value in fields.values()):
                raise ValueError("FDS snapshot shape does not match manifest")
            result[time_s] = FireField(
                **fields,
                timestamp=time_s,
                clear_air_visibility_m=float(data["clear_air_visibility_m"]),
                smoke_extinction_coefficient=float(data["smoke_extinction_coefficient"]),
            )
    return result

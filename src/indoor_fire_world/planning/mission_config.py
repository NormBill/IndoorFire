"""Explicit P5 mission endpoints and weight-sensitivity configuration."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from indoor_fire_world.scene.models import Scene

from .result import GridCell


@dataclass(frozen=True)
class MissionEndpoint:
    id: str
    position_m: tuple[float, float]


@dataclass(frozen=True)
class MissionParameters:
    entry_start: MissionEndpoint
    inspection_target: MissionEndpoint
    retreat_exit_ids: tuple[str, ...]
    hazard_weights: tuple[float, ...]
    uncertainty_weights: tuple[float, ...]
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.retreat_exit_ids or len(set(self.retreat_exit_ids)) != len(
            self.retreat_exit_ids
        ):
            raise ValueError("Retreat exit IDs must be non-empty and unique")
        for name, values in (
            ("hazard", self.hazard_weights),
            ("uncertainty", self.uncertainty_weights),
        ):
            if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
                raise ValueError(f"{name} sensitivity weights must be finite and non-negative")


def _endpoint(data: dict[str, Any]) -> MissionEndpoint:
    position = data["position_m"]
    if len(position) != 2:
        raise ValueError("Mission position_m must contain [x, y]")
    point = (float(position[0]), float(position[1]))
    if not all(math.isfinite(value) for value in point):
        raise ValueError("Mission endpoint coordinates must be finite")
    return MissionEndpoint(id=str(data["id"]), position_m=point)


def load_mission_parameters(config_path: str | Path) -> MissionParameters:
    with Path(config_path).open("r", encoding="utf-8") as stream:
        data: dict[str, Any] = json.load(stream)
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported mission configuration schema_version")
    sensitivity = data["weight_sensitivity"]
    return MissionParameters(
        entry_start=_endpoint(data["entry_start"]),
        inspection_target=_endpoint(data["inspection_target"]),
        retreat_exit_ids=tuple(str(value) for value in data["retreat_exit_ids"]),
        hazard_weights=tuple(float(value) for value in sensitivity["hazard_weights"]),
        uncertainty_weights=tuple(
            float(value) for value in sensitivity["uncertainty_weights"]
        ),
        notes=str(data.get("notes", "")),
    )


def resolve_mission_cells(
    scene: Scene,
    parameters: MissionParameters,
) -> tuple[GridCell, GridCell, dict[str, GridCell]]:
    entry = scene.navigation.metric_to_grid(*parameters.entry_start.position_m)
    target = scene.navigation.metric_to_grid(*parameters.inspection_target.position_m)
    exits: dict[str, GridCell] = {}
    for exit_id in parameters.retreat_exit_ids:
        semantic = scene.semantic.by_id(exit_id)
        if semantic.kind != "exit" or semantic.position is None:
            raise ValueError(f"Configured retreat exit {exit_id!r} is not a positioned exit")
        exits[exit_id] = scene.navigation.metric_to_grid(*semantic.position)
    return entry, target, exits

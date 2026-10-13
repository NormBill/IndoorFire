"""Validated, immutable configuration for the controlled P6 experiment."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from indoor_fire_world.p3_config import (
    AssimilationParameters,
    InitialUncertaintyParameters,
    InspectionWaypoint,
    SensorNoiseParameters,
)
from indoor_fire_world.planning import MissionEndpoint, PlanningCostParameters
from indoor_fire_world.scene import Pose2D


@dataclass(frozen=True)
class P6ScenarioParameters:
    scenario_id: str
    scene_config: Path
    truth_config: Path
    prior_config: Path
    hazard_config: Path
    inspection_time_s: float
    random_seed: int
    sensor: SensorNoiseParameters
    initial_uncertainty: InitialUncertaintyParameters
    assimilation: AssimilationParameters
    inspection_waypoint: InspectionWaypoint
    entry_start: MissionEndpoint
    inspection_target: MissionEndpoint
    retreat_exit_ids: tuple[str, ...]
    planning: PlanningCostParameters
    local_comparison_radius_m: float
    require_route_change: bool
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("P6 scenario_id must not be empty")
        if not math.isfinite(self.inspection_time_s) or self.inspection_time_s <= 0.0:
            raise ValueError("P6 inspection time must be finite and positive")
        if self.random_seed < 0:
            raise ValueError("P6 random seed must be non-negative")
        if not self.retreat_exit_ids or len(set(self.retreat_exit_ids)) != len(
            self.retreat_exit_ids
        ):
            raise ValueError("P6 retreat exit IDs must be non-empty and unique")
        if (
            not math.isfinite(self.local_comparison_radius_m)
            or self.local_comparison_radius_m <= 0.0
        ):
            raise ValueError("P6 local comparison radius must be finite and positive")


def _endpoint(data: dict[str, Any]) -> MissionEndpoint:
    position = data["position_m"]
    if len(position) != 2:
        raise ValueError("P6 mission position_m must contain [x, y]")
    point = (float(position[0]), float(position[1]))
    if not all(math.isfinite(value) for value in point):
        raise ValueError("P6 mission endpoint coordinates must be finite")
    return MissionEndpoint(id=str(data["id"]), position_m=point)


def load_p6_parameters(config_path: str | Path) -> P6ScenarioParameters:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as stream:
        data: dict[str, Any] = json.load(stream)
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported P6 configuration schema_version")
    root = path.parent
    sensor = data["sensor"]
    uncertainty = data["uncertainty"]
    assimilation = data["assimilation"]
    waypoint = data["inspection_waypoint"]
    mission = data["mission"]
    planning = data["planning"]
    pose = waypoint["pose"]
    return P6ScenarioParameters(
        scenario_id=str(data["scenario_id"]),
        scene_config=(root / str(data["scene_config"])).resolve(),
        truth_config=(root / str(data["truth_config"])).resolve(),
        prior_config=(root / str(data["prior_config"])).resolve(),
        hazard_config=(root / str(data["hazard_config"])).resolve(),
        inspection_time_s=float(data["inspection_time_s"]),
        random_seed=int(data["random_seed"]),
        sensor=SensorNoiseParameters(
            temperature_std=float(sensor["temperature_std"]),
            co_std=float(sensor["co_std"]),
            smoke_std=float(sensor["smoke_std"]),
            visibility_std=float(sensor["visibility_std"]),
        ),
        initial_uncertainty=InitialUncertaintyParameters(
            temperature=float(uncertainty["temperature_initial"]),
            co=float(uncertainty["co_initial"]),
            smoke=float(uncertainty["smoke_initial"]),
            visibility=float(uncertainty["visibility_initial"]),
            meaning=str(uncertainty["meaning"]),
        ),
        assimilation=AssimilationParameters(
            gain=float(assimilation["gain"]),
            kernel_sigma_m=float(assimilation["kernel_sigma_m"]),
            max_radius_m=float(assimilation["max_radius_m"]),
            uncertainty_reduction_fraction=float(
                assimilation["uncertainty_reduction_fraction"]
            ),
            minimum_temperature_c=float(assimilation["minimum_temperature_c"]),
        ),
        inspection_waypoint=InspectionWaypoint(
            id=str(waypoint["id"]),
            pose=Pose2D(
                x=float(pose["x_m"]),
                y=float(pose["y_m"]),
                yaw=float(pose["yaw_rad"]),
            ),
        ),
        entry_start=_endpoint(mission["entry_start"]),
        inspection_target=_endpoint(mission["inspection_target"]),
        retreat_exit_ids=tuple(str(value) for value in mission["retreat_exit_ids"]),
        planning=PlanningCostParameters(
            base_cost=float(planning["base_cost"]),
            hazard_weight=float(planning["hazard_weight"]),
            uncertainty_weight=float(planning["uncertainty_weight"]),
            untenable_hazard_threshold=float(planning["untenable_hazard_threshold"]),
            forbidden_representation=str(planning["forbidden_representation"]),
            notes="Frozen P6 planning weights loaded before simulation.",
        ),
        local_comparison_radius_m=float(data["local_comparison_radius_m"]),
        require_route_change=bool(data["require_route_change"]),
        notes=str(data.get("notes", "")),
    )

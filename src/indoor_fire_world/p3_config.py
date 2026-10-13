"""Validated P3 sensing, belief, and assimilation configuration."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from indoor_fire_world.scene import Pose2D


@dataclass(frozen=True)
class SensorNoiseParameters:
    temperature_std: float
    co_std: float
    smoke_std: float
    visibility_std: float

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class InitialUncertaintyParameters:
    temperature: float
    co: float
    smoke: float
    visibility: float
    meaning: str

    def __post_init__(self) -> None:
        for name in ("temperature", "co", "smoke", "visibility"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"Initial {name} uncertainty must be finite and non-negative")
        if not self.meaning:
            raise ValueError("Uncertainty meaning must be documented")


@dataclass(frozen=True)
class AssimilationParameters:
    gain: float
    kernel_sigma_m: float
    max_radius_m: float
    uncertainty_reduction_fraction: float
    minimum_temperature_c: float

    def __post_init__(self) -> None:
        values = (
            self.gain,
            self.kernel_sigma_m,
            self.max_radius_m,
            self.uncertainty_reduction_fraction,
            self.minimum_temperature_c,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Assimilation parameters must be finite")
        if not 0.0 < self.gain <= 1.0:
            raise ValueError("Assimilation gain must be in (0, 1]")
        if self.kernel_sigma_m <= 0.0 or self.max_radius_m <= 0.0:
            raise ValueError("Kernel sigma and maximum radius must be positive")
        if not 0.0 <= self.uncertainty_reduction_fraction <= 1.0:
            raise ValueError("Uncertainty reduction fraction must be in [0, 1]")


@dataclass(frozen=True)
class InspectionWaypoint:
    id: str
    pose: Pose2D


@dataclass(frozen=True)
class P3Parameters:
    random_seed: int
    sensor: SensorNoiseParameters
    initial_uncertainty: InitialUncertaintyParameters
    assimilation: AssimilationParameters
    inspection_waypoints: tuple[InspectionWaypoint, ...]
    demo_target_timestamp_s: float
    demo_waypoint_id: str
    notes: str = ""

    def __post_init__(self) -> None:
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")
        if not math.isfinite(self.demo_target_timestamp_s) or self.demo_target_timestamp_s < 0.0:
            raise ValueError("demo target timestamp must be finite and non-negative")
        ids = [waypoint.id for waypoint in self.inspection_waypoints]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Inspection waypoint IDs must be non-empty and unique")
        if self.demo_waypoint_id not in ids:
            raise ValueError("Demo waypoint must reference an inspection waypoint")

    def waypoint(self, waypoint_id: str) -> InspectionWaypoint:
        for waypoint in self.inspection_waypoints:
            if waypoint.id == waypoint_id:
                return waypoint
        raise KeyError(waypoint_id)


def load_p3_parameters(config_path: str | Path) -> P3Parameters:
    """Load JSON-subset YAML P3 configuration."""
    with Path(config_path).open("r", encoding="utf-8") as stream:
        data: dict[str, Any] = json.load(stream)
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported P3 configuration schema_version")
    sensor = data["sensor"]
    uncertainty = data["uncertainty"]
    assimilation = data["assimilation"]
    demo = data["demo"]
    waypoints = tuple(
        InspectionWaypoint(
            id=str(item["id"]),
            pose=Pose2D(
                x=float(item["pose"]["x_m"]),
                y=float(item["pose"]["y_m"]),
                yaw=float(item["pose"]["yaw_rad"]),
            ),
        )
        for item in data["inspection_waypoints"]
    )
    return P3Parameters(
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
            uncertainty_reduction_fraction=float(assimilation["uncertainty_reduction_fraction"]),
            minimum_temperature_c=float(assimilation["minimum_temperature_c"]),
        ),
        inspection_waypoints=waypoints,
        demo_target_timestamp_s=float(demo["target_timestamp_s"]),
        demo_waypoint_id=str(demo["waypoint_id"]),
        notes=str(data.get("notes", "")),
    )

"""Validated configuration for P8 registered-map snapshot experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from indoor_fire_world.p3_config import (
    AssimilationParameters,
    InitialUncertaintyParameters,
    SensorNoiseParameters,
)
from indoor_fire_world.planning import PlanningCostParameters


@dataclass(frozen=True)
class ObservationOffset:
    id: str
    dx_m: float
    dy_m: float
    yaw_rad: float = 0.0

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("P8 observation point ID must not be empty")
        if not all(math.isfinite(value) for value in (self.dx_m, self.dy_m, self.yaw_rad)):
            raise ValueError("P8 observation offsets and yaw must be finite")


@dataclass(frozen=True)
class P8ScenarioParameters:
    config_path: Path
    scenario_id: str
    scene_config: Path
    truth_config: Path
    prior_config: Path
    hazard_config: Path
    fire_age_at_dispatch_s: float
    pre_response_research_horizon_s: float
    evaluation_fire_times_s: tuple[float, ...]
    primary_fire_time_s: float
    random_seed: int
    sensor: SensorNoiseParameters
    initial_uncertainty: InitialUncertaintyParameters
    assimilation: AssimilationParameters
    observation_points: tuple[ObservationOffset, ...]
    primary_observation_id: str
    retreat_exit_ids: tuple[str, ...]
    base_cost: float
    hazard_weight: float
    uncertainty_weight: float
    ugv_operability_threshold: float
    retreat_hazard_threshold: float
    forbidden_representation: str
    local_comparison_radius_m: float
    horizon_definition: str
    horizon_evidence_status: str
    horizon_motivation_source: str
    snapshot_semantics: str
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("P8 scenario_id must not be empty")
        scalar_times = (
            self.fire_age_at_dispatch_s,
            self.pre_response_research_horizon_s,
            self.primary_fire_time_s,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in scalar_times):
            raise ValueError("P8 clock values must be finite and non-negative")
        if self.pre_response_research_horizon_s <= 0.0:
            raise ValueError("P8 research horizon must be positive")
        if not self.evaluation_fire_times_s or any(
            not math.isfinite(value) or value < self.fire_age_at_dispatch_s
            for value in self.evaluation_fire_times_s
        ):
            raise ValueError("P8 evaluation times must be finite and no earlier than dispatch")
        if tuple(sorted(set(self.evaluation_fire_times_s))) != self.evaluation_fire_times_s:
            raise ValueError("P8 evaluation times must be unique and increasing")
        if self.primary_fire_time_s not in self.evaluation_fire_times_s:
            raise ValueError("P8 primary fire time must be an evaluation time")
        if any(
            value - self.fire_age_at_dispatch_s > self.pre_response_research_horizon_s + 1e-12
            for value in self.evaluation_fire_times_s
        ):
            raise ValueError("P8 evaluation times must fit inside the research horizon")
        if self.random_seed < 0:
            raise ValueError("P8 random seed must be non-negative")
        ids = tuple(point.id for point in self.observation_points)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("P8 observation point IDs must be non-empty and unique")
        if self.primary_observation_id not in ids:
            raise ValueError("P8 primary observation point must be configured")
        if not self.retreat_exit_ids or len(set(self.retreat_exit_ids)) != len(
            self.retreat_exit_ids
        ):
            raise ValueError("P8 retreat exit IDs must be non-empty and unique")
        for name, value in (
            ("base_cost", self.base_cost),
            ("hazard_weight", self.hazard_weight),
            ("uncertainty_weight", self.uncertainty_weight),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"P8 {name} must be finite and non-negative")
        for name, value in (
            ("ugv_operability_threshold", self.ugv_operability_threshold),
            ("retreat_hazard_threshold", self.retreat_hazard_threshold),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"P8 {name} must be in [0, 1]")
        if self.forbidden_representation != "infinity":
            raise ValueError("P8 requires infinity forbidden representation")
        if not math.isfinite(self.local_comparison_radius_m) or self.local_comparison_radius_m <= 0.0:
            raise ValueError("P8 local comparison radius must be finite and positive")
        if not (
            self.horizon_definition
            and self.horizon_evidence_status
            and self.horizon_motivation_source
        ):
            raise ValueError("P8 research-horizon provenance must be declared")
        if not self.snapshot_semantics:
            raise ValueError("P8 snapshot semantics must be declared")

    def planning_parameters(self, hazard_threshold: float, purpose: str) -> PlanningCostParameters:
        return PlanningCostParameters(
            base_cost=self.base_cost,
            hazard_weight=self.hazard_weight,
            uncertainty_weight=self.uncertainty_weight,
            untenable_hazard_threshold=hazard_threshold,
            forbidden_representation=self.forbidden_representation,
            notes=f"P8 {purpose}; threshold semantics are independent.",
        )


def load_p8_parameters(config_path: str | Path) -> P8ScenarioParameters:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as stream:
        data: dict[str, Any] = json.load(stream)
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported P8 configuration schema_version")
    root = path.parent
    clocks = data["clocks"]
    sensor = data["sensor"]
    uncertainty = data["uncertainty"]
    assimilation = data["assimilation"]
    observations = data["observation_points"]
    planning = data["planning"]
    return P8ScenarioParameters(
        config_path=path,
        scenario_id=str(data["scenario_id"]),
        scene_config=(root / str(data["scene_config"])).resolve(),
        truth_config=(root / str(data["truth_config"])).resolve(),
        prior_config=(root / str(data["prior_config"])).resolve(),
        hazard_config=(root / str(data["hazard_config"])).resolve(),
        fire_age_at_dispatch_s=float(clocks["fire_age_at_dispatch_s"]),
        pre_response_research_horizon_s=float(clocks["pre_response_research_horizon_s"]),
        evaluation_fire_times_s=tuple(float(value) for value in clocks["evaluation_fire_times_s"]),
        primary_fire_time_s=float(clocks["primary_fire_time_s"]),
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
        observation_points=tuple(
            ObservationOffset(
                id=str(item["id"]),
                dx_m=float(item["offset_m"][0]),
                dy_m=float(item["offset_m"][1]),
                yaw_rad=float(item.get("yaw_rad", 0.0)),
            )
            for item in observations["points"]
        ),
        primary_observation_id=str(observations["primary_id"]),
        retreat_exit_ids=tuple(str(value) for value in data["retreat_exit_ids"]),
        base_cost=float(planning["base_cost"]),
        hazard_weight=float(planning["hazard_weight"]),
        uncertainty_weight=float(planning["uncertainty_weight"]),
        ugv_operability_threshold=float(planning["ugv_operability_threshold"]),
        retreat_hazard_threshold=float(planning["retreat_hazard_threshold"]),
        forbidden_representation=str(planning["forbidden_representation"]),
        local_comparison_radius_m=float(data["local_comparison_radius_m"]),
        horizon_definition=str(clocks["horizon_definition"]),
        horizon_evidence_status=str(clocks["horizon_evidence_status"]),
        horizon_motivation_source=str(clocks["horizon_motivation_source"]),
        snapshot_semantics=str(data["snapshot_semantics"]),
        notes=str(data.get("notes", "")),
    )

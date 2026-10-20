"""Configuration contract for the simulation-only P8 v2 causal experiment."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from indoor_fire_world.p3_config import SensorNoiseParameters


@dataclass(frozen=True)
class SyntheticScenario:
    id: str
    reference_config: Path
    fds_input: Path
    fds_manifest: Path


@dataclass(frozen=True)
class ObservationPointSpec:
    id: str
    offset_m: tuple[float, float]


@dataclass(frozen=True)
class EnsembleAxes:
    hrr_factors: tuple[float, ...]
    yield_factors: tuple[float, ...]
    ventilation_factors: tuple[float, ...]


@dataclass(frozen=True)
class ExposureLevel:
    id: str
    upper_bound: float | None
    color: str


@dataclass(frozen=True)
class FireSourceSemantics:
    ignition_model: str
    primary_source: str
    secondary_ignition_mode: str
    fuel_chemistry_semantics: str


@dataclass(frozen=True)
class P8V2Parameters:
    config_path: Path
    experiment_id: str
    scene_config: Path
    corridor_config: Path
    prior_config: Path
    hazard_config: Path
    scenarios: tuple[SyntheticScenario, ...]
    observation_times_s: tuple[float, ...]
    handoff_time_s: float
    forecast_times_s: tuple[float, ...]
    observation_points: tuple[ObservationPointSpec, ...]
    primary_observation_id: str
    primary_scenario_id: str
    sensor: SensorNoiseParameters
    likelihood_modalities: tuple[str, ...]
    ensemble: EnsembleAxes
    entrances: tuple[str, ...]
    target_node: str
    base_cost: float
    exposure_weight: float
    uncertainty_weight: float
    ugv_operability_threshold: float
    ingress_exposure_threshold: float
    random_seed: int
    reference_provider: str
    assumption_status: str
    exposure_levels: tuple[ExposureLevel, ...]
    fire_source_semantics: FireSourceSemantics
    spread_profile_id: str
    spread_profile_semantics: str

    def __post_init__(self) -> None:
        if not self.experiment_id or self.assumption_status != "simulation_assumed":
            raise ValueError("P8 v2 requires an ID and simulation_assumed status")
        if len(self.scenarios) != 3 or len({item.id for item in self.scenarios}) != 3:
            raise ValueError("P8 v2 requires exactly three unique synthetic scenarios")
        if len(self.observation_times_s) != 4 or tuple(sorted(set(self.observation_times_s))) != self.observation_times_s:
            raise ValueError("P8 v2 requires four unique increasing observation times")
        if len(self.observation_points) != 4 or len({item.id for item in self.observation_points}) != 4:
            raise ValueError("P8 v2 requires four unique frozen observation points")
        if self.primary_observation_id not in {item.id for item in self.observation_points}:
            raise ValueError("Unknown primary observation point")
        if self.primary_scenario_id not in {item.id for item in self.scenarios}:
            raise ValueError("Unknown primary synthetic scenario")
        if self.forecast_times_s != (300.0, 360.0, 420.0, 480.0):
            raise ValueError("P8 v2 forecast times must be 300/360/420/480 s")
        if self.handoff_time_s != self.forecast_times_s[0]:
            raise ValueError("Handoff time must equal the first forecast time")
        if any(value > self.handoff_time_s for value in self.observation_times_s):
            raise ValueError("Observation times must not exceed handoff")
        if self.likelihood_modalities != ("temperature", "co", "smoke"):
            raise ValueError("P8 v2 likelihood uses independent temperature/CO/smoke channels")
        for axis in (self.ensemble.hrr_factors, self.ensemble.yield_factors, self.ensemble.ventilation_factors):
            if not axis or any(not math.isfinite(value) or value <= 0.0 for value in axis):
                raise ValueError("Ensemble factors must be finite and positive")
        if len(self.entrances) != 3 or len(set(self.entrances)) != 3:
            raise ValueError("P8 v2 requires three unique candidate entrances")
        if not self.target_node:
            raise ValueError("P8 v2 requires a configured task target")
        for value in (self.base_cost, self.exposure_weight, self.uncertainty_weight):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("Planning weights must be finite and non-negative")
        for value in (self.ugv_operability_threshold, self.ingress_exposure_threshold):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("P8 v2 thresholds must be in [0, 1]")
        if self.reference_provider not in {"reduced_order_fixture", "fds_snapshot"}:
            raise ValueError("Unsupported P8 v2 reference provider")
        if len(self.exposure_levels) < 2 or self.exposure_levels[-1].upper_bound is not None:
            raise ValueError("P8 v2 exposure levels require an open-ended terminal band")
        finite_bounds = tuple(item.upper_bound for item in self.exposure_levels[:-1])
        if any(value is None for value in finite_bounds) or tuple(sorted(set(finite_bounds))) != finite_bounds:
            raise ValueError("P8 v2 exposure bounds must be unique and increasing")
        semantics = self.fire_source_semantics
        if (
            semantics.ignition_model != "single_primary_source"
            or semantics.primary_source != "microwave"
            or semantics.secondary_ignition_mode != "disabled"
            or semantics.fuel_chemistry_semantics != "synthetic_surrogate"
        ):
            raise ValueError("P8 v2 requires explicit single-source synthetic fire semantics")
        if not self.spread_profile_id or self.spread_profile_semantics != "simulation_assumed_not_calibrated":
            raise ValueError("P8 v2 spread severity must remain explicit and simulation-assumed")


def load_p8_v2_parameters(config_path: str | Path) -> P8V2Parameters:
    path = Path(config_path).resolve()
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 2:
        raise ValueError("Unsupported P8 v2 schema_version")
    root = path.parent
    clocks = data["clocks"]
    sensor = data["sensor"]
    axes = data["candidate_ensemble"]
    planning = data["planning"]
    observations = data["observation_points"]
    source_semantics = data["fire_source_semantics"]
    return P8V2Parameters(
        config_path=path,
        experiment_id=str(data["experiment_id"]),
        scene_config=(root / data["scene_config"]).resolve(),
        corridor_config=(root / data["corridor_config"]).resolve(),
        prior_config=(root / data["prior_config"]).resolve(),
        hazard_config=(root / data["hazard_config"]).resolve(),
        scenarios=tuple(
            SyntheticScenario(
                id=str(item["id"]),
                reference_config=(root / item["reference_config"]).resolve(),
                fds_input=(root / item["fds_input"]).resolve(),
                fds_manifest=(root / item["fds_manifest"]).resolve(),
            )
            for item in data["synthetic_scenarios"]
        ),
        observation_times_s=tuple(float(value) for value in clocks["observation_times_s"]),
        handoff_time_s=float(clocks["handoff_time_s"]),
        forecast_times_s=tuple(float(value) for value in clocks["forecast_times_s"]),
        observation_points=tuple(
            ObservationPointSpec(str(item["id"]), (float(item["offset_m"][0]), float(item["offset_m"][1])))
            for item in observations["points"]
        ),
        primary_observation_id=str(observations["primary_id"]),
        primary_scenario_id=str(data["primary_scenario_id"]),
        sensor=SensorNoiseParameters(
            temperature_std=float(sensor["temperature_std"]),
            co_std=float(sensor["co_std"]),
            smoke_std=float(sensor["smoke_std"]),
            visibility_std=float(sensor["visibility_std"]),
        ),
        likelihood_modalities=tuple(data["likelihood_modalities"]),
        ensemble=EnsembleAxes(
            tuple(float(value) for value in axes["hrr_factors"]),
            tuple(float(value) for value in axes["yield_factors"]),
            tuple(float(value) for value in axes["ventilation_factors"]),
        ),
        entrances=tuple(str(value) for value in planning["entrance_ids"]),
        target_node=str(planning["target_node"]),
        base_cost=float(planning["base_cost"]),
        exposure_weight=float(planning["exposure_weight"]),
        uncertainty_weight=float(planning["uncertainty_weight"]),
        ugv_operability_threshold=float(planning["ugv_operability_threshold"]),
        ingress_exposure_threshold=float(planning["ingress_exposure_threshold"]),
        random_seed=int(data["random_seed"]),
        reference_provider=str(data["reference_provider"]),
        assumption_status=str(data["assumption_status"]),
        exposure_levels=tuple(
            ExposureLevel(
                id=str(item["id"]),
                upper_bound=None if item["upper_bound"] is None else float(item["upper_bound"]),
                color=str(item["color"]),
            )
            for item in data["forecast_exposure_levels"]
        ),
        fire_source_semantics=FireSourceSemantics(
            ignition_model=str(source_semantics["ignition_model"]),
            primary_source=str(source_semantics["primary_source"]),
            secondary_ignition_mode=str(source_semantics["secondary_ignition_mode"]),
            fuel_chemistry_semantics=str(source_semantics["fuel_chemistry_semantics"]),
        ),
        spread_profile_id=str(data["spread_severity_profile"]["id"]),
        spread_profile_semantics=str(data["spread_severity_profile"]["semantics"]),
    )

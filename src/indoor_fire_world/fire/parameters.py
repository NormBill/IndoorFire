"""Validated configuration for the reduced-order fire model."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FireSourcePhase:
    phase_id: str
    start_s: float
    end_s: float | None
    hrr_model: str
    initial_heat_release_rate_kw: float
    fire_growth_rate_kw_per_s2: float
    max_heat_release_rate_kw: float
    temperature_gain_c_per_kw_s: float
    smoke_yield_per_kw_s: float
    co_yield_per_kw_s: float
    physical_interpretation: str
    evidence_status: str

    def __post_init__(self) -> None:
        if not self.phase_id:
            raise ValueError("Fire source phase_id must not be empty")
        if self.hrr_model not in {"inactive", "constant", "t_squared"}:
            raise ValueError("Fire source hrr_model must be inactive, constant, or t_squared")
        values = {
            "start_s": self.start_s,
            "initial_heat_release_rate_kw": self.initial_heat_release_rate_kw,
            "fire_growth_rate_kw_per_s2": self.fire_growth_rate_kw_per_s2,
            "max_heat_release_rate_kw": self.max_heat_release_rate_kw,
            "temperature_gain_c_per_kw_s": self.temperature_gain_c_per_kw_s,
            "smoke_yield_per_kw_s": self.smoke_yield_per_kw_s,
            "co_yield_per_kw_s": self.co_yield_per_kw_s,
        }
        if self.end_s is not None:
            values["end_s"] = self.end_s
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("Fire source phase values must be finite")
        if any(value < 0.0 for value in values.values()):
            raise ValueError("Fire source phase values must be non-negative")
        if self.end_s is not None and self.end_s <= self.start_s:
            raise ValueError("Fire source phase end_s must be greater than start_s")
        if self.max_heat_release_rate_kw < self.initial_heat_release_rate_kw:
            raise ValueError("Phase maximum HRR must not be below its initial HRR")


@dataclass(frozen=True)
class FireParameters:
    model_name: str
    time_step_s: float
    ambient_temperature_c: float
    ventilation_x_mps: float
    ventilation_y_mps: float
    temperature_diffusion_m2_per_s: float
    smoke_diffusion_m2_per_s: float
    co_diffusion_m2_per_s: float
    smoke_decay_per_s: float
    co_decay_per_s: float
    temperature_relaxation_per_s: float
    source_enabled: bool
    source_type: str
    source_semantic_id: str
    fire_growth_rate_kw_per_s2: float
    max_heat_release_rate_kw: float
    temperature_gain_c_per_kw_s: float
    smoke_yield_per_kw_s: float
    co_yield_per_kw_s: float
    clear_air_visibility_m: float
    smoke_extinction_per_concentration: float
    source_phases: tuple[FireSourcePhase, ...] = ()
    assumption_status: str = "unspecified"
    notes: str = ""

    def __post_init__(self) -> None:
        positive = {
            "time_step_s": self.time_step_s,
            "clear_air_visibility_m": self.clear_air_visibility_m,
        }
        nonnegative = {
            "temperature_diffusion_m2_per_s": self.temperature_diffusion_m2_per_s,
            "smoke_diffusion_m2_per_s": self.smoke_diffusion_m2_per_s,
            "co_diffusion_m2_per_s": self.co_diffusion_m2_per_s,
            "smoke_decay_per_s": self.smoke_decay_per_s,
            "co_decay_per_s": self.co_decay_per_s,
            "temperature_relaxation_per_s": self.temperature_relaxation_per_s,
            "fire_growth_rate_kw_per_s2": self.fire_growth_rate_kw_per_s2,
            "max_heat_release_rate_kw": self.max_heat_release_rate_kw,
            "temperature_gain_c_per_kw_s": self.temperature_gain_c_per_kw_s,
            "smoke_yield_per_kw_s": self.smoke_yield_per_kw_s,
            "co_yield_per_kw_s": self.co_yield_per_kw_s,
            "smoke_extinction_per_concentration": self.smoke_extinction_per_concentration,
        }
        finite = {
            **positive,
            **nonnegative,
            "ambient_temperature_c": self.ambient_temperature_c,
            "ventilation_x_mps": self.ventilation_x_mps,
            "ventilation_y_mps": self.ventilation_y_mps,
        }
        for name, value in finite.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name, value in positive.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name, value in nonnegative.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.source_type != "microwave_electrical_fault":
            raise ValueError("P2 supports only source_type='microwave_electrical_fault'")
        if not self.source_semantic_id:
            raise ValueError("source_semantic_id must not be empty")
        if self.source_phases:
            if self.source_phases[0].start_s != 0.0:
                raise ValueError("The first fire source phase must start at 0 s")
            for index, phase in enumerate(self.source_phases):
                is_last = index == len(self.source_phases) - 1
                if phase.end_s is None and not is_last:
                    raise ValueError("Only the final fire source phase may be open-ended")
                if not is_last and not math.isclose(
                    phase.end_s or -1.0,
                    self.source_phases[index + 1].start_s,
                    abs_tol=1e-9,
                ):
                    raise ValueError("Fire source phases must be contiguous")


def load_fire_parameters(config_path: str | Path) -> FireParameters:
    """Load JSON-subset YAML without adding a YAML runtime dependency."""
    with Path(config_path).open("r", encoding="utf-8") as stream:
        data: dict[str, Any] = json.load(stream)
    schema_version = data.get("schema_version")
    if schema_version not in {1, 2}:
        raise ValueError("Unsupported fire configuration schema_version")
    diffusion = data["diffusion_m2_per_s"]
    decay = data["decay_per_s"]
    source = data["source"]
    visibility = data["visibility"]
    ventilation = data["ventilation_mps"]
    if len(ventilation) != 2:
        raise ValueError("ventilation_mps must contain [x, y]")
    phases = tuple(
        FireSourcePhase(
            phase_id=str(item["phase_id"]),
            start_s=float(item["start_s"]),
            end_s=None if item.get("end_s") is None else float(item["end_s"]),
            hrr_model=str(item["hrr_model"]),
            initial_heat_release_rate_kw=float(
                item.get("initial_heat_release_rate_kw", 0.0)
            ),
            fire_growth_rate_kw_per_s2=float(
                item.get("fire_growth_rate_kw_per_s2", source["fire_growth_rate_kw_per_s2"])
            ),
            max_heat_release_rate_kw=float(
                item.get("max_heat_release_rate_kw", source["max_heat_release_rate_kw"])
            ),
            temperature_gain_c_per_kw_s=float(
                item.get("temperature_gain_c_per_kw_s", source["temperature_gain_c_per_kw_s"])
            ),
            smoke_yield_per_kw_s=float(
                item.get("smoke_yield_per_kw_s", source["smoke_yield_per_kw_s"])
            ),
            co_yield_per_kw_s=float(
                item.get("co_yield_per_kw_s", source["co_yield_per_kw_s"])
            ),
            physical_interpretation=str(item.get("physical_interpretation", "")),
            evidence_status=str(item.get("evidence_status", "unspecified")),
        )
        for item in source.get("phases", [])
    )
    return FireParameters(
        model_name=str(data["model_name"]),
        time_step_s=float(data["time_step_s"]),
        ambient_temperature_c=float(data["ambient_temperature_c"]),
        ventilation_x_mps=float(ventilation[0]),
        ventilation_y_mps=float(ventilation[1]),
        temperature_diffusion_m2_per_s=float(diffusion["temperature"]),
        smoke_diffusion_m2_per_s=float(diffusion["smoke"]),
        co_diffusion_m2_per_s=float(diffusion["co"]),
        smoke_decay_per_s=float(decay["smoke"]),
        co_decay_per_s=float(decay["co"]),
        temperature_relaxation_per_s=float(data["temperature_relaxation_per_s"]),
        source_enabled=bool(source["enabled"]),
        source_type=str(source["source_type"]),
        source_semantic_id=str(source["semantic_id"]),
        fire_growth_rate_kw_per_s2=float(source["fire_growth_rate_kw_per_s2"]),
        max_heat_release_rate_kw=float(source["max_heat_release_rate_kw"]),
        temperature_gain_c_per_kw_s=float(source["temperature_gain_c_per_kw_s"]),
        smoke_yield_per_kw_s=float(source["smoke_yield_per_kw_s"]),
        co_yield_per_kw_s=float(source["co_yield_per_kw_s"]),
        clear_air_visibility_m=float(visibility["clear_air_visibility_m"]),
        smoke_extinction_per_concentration=float(visibility["smoke_extinction_per_concentration"]),
        source_phases=phases,
        assumption_status=str(data.get("assumption_status", "unspecified")),
        notes=str(data.get("notes", "")),
    )

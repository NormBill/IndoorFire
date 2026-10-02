"""Validated configuration for the reduced-order fire model."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


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


def load_fire_parameters(config_path: str | Path) -> FireParameters:
    """Load JSON-subset YAML without adding a YAML runtime dependency."""
    with Path(config_path).open("r", encoding="utf-8") as stream:
        data: dict[str, Any] = json.load(stream)
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported fire configuration schema_version")
    diffusion = data["diffusion_m2_per_s"]
    decay = data["decay_per_s"]
    source = data["source"]
    visibility = data["visibility"]
    ventilation = data["ventilation_mps"]
    if len(ventilation) != 2:
        raise ValueError("ventilation_mps must contain [x, y]")
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
        assumption_status=str(data.get("assumption_status", "unspecified")),
        notes=str(data.get("notes", "")),
    )

"""Validated, unit-documented P4 threshold configuration."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class IncreasingThreshold:
    safe: float
    danger: float
    unit: str
    notes: str

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.safe, self.danger)):
            raise ValueError("Hazard thresholds must be finite")
        if self.safe >= self.danger:
            raise ValueError("Increasing hazard threshold requires safe < danger")
        if not self.unit or not self.notes:
            raise ValueError("Hazard threshold units and notes are required")


@dataclass(frozen=True)
class DecreasingThreshold:
    safe: float
    danger: float
    unit: str
    notes: str

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.safe, self.danger)):
            raise ValueError("Hazard thresholds must be finite")
        if self.safe <= self.danger:
            raise ValueError("Decreasing hazard threshold requires safe > danger")
        if not self.unit or not self.notes:
            raise ValueError("Hazard threshold units and notes are required")


@dataclass(frozen=True)
class UncertaintyReference:
    value: float
    unit: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.value) or self.value <= 0.0:
            raise ValueError("Uncertainty reference values must be finite and positive")
        if not self.unit:
            raise ValueError("Uncertainty reference units are required")


@dataclass(frozen=True)
class HazardParameters:
    temperature: IncreasingThreshold
    co: IncreasingThreshold
    smoke: IncreasingThreshold
    visibility: DecreasingThreshold
    temperature_uncertainty: UncertaintyReference
    co_uncertainty: UncertaintyReference
    smoke_uncertainty: UncertaintyReference
    visibility_uncertainty: UncertaintyReference
    assumption_status: str
    notes: str


def _increasing(data: dict[str, Any]) -> IncreasingThreshold:
    return IncreasingThreshold(
        safe=float(data["safe"]),
        danger=float(data["danger"]),
        unit=str(data["unit"]),
        notes=str(data["notes"]),
    )


def _reference(data: dict[str, Any]) -> UncertaintyReference:
    return UncertaintyReference(value=float(data["value"]), unit=str(data["unit"]))


def load_hazard_parameters(config_path: str | Path) -> HazardParameters:
    with Path(config_path).open("r", encoding="utf-8") as stream:
        data: dict[str, Any] = json.load(stream)
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported hazard configuration schema_version")
    if data.get("aggregation") != "maximum":
        raise ValueError("P4 supports only conservative maximum aggregation")
    thresholds = data["thresholds"]
    references = data["uncertainty_reference"]
    visibility = thresholds["visibility"]
    return HazardParameters(
        temperature=_increasing(thresholds["temperature"]),
        co=_increasing(thresholds["co"]),
        smoke=_increasing(thresholds["smoke"]),
        visibility=DecreasingThreshold(
            safe=float(visibility["safe"]),
            danger=float(visibility["danger"]),
            unit=str(visibility["unit"]),
            notes=str(visibility["notes"]),
        ),
        temperature_uncertainty=_reference(references["temperature"]),
        co_uncertainty=_reference(references["co"]),
        smoke_uncertainty=_reference(references["smoke"]),
        visibility_uncertainty=_reference(references["visibility"]),
        assumption_status=str(data["assumption_status"]),
        notes=str(data.get("notes", "")),
    )

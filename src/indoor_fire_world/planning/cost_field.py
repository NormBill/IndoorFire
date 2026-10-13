"""Planning-cost construction without route search."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from indoor_fire_world.hazard.result import HazardResult, UncertaintyAggregationResult
from indoor_fire_world.scene import Occupancy
from indoor_fire_world.scene.models import NavigationLayer


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class PlanningCostParameters:
    base_cost: float
    hazard_weight: float
    uncertainty_weight: float
    untenable_hazard_threshold: float
    forbidden_representation: str = "infinity"
    notes: str = ""

    def __post_init__(self) -> None:
        values = (self.base_cost, self.hazard_weight, self.uncertainty_weight)
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("Planning cost and weights must be finite and non-negative")
        if not math.isfinite(self.untenable_hazard_threshold) or not (
            0.0 <= self.untenable_hazard_threshold <= 1.0
        ):
            raise ValueError("Untenable hazard threshold must be in [0, 1]")
        if self.forbidden_representation != "infinity":
            raise ValueError("P4 supports only infinity as the forbidden representation")


@dataclass(frozen=True)
class PlanningCostResult:
    field: FloatArray
    forbidden: BoolArray
    obstacle_forbidden: BoolArray
    hazard_forbidden: BoolArray

    def __post_init__(self) -> None:
        field = np.array(self.field, dtype=np.float64, copy=True)
        masks = tuple(
            np.array(getattr(self, name), dtype=np.bool_, copy=True)
            for name in ("forbidden", "obstacle_forbidden", "hazard_forbidden")
        )
        if field.ndim != 2 or any(mask.shape != field.shape for mask in masks):
            raise ValueError("Planning field and masks must be matching 2D arrays")
        if np.any(~np.isfinite(field[~masks[0]])) or np.any(field[~masks[0]] < 0.0):
            raise ValueError("Traversable planning costs must be finite and non-negative")
        if np.any(~np.isinf(field[masks[0]])):
            raise ValueError("Forbidden planning cells must have infinite cost")
        field.setflags(write=False)
        for mask in masks:
            mask.setflags(write=False)
        object.__setattr__(self, "field", field)
        for name, mask in zip(("forbidden", "obstacle_forbidden", "hazard_forbidden"), masks):
            object.__setattr__(self, name, mask)

    @property
    def traversable(self) -> BoolArray:
        traversable = np.logical_not(self.forbidden)
        traversable.setflags(write=False)
        return traversable

    @property
    def shape(self) -> tuple[int, int]:
        return self.field.shape

    def value_at(self, row: int, col: int) -> float:
        return float(self.field[row, col])


class PlanningCostModel:
    def __init__(self, parameters: PlanningCostParameters) -> None:
        self.parameters = parameters

    def compute(
        self,
        hazard: HazardResult | NDArray[np.floating],
        uncertainty: UncertaintyAggregationResult | NDArray[np.floating],
        navigation: NavigationLayer,
    ) -> PlanningCostResult:
        hazard_field = np.asarray(
            hazard.field if isinstance(hazard, HazardResult) else hazard,
            dtype=np.float64,
        )
        uncertainty_field = np.asarray(
            uncertainty.field
            if isinstance(uncertainty, UncertaintyAggregationResult)
            else uncertainty,
            dtype=np.float64,
        )
        if hazard_field.shape != uncertainty_field.shape:
            raise ValueError("Hazard and uncertainty shapes must match")
        geometry = navigation.occupancy.geometry
        if hazard_field.shape != (geometry.rows, geometry.cols):
            raise ValueError("Planning inputs must match navigation geometry")
        for name, field in (("hazard", hazard_field), ("uncertainty", uncertainty_field)):
            if not np.all(np.isfinite(field)) or np.any((field < 0.0) | (field > 1.0)):
                raise ValueError(f"Normalized {name} field must be finite and in [0, 1]")
        obstacle_forbidden = np.array(
            [
                [cell != Occupancy.FREE for cell in row]
                for row in navigation.occupancy.cells
            ],
            dtype=np.bool_,
        )
        hazard_forbidden = hazard_field >= self.parameters.untenable_hazard_threshold
        forbidden = obstacle_forbidden | hazard_forbidden
        field = (
            self.parameters.base_cost
            + self.parameters.hazard_weight * hazard_field
            + self.parameters.uncertainty_weight * uncertainty_field
        )
        field = np.array(field, dtype=np.float64, copy=True)
        field[forbidden] = np.inf
        return PlanningCostResult(field, forbidden, obstacle_forbidden, hazard_forbidden)


def load_planning_cost_parameters(config_path: str | Path) -> PlanningCostParameters:
    with Path(config_path).open("r", encoding="utf-8") as stream:
        data: dict[str, Any] = json.load(stream)
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported planning cost configuration schema_version")
    return PlanningCostParameters(
        base_cost=float(data["base_cost"]),
        hazard_weight=float(data["hazard_weight"]),
        uncertainty_weight=float(data["uncertainty_weight"]),
        untenable_hazard_threshold=float(data["untenable_hazard_threshold"]),
        forbidden_representation=str(data["forbidden_representation"]),
        notes=str(data.get("notes", "")),
    )

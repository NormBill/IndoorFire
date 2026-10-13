"""Physical hazard and unit-aware uncertainty aggregation models."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from indoor_fire_world.world_model import BeliefMean
from indoor_fire_world.world_model.uncertainty import UncertaintyField

from .result import HazardResult, UncertaintyAggregationResult
from .thresholds import HazardParameters, IncreasingThreshold, DecreasingThreshold


MODALITIES = ("temperature", "co", "smoke", "visibility")


def increasing_severity(
    values: NDArray[np.floating],
    threshold: IncreasingThreshold,
) -> NDArray[np.float64]:
    return np.clip(
        (np.asarray(values, dtype=np.float64) - threshold.safe)
        / (threshold.danger - threshold.safe),
        0.0,
        1.0,
    )


def decreasing_severity(
    values: NDArray[np.floating],
    threshold: DecreasingThreshold,
) -> NDArray[np.float64]:
    return np.clip(
        (threshold.safe - np.asarray(values, dtype=np.float64))
        / (threshold.safe - threshold.danger),
        0.0,
        1.0,
    )


class PhysicalHazardModel:
    def __init__(self, parameters: HazardParameters) -> None:
        self.parameters = parameters

    def compute(self, mean: BeliefMean) -> HazardResult:
        severities = {
            "temperature": increasing_severity(mean.temperature, self.parameters.temperature),
            "co": increasing_severity(mean.co, self.parameters.co),
            "smoke": increasing_severity(mean.smoke, self.parameters.smoke),
            "visibility": decreasing_severity(mean.visibility, self.parameters.visibility),
        }
        stacked = np.stack([severities[name] for name in MODALITIES], axis=0)
        dominant_indices = np.argmax(stacked, axis=0)
        modality_names = np.asarray(MODALITIES)
        return HazardResult(
            field=np.max(stacked, axis=0),
            dominant_modality=modality_names[dominant_indices],
            **severities,
        )


class UncertaintyAggregationModel:
    def __init__(self, parameters: HazardParameters) -> None:
        self.parameters = parameters

    def compute(self, uncertainty: UncertaintyField) -> UncertaintyAggregationResult:
        normalized = {
            name: np.clip(
                np.asarray(getattr(uncertainty, name), dtype=np.float64)
                / getattr(self.parameters, f"{name}_uncertainty").value,
                0.0,
                1.0,
            )
            for name in MODALITIES
        }
        return UncertaintyAggregationResult(
            field=np.maximum.reduce([normalized[name] for name in MODALITIES]),
            **normalized,
        )

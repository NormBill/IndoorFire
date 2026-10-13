"""Belief mean and uncertainty initialized from a physics-prior snapshot."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from indoor_fire_world.fire import FireField
from indoor_fire_world.p3_config import InitialUncertaintyParameters

from .uncertainty import UncertaintyField, initialize_uncertainty, readonly_array


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BeliefMean:
    """Estimated environmental fields, distinct from hidden ground truth."""

    temperature: FloatArray
    co: FloatArray
    smoke: FloatArray
    visibility: FloatArray

    def __post_init__(self) -> None:
        arrays = tuple(
            readonly_array(getattr(self, name))
            for name in ("temperature", "co", "smoke", "visibility")
        )
        shape = arrays[0].shape
        if len(shape) != 2 or any(array.shape != shape for array in arrays):
            raise ValueError("Belief mean fields must be matching two-dimensional arrays")
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("Belief mean values must be finite")
        for name, array in zip(("temperature", "co", "smoke", "visibility"), arrays):
            object.__setattr__(self, name, array)

    @classmethod
    def from_fire_field(cls, field: FireField) -> "BeliefMean":
        return cls(
            temperature=field.temperature,
            co=field.co,
            smoke=field.smoke,
            visibility=field.visibility,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.temperature.shape


@dataclass(frozen=True)
class BeliefState:
    timestamp: float
    mean: BeliefMean
    uncertainty: UncertaintyField

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp) or self.timestamp < 0.0:
            raise ValueError("Belief timestamp must be finite and non-negative")
        if self.mean.shape != self.uncertainty.shape:
            raise ValueError("Belief mean and uncertainty shapes must match")

    @property
    def fire_field(self) -> BeliefMean:
        """Compatibility alias for the P0-P2 belief contract."""
        return self.mean

    @property
    def uncertainty_field(self) -> UncertaintyField:
        """Compatibility alias for the P0-P2 belief contract."""
        return self.uncertainty


def initialize_belief(
    prior_state: FireField,
    uncertainty_parameters: InitialUncertaintyParameters,
) -> BeliefState:
    mean = BeliefMean.from_fire_field(prior_state)
    return BeliefState(
        timestamp=prior_state.timestamp,
        mean=mean,
        uncertainty=initialize_uncertainty(mean.shape, uncertainty_parameters),
    )

"""Immutable environmental snapshots produced by fire simulators."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def visibility_from_smoke(
    smoke: NDArray[np.floating],
    clear_air_visibility_m: float,
    extinction_coefficient: float,
) -> FloatArray:
    """Derive visibility from smoke; visibility is never independently evolved."""
    smoke_nonnegative = np.maximum(np.asarray(smoke, dtype=np.float64), 0.0)
    return clear_air_visibility_m * np.exp(-extinction_coefficient * smoke_nonnegative)


def _readonly_copy(values: NDArray[np.floating]) -> FloatArray:
    copied = np.array(values, dtype=np.float64, copy=True)
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True)
class FireField:
    """Read-only temperature, pollutant, and derived-visibility snapshot."""

    temperature: FloatArray
    co: FloatArray
    smoke: FloatArray
    timestamp: float
    clear_air_visibility_m: float = field(repr=False)
    smoke_extinction_coefficient: float = field(repr=False)
    visibility: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        temperature = _readonly_copy(self.temperature)
        co = _readonly_copy(self.co)
        smoke = _readonly_copy(self.smoke)
        if temperature.shape != co.shape or temperature.shape != smoke.shape:
            raise ValueError("All fire fields must have the same shape")
        if temperature.ndim != 2:
            raise ValueError("Fire fields must be two-dimensional")
        if self.timestamp < 0.0:
            raise ValueError("timestamp must be non-negative")
        if self.clear_air_visibility_m <= 0.0:
            raise ValueError("clear_air_visibility_m must be positive")
        if self.smoke_extinction_coefficient < 0.0:
            raise ValueError("smoke_extinction_coefficient must be non-negative")
        visibility = _readonly_copy(
            visibility_from_smoke(
                smoke,
                self.clear_air_visibility_m,
                self.smoke_extinction_coefficient,
            )
        )
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "co", co)
        object.__setattr__(self, "smoke", smoke)
        object.__setattr__(self, "visibility", visibility)

    @property
    def shape(self) -> tuple[int, int]:
        return self.temperature.shape

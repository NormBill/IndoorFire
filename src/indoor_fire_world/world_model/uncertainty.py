"""Spatial epistemic uncertainty kept separate from physical and hazard fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from indoor_fire_world.p3_config import InitialUncertaintyParameters


FloatArray = NDArray[np.float64]


def readonly_array(values: NDArray[np.floating]) -> FloatArray:
    copied = np.array(values, dtype=np.float64, copy=True)
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True)
class UncertaintyField:
    """Per-cell diagonal epistemic variance proxies in squared native units."""

    temperature: FloatArray
    co: FloatArray
    smoke: FloatArray
    visibility: FloatArray
    meaning: str

    def __post_init__(self) -> None:
        arrays = tuple(
            readonly_array(getattr(self, name))
            for name in ("temperature", "co", "smoke", "visibility")
        )
        shape = arrays[0].shape
        if len(shape) != 2 or any(array.shape != shape for array in arrays):
            raise ValueError("Uncertainty fields must be matching two-dimensional arrays")
        if any(not np.all(np.isfinite(array)) or np.any(array < 0.0) for array in arrays):
            raise ValueError("Uncertainty values must be finite and non-negative")
        if not self.meaning:
            raise ValueError("Uncertainty meaning must not be empty")
        for name, array in zip(("temperature", "co", "smoke", "visibility"), arrays):
            object.__setattr__(self, name, array)

    @property
    def shape(self) -> tuple[int, int]:
        return self.temperature.shape


def initialize_uncertainty(
    shape: tuple[int, int],
    parameters: InitialUncertaintyParameters,
) -> UncertaintyField:
    return UncertaintyField(
        temperature=np.full(shape, parameters.temperature, dtype=np.float64),
        co=np.full(shape, parameters.co, dtype=np.float64),
        smoke=np.full(shape, parameters.smoke, dtype=np.float64),
        visibility=np.full(shape, parameters.visibility, dtype=np.float64),
        meaning=parameters.meaning,
    )

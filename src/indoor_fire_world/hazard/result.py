"""Immutable and explainable P4 field results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
StringArray = NDArray[np.str_]


def readonly_float(values: NDArray[np.floating]) -> FloatArray:
    copied = np.array(values, dtype=np.float64, copy=True)
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True)
class HazardResult:
    field: FloatArray
    dominant_modality: StringArray
    temperature: FloatArray
    co: FloatArray
    smoke: FloatArray
    visibility: FloatArray

    def __post_init__(self) -> None:
        names = ("field", "temperature", "co", "smoke", "visibility")
        arrays = tuple(readonly_float(getattr(self, name)) for name in names)
        shape = arrays[0].shape
        dominant = np.array(self.dominant_modality, dtype=np.str_, copy=True)
        if len(shape) != 2 or any(array.shape != shape for array in arrays):
            raise ValueError("Hazard fields must be matching two-dimensional arrays")
        if dominant.shape != shape:
            raise ValueError("Dominant hazard modality shape must match hazard field")
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("Hazard values must be finite")
        if any(np.any((array < 0.0) | (array > 1.0)) for array in arrays):
            raise ValueError("Hazard severity values must remain in [0, 1]")
        if not np.allclose(arrays[0], np.maximum.reduce(arrays[1:])):
            raise ValueError("Hazard field must equal the maximum modality severity")
        if not np.all(np.isin(dominant, ("temperature", "co", "smoke", "visibility"))):
            raise ValueError("Dominant hazard modality contains an unsupported label")
        dominant.setflags(write=False)
        for name, array in zip(names, arrays):
            object.__setattr__(self, name, array)
        object.__setattr__(self, "dominant_modality", dominant)

    @property
    def shape(self) -> tuple[int, int]:
        return self.field.shape

    def value_at(self, row: int, col: int) -> float:
        return float(self.field[row, col])


@dataclass(frozen=True)
class UncertaintyAggregationResult:
    field: FloatArray
    temperature: FloatArray
    co: FloatArray
    smoke: FloatArray
    visibility: FloatArray

    def __post_init__(self) -> None:
        names = ("field", "temperature", "co", "smoke", "visibility")
        arrays = tuple(readonly_float(getattr(self, name)) for name in names)
        shape = arrays[0].shape
        if len(shape) != 2 or any(array.shape != shape for array in arrays):
            raise ValueError("Aggregated uncertainty fields must be matching 2D arrays")
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("Normalized uncertainty values must be finite")
        if any(np.any((array < 0.0) | (array > 1.0)) for array in arrays):
            raise ValueError("Normalized uncertainty values must remain in [0, 1]")
        if not np.allclose(arrays[0], np.maximum.reduce(arrays[1:])):
            raise ValueError("Uncertainty field must equal the maximum normalized modality")
        for name, array in zip(names, arrays):
            object.__setattr__(self, name, array)

    @property
    def shape(self) -> tuple[int, int]:
        return self.field.shape

    def value_at(self, row: int, col: int) -> float:
        return float(self.field[row, col])

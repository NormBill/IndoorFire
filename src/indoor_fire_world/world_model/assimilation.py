"""Deterministic local residual correction baseline for P3."""

from __future__ import annotations

import math

import numpy as np

from indoor_fire_world.p3_config import AssimilationParameters
from indoor_fire_world.scene import Occupancy, Scene
from indoor_fire_world.sensing import Observation

from .belief import BeliefMean, BeliefState
from .uncertainty import UncertaintyField


MODALITIES = ("temperature", "co", "smoke", "visibility")


class SpatialAssimilator:
    def __init__(self, scene: Scene, parameters: AssimilationParameters) -> None:
        self.scene = scene
        self.parameters = parameters
        geometry = scene.occupancy.geometry
        x = geometry.origin_x + (np.arange(geometry.cols) + 0.5) * geometry.resolution_m
        y = geometry.origin_y + (np.arange(geometry.rows) + 0.5) * geometry.resolution_m
        self._grid_x, self._grid_y = np.meshgrid(x, y)
        self._free_mask = np.array(
            [[cell == Occupancy.FREE for cell in row] for row in scene.occupancy.cells],
            dtype=np.bool_,
        )

    def assimilate(self, prior: BeliefState, observation: Observation) -> BeliefState:
        if not math.isclose(prior.timestamp, observation.timestamp, abs_tol=1e-9):
            raise ValueError("Observation and prior timestamps must match")
        sensor_cell = self.scene.navigation.metric_to_grid(*observation.position)
        if not self.scene.navigation.is_traversable(sensor_cell):
            raise ValueError("Observation location must be traversable")
        kernel = self._kernel(sensor_cell)

        mean_arrays = {
            name: np.array(getattr(prior.mean, name), copy=True)
            for name in MODALITIES
        }
        uncertainty_arrays = {
            name: np.array(getattr(prior.uncertainty, name), copy=True)
            for name in MODALITIES
        }
        for name in MODALITIES:
            observed = getattr(observation, name)
            if observed is None:
                continue
            if not math.isfinite(observed):
                raise ValueError(f"Observed {name} must be finite")
            residual = observed - float(getattr(prior.mean, name)[sensor_cell])
            mean_arrays[name] += self.parameters.gain * kernel * residual
            uncertainty_arrays[name] *= (
                1.0 - self.parameters.uncertainty_reduction_fraction * kernel
            )

        mean_arrays["temperature"] = np.maximum(
            mean_arrays["temperature"],
            self.parameters.minimum_temperature_c,
        )
        for name in ("co", "smoke", "visibility"):
            mean_arrays[name] = np.maximum(mean_arrays[name], 0.0)
        for name in MODALITIES:
            uncertainty_arrays[name] = np.maximum(uncertainty_arrays[name], 0.0)

        return BeliefState(
            timestamp=observation.timestamp,
            mean=BeliefMean(**mean_arrays),
            uncertainty=UncertaintyField(
                **uncertainty_arrays,
                meaning=prior.uncertainty.meaning,
            ),
        )

    def _kernel(self, sensor_cell: tuple[int, int]) -> np.ndarray:
        sensor_x, sensor_y = self.scene.navigation.grid_to_metric(*sensor_cell)
        distance_squared = (self._grid_x - sensor_x) ** 2 + (self._grid_y - sensor_y) ** 2
        kernel = np.exp(
            -distance_squared / (2.0 * self.parameters.kernel_sigma_m**2)
        )
        kernel[distance_squared > self.parameters.max_radius_m**2] = 0.0
        kernel[~self._free_mask] = 0.0
        return kernel

"""Seeded point sensor that reads immutable hidden ground-truth snapshots."""

from __future__ import annotations

import numpy as np

from indoor_fire_world.fire import FireField
from indoor_fire_world.p3_config import SensorNoiseParameters
from indoor_fire_world.scene import Pose2D, Scene

from .models import Observation


class UGVSensor:
    def __init__(self, scene: Scene, noise: SensorNoiseParameters, seed: int) -> None:
        if seed < 0:
            raise ValueError("Sensor random seed must be non-negative")
        self.scene = scene
        self.noise = noise
        self._rng = np.random.default_rng(seed)

    def observe(self, truth_state: FireField, pose: Pose2D) -> Observation:
        """Sample one grid cell without retaining or mutating the truth state."""
        cell = self.scene.navigation.metric_to_grid(pose.x, pose.y)
        if not self.scene.navigation.is_traversable(cell):
            raise ValueError("UGV sensor pose must be traversable")
        return Observation(
            timestamp=truth_state.timestamp,
            position=(pose.x, pose.y),
            temperature=float(truth_state.temperature[cell] + self._noise(self.noise.temperature_std)),
            co=max(0.0, float(truth_state.co[cell] + self._noise(self.noise.co_std))),
            smoke=max(0.0, float(truth_state.smoke[cell] + self._noise(self.noise.smoke_std))),
            visibility=max(0.0, float(truth_state.visibility[cell] + self._noise(self.noise.visibility_std))),
            source="ugv",
        )

    def _noise(self, standard_deviation: float) -> float:
        if standard_deviation == 0.0:
            return 0.0
        return float(self._rng.normal(0.0, standard_deviation))

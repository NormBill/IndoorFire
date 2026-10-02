"""Deterministic reduced-order ground-truth fire simulator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from indoor_fire_world.scene.models import Occupancy, Scene

from .parameters import FireParameters, load_fire_parameters
from .source import MicrowaveFireSource
from .state import FireField
from .transport import advance_scalar, validate_stable_timestep


ParameterInput = FireParameters | str | Path


class ReducedOrderFireSimulator:
    """Shared solver implementation; each instance exclusively owns its state."""

    def __init__(self, scene: Scene, config: ParameterInput) -> None:
        self.scene = scene
        self.parameters = config if isinstance(config, FireParameters) else load_fire_parameters(config)
        self.source = MicrowaveFireSource.from_scene(scene, self.parameters)
        shape = (scene.occupancy.geometry.rows, scene.occupancy.geometry.cols)
        self._fluid_mask = np.fromiter(
            (cell == Occupancy.FREE for row in scene.occupancy.cells for cell in row),
            dtype=np.bool_,
            count=shape[0] * shape[1],
        ).reshape(shape)
        self._temperature = np.full(shape, self.parameters.ambient_temperature_c, dtype=np.float64)
        self._co = np.zeros(shape, dtype=np.float64)
        self._smoke = np.zeros(shape, dtype=np.float64)
        self._timestamp = 0.0
        self._validate_timestep(self.parameters.time_step_s)

    def _validate_timestep(self, dt: float) -> None:
        spacing = self.scene.occupancy.geometry.resolution_m
        velocity = (self.parameters.ventilation_x_mps, self.parameters.ventilation_y_mps)
        fields = (
            ("temperature", self.parameters.temperature_diffusion_m2_per_s, self.parameters.temperature_relaxation_per_s),
            ("smoke", self.parameters.smoke_diffusion_m2_per_s, self.parameters.smoke_decay_per_s),
            ("co", self.parameters.co_diffusion_m2_per_s, self.parameters.co_decay_per_s),
        )
        for name, diffusion, decay in fields:
            validate_stable_timestep(dt, spacing, diffusion, velocity, decay, name)

    @property
    def current_state(self) -> FireField:
        """Return a read-only deep snapshot, never the simulator's mutable arrays."""
        return FireField(
            temperature=self._temperature,
            co=self._co,
            smoke=self._smoke,
            timestamp=self._timestamp,
            clear_air_visibility_m=self.parameters.clear_air_visibility_m,
            smoke_extinction_coefficient=self.parameters.smoke_extinction_per_concentration,
        )

    @property
    def fire_field(self) -> FireField:
        return self.current_state

    def step(self, dt: float | None = None) -> FireField:
        step_s = self.parameters.time_step_s if dt is None else float(dt)
        self._validate_timestep(step_s)
        source_rate = self._source_rate(self._timestamp + step_s)
        spacing = self.scene.occupancy.geometry.resolution_m
        velocity = (self.parameters.ventilation_x_mps, self.parameters.ventilation_y_mps)

        temperature_anomaly = self._temperature - self.parameters.ambient_temperature_c
        self._temperature = self.parameters.ambient_temperature_c + advance_scalar(
            temperature_anomaly,
            self._fluid_mask,
            source_rate * self.parameters.temperature_gain_c_per_kw_s,
            dt=step_s,
            cell_size_m=spacing,
            diffusion_m2_per_s=self.parameters.temperature_diffusion_m2_per_s,
            velocity_mps=velocity,
            decay_per_s=self.parameters.temperature_relaxation_per_s,
            solid_value=0.0,
        )
        self._smoke = advance_scalar(
            self._smoke,
            self._fluid_mask,
            source_rate * self.parameters.smoke_yield_per_kw_s,
            dt=step_s,
            cell_size_m=spacing,
            diffusion_m2_per_s=self.parameters.smoke_diffusion_m2_per_s,
            velocity_mps=velocity,
            decay_per_s=self.parameters.smoke_decay_per_s,
            solid_value=0.0,
        )
        self._co = advance_scalar(
            self._co,
            self._fluid_mask,
            source_rate * self.parameters.co_yield_per_kw_s,
            dt=step_s,
            cell_size_m=spacing,
            diffusion_m2_per_s=self.parameters.co_diffusion_m2_per_s,
            velocity_mps=velocity,
            decay_per_s=self.parameters.co_decay_per_s,
            solid_value=0.0,
        )
        self._timestamp += step_s
        return self.current_state

    def _source_rate(self, evaluation_time: float) -> NDArray[np.float64]:
        rate = np.zeros_like(self._smoke)
        rate[self.source.cell] = self.source.heat_release_rate_kw(evaluation_time)
        return rate


class GroundTruthSimulator(ReducedOrderFireSimulator):
    """Hidden ground-truth simulator with its own configuration and state."""


"""Deterministic microwave ignition source."""

from __future__ import annotations

from dataclasses import dataclass

from indoor_fire_world.scene.models import Occupancy, Scene

from .parameters import FireParameters, FireSourcePhase


@dataclass(frozen=True)
class MicrowaveFireSource:
    cell: tuple[int, int]
    parameters: FireParameters

    @classmethod
    def from_scene(cls, scene: Scene, parameters: FireParameters) -> "MicrowaveFireSource":
        semantic = scene.semantic.by_id(parameters.source_semantic_id)
        if semantic.kind != "microwave" or semantic.position is None:
            raise ValueError("Configured source must reference a positioned microwave semantic object")
        cell = scene.navigation.metric_to_grid(*semantic.position)
        if scene.occupancy.at(cell) != Occupancy.FREE:
            raise ValueError("Microwave fire source must be located in FREE occupancy")
        return cls(cell=cell, parameters=parameters)

    def heat_release_rate_kw(self, timestamp: float) -> float:
        if not self.parameters.source_enabled:
            return 0.0
        phase = self.phase_at(timestamp)
        if phase is not None:
            if phase.hrr_model == "inactive":
                return 0.0
            if phase.hrr_model == "constant":
                return phase.initial_heat_release_rate_kw
            elapsed = timestamp - phase.start_s
            growth = (
                phase.initial_heat_release_rate_kw
                + phase.fire_growth_rate_kw_per_s2 * elapsed * elapsed
            )
            return min(growth, phase.max_heat_release_rate_kw)
        growth = self.parameters.fire_growth_rate_kw_per_s2 * timestamp * timestamp
        return min(growth, self.parameters.max_heat_release_rate_kw)

    def phase_at(self, timestamp: float) -> FireSourcePhase | None:
        if timestamp < 0.0:
            raise ValueError("Fire source timestamp must be non-negative")
        for phase in self.parameters.source_phases:
            if timestamp >= phase.start_s and (
                phase.end_s is None or timestamp < phase.end_s
            ):
                return phase
        return None

    def emission_coefficients(self, timestamp: float) -> tuple[float, float, float]:
        phase = self.phase_at(timestamp)
        if phase is None:
            return (
                self.parameters.temperature_gain_c_per_kw_s,
                self.parameters.smoke_yield_per_kw_s,
                self.parameters.co_yield_per_kw_s,
            )
        return (
            phase.temperature_gain_c_per_kw_s,
            phase.smoke_yield_per_kw_s,
            phase.co_yield_per_kw_s,
        )

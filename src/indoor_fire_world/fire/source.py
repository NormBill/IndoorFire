"""Deterministic microwave ignition source."""

from __future__ import annotations

from dataclasses import dataclass

from indoor_fire_world.scene.models import Occupancy, Scene

from .parameters import FireParameters


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
        growth = self.parameters.fire_growth_rate_kw_per_s2 * timestamp * timestamp
        return min(growth, self.parameters.max_heat_release_rate_kw)

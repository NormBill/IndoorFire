"""Action-conditioned extension of the frozen reduced-order fire simulator."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from indoor_fire_world.fire import FireField, FireParameters, ReducedOrderFireSimulator
from indoor_fire_world.scene.models import Scene

from .actions import InterventionAction


class ActionConditionedFireSimulator(ReducedOrderFireSimulator):
    """Apply a post-arrival action only to future source emissions.

    Existing temperature, smoke, and CO fields are never cleared or rescaled.
    For ``no_intervention`` this class is numerically identical to the frozen P8
    simulator because the inherited transport implementation is unchanged.
    """

    def __init__(
        self,
        scene: Scene,
        parameters: FireParameters,
        action: InterventionAction,
    ) -> None:
        super().__init__(scene, parameters)
        self.action = action

    @classmethod
    def resume_from_snapshot(
        cls,
        scene: Scene,
        parameters: FireParameters,
        action: InterventionAction,
        snapshot: FireField,
    ) -> "ActionConditionedFireSimulator":
        """Create an independent post-handoff branch from one frozen snapshot."""
        simulator = cls(scene, parameters, action)
        expected = simulator.current_state.shape
        if snapshot.shape != expected:
            raise ValueError("Intervention snapshot shape does not match the scene")
        if snapshot.timestamp > (
            action.activation_time_s
            if action.activation_time_s is not None
            else snapshot.timestamp
        ):
            raise ValueError("Cannot resume after an intervention should already be active")
        simulator._temperature = np.array(snapshot.temperature, dtype=np.float64, copy=True)
        simulator._smoke = np.array(snapshot.smoke, dtype=np.float64, copy=True)
        simulator._co = np.array(snapshot.co, dtype=np.float64, copy=True)
        simulator._timestamp = float(snapshot.timestamp)
        return simulator

    def effective_heat_release_rate_kw(self, timestamp_s: float) -> float:
        return self.source.heat_release_rate_kw(timestamp_s) * self.action.source_multiplier(
            timestamp_s
        )

    def _source_rate(self, evaluation_time: float) -> NDArray[np.float64]:
        rate = np.zeros_like(self._smoke)
        rate[self.source.cell] = self.effective_heat_release_rate_kw(evaluation_time)
        return rate


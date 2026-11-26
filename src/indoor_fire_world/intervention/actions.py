"""Typed, simulation-assumed post-arrival intervention actions."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class InterventionAction:
    """One immutable action schedule for counterfactual rollout."""

    id: str
    action_type: str
    firefighter_available_time_s: float
    activation_time_s: float | None
    suppression_rate_per_s: float = 0.0
    residual_hrr_fraction: float = 1.0
    assumption_status: str = "simulation_assumed"

    def __post_init__(self) -> None:
        if not self.id or self.action_type not in {
            "no_intervention",
            "localized_suppression",
        }:
            raise ValueError("Unsupported P9 intervention action")
        if self.assumption_status != "simulation_assumed":
            raise ValueError("P9 intervention effects must remain simulation-assumed")
        if not math.isfinite(self.firefighter_available_time_s) or (
            self.firefighter_available_time_s < 0.0
        ):
            raise ValueError("Firefighter availability time must be finite and non-negative")
        if self.action_type == "no_intervention":
            if self.activation_time_s is not None:
                raise ValueError("No-intervention action must not have an activation time")
            if self.suppression_rate_per_s != 0.0 or self.residual_hrr_fraction != 1.0:
                raise ValueError("No-intervention action must preserve the source exactly")
            return
        if self.activation_time_s is None or not math.isfinite(self.activation_time_s):
            raise ValueError("Localized suppression requires a finite activation time")
        if self.activation_time_s < self.firefighter_available_time_s:
            raise ValueError("Intervention cannot precede firefighter availability")
        if not math.isfinite(self.suppression_rate_per_s) or self.suppression_rate_per_s <= 0.0:
            raise ValueError("Suppression rate must be finite and positive")
        if not 0.0 <= self.residual_hrr_fraction <= 1.0:
            raise ValueError("Residual HRR fraction must be in [0, 1]")

    def source_multiplier(self, timestamp_s: float) -> float:
        """Return the causal source multiplier at one timestamp."""
        if timestamp_s < 0.0:
            raise ValueError("Action timestamp must be non-negative")
        if self.action_type == "no_intervention" or timestamp_s < float(
            self.activation_time_s
        ):
            return 1.0
        elapsed = timestamp_s - float(self.activation_time_s)
        return self.residual_hrr_fraction + (
            1.0 - self.residual_hrr_fraction
        ) * math.exp(-self.suppression_rate_per_s * elapsed)


"""Belief container kept independent from hidden simulator truth."""

from dataclasses import dataclass
from indoor_fire_world.fire import FireField


@dataclass(frozen=True)
class BeliefState:
    timestamp: float
    fire_field: FireField | None = None
    uncertainty_field: object | None = None


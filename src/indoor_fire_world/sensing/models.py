"""Measurements shared by fixed sensors and mobile platforms."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Observation:
    timestamp: float
    position: tuple[float, float]
    temperature: float | None = None
    co: float | None = None
    smoke: float | None = None
    visibility: float | None = None
    source: str = ""


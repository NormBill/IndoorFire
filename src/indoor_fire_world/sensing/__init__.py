"""Sensor-domain data contracts."""

from .models import Observation
from .ugv_sensor import UGVSensor

__all__ = ["Observation", "UGVSensor"]


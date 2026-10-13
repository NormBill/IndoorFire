"""Controlled phase demos composed from independent domain components."""

from .p3_demo import run_assimilation_demo
from .p4_demo import P4DemoOutput, run_p4_demo
from .p5_demo import P5DemoOutput, run_p5_demo, run_p5_weight_sensitivity
from .p6_demo import P6DemoOutput, run_p6_closed_loop_demo

__all__ = [
    "P4DemoOutput",
    "P5DemoOutput",
    "P6DemoOutput",
    "run_assimilation_demo",
    "run_p4_demo",
    "run_p5_demo",
    "run_p5_weight_sensitivity",
    "run_p6_closed_loop_demo",
]


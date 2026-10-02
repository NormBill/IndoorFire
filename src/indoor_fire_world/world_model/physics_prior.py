"""Reduced-order physics prior kept independent from hidden ground truth."""

from indoor_fire_world.fire.simulator import ReducedOrderFireSimulator


class PhysicsPrior(ReducedOrderFireSimulator):
    """Prediction-only P2 simulator; observation assimilation is not implemented."""


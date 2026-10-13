"""Controlled P4 propagation from P3 belief correction into decision fields."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from indoor_fire_world.fire import GroundTruthSimulator
from indoor_fire_world.hazard import (
    PhysicalHazardModel,
    UncertaintyAggregationModel,
    load_hazard_parameters,
)
from indoor_fire_world.p3_config import load_p3_parameters
from indoor_fire_world.planning import PlanningCostModel, load_planning_cost_parameters
from indoor_fire_world.robotics import UGV
from indoor_fire_world.scene import load_scene
from indoor_fire_world.sensing import UGVSensor
from indoor_fire_world.world_model import PhysicsPrior, SpatialAssimilator, initialize_belief


@dataclass(frozen=True)
class P4DemoOutput:
    summary: dict[str, Any]
    arrays: dict[str, NDArray[Any]]


def _advance_to(simulator: GroundTruthSimulator | PhysicsPrior, target: float) -> None:
    state = simulator.current_state
    while state.timestamp < target - 1e-12:
        remaining = target - state.timestamp
        state = simulator.step(min(simulator.parameters.time_step_s, remaining))


def run_p4_demo(
    scene_config: str | Path,
    truth_config: str | Path,
    prior_config: str | Path,
    p3_config: str | Path,
    hazard_config: str | Path,
    planning_config: str | Path,
) -> P4DemoOutput:
    scene = load_scene(scene_config)
    p3 = load_p3_parameters(p3_config)
    truth = GroundTruthSimulator(scene, truth_config)
    prior_model = PhysicsPrior(scene, prior_config)
    _advance_to(truth, p3.demo_target_timestamp_s)
    _advance_to(prior_model, p3.demo_target_timestamp_s)

    ugv = UGV(scene)
    waypoint = p3.waypoint(p3.demo_waypoint_id)
    ugv.move_to(waypoint.pose)
    observation = UGVSensor(scene, p3.sensor, p3.random_seed).observe(
        truth.current_state,
        ugv.pose,
    )
    prior_belief = initialize_belief(prior_model.current_state, p3.initial_uncertainty)
    posterior_belief = SpatialAssimilator(scene, p3.assimilation).assimilate(
        prior_belief,
        observation,
    )

    hazard_model = PhysicalHazardModel(load_hazard_parameters(hazard_config))
    uncertainty_model = UncertaintyAggregationModel(hazard_model.parameters)
    cost_model = PlanningCostModel(load_planning_cost_parameters(planning_config))
    prior_hazard = hazard_model.compute(prior_belief.mean)
    posterior_hazard = hazard_model.compute(posterior_belief.mean)
    prior_uncertainty = uncertainty_model.compute(prior_belief.uncertainty)
    posterior_uncertainty = uncertainty_model.compute(posterior_belief.uncertainty)
    prior_cost = cost_model.compute(prior_hazard, prior_uncertainty, scene.navigation)
    posterior_cost = cost_model.compute(
        posterior_hazard,
        posterior_uncertainty,
        scene.navigation,
    )

    observation_cell = scene.navigation.metric_to_grid(*observation.position)
    source_cell = truth.source.cell
    source_neighbor = next(scene.navigation.neighbors(source_cell))
    distant_cell = scene.navigation.metric_to_grid(
        scene.ugv_initial_pose.x,
        scene.ugv_initial_pose.y,
    )
    locations = {
        "ugv_observation": observation_cell,
        "microwave_neighborhood": source_neighbor,
        "distant_unaffected": distant_cell,
    }

    def stage_values(cell: tuple[int, int], stage: str) -> dict[str, Any]:
        if stage == "prior":
            hazard, uncertainty, cost = prior_hazard, prior_uncertainty, prior_cost
        else:
            hazard, uncertainty, cost = posterior_hazard, posterior_uncertainty, posterior_cost
        return {
            "hazard": float(hazard.field[cell]),
            "dominant_hazard": str(hazard.dominant_modality[cell]),
            "uncertainty": float(uncertainty.field[cell]),
            "planning_cost": float(cost.field[cell]),
            "planning_forbidden": bool(cost.forbidden[cell]),
        }

    location_summary: dict[str, Any] = {}
    for label, cell in locations.items():
        x, y = scene.navigation.grid_to_metric(*cell)
        prior_values = stage_values(cell, "prior")
        posterior_values = stage_values(cell, "posterior")
        location_summary[label] = {
            "cell": {"row": cell[0], "col": cell[1]},
            "cell_center_m": {"x": x, "y": y},
            "prior": prior_values,
            "posterior": posterior_values,
            "change": {
                "hazard": posterior_values["hazard"] - prior_values["hazard"],
                "uncertainty": posterior_values["uncertainty"] - prior_values["uncertainty"],
                "planning_cost": posterior_values["planning_cost"]
                - prior_values["planning_cost"],
            },
        }

    arrays: dict[str, NDArray[Any]] = {
        "prior_hazard": prior_hazard.field,
        "posterior_hazard": posterior_hazard.field,
        "prior_dominant_hazard": prior_hazard.dominant_modality,
        "posterior_dominant_hazard": posterior_hazard.dominant_modality,
        "prior_uncertainty": prior_uncertainty.field,
        "posterior_uncertainty": posterior_uncertainty.field,
        "prior_planning_cost": prior_cost.field,
        "posterior_planning_cost": posterior_cost.field,
        "prior_forbidden": prior_cost.forbidden,
        "posterior_forbidden": posterior_cost.forbidden,
    }
    return P4DemoOutput(
        summary={
            "timestamp": observation.timestamp,
            "waypoint_id": waypoint.id,
            "ugv_position": {"x": ugv.pose.x, "y": ugv.pose.y, "yaw": ugv.pose.yaw},
            "locations": location_summary,
            "observation_uncertainty_reduced": (
                posterior_uncertainty.field[observation_cell]
                < prior_uncertainty.field[observation_cell]
            ).item(),
            "notes": "Hazard and uncertainty are separate; planning cost may increase or decrease after observation.",
        },
        arrays=arrays,
    )

"""Reusable controlled P3 assimilation scenario."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from indoor_fire_world.fire import GroundTruthSimulator
from indoor_fire_world.p3_config import load_p3_parameters
from indoor_fire_world.robotics import UGV
from indoor_fire_world.scene import load_scene
from indoor_fire_world.sensing import UGVSensor
from indoor_fire_world.world_model import PhysicsPrior, SpatialAssimilator, initialize_belief


MODALITIES = ("temperature", "co", "smoke", "visibility")


def _advance_to(simulator: GroundTruthSimulator | PhysicsPrior, target: float) -> None:
    state = simulator.current_state
    while state.timestamp < target - 1e-12:
        remaining = target - state.timestamp
        state = simulator.step(min(simulator.parameters.time_step_s, remaining))


def run_assimilation_demo(
    scene_config: str | Path,
    truth_config: str | Path,
    prior_config: str | Path,
    p3_config: str | Path,
) -> dict[str, Any]:
    scene = load_scene(scene_config)
    parameters = load_p3_parameters(p3_config)
    truth = GroundTruthSimulator(scene, truth_config)
    prior_model = PhysicsPrior(scene, prior_config)
    _advance_to(truth, parameters.demo_target_timestamp_s)
    _advance_to(prior_model, parameters.demo_target_timestamp_s)

    ugv = UGV(scene)
    waypoint = parameters.waypoint(parameters.demo_waypoint_id)
    ugv.move_to(waypoint.pose)
    sensor = UGVSensor(scene, parameters.sensor, parameters.random_seed)
    observation = sensor.observe(truth.current_state, ugv.pose)
    prior_belief = initialize_belief(
        prior_model.current_state,
        parameters.initial_uncertainty,
    )
    posterior = SpatialAssimilator(scene, parameters.assimilation).assimilate(
        prior_belief,
        observation,
    )
    cell = scene.navigation.metric_to_grid(*observation.position)

    observed = {name: float(getattr(observation, name)) for name in MODALITIES}
    prior_prediction = {
        name: float(getattr(prior_belief.mean, name)[cell]) for name in MODALITIES
    }
    posterior_prediction = {
        name: float(getattr(posterior.mean, name)[cell]) for name in MODALITIES
    }
    prior_error = {
        name: abs(observed[name] - prior_prediction[name]) for name in MODALITIES
    }
    posterior_error = {
        name: abs(observed[name] - posterior_prediction[name]) for name in MODALITIES
    }
    uncertainty_before = {
        name: float(getattr(prior_belief.uncertainty, name)[cell]) for name in MODALITIES
    }
    uncertainty_after = {
        name: float(getattr(posterior.uncertainty, name)[cell]) for name in MODALITIES
    }
    error_improved = {
        name: posterior_error[name] < prior_error[name] for name in MODALITIES
    }
    uncertainty_reduced = {
        name: uncertainty_after[name] < uncertainty_before[name] for name in MODALITIES
    }
    return {
        "timestamp": observation.timestamp,
        "ugv_position": {"x": ugv.pose.x, "y": ugv.pose.y, "yaw": ugv.pose.yaw},
        "waypoint_id": waypoint.id,
        "observation": observed,
        "prior_prediction": prior_prediction,
        "posterior_prediction": posterior_prediction,
        "prior_error": prior_error,
        "posterior_error": posterior_error,
        "uncertainty_before": uncertainty_before,
        "uncertainty_after": uncertainty_after,
        "acceptance": {
            "posterior_error_lower": error_improved,
            "posterior_uncertainty_lower": uncertainty_reduced,
            "all_modalities_pass": all(error_improved.values()) and all(uncertainty_reduced.values()),
        },
    }

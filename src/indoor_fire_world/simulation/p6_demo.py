"""Controlled P6 orchestration from hidden model mismatch to route replanning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from indoor_fire_world.fire import GroundTruthSimulator, load_fire_parameters
from indoor_fire_world.hazard import (
    HazardResult,
    PhysicalHazardModel,
    UncertaintyAggregationModel,
    UncertaintyAggregationResult,
    load_hazard_parameters,
)
from indoor_fire_world.p6_config import P6ScenarioParameters, load_p6_parameters
from indoor_fire_world.planning import (
    GridAStarPlanner,
    MissionPlan,
    MissionPlanner,
    PlanningCostModel,
    PlanningCostResult,
    RouteResult,
)
from indoor_fire_world.robotics import UGV
from indoor_fire_world.scene import load_scene
from indoor_fire_world.scene.models import Scene
from indoor_fire_world.sensing import UGVSensor
from indoor_fire_world.world_model import PhysicsPrior, SpatialAssimilator, initialize_belief


MODALITIES = ("temperature", "co", "smoke", "visibility")


@dataclass(frozen=True)
class DecisionFields:
    hazard: HazardResult
    uncertainty: UncertaintyAggregationResult
    cost: PlanningCostResult


@dataclass(frozen=True)
class P6DemoOutput:
    summary: dict[str, Any]
    arrays: dict[str, NDArray[Any]]


def _advance_to(simulator: GroundTruthSimulator | PhysicsPrior, target: float) -> None:
    state = simulator.current_state
    while state.timestamp < target - 1e-12:
        remaining = target - state.timestamp
        state = simulator.step(min(simulator.parameters.time_step_s, remaining))


def _decision_fields(
    belief,
    hazard_model: PhysicalHazardModel,
    uncertainty_model: UncertaintyAggregationModel,
    cost_model: PlanningCostModel,
    scene: Scene,
) -> DecisionFields:
    hazard = hazard_model.compute(belief.mean)
    uncertainty = uncertainty_model.compute(belief.uncertainty)
    cost = cost_model.compute(hazard, uncertainty, scene.navigation)
    return DecisionFields(hazard=hazard, uncertainty=uncertainty, cost=cost)


def _resolve_mission_cells(
    scene: Scene,
    parameters: P6ScenarioParameters,
) -> tuple[tuple[int, int], tuple[int, int], dict[str, tuple[int, int]]]:
    entry = scene.navigation.metric_to_grid(*parameters.entry_start.position_m)
    target = scene.navigation.metric_to_grid(*parameters.inspection_target.position_m)
    exits: dict[str, tuple[int, int]] = {}
    for exit_id in parameters.retreat_exit_ids:
        semantic = scene.semantic.by_id(exit_id)
        if semantic.kind != "exit" or semantic.position is None:
            raise ValueError(f"Configured P6 retreat exit {exit_id!r} is invalid")
        exits[exit_id] = scene.navigation.metric_to_grid(*semantic.position)
    return entry, target, exits


def _mission_identity(mission: MissionPlan) -> str | None:
    if not mission.feasible:
        return None
    value = (
        f"{mission.entry_route.path_hash}|{mission.retreat_route.path_hash}|"
        f"{mission.selected_exit}"
    )
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _route_dict(route: RouteResult) -> dict[str, Any]:
    return {
        "feasible": route.feasible,
        "path": [{"row": row, "col": col} for row, col in route.path],
        "path_hash": route.path_hash,
        "geometric_length_m": route.geometric_length_m,
        "total_cost": route.total_cost if math.isfinite(route.total_cost) else None,
        "integrated_hazard": route.integrated_hazard,
        "integrated_uncertainty": route.integrated_uncertainty,
        "max_hazard": route.max_hazard,
        "cost_decomposition": {
            "base": route.base_contribution,
            "hazard": route.hazard_contribution,
            "uncertainty": route.uncertainty_contribution,
        },
        "rejection_reasons": list(route.rejection_reasons),
    }


def _mission_dict(mission: MissionPlan) -> dict[str, Any]:
    return {
        "feasible": mission.feasible,
        "mission_path_hash": _mission_identity(mission),
        "selected_exit": mission.selected_exit,
        "entry_route": _route_dict(mission.entry_route),
        "retreat_route": _route_dict(mission.retreat_route),
        "retreat_candidates": {
            candidate.exit_id: _route_dict(candidate.route)
            for candidate in mission.retreat_candidates
        },
        "rejection_reasons": list(mission.rejection_reasons),
    }


def _evaluate_fixed_mission(
    template: MissionPlan,
    fields: DecisionFields,
    planner: GridAStarPlanner,
    scene: Scene,
) -> dict[str, Any]:
    entry = planner.evaluate_path(
        template.entry_route.path,
        fields.cost.field,
        fields.cost.forbidden,
        scene.navigation,
        fields.hazard.field,
        fields.uncertainty.field,
    )
    retreat = planner.evaluate_path(
        template.retreat_route.path,
        fields.cost.field,
        fields.cost.forbidden,
        scene.navigation,
        fields.hazard.field,
        fields.uncertainty.field,
    )
    feasible = entry.feasible and retreat.feasible
    aggregate = None
    if feasible:
        aggregate = {
            "geometric_length_m": entry.geometric_length_m + retreat.geometric_length_m,
            "total_cost": entry.total_cost + retreat.total_cost,
            "integrated_hazard": entry.integrated_hazard + retreat.integrated_hazard,
            "integrated_uncertainty": (
                entry.integrated_uncertainty + retreat.integrated_uncertainty
            ),
            "max_hazard": max(entry.max_hazard, retreat.max_hazard),
            "cost_decomposition": {
                "base": entry.base_contribution + retreat.base_contribution,
                "hazard": entry.hazard_contribution + retreat.hazard_contribution,
                "uncertainty": (
                    entry.uncertainty_contribution + retreat.uncertainty_contribution
                ),
            },
        }
    return {
        "feasible": feasible,
        "selected_exit": template.selected_exit,
        "mission_path_hash": _mission_identity(template),
        "entry_route": _route_dict(entry),
        "retreat_route": _route_dict(retreat),
        "aggregate": aggregate,
    }


def _parameter_snapshot(
    parameters: P6ScenarioParameters,
    truth_parameters,
    prior_parameters,
    hazard_parameters,
) -> dict[str, Any]:
    return {
        "scenario_id": parameters.scenario_id,
        "inspection_time_s": parameters.inspection_time_s,
        "random_seed": parameters.random_seed,
        "config_files": {
            "scene": parameters.scene_config.name,
            "truth": parameters.truth_config.name,
            "prior": parameters.prior_config.name,
            "hazard": parameters.hazard_config.name,
        },
        "truth_parameters": asdict(truth_parameters),
        "prior_parameters": asdict(prior_parameters),
        "hazard_parameters": asdict(hazard_parameters),
        "sensor": asdict(parameters.sensor),
        "initial_uncertainty": asdict(parameters.initial_uncertainty),
        "assimilation": asdict(parameters.assimilation),
        "inspection_waypoint": {
            "id": parameters.inspection_waypoint.id,
            "pose": asdict(parameters.inspection_waypoint.pose),
        },
        "mission": {
            "entry_start": asdict(parameters.entry_start),
            "inspection_target": asdict(parameters.inspection_target),
            "retreat_exit_ids": list(parameters.retreat_exit_ids),
        },
        "planning_weights": asdict(parameters.planning),
        "local_comparison_radius_m": parameters.local_comparison_radius_m,
        "require_route_change": parameters.require_route_change,
        "notes": parameters.notes,
    }


def _snapshot_hash(snapshot: dict[str, Any]) -> str:
    encoded = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _stage_at(fields: DecisionFields, cell: tuple[int, int]) -> dict[str, Any]:
    return {
        "hazard": float(fields.hazard.field[cell]),
        "dominant_hazard": str(fields.hazard.dominant_modality[cell]),
        "uncertainty": float(fields.uncertainty.field[cell]),
        "planning_cost": _finite(float(fields.cost.field[cell])),
        "planning_forbidden": bool(fields.cost.forbidden[cell]),
    }


def _matches_standard_cost(
    fields: DecisionFields,
    parameters: P6ScenarioParameters,
) -> bool:
    expected = (
        parameters.planning.base_cost
        + parameters.planning.hazard_weight * fields.hazard.field
        + parameters.planning.uncertainty_weight * fields.uncertainty.field
    )
    allowed = ~fields.cost.forbidden
    return bool(
        np.allclose(fields.cost.field[allowed], expected[allowed])
        and np.all(np.isinf(fields.cost.field[fields.cost.forbidden]))
    )


def _local_comparison(
    scene: Scene,
    parameters: P6ScenarioParameters,
    initial: MissionPlan,
    prior: DecisionFields,
    posterior: DecisionFields,
    observation_cell: tuple[int, int],
) -> list[dict[str, Any]]:
    route_cells = set(initial.entry_route.path) | set(initial.retreat_route.path)
    waypoint = parameters.inspection_waypoint.pose
    cells = set()
    for cell in route_cells:
        x, y = scene.navigation.grid_to_metric(*cell)
        if math.hypot(x - waypoint.x, y - waypoint.y) <= (
            parameters.local_comparison_radius_m + 1e-12
        ):
            cells.add(cell)
    cells.add(observation_cell)
    records: list[dict[str, Any]] = []
    for cell in sorted(cells):
        x, y = scene.navigation.grid_to_metric(*cell)
        prior_values = _stage_at(prior, cell)
        posterior_values = _stage_at(posterior, cell)
        prior_cost = prior_values["planning_cost"]
        posterior_cost = posterior_values["planning_cost"]
        records.append(
            {
                "cell": {"row": cell[0], "col": cell[1]},
                "cell_center_m": {"x": x, "y": y},
                "on_original_route_a": cell in route_cells,
                "is_observation_cell": cell == observation_cell,
                "prior": prior_values,
                "posterior": posterior_values,
                "change": {
                    "hazard": posterior_values["hazard"] - prior_values["hazard"],
                    "uncertainty": (
                        posterior_values["uncertainty"] - prior_values["uncertainty"]
                    ),
                    "planning_cost": (
                        posterior_cost - prior_cost
                        if prior_cost is not None and posterior_cost is not None
                        else None
                    ),
                    "planning_cost_changed": (
                        prior_cost != posterior_cost
                        or prior_values["planning_forbidden"]
                        != posterior_values["planning_forbidden"]
                    ),
                },
            }
        )
    return records


def run_p6_closed_loop_demo(config_path: str | Path) -> P6DemoOutput:
    parameters = load_p6_parameters(config_path)
    scene = load_scene(parameters.scene_config)
    truth_parameters = load_fire_parameters(parameters.truth_config)
    prior_parameters = load_fire_parameters(parameters.prior_config)
    hazard_parameters = load_hazard_parameters(parameters.hazard_config)
    parameter_snapshot = _parameter_snapshot(
        parameters,
        truth_parameters,
        prior_parameters,
        hazard_parameters,
    )
    parameters_hash_before = _snapshot_hash(parameter_snapshot)

    truth = GroundTruthSimulator(scene, truth_parameters)
    prior_model = PhysicsPrior(scene, prior_parameters)
    hazard_model = PhysicalHazardModel(hazard_parameters)
    uncertainty_model = UncertaintyAggregationModel(hazard_parameters)
    cost_model = PlanningCostModel(parameters.planning)
    route_planner = GridAStarPlanner(parameters.planning)
    mission_planner = MissionPlanner(route_planner)
    entry, target, exits = _resolve_mission_cells(scene, parameters)

    _advance_to(truth, parameters.inspection_time_s)
    _advance_to(prior_model, parameters.inspection_time_s)
    prior_belief = initialize_belief(
        prior_model.current_state,
        parameters.initial_uncertainty,
    )
    prior_fields = _decision_fields(
        prior_belief,
        hazard_model,
        uncertainty_model,
        cost_model,
        scene,
    )
    initial_mission = mission_planner.plan(
        prior_fields.cost.field,
        prior_fields.cost.forbidden,
        prior_fields.hazard.field,
        prior_fields.uncertainty.field,
        scene.navigation,
        entry,
        target,
        exits,
    )

    ugv = UGV(scene)
    ugv.move_to(parameters.inspection_waypoint.pose)
    observation = UGVSensor(
        scene,
        parameters.sensor,
        parameters.random_seed,
    ).observe(truth.current_state, ugv.pose)
    posterior_belief = SpatialAssimilator(scene, parameters.assimilation).assimilate(
        prior_belief,
        observation,
    )
    posterior_fields = _decision_fields(
        posterior_belief,
        hazard_model,
        uncertainty_model,
        cost_model,
        scene,
    )
    posterior_mission = mission_planner.plan(
        posterior_fields.cost.field,
        posterior_fields.cost.forbidden,
        posterior_fields.hazard.field,
        posterior_fields.uncertainty.field,
        scene.navigation,
        entry,
        target,
        exits,
    )

    observation_cell = scene.navigation.metric_to_grid(*observation.position)
    observed = {name: float(getattr(observation, name)) for name in MODALITIES}
    prior_prediction = {
        name: float(getattr(prior_belief.mean, name)[observation_cell])
        for name in MODALITIES
    }
    posterior_prediction = {
        name: float(getattr(posterior_belief.mean, name)[observation_cell])
        for name in MODALITIES
    }
    prior_error = {
        name: abs(observed[name] - prior_prediction[name]) for name in MODALITIES
    }
    posterior_error = {
        name: abs(observed[name] - posterior_prediction[name]) for name in MODALITIES
    }
    uncertainty_before = {
        name: float(getattr(prior_belief.uncertainty, name)[observation_cell])
        for name in MODALITIES
    }
    uncertainty_after = {
        name: float(getattr(posterior_belief.uncertainty, name)[observation_cell])
        for name in MODALITIES
    }

    route_a_under_prior = _evaluate_fixed_mission(
        initial_mission, prior_fields, route_planner, scene
    )
    route_a_under_posterior = _evaluate_fixed_mission(
        initial_mission, posterior_fields, route_planner, scene
    )
    route_b_under_prior = _evaluate_fixed_mission(
        posterior_mission, prior_fields, route_planner, scene
    )
    route_b_under_posterior = _evaluate_fixed_mission(
        posterior_mission, posterior_fields, route_planner, scene
    )
    route_changed = _mission_identity(initial_mission) != _mission_identity(
        posterior_mission
    )
    if not initial_mission.feasible:
        change_reason = "initial_mission_infeasible"
    elif not posterior_mission.feasible:
        change_reason = "posterior_mission_infeasible"
    elif not route_a_under_posterior["feasible"]:
        change_reason = "original_route_a_infeasible_under_posterior_fields"
    elif route_changed and initial_mission.selected_exit != posterior_mission.selected_exit:
        change_reason = "updated_fields_select_lower_cost_route_and_exit"
    elif route_changed:
        route_a_cost = route_a_under_posterior["aggregate"]["total_cost"]
        route_b_cost = route_b_under_posterior["aggregate"]["total_cost"]
        change_reason = (
            "updated_fields_select_lower_cost_feasible_route"
            if route_b_cost < route_a_cost - 1e-12
            else "updated_fields_change_deterministic_optimal_route"
        )
    else:
        change_reason = "no_route_change"

    local = _local_comparison(
        scene,
        parameters,
        initial_mission,
        prior_fields,
        posterior_fields,
        observation_cell,
    )
    local_observation = next(record for record in local if record["is_observation_cell"])
    initial_cells = set(initial_mission.entry_route.path) | set(
        initial_mission.retreat_route.path
    )
    geometry = scene.navigation.occupancy.geometry
    field_finite = np.isfinite(prior_fields.cost.field) & np.isfinite(
        posterior_fields.cost.field
    )
    planning_field_changed = bool(
        np.any(
            np.abs(
                posterior_fields.cost.field[field_finite]
                - prior_fields.cost.field[field_finite]
            )
            > 1e-12
        )
        or np.any(prior_fields.cost.forbidden != posterior_fields.cost.forbidden)
    )
    parameters_hash_after = _snapshot_hash(
        _parameter_snapshot(
            parameters,
            truth.parameters,
            prior_model.parameters,
            hazard_model.parameters,
        )
    )
    posterior_recheck = _evaluate_fixed_mission(
        posterior_mission,
        posterior_fields,
        route_planner,
        scene,
    )
    acceptance = {
        "observation_differs_from_prior": any(
            value > 1e-12 for value in prior_error.values()
        ),
        "posterior_error_lower": all(
            posterior_error[name] < prior_error[name] for name in MODALITIES
        ),
        "local_uncertainty_decreases": (
            posterior_fields.uncertainty.field[observation_cell]
            < prior_fields.uncertainty.field[observation_cell]
        ).item(),
        "posterior_hazard_near_route_a_increases": (
            local_observation["on_original_route_a"]
            and local_observation["posterior"]["hazard"]
            > local_observation["prior"]["hazard"]
        ),
        "planning_field_changes": planning_field_changed,
        "initial_mission_feasible": initial_mission.feasible,
        "posterior_mission_feasible": posterior_mission.feasible,
        "posterior_replanning_uses_posterior_cost": (
            posterior_recheck["feasible"]
            and posterior_recheck["aggregate"] is not None
            and posterior_recheck["mission_path_hash"]
            == _mission_identity(posterior_mission)
        ),
        "observation_on_initial_route_a": observation_cell in initial_cells,
        "route_hash_changes": route_changed,
        "route_change_requirement_satisfied": (
            route_changed or not parameters.require_route_change
        ),
        "parameters_frozen": parameters_hash_before == parameters_hash_after,
        "no_manual_route_penalty": (
            _matches_standard_cost(prior_fields, parameters)
            and _matches_standard_cost(posterior_fields, parameters)
        ),
    }
    acceptance["controlled_scenario_pass"] = all(
        passed
        for name, passed in acceptance.items()
        if name != "route_hash_changes"
    )

    arrays: dict[str, NDArray[Any]] = {
        "prior_hazard": prior_fields.hazard.field,
        "posterior_hazard": posterior_fields.hazard.field,
        "prior_uncertainty": prior_fields.uncertainty.field,
        "posterior_uncertainty": posterior_fields.uncertainty.field,
        "prior_planning_cost": prior_fields.cost.field,
        "posterior_planning_cost": posterior_fields.cost.field,
        "prior_forbidden": prior_fields.cost.forbidden,
        "posterior_forbidden": posterior_fields.cost.forbidden,
    }
    return P6DemoOutput(
        summary={
            "scenario_parameters": parameter_snapshot,
            "parameter_fingerprint_before": parameters_hash_before,
            "parameter_fingerprint_after": parameters_hash_after,
            "timestamp": observation.timestamp,
            "grid_resolution_m": geometry.resolution_m,
            "ugv_observation": {
                "waypoint_id": parameters.inspection_waypoint.id,
                "position_m": {"x": ugv.pose.x, "y": ugv.pose.y},
                "cell": {"row": observation_cell[0], "col": observation_cell[1]},
                "observed": observed,
                "prior_prediction": prior_prediction,
                "posterior_prediction": posterior_prediction,
                "prior_error": prior_error,
                "posterior_error": posterior_error,
                "uncertainty_before": uncertainty_before,
                "uncertainty_after": uncertainty_after,
            },
            "prior_posterior_local_fields": local,
            "initial_route_a": _mission_dict(initial_mission),
            "posterior_route_b": _mission_dict(posterior_mission),
            "counterfactual": {
                "route_a_under_prior": route_a_under_prior,
                "route_a_under_posterior": route_a_under_posterior,
                "route_b_under_prior": route_b_under_prior,
                "route_b_under_posterior": route_b_under_posterior,
            },
            "route_changed": route_changed,
            "change_reason": change_reason,
            "acceptance": acceptance,
            "planning_field_provenance": {
                "formula": "base_cost + hazard_weight*hazard + uncertainty_weight*uncertainty",
                "manual_route_penalty": False,
                "weights_loaded_before_simulation": True,
            },
        },
        arrays=arrays,
    )

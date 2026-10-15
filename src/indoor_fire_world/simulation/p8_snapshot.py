"""P8 registered-map snapshot closed loop with retreat-only replanning."""

from __future__ import annotations

from dataclasses import dataclass
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
from indoor_fire_world.p3_config import InitialUncertaintyParameters
from indoor_fire_world.p8_config import P8ScenarioParameters, load_p8_parameters
from indoor_fire_world.planning import (
    GridAStarPlanner,
    MissionPlanner,
    PlanningCostModel,
    PlanningCostResult,
    RetreatPlan,
    RouteResult,
)
from indoor_fire_world.scene import Pose2D, load_scene
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
class FrozenObservationPoint:
    id: str
    offset_m: tuple[float, float]
    pose: Pose2D
    cell: tuple[int, int] | None
    valid: bool
    invalid_reason: str | None


@dataclass(frozen=True)
class P8CaseOutput:
    summary: dict[str, Any]
    arrays: dict[str, NDArray[Any]]


def _advance_to(simulator: GroundTruthSimulator | PhysicsPrior, target: float) -> None:
    state = simulator.current_state
    while state.timestamp < target - 1e-12:
        state = simulator.step(min(simulator.parameters.time_step_s, target - state.timestamp))


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_observation_points(
    scene: Scene,
    parameters: P8ScenarioParameters,
) -> tuple[FrozenObservationPoint, ...]:
    """Resolve configured geometry offsets without accepting any fire-field input."""
    microwave = scene.semantic.by_id("microwave")
    if microwave.position is None:
        raise ValueError("Registered microwave semantic requires a point position")
    frozen: list[FrozenObservationPoint] = []
    for item in parameters.observation_points:
        pose = Pose2D(
            x=microwave.position[0] + item.dx_m,
            y=microwave.position[1] + item.dy_m,
            yaw=item.yaw_rad,
        )
        try:
            cell = scene.navigation.metric_to_grid(pose.x, pose.y)
        except ValueError:
            frozen.append(
                FrozenObservationPoint(
                    id=item.id,
                    offset_m=(item.dx_m, item.dy_m),
                    pose=pose,
                    cell=None,
                    valid=False,
                    invalid_reason="observation_point_occupied",
                )
            )
            continue
        traversable = scene.navigation.is_traversable(cell)
        frozen.append(
            FrozenObservationPoint(
                id=item.id,
                offset_m=(item.dx_m, item.dy_m),
                pose=pose,
                cell=cell,
                valid=traversable,
                invalid_reason=None if traversable else "observation_point_occupied",
            )
        )
    return tuple(frozen)


def _resolve_static_geometry(
    scene: Scene,
    parameters: P8ScenarioParameters,
) -> tuple[tuple[int, int] | None, dict[str, tuple[int, int]], str | None]:
    try:
        ugv_start = scene.navigation.metric_to_grid(
            scene.ugv_initial_pose.x,
            scene.ugv_initial_pose.y,
        )
    except ValueError:
        return None, {}, "invalid_ugv_start"
    if not scene.navigation.is_traversable(ugv_start):
        return None, {}, "invalid_ugv_start"
    exits: dict[str, tuple[int, int]] = {}
    for exit_id in parameters.retreat_exit_ids:
        try:
            semantic = scene.semantic.by_id(exit_id)
        except KeyError:
            return ugv_start, {}, "invalid_configured_exit"
        if semantic.kind != "exit" or semantic.position is None:
            return ugv_start, {}, "invalid_configured_exit"
        try:
            cell = scene.navigation.metric_to_grid(*semantic.position)
        except ValueError:
            return ugv_start, {}, "invalid_configured_exit"
        if not scene.navigation.is_traversable(cell):
            return ugv_start, {}, "invalid_configured_exit"
        exits[exit_id] = cell
    return ugv_start, exits, None


def _decision_fields(
    belief,
    hazard_model: PhysicalHazardModel,
    uncertainty_model: UncertaintyAggregationModel,
    cost_model: PlanningCostModel,
    scene: Scene,
) -> DecisionFields:
    hazard = hazard_model.compute(belief.mean)
    uncertainty = uncertainty_model.compute(belief.uncertainty)
    return DecisionFields(
        hazard=hazard,
        uncertainty=uncertainty,
        cost=cost_model.compute(hazard, uncertainty, scene.navigation),
    )


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


def _retreat_dict(plan: RetreatPlan) -> dict[str, Any]:
    return {
        "feasible": plan.feasible,
        "selected_exit": plan.selected_exit,
        "route": _route_dict(plan.route),
        "candidates": {
            candidate.exit_id: _route_dict(candidate.route) for candidate in plan.candidates
        },
        "rejection_reasons": list(plan.rejection_reasons),
    }


def _invalid_summary(
    parameters: P8ScenarioParameters,
    point: FrozenObservationPoint,
    fire_time_s: float,
    reason: str,
    frozen_hash: str,
    *,
    observation_enabled: bool,
    assimilation_enabled: bool,
    oracle_planning: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scenario_id": parameters.scenario_id,
        "case_status": "invalid_observation_case",
        "invalid_reason": reason,
        "route_outcome": "invalid_observation_case",
        "fire_time_s": fire_time_s,
        "seconds_after_dispatch": fire_time_s - parameters.fire_age_at_dispatch_s,
        "snapshot_semantics": parameters.snapshot_semantics,
        "truth_simulator_constructed": False,
        "observation_point": _point_dict(point, frozen_hash),
        "run_controls": {
            "observation_enabled": observation_enabled,
            "assimilation_enabled": assimilation_enabled,
            "oracle_planning": oracle_planning,
        },
        "thresholds": {
            "ugv_operability_threshold": parameters.ugv_operability_threshold,
            "retreat_hazard_threshold": parameters.retreat_hazard_threshold,
        },
        "research_horizon": _horizon_dict(parameters),
    }


def _point_dict(point: FrozenObservationPoint, frozen_hash: str) -> dict[str, Any]:
    return {
        "id": point.id,
        "offset_m": list(point.offset_m),
        "resolved_position_m": {"x": point.pose.x, "y": point.pose.y},
        "cell": (
            {"row": point.cell[0], "col": point.cell[1]} if point.cell is not None else None
        ),
        "selection_provenance": "registered_geometry_only",
        "frozen_point_set_hash": frozen_hash,
    }


def _horizon_dict(parameters: P8ScenarioParameters) -> dict[str, Any]:
    return {
        "fire_age_at_dispatch_s": parameters.fire_age_at_dispatch_s,
        "pre_response_research_horizon_s": parameters.pre_response_research_horizon_s,
        "definition": parameters.horizon_definition,
        "evidence_status": parameters.horizon_evidence_status,
        "motivation_source": parameters.horizon_motivation_source,
        "explicit_exclusions": [
            "not_w1_measured_response_time",
            "not_actual_fire_engine_arrival_time",
            "not_firefighter_interior_deployment_completion_time",
        ],
    }


def _frozen_hash(points: tuple[FrozenObservationPoint, ...]) -> str:
    payload = [
        {
            "id": point.id,
            "offset_m": point.offset_m,
            "position": (point.pose.x, point.pose.y),
            "cell": point.cell,
            "valid": point.valid,
        }
        for point in points
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _local_mask(scene: Scene, cell: tuple[int, int], radius_m: float) -> NDArray[np.bool_]:
    rows, cols = np.indices((scene.occupancy.geometry.rows, scene.occupancy.geometry.cols))
    distance = np.hypot(rows - cell[0], cols - cell[1]) * scene.occupancy.geometry.resolution_m
    free = np.asarray(scene.occupancy.cells) == 0
    return free & (distance <= radius_m + 1e-12)


def _field_metrics(truth_state, prior_belief, posterior_belief, mask) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in MODALITIES:
        truth = np.asarray(getattr(truth_state, name))
        prior = np.asarray(getattr(prior_belief.mean, name))
        posterior = np.asarray(getattr(posterior_belief.mean, name))
        result[name] = {
            "local_mae_prior": float(np.mean(np.abs(prior[mask] - truth[mask]))),
            "local_mae_posterior": float(np.mean(np.abs(posterior[mask] - truth[mask]))),
        }
    return result


def _classify_route(prior: RetreatPlan, updated: RetreatPlan) -> str:
    if not prior.feasible:
        return "prior_no_feasible_retreat"
    if not updated.feasible:
        return "posterior_no_feasible_exit"
    if prior.selected_exit != updated.selected_exit:
        return "exit_switch"
    if prior.route.path_hash != updated.route.path_hash:
        return "same_exit_reroute"
    return "unchanged"


def _counterfactual(
    prior: RetreatPlan,
    updated: RetreatPlan,
    fields: DecisionFields,
    planner: GridAStarPlanner,
    scene: Scene,
) -> dict[str, Any]:
    old = planner.evaluate_path(
        prior.route.path,
        fields.cost.field,
        fields.cost.forbidden,
        scene.navigation,
        fields.hazard.field,
        fields.uncertainty.field,
    )
    new = planner.evaluate_path(
        updated.route.path,
        fields.cost.field,
        fields.cost.forbidden,
        scene.navigation,
        fields.hazard.field,
        fields.uncertainty.field,
    )
    delta_j: float | None = None
    if old.feasible and new.feasible:
        delta_j = old.total_cost - new.total_cost
        improvement_type = (
            "posterior_objective_improvement" if delta_j > 1e-12 else "no_objective_improvement"
        )
    elif not old.feasible and new.feasible:
        improvement_type = "feasibility_improvement"
    elif not new.feasible:
        improvement_type = (
            "posterior_replanning_failure" if old.feasible else "no_feasible_updated_retreat"
        )
    else:
        improvement_type = "no_feasible_updated_retreat"
    return {
        "definition": "delta_j = J_post(prior_route) - J_post(updated_route)",
        "field_identity": "same_effective_posterior_planning_field",
        "old_route_posterior_feasible": old.feasible,
        "updated_route_posterior_feasible": new.feasible,
        "old_route_under_posterior": _route_dict(old),
        "updated_route_under_posterior": _route_dict(new),
        "delta_j": delta_j,
        "improvement_type": improvement_type,
    }


def run_p8_snapshot_case(
    config_path: str | Path,
    fire_time_s: float,
    observation_id: str,
    *,
    observation_enabled: bool = True,
    assimilation_enabled: bool = True,
    oracle_planning: bool = False,
) -> P8CaseOutput:
    parameters = load_p8_parameters(config_path)
    if fire_time_s not in parameters.evaluation_fire_times_s:
        raise ValueError("P8 fire_time_s must be one of the configured evaluation times")
    scene = load_scene(parameters.scene_config)

    # This is deliberately completed before a truth simulator is constructed.
    frozen_points = freeze_observation_points(scene, parameters)
    point_set_hash = _frozen_hash(frozen_points)
    try:
        point = next(item for item in frozen_points if item.id == observation_id)
    except StopIteration as error:
        raise ValueError(f"Unknown P8 observation point {observation_id!r}") from error
    ugv_start, exits, static_reason = _resolve_static_geometry(scene, parameters)
    invalid_reason = point.invalid_reason or static_reason
    if invalid_reason is not None or ugv_start is None or point.cell is None:
        return P8CaseOutput(
            summary=_invalid_summary(
                parameters,
                point,
                fire_time_s,
                invalid_reason or "invalid_observation_case",
                point_set_hash,
                observation_enabled=observation_enabled,
                assimilation_enabled=assimilation_enabled,
                oracle_planning=oracle_planning,
            ),
            arrays={},
        )

    hazard_parameters = load_hazard_parameters(parameters.hazard_config)
    prior_model = PhysicsPrior(scene, load_fire_parameters(parameters.prior_config))
    _advance_to(prior_model, fire_time_s)
    prior_belief = initialize_belief(prior_model.current_state, parameters.initial_uncertainty)
    hazard_model = PhysicalHazardModel(hazard_parameters)
    uncertainty_model = UncertaintyAggregationModel(hazard_parameters)
    approach_parameters = parameters.planning_parameters(
        parameters.ugv_operability_threshold,
        "UGV approach operability",
    )
    retreat_parameters = parameters.planning_parameters(
        parameters.retreat_hazard_threshold,
        "retreat safety",
    )
    approach_fields = _decision_fields(
        prior_belief,
        hazard_model,
        uncertainty_model,
        PlanningCostModel(approach_parameters),
        scene,
    )
    approach_planner = GridAStarPlanner(approach_parameters)
    approach = approach_planner.plan(
        approach_fields.cost.field,
        approach_fields.cost.forbidden,
        ugv_start,
        point.cell,
        scene.navigation,
        approach_fields.hazard.field,
        approach_fields.uncertainty.field,
    )
    if not approach.feasible:
        return P8CaseOutput(
            summary=_invalid_summary(
                parameters,
                point,
                fire_time_s,
                "ugv_approach_unreachable",
                point_set_hash,
                observation_enabled=observation_enabled,
                assimilation_enabled=assimilation_enabled,
                oracle_planning=oracle_planning,
            ),
            arrays={},
        )

    retreat_cost_model = PlanningCostModel(retreat_parameters)
    prior_fields = _decision_fields(
        prior_belief,
        hazard_model,
        uncertainty_model,
        retreat_cost_model,
        scene,
    )
    retreat_planner = GridAStarPlanner(retreat_parameters)
    mission_planner = MissionPlanner(retreat_planner)
    prior_retreat = mission_planner.plan_retreat(
        prior_fields.cost.field,
        prior_fields.cost.forbidden,
        prior_fields.hazard.field,
        prior_fields.uncertainty.field,
        scene.navigation,
        point.cell,
        exits,
    )

    # Truth construction occurs only after geometry-only point freezing and approach validation.
    truth = GroundTruthSimulator(scene, load_fire_parameters(parameters.truth_config))
    _advance_to(truth, fire_time_s)
    truth_time_before_observation = truth.current_state.timestamp
    observation = None
    if observation_enabled:
        observation = UGVSensor(scene, parameters.sensor, parameters.random_seed).observe(
            truth.current_state,
            point.pose,
        )
    if observation is not None and assimilation_enabled:
        posterior_belief = SpatialAssimilator(scene, parameters.assimilation).assimilate(
            prior_belief,
            observation,
        )
    else:
        posterior_belief = prior_belief

    planner_field_source = "posterior_belief"
    planning_belief = posterior_belief
    if oracle_planning:
        zero = InitialUncertaintyParameters(
            temperature=0.0,
            co=0.0,
            smoke=0.0,
            visibility=0.0,
            meaning="P8 truth oracle uses zero epistemic uncertainty.",
        )
        planning_belief = initialize_belief(truth.current_state, zero)
        planner_field_source = "ground_truth_oracle"
    posterior_fields = _decision_fields(
        planning_belief,
        hazard_model,
        uncertainty_model,
        retreat_cost_model,
        scene,
    )
    updated_retreat = mission_planner.plan_retreat(
        posterior_fields.cost.field,
        posterior_fields.cost.forbidden,
        posterior_fields.hazard.field,
        posterior_fields.uncertainty.field,
        scene.navigation,
        point.cell,
        exits,
    )
    counterfactual = _counterfactual(
        prior_retreat,
        updated_retreat,
        posterior_fields,
        retreat_planner,
        scene,
    )
    local_mask = _local_mask(scene, point.cell, parameters.local_comparison_radius_m)
    observed = (
        {name: float(getattr(observation, name)) for name in MODALITIES}
        if observation is not None
        else None
    )
    prior_prediction = {
        name: float(getattr(prior_belief.mean, name)[point.cell]) for name in MODALITIES
    }
    posterior_prediction = {
        name: float(getattr(posterior_belief.mean, name)[point.cell]) for name in MODALITIES
    }
    uncertainty_before = {
        name: float(getattr(prior_belief.uncertainty, name)[point.cell]) for name in MODALITIES
    }
    uncertainty_after = {
        name: float(getattr(posterior_belief.uncertainty, name)[point.cell]) for name in MODALITIES
    }
    summary = {
        "schema_version": 1,
        "scenario_id": parameters.scenario_id,
        "case_status": "valid",
        "invalid_reason": None,
        "route_outcome": _classify_route(prior_retreat, updated_retreat),
        "fire_time_s": fire_time_s,
        "seconds_after_dispatch": fire_time_s - parameters.fire_age_at_dispatch_s,
        "snapshot_semantics": parameters.snapshot_semantics,
        "fire_frozen_during_approach_and_comparison": True,
        "simulator_timestamps": {
            "prior_snapshot_s": prior_model.current_state.timestamp,
            "truth_before_observation_s": truth_time_before_observation,
            "truth_after_replanning_s": truth.current_state.timestamp,
        },
        "truth_simulator_constructed": True,
        "research_horizon": _horizon_dict(parameters),
        "thresholds": {
            "ugv_operability_threshold": parameters.ugv_operability_threshold,
            "retreat_hazard_threshold": parameters.retreat_hazard_threshold,
        },
        "run_controls": {
            "observation_enabled": observation_enabled,
            "assimilation_enabled": assimilation_enabled,
            "oracle_planning": oracle_planning,
        },
        "provenance": {
            "scene_config_sha256": _file_hash(parameters.scene_config),
            "p8_config_sha256": _file_hash(parameters.config_path),
            "planner_field_source": planner_field_source,
            "truth_used_for_observation_point_selection": False,
            "manual_route_penalty": False,
        },
        "observation_point": _point_dict(point, point_set_hash),
        "ugv_approach": _route_dict(approach),
        "prior_retreat": _retreat_dict(prior_retreat),
        "updated_retreat": _retreat_dict(updated_retreat),
        "posterior_counterfactual": counterfactual,
        "observation": {
            "observed": observed,
            "prior_prediction": prior_prediction,
            "posterior_prediction": posterior_prediction,
            "residual": (
                {name: observed[name] - prior_prediction[name] for name in MODALITIES}
                if observed is not None
                else None
            ),
            "uncertainty_before": uncertainty_before,
            "uncertainty_after": uncertainty_after,
        },
        "field_metrics": _field_metrics(
            truth.current_state,
            prior_belief,
            posterior_belief,
            local_mask,
        ),
        "explanation": {
            "dominant_posterior_hazard_at_observation": str(
                posterior_fields.hazard.dominant_modality[point.cell]
            ),
            "route_change": _classify_route(prior_retreat, updated_retreat),
            "selected_exit_before": prior_retreat.selected_exit,
            "selected_exit_after": updated_retreat.selected_exit,
            "improvement_type": counterfactual["improvement_type"],
        },
    }
    arrays: dict[str, NDArray[Any]] = {
        "truth_temperature": np.asarray(truth.current_state.temperature),
        "truth_co": np.asarray(truth.current_state.co),
        "truth_smoke": np.asarray(truth.current_state.smoke),
        "truth_visibility": np.asarray(truth.current_state.visibility),
        "prior_hazard": np.asarray(prior_fields.hazard.field),
        "posterior_hazard": np.asarray(posterior_fields.hazard.field),
        "prior_uncertainty": np.asarray(prior_fields.uncertainty.field),
        "posterior_uncertainty": np.asarray(posterior_fields.uncertainty.field),
        "prior_planning_cost": np.asarray(prior_fields.cost.field),
        "posterior_planning_cost": np.asarray(posterior_fields.cost.field),
        "prior_forbidden": np.asarray(prior_fields.cost.forbidden),
        "posterior_forbidden": np.asarray(posterior_fields.cost.forbidden),
    }
    for name in MODALITIES:
        arrays[f"prior_mean_{name}"] = np.asarray(getattr(prior_belief.mean, name))
        arrays[f"posterior_mean_{name}"] = np.asarray(getattr(posterior_belief.mean, name))
    return P8CaseOutput(summary=summary, arrays=arrays)

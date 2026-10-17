"""P8 v2: synthetic observation, parameter posterior, forecast and ingress."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from indoor_fire_world.fire import (
    FireField,
    GroundTruthSimulator,
    load_fds_snapshot_manifest,
    load_fds_snapshots,
    load_fire_parameters,
)
from indoor_fire_world.hazard import PhysicalHazardModel, UncertaintyAggregationModel, load_hazard_parameters
from indoor_fire_world.p8_v2_config import P8V2Parameters, load_p8_v2_parameters
from indoor_fire_world.planning import (
    DirectedRoutePlanner,
    FirefighterIngressPlan,
    GridAStarPlanner,
    PlanningCostModel,
    PlanningCostParameters,
    RouteResult,
    load_directed_evacuation_graph,
)
from indoor_fire_world.scene import Pose2D, load_scene
from indoor_fire_world.scene.models import Scene
from indoor_fire_world.sensing import UGVSensor
from indoor_fire_world.world_model import (
    BeliefMean,
    UncertaintyField,
    build_candidates,
    posterior_weights,
    simulate_candidate_bank,
    weight_diagnostics,
    weighted_forecast,
    WeightedForecast,
)


MODALITIES = ("temperature", "co", "smoke", "visibility")


@dataclass(frozen=True)
class FrozenV2ObservationPoint:
    id: str
    offset_m: tuple[float, float]
    pose: Pose2D
    cell: tuple[int, int]


@dataclass(frozen=True)
class P8V2CaseOutput:
    summary: dict[str, Any]
    arrays: dict[str, NDArray[Any]]


@dataclass
class P8V2Context:
    parameters: P8V2Parameters
    scene: Scene
    graph: Any
    points: tuple[FrozenV2ObservationPoint, ...]
    point_hash: str
    candidates: tuple[Any, ...]
    candidate_bank: dict[str, dict[float, FireField]]
    reference_banks: dict[str, dict[float, FireField]]


def _advance(simulator: GroundTruthSimulator, target: float) -> None:
    while simulator.current_state.timestamp < target - 1e-12:
        simulator.step(min(simulator.parameters.time_step_s, target - simulator.current_state.timestamp))


def _simulate_reference(scene: Scene, path: Path, times: tuple[float, ...]) -> dict[float, FireField]:
    simulator = GroundTruthSimulator(scene, load_fire_parameters(path))
    snapshots: dict[float, FireField] = {}
    for target in sorted(set(times)):
        _advance(simulator, target)
        snapshots[target] = simulator.current_state
    return snapshots


def _freeze_points(scene: Scene, parameters: P8V2Parameters, graph: Any) -> tuple[tuple[FrozenV2ObservationPoint, ...], str]:
    microwave = scene.semantic.by_id("microwave")
    if microwave.position is None:
        raise ValueError("Microwave anchor requires a registered position")
    points: list[FrozenV2ObservationPoint] = []
    for spec in parameters.observation_points:
        pose = Pose2D(microwave.position[0] + spec.offset_m[0], microwave.position[1] + spec.offset_m[1], 0.0)
        cell = scene.navigation.metric_to_grid(pose.x, pose.y)
        if not scene.navigation.is_traversable(cell):
            raise ValueError(f"Observation point {spec.id!r} is occupied")
        node = graph.nodes.get(spec.id)
        if node is None or node.cell != cell:
            raise ValueError(f"Observation point {spec.id!r} is not on the registered corridor")
        points.append(FrozenV2ObservationPoint(spec.id, spec.offset_m, pose, cell))
    payload = [(item.id, item.offset_m, item.cell) for item in points]
    digest = hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()
    return tuple(points), digest


def build_p8_v2_context(config_path: str | Path) -> P8V2Context:
    parameters = load_p8_v2_parameters(config_path)
    scene = load_scene(parameters.scene_config)
    graph = load_directed_evacuation_graph(parameters.corridor_config, scene)
    # Causal guard: points and hashes are frozen before any reference simulator exists.
    points, point_hash = _freeze_points(scene, parameters, graph)
    all_times = tuple(sorted(set(parameters.observation_times_s + parameters.forecast_times_s)))
    base = load_fire_parameters(parameters.prior_config)
    candidates = build_candidates(base, parameters.ensemble.hrr_factors, parameters.ensemble.yield_factors, parameters.ensemble.ventilation_factors)
    candidate_bank = simulate_candidate_bank(scene, candidates, all_times)
    if parameters.reference_provider == "reduced_order_fixture":
        reference_banks = {
            scenario.id: _simulate_reference(scene, scenario.reference_config, all_times)
            for scenario in parameters.scenarios
        }
    else:
        reference_banks = {}
        expected_shape = (scene.occupancy.geometry.rows, scene.occupancy.geometry.cols)
        for scenario in parameters.scenarios:
            manifest = load_fds_snapshot_manifest(scenario.fds_manifest)
            if manifest.scenario_id != scenario.id or manifest.shape != expected_shape:
                raise ValueError("FDS manifest scenario or registered-grid shape mismatch")
            snapshots = load_fds_snapshots(manifest)
            if any(time_s not in snapshots for time_s in all_times):
                raise ValueError("FDS manifest is missing a required observation or forecast time")
            reference_banks[scenario.id] = snapshots
    return P8V2Context(parameters, scene, graph, points, point_hash, candidates, candidate_bank, reference_banks)


def _route_dict(route: RouteResult) -> dict[str, Any]:
    return {
        "feasible": route.feasible,
        "path": [{"row": row, "col": col} for row, col in route.path],
        "path_hash": route.path_hash,
        "geometric_length_m": route.geometric_length_m,
        "total_cost": route.total_cost if math.isfinite(route.total_cost) else None,
        "integrated_exposure": route.integrated_hazard,
        "integrated_uncertainty": route.integrated_uncertainty,
        "max_exposure": route.max_hazard,
        "rejection_reasons": list(route.rejection_reasons),
    }


def _ingress_dict(plan: FirefighterIngressPlan) -> dict[str, Any]:
    return {
        "feasible": plan.feasible,
        "selected_entrance": plan.selected_entrance,
        "target_id": plan.target_id,
        "route": _route_dict(plan.route),
        "candidates": {item.entrance_id: _route_dict(item.route) for item in plan.candidates},
        "rejection_reasons": list(plan.rejection_reasons),
    }


def _forecast_fields(context: P8V2Context, weights: NDArray[np.floating]) -> tuple[dict[float, Any], NDArray[np.float64], NDArray[np.float64]]:
    hazard_model = PhysicalHazardModel(load_hazard_parameters(context.parameters.hazard_config))
    uncertainty_model = UncertaintyAggregationModel(load_hazard_parameters(context.parameters.hazard_config))
    forecasts = {
        time_s: weighted_forecast(context.candidates, context.candidate_bank, time_s, weights)
        for time_s in context.parameters.forecast_times_s
    }
    exposure = np.maximum.reduce([hazard_model.compute(item.mean).field for item in forecasts.values()])
    uncertainty = np.maximum.reduce([uncertainty_model.compute(item.uncertainty).field for item in forecasts.values()])
    return forecasts, exposure, uncertainty


def _plan_ingress(context: P8V2Context, exposure: NDArray[np.floating], uncertainty: NDArray[np.floating], *, uncertainty_weight: float | None = None) -> FirefighterIngressPlan:
    parameters = PlanningCostParameters(
        context.parameters.base_cost,
        context.parameters.exposure_weight,
        context.parameters.uncertainty_weight if uncertainty_weight is None else uncertainty_weight,
        context.parameters.ingress_exposure_threshold,
        notes="P8 v2 simulation-only forecast exposure index",
    )
    cost = PlanningCostModel(parameters).compute(exposure, uncertainty, context.scene.navigation)
    return DirectedRoutePlanner(parameters, context.graph).plan_ingress(
        cost.field,
        cost.forbidden,
        exposure,
        uncertainty,
        context.scene.navigation,
        context.parameters.entrances,
        context.parameters.target_node,
    )


def _forecast_errors(reference: dict[float, FireField], prior: dict[float, Any], posterior: dict[float, Any]) -> tuple[dict[str, Any], float, float]:
    metrics: dict[str, Any] = {}
    prior_values: list[float] = []
    posterior_values: list[float] = []
    scales = {"temperature": 50.0, "co": 0.30, "smoke": 1.0, "visibility": 30.0}
    for time_s in sorted(prior):
        per_time: dict[str, Any] = {}
        for modality in MODALITIES:
            truth = np.asarray(getattr(reference[time_s], modality))
            before = np.asarray(getattr(prior[time_s].mean, modality))
            after = np.asarray(getattr(posterior[time_s].mean, modality))
            mae_prior = float(np.mean(np.abs(before - truth)))
            mae_posterior = float(np.mean(np.abs(after - truth)))
            per_time[modality] = {
                "mae_prior": mae_prior,
                "mae_posterior": mae_posterior,
                "rmse_prior": float(np.sqrt(np.mean((before - truth) ** 2))),
                "rmse_posterior": float(np.sqrt(np.mean((after - truth) ** 2))),
            }
            prior_values.append(mae_prior / scales[modality])
            posterior_values.append(mae_posterior / scales[modality])
        metrics[f"t{time_s:g}"] = per_time
    aggregate_prior = float(np.mean(prior_values))
    aggregate_posterior = float(np.mean(posterior_values))
    return metrics, aggregate_prior, aggregate_posterior


def _risk_level_grid(
    context: P8V2Context,
    exposure: NDArray[np.floating],
) -> tuple[NDArray[np.int8], dict[str, int]]:
    """Classify every registered cell as blocked (-1) or one configured level."""
    values = np.asarray(exposure, dtype=np.float64)
    traversable = np.asarray(context.scene.occupancy.cells) == 0
    result = np.full(values.shape, -1, dtype=np.int8)
    unassigned = traversable.copy()
    counts: dict[str, int] = {}
    for index, level in enumerate(context.parameters.exposure_levels):
        selected = unassigned if level.upper_bound is None else (
            unassigned & (values <= level.upper_bound + 1e-12)
        )
        result[selected] = index
        counts[level.id] = int(np.count_nonzero(selected))
        unassigned[selected] = False
    if np.any(unassigned):
        raise RuntimeError("Forecast exposure levels did not classify every traversable cell")
    counts["blocked"] = int(np.count_nonzero(~traversable))
    if sum(counts.values()) != values.size:
        raise RuntimeError("P8 v2 risk grid must classify every registered cell exactly once")
    return result, counts


def _local_gaussian_baseline(context: P8V2Context, prior_forecasts: dict[float, Any], observation: Any, point: FrozenV2ObservationPoint) -> dict[float, WeightedForecast]:
    geometry = context.scene.occupancy.geometry
    rows, cols = np.indices((geometry.rows, geometry.cols))
    distance = np.hypot(rows - point.cell[0], cols - point.cell[1]) * geometry.resolution_m
    kernel = np.exp(-(distance**2) / (2.0 * 0.35**2))
    kernel[distance > 0.50] = 0.0
    kernel[np.asarray(context.scene.occupancy.cells) != 0] = 0.0
    prior_at_observation = weighted_forecast(
        context.candidates,
        context.candidate_bank,
        observation.timestamp,
        np.full(len(context.candidates), 1.0 / len(context.candidates)),
    )
    residuals = {
        name: float(getattr(observation, name)) - float(getattr(prior_at_observation.mean, name)[point.cell])
        for name in MODALITIES
    }
    result: dict[float, WeightedForecast] = {}
    for time_s, forecast in prior_forecasts.items():
        values = {
            name: np.asarray(getattr(forecast.mean, name)) + 0.90 * kernel * residuals[name]
            for name in MODALITIES
        }
        values["temperature"] = np.maximum(values["temperature"], -273.15)
        for name in ("co", "smoke", "visibility"):
            values[name] = np.maximum(values[name], 0.0)
        result[time_s] = WeightedForecast(BeliefMean(**values), forecast.uncertainty)
    return result


def _counterfactual(context: P8V2Context, prior_plan: FirefighterIngressPlan, updated_plan: FirefighterIngressPlan, exposure: NDArray[np.floating], uncertainty: NDArray[np.floating]) -> dict[str, Any]:
    parameters = PlanningCostParameters(
        context.parameters.base_cost,
        context.parameters.exposure_weight,
        context.parameters.uncertainty_weight,
        context.parameters.ingress_exposure_threshold,
    )
    cost = PlanningCostModel(parameters).compute(exposure, uncertainty, context.scene.navigation)
    evaluator = GridAStarPlanner(parameters)
    old = evaluator.evaluate_path(prior_plan.route.path, cost.field, cost.forbidden, context.scene.navigation, exposure, uncertainty)
    new = evaluator.evaluate_path(updated_plan.route.path, cost.field, cost.forbidden, context.scene.navigation, exposure, uncertainty)
    delta_j = old.total_cost - new.total_cost if old.feasible and new.feasible else None
    if not old.feasible and new.feasible:
        improvement = "feasibility_improvement"
    elif old.feasible and not new.feasible:
        improvement = "posterior_replanning_failure"
    elif not old.feasible and not new.feasible:
        improvement = "no_feasible_firefighter_ingress"
    elif delta_j is not None and delta_j > 1e-12:
        improvement = "posterior_objective_improvement"
    else:
        improvement = "no_objective_improvement"
    return {
        "definition": "delta_j = J_post(prior_route) - J_post(updated_route)",
        "field_identity": "same_posterior_forecast_exposure_sequence",
        "old_route_posterior_feasible": old.feasible,
        "updated_route_posterior_feasible": new.feasible,
        "delta_j": delta_j,
        "improvement_type": improvement,
        "old_route_under_posterior": _route_dict(old),
        "updated_route_under_posterior": _route_dict(new),
    }


def run_p8_v2_case(context: P8V2Context, scenario_id: str, observation_time_s: float, observation_id: str) -> P8V2CaseOutput:
    parameters = context.parameters
    if scenario_id not in context.reference_banks:
        raise ValueError("Unknown P8 v2 synthetic scenario")
    if observation_time_s not in parameters.observation_times_s:
        raise ValueError("Unknown P8 v2 observation time")
    point = next((item for item in context.points if item.id == observation_id), None)
    if point is None:
        raise ValueError("Unknown P8 v2 observation point")

    shape = (context.scene.occupancy.geometry.rows, context.scene.occupancy.geometry.cols)
    zeros = np.zeros(shape, dtype=np.float64)
    free_cost = np.ones(shape, dtype=np.float64)
    forbidden = np.asarray(context.scene.occupancy.cells) != 0
    free_cost[forbidden] = np.inf
    approach = DirectedRoutePlanner(
        PlanningCostParameters(1.0, 0.0, 0.0, parameters.ugv_operability_threshold),
        context.graph,
    ).plan(free_cost, forbidden, "ugv_start", point.id, context.scene.navigation, zeros, zeros)
    if not approach.feasible:
        return P8V2CaseOutput({
            "schema_version": 2,
            "case_status": "invalid_observation_case",
            "invalid_reason": "ugv_approach_unreachable",
            "scenario_id": scenario_id,
            "observation_time_s": observation_time_s,
            "observation_id": observation_id,
        }, {})

    reference = context.reference_banks[scenario_id]
    seed_payload = f"{parameters.random_seed}:{scenario_id}:{observation_time_s}:{observation_id}"
    seed = int(hashlib.sha256(seed_payload.encode()).hexdigest()[:8], 16)
    observation = UGVSensor(context.scene, parameters.sensor, seed).observe(reference[observation_time_s], point.pose)
    prior_weights = np.full(len(context.candidates), 1.0 / len(context.candidates), dtype=np.float64)
    updated_weights = posterior_weights(context.candidates, context.candidate_bank, observation, point.cell, parameters.sensor)
    hrr_only_weights = posterior_weights(
        context.candidates,
        context.candidate_bank,
        observation,
        point.cell,
        parameters.sensor,
        modalities=("temperature",),
    )
    prior_forecasts, prior_exposure, prior_uncertainty = _forecast_fields(context, prior_weights)
    posterior_forecasts, posterior_exposure, posterior_uncertainty = _forecast_fields(context, updated_weights)
    hrr_only_forecasts, _, _ = _forecast_fields(context, hrr_only_weights)
    gaussian_forecasts = _local_gaussian_baseline(context, prior_forecasts, observation, point)
    hazard_model = PhysicalHazardModel(load_hazard_parameters(parameters.hazard_config))
    oracle_exposure = np.maximum.reduce([
        hazard_model.compute(BeliefMean.from_fire_field(reference[time_s])).field
        for time_s in parameters.forecast_times_s
    ])
    oracle_plan = _plan_ingress(context, oracle_exposure, np.zeros_like(oracle_exposure))
    prior_plan = _plan_ingress(context, prior_exposure, prior_uncertainty)
    updated_plan = _plan_ingress(context, posterior_exposure, posterior_uncertainty)
    no_uncertainty_plan = _plan_ingress(
        context,
        posterior_exposure,
        posterior_uncertainty,
        uncertainty_weight=0.0,
    )
    counterfactual = _counterfactual(context, prior_plan, updated_plan, posterior_exposure, posterior_uncertainty)
    metrics, prior_error, posterior_error = _forecast_errors(reference, prior_forecasts, posterior_forecasts)
    _, _, hrr_only_error = _forecast_errors(reference, prior_forecasts, hrr_only_forecasts)
    _, _, gaussian_error = _forecast_errors(reference, prior_forecasts, gaussian_forecasts)

    if not updated_plan.feasible:
        route_outcome = "posterior_no_feasible_ingress"
    elif not prior_plan.feasible:
        route_outcome = "feasibility_improvement"
    elif prior_plan.selected_entrance != updated_plan.selected_entrance:
        route_outcome = "entrance_switch"
    elif prior_plan.route.path_hash != updated_plan.route.path_hash:
        route_outcome = "same_entrance_reroute"
    else:
        route_outcome = "unchanged"

    summary = {
        "schema_version": 2,
        "experiment_id": parameters.experiment_id,
        "case_status": "valid",
        "scenario_id": scenario_id,
        "reference_semantics": "high_fidelity_synthetic_reference" if parameters.reference_provider == "fds_snapshot" else "reduced_order_synthetic_fixture_not_fds",
        "assumption_status": parameters.assumption_status,
        "observation_time_s": observation_time_s,
        "handoff_time_s": parameters.handoff_time_s,
        "forecast_times_s": list(parameters.forecast_times_s),
        "observation_point": {
            "id": point.id,
            "offset_m": list(point.offset_m),
            "position_m": [point.pose.x, point.pose.y],
            "cell": list(point.cell),
            "selection_provenance": "registered_geometry_only_frozen_before_reference",
            "frozen_point_set_hash": context.point_hash,
        },
        "observation": {name: getattr(observation, name) for name in ("temperature", "co", "smoke", "visibility")},
        "candidate_count": len(context.candidates),
        "prior_weight_diagnostics": weight_diagnostics(prior_weights),
        "posterior_weight_diagnostics": weight_diagnostics(updated_weights),
        "posterior_weights": {item.id: float(weight) for item, weight in zip(context.candidates, updated_weights)},
        "forecast_metrics": metrics,
        "aggregate_normalized_forecast_error": {
            "prior": prior_error,
            "posterior": posterior_error,
            "local_gaussian_baseline": gaussian_error,
            "hrr_only_ablation": hrr_only_error,
            "reduction": prior_error - posterior_error,
            "posterior_better": posterior_error < prior_error,
        },
        "forecast_exposure_semantics": "simulation planning index; not certified firefighter safety or physiological exposure",
        "risk_grid": {
            "shape": [context.scene.occupancy.geometry.rows, context.scene.occupancy.geometry.cols],
            "resolution_m": context.scene.occupancy.geometry.resolution_m,
            "cell_area_m2": context.scene.occupancy.geometry.resolution_m**2,
            "classification": "every registered cell is one configured exposure level or blocked",
            "blocked_index": -1,
            "levels": [
                {"index": index, "id": level.id, "upper_bound": level.upper_bound, "color": level.color}
                for index, level in enumerate(parameters.exposure_levels)
            ],
            "counts_by_forecast_time": {},
        },
        "corridor_semantics": "bidirectional physical corridor; arrow orientation retained only as evacuation metadata",
        "ugv_approach": _route_dict(approach),
        "prior_ingress": _ingress_dict(prior_plan),
        "updated_ingress": _ingress_dict(updated_plan),
        "oracle_ingress": _ingress_dict(oracle_plan),
        "planning_ablation_without_uncertainty_penalty": _ingress_dict(no_uncertainty_plan),
        "causal_controls": {
            "no_observation_uses_uniform_prior_weights": True,
            "observation_without_fusion_equals_prior": True,
            "oracle_reference_is_evaluation_only": True,
        },
        "route_outcome": route_outcome,
        "posterior_counterfactual": counterfactual,
        "explicit_exclusions": [
            "real_response_time", "physiological_tenability", "robot_dynamics",
            "suppression_feedback", "crowd_evacuation", "operational_deployment",
        ],
    }
    arrays: dict[str, NDArray[Any]] = {
        "prior_forecast_exposure": prior_exposure,
        "posterior_forecast_exposure": posterior_exposure,
        "posterior_forecast_uncertainty": posterior_uncertainty,
        "posterior_weights": updated_weights,
        "oracle_forecast_exposure": oracle_exposure,
    }
    for time_s in parameters.forecast_times_s:
        key = f"t{time_s:g}"
        arrays[f"truth_temperature_{key}"] = reference[time_s].temperature
        arrays[f"truth_smoke_{key}"] = reference[time_s].smoke
        arrays[f"posterior_temperature_{key}"] = posterior_forecasts[time_s].mean.temperature
        arrays[f"posterior_smoke_{key}"] = posterior_forecasts[time_s].mean.smoke
        exposure = hazard_model.compute(posterior_forecasts[time_s].mean).field
        risk_levels, level_counts = _risk_level_grid(context, exposure)
        arrays[f"posterior_exposure_{key}"] = exposure
        arrays[f"posterior_risk_level_{key}"] = risk_levels
        summary["risk_grid"]["counts_by_forecast_time"][key] = level_counts
    return P8V2CaseOutput(summary, arrays)

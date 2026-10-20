"""P8 v2: synthetic observation, parameter posterior, forecast and ingress."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
from indoor_fire_world.hazard import (
    HazardResult,
    PhysicalHazardModel,
    UncertaintyAggregationModel,
    load_hazard_parameters,
)
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
MODALITY_CODES = {"none": 0, "temperature": 1, "co": 2, "smoke": 3, "visibility": 4}


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
    source_cell: tuple[int, int]
    ambient_temperature_c: float


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
    all_times = tuple(
        sorted(
            set(
                parameters.observation_times_s
                + parameters.forecast_times_s
                + parameters.dense_prediction_times_s
                + parameters.online_evaluation_times_s
            )
        )
    )
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
    microwave = scene.semantic.by_id(parameters.fire_source_semantics.primary_source)
    if microwave.position is None:
        raise ValueError("Configured primary fire source requires a registered position")
    source_cell = scene.navigation.metric_to_grid(*microwave.position)
    if not scene.navigation.is_traversable(source_cell):
        raise ValueError("Configured primary fire source must occupy a traversable simulation cell")
    return P8V2Context(
        parameters,
        scene,
        graph,
        points,
        point_hash,
        candidates,
        candidate_bank,
        reference_banks,
        source_cell,
        base.ambient_temperature_c,
    )


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


def _label_components(mask: NDArray[np.bool_]) -> tuple[NDArray[np.int32], list[list[tuple[int, int]]]]:
    """Return deterministic row-major four-neighbour components."""
    active = np.asarray(mask, dtype=np.bool_)
    labels = np.full(active.shape, -1, dtype=np.int32)
    components: list[list[tuple[int, int]]] = []
    rows, cols = active.shape
    for row in range(rows):
        for col in range(cols):
            if not active[row, col] or labels[row, col] >= 0:
                continue
            component_id = len(components)
            labels[row, col] = component_id
            queue = [(row, col)]
            cells: list[tuple[int, int]] = []
            cursor = 0
            while cursor < len(queue):
                current_row, current_col = queue[cursor]
                cursor += 1
                cells.append((current_row, current_col))
                for next_row, next_col in (
                    (current_row - 1, current_col),
                    (current_row, current_col - 1),
                    (current_row, current_col + 1),
                    (current_row + 1, current_col),
                ):
                    if (
                        0 <= next_row < rows
                        and 0 <= next_col < cols
                        and active[next_row, next_col]
                        and labels[next_row, next_col] < 0
                    ):
                        labels[next_row, next_col] = component_id
                        queue.append((next_row, next_col))
            components.append(cells)
    return labels, components


def _dominant_component_modality(
    dominant: NDArray[np.str_],
    cells: list[tuple[int, int]],
) -> str:
    counts = {
        name: sum(str(dominant[row, col]) == name for row, col in cells)
        for name in MODALITIES
    }
    return min(MODALITIES, key=lambda name: (-counts[name], MODALITIES.index(name)))


def _component_record(
    context: P8V2Context,
    field: NDArray[np.float64],
    dominant: NDArray[np.str_],
    cells: list[tuple[int, int]],
    component_id: int,
    *,
    source_connected_at_threshold: bool,
    source_connected_in_meaningful_support: bool,
    source_connected_in_continuous_support: bool,
    source_connected_in_transport_domain: bool,
) -> dict[str, Any]:
    if source_connected_at_threshold:
        classification = "source_connected_exposure"
    elif source_connected_in_meaningful_support:
        classification = "threshold_disconnected_transport_region"
    elif source_connected_in_continuous_support:
        classification = "weakly_supported_transport_region"
    else:
        classification = "unexpected_disconnected_transport"
    rows = np.asarray([cell[0] for cell in cells], dtype=np.float64)
    cols = np.asarray([cell[1] for cell in cells], dtype=np.float64)
    centroid_row = float(np.mean(rows))
    centroid_col = float(np.mean(cols))
    geometry = context.scene.occupancy.geometry
    centroid_x = geometry.origin_x + (centroid_col + 0.5) * geometry.resolution_m
    centroid_y = geometry.origin_y + (centroid_row + 0.5) * geometry.resolution_m
    return {
        "component_id": component_id,
        "cell_count": len(cells),
        "area_m2": len(cells) * geometry.resolution_m**2,
        "centroid_cell": [centroid_row, centroid_col],
        "centroid_m": [centroid_x, centroid_y],
        "peak_exposure": max(float(field[row, col]) for row, col in cells),
        "dominant_modality": _dominant_component_modality(dominant, cells),
        "source_connected_at_threshold": source_connected_at_threshold,
        "source_connected_in_meaningful_support": source_connected_in_meaningful_support,
        "source_connected_in_continuous_support": source_connected_in_continuous_support,
        "source_connected_in_transport_domain": source_connected_in_transport_domain,
        "classification": classification,
    }


def _exposure_component_diagnostics(
    context: P8V2Context,
    field: NDArray[np.floating],
    dominant: NDArray[np.str_],
    transport_support: NDArray[np.floating],
) -> tuple[NDArray[np.int32], dict[str, Any]]:
    """Audit categorical islands against unthresholded transported fire signals."""
    values = np.asarray(field, dtype=np.float64)
    support_values = np.asarray(transport_support, dtype=np.float64)
    traversable = np.asarray(context.scene.occupancy.cells) == 0
    maximum = float(np.max(values[traversable]))
    support_epsilon = max(1e-12, 1e-6 * maximum)
    meaningful_support_threshold = max(1e-12, 1e-2 * maximum)
    support_labels, support_components = _label_components(
        traversable & (support_values > support_epsilon)
    )
    source_support_id = int(support_labels[context.source_cell])
    meaningful_labels, meaningful_components = _label_components(
        traversable & (support_values > meaningful_support_threshold)
    )
    source_meaningful_id = int(meaningful_labels[context.source_cell])
    transport_domain_labels, _ = _label_components(traversable)
    source_transport_domain_id = int(transport_domain_labels[context.source_cell])
    support_records = [
        _component_record(
            context,
            values,
            dominant,
            cells,
            component_id,
            source_connected_at_threshold=component_id == source_support_id,
            source_connected_in_meaningful_support=all(
                source_meaningful_id >= 0
                and int(meaningful_labels[row, col]) == source_meaningful_id
                for row, col in cells
            ),
            source_connected_in_continuous_support=component_id == source_support_id,
            source_connected_in_transport_domain=all(
                int(transport_domain_labels[row, col]) == source_transport_domain_id
                for row, col in cells
            ),
        )
        for component_id, cells in enumerate(support_components)
    ]

    threshold_records: dict[str, Any] = {}
    primary_component_grid = np.full(values.shape, -1, dtype=np.int32)
    primary_component_grid[~traversable] = -2
    transport_only_count = 0
    weakly_supported_count = 0
    unexpected_count = sum(
        record["classification"] == "unexpected_disconnected_transport"
        for record in support_records
    )
    finite_levels = [level for level in context.parameters.exposure_levels if level.upper_bound is not None]
    for threshold_index, level in enumerate(finite_levels):
        threshold = float(level.upper_bound)
        labels, components = _label_components(traversable & (values > threshold + 1e-12))
        if threshold_index == 0:
            primary_component_grid[labels >= 0] = labels[labels >= 0]
        source_component_id = int(labels[context.source_cell])
        records: list[dict[str, Any]] = []
        for component_id, cells in enumerate(components):
            support_ids = {int(support_labels[row, col]) for row, col in cells}
            connected_in_support = source_support_id >= 0 and support_ids == {source_support_id}
            meaningful_ids = {int(meaningful_labels[row, col]) for row, col in cells}
            connected_in_meaningful_support = (
                source_meaningful_id >= 0
                and meaningful_ids == {source_meaningful_id}
            )
            connected_in_domain = all(
                int(transport_domain_labels[row, col]) == source_transport_domain_id
                for row, col in cells
            )
            record = _component_record(
                context,
                values,
                dominant,
                cells,
                component_id,
                source_connected_at_threshold=component_id == source_component_id,
                source_connected_in_meaningful_support=connected_in_meaningful_support,
                source_connected_in_continuous_support=connected_in_support,
                source_connected_in_transport_domain=connected_in_domain,
            )
            transport_only_count += record["classification"] == "threshold_disconnected_transport_region"
            weakly_supported_count += record["classification"] == "weakly_supported_transport_region"
            records.append(record)
        threshold_records[level.id] = {
            "superlevel_definition": f"exposure > {threshold:g}",
            "component_count": len(records),
            "components": records,
        }
    return primary_component_grid, {
        "continuous_support_epsilon": support_epsilon,
        "meaningful_support_threshold": meaningful_support_threshold,
        "meaningful_support_definition": "transport support > 1% of snapshot peak; simulation diagnostic, not a certified physical threshold",
        "meaningful_support_component_count": len(meaningful_components),
        "continuous_support_basis": "max(normalized_temperature_anomaly, normalized_raw_smoke, normalized_raw_co); no hazard safe-threshold clipping",
        "causal_connectivity_basis": "four-neighbour traversable transport domain used by the no-flux solver",
        "continuous_support_component_count": len(support_records),
        "continuous_support_components": support_records,
        "thresholds": threshold_records,
        "transport_only_component_count_across_thresholds": int(transport_only_count),
        "weakly_supported_component_count_across_thresholds": int(weakly_supported_count),
        "unexpected_disconnected_component_count": int(unexpected_count),
    }


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
    observation = UGVSensor(context.scene, parameters.sensor, seed).observe(
        reference[observation_time_s], point.pose
    )
    # Snapshot-bank keys are configured evaluation times. Normalize harmless
    # floating-point integration drift in the simulator's internal clock so
    # posterior lookup remains exact and deterministic.
    observation = replace(observation, timestamp=observation_time_s)
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
        "fire_source_semantics": {
            "ignition_model": parameters.fire_source_semantics.ignition_model,
            "primary_source": parameters.fire_source_semantics.primary_source,
            "primary_source_cell": list(context.source_cell),
            "primary_source_count": 1,
            "secondary_ignition_mode": parameters.fire_source_semantics.secondary_ignition_mode,
            "secondary_ignition_count": 0,
            "fuel_chemistry_semantics": parameters.fire_source_semantics.fuel_chemistry_semantics,
            "interpretation": "Only the microwave cell is an ignition source; colored remote cells are forecast exposure, not burning cells.",
        },
        "spread_severity_profile": {
            "id": parameters.spread_profile_id,
            "semantics": parameters.spread_profile_semantics,
            "thresholds_or_palette_modified": False,
            "secondary_ignition_added": False,
        },
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
        "risk_map_provenance": {
            "field_source": "posterior_weighted_candidate_ensemble_forecast",
            "ground_truth_used_to_construct_map": False,
            "ground_truth_role": "synthetic observation generation and forecast-error evaluation only",
            "temporal_semantics": "independent posterior forecast level grid at each configured forecast time",
            "aggregation": "maximum normalized severity across temperature, co, smoke, and visibility",
        },
        "planning_field_semantics": {
            "aggregation": "cellwise_maximum_across_forecast_times",
            "definition": "planning_exposure_envelope = max(exposure_t300, exposure_t360, exposure_t420, exposure_t480)",
            "time_coupling": "none; route planning uses a worst-case forecast envelope",
        },
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
            "dominant_modality_codes": MODALITY_CODES,
            "component_grid_semantics": {
                "blocked": -2,
                "below_first_threshold": -1,
                "nonnegative": "deterministic row-major component ID for exposure above the first configured boundary",
            },
        },
        "synthetic_reference_risk_grid": {
            "role": "model-comparison reference only; not real W1 ground truth",
            "field_source": parameters.reference_provider,
            "threshold_identity": "same configured exposure thresholds as posterior risk grid",
            "counts_by_forecast_time": {},
            "posterior_comparison_by_forecast_time": {},
        },
        "risk_component_diagnostics": {},
        "transport_only_component_count": 0,
        "weakly_supported_component_count": 0,
        "unexpected_disconnected_component_count": 0,
        "exposure_change_from_previous_time": {},
        "risk_level_change_from_previous_time": {},
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
        "planning_exposure_envelope": posterior_exposure,
        "posterior_forecast_uncertainty": posterior_uncertainty,
        "posterior_weights": updated_weights,
        "oracle_forecast_exposure": oracle_exposure,
    }
    source_mask = np.zeros(shape, dtype=np.uint8)
    source_mask[context.source_cell] = 1
    arrays["primary_source_mask"] = source_mask
    arrays["secondary_ignition_mask"] = np.zeros(shape, dtype=np.uint8)
    previous_exposure: NDArray[np.float64] | None = None
    previous_risk_levels: NDArray[np.int8] | None = None
    for time_s in parameters.forecast_times_s:
        key = f"t{time_s:g}"
        arrays[f"truth_temperature_{key}"] = reference[time_s].temperature
        arrays[f"truth_smoke_{key}"] = reference[time_s].smoke
        arrays[f"posterior_temperature_{key}"] = posterior_forecasts[time_s].mean.temperature
        arrays[f"posterior_smoke_{key}"] = posterior_forecasts[time_s].mean.smoke
        hazard = hazard_model.compute(posterior_forecasts[time_s].mean)
        exposure = hazard.field
        reference_hazard = hazard_model.compute(reference[time_s])
        reference_exposure = reference_hazard.field
        forecast_mean = posterior_forecasts[time_s].mean
        transport_support = np.maximum.reduce([
            np.maximum(np.asarray(forecast_mean.temperature) - context.ambient_temperature_c, 0.0) / 50.0,
            np.asarray(forecast_mean.smoke) / 1.0,
            np.asarray(forecast_mean.co) / 0.30,
        ])
        risk_levels, level_counts = _risk_level_grid(context, exposure)
        reference_risk_levels, reference_level_counts = _risk_level_grid(
            context, reference_exposure
        )
        component_grid, component_diagnostics = _exposure_component_diagnostics(
            context,
            exposure,
            hazard.dominant_modality,
            transport_support,
        )
        dominant_codes = np.zeros(shape, dtype=np.int8)
        for modality, code in MODALITY_CODES.items():
            dominant_codes[np.asarray(hazard.dominant_modality) == modality] = code
        arrays[f"posterior_exposure_{key}"] = exposure
        arrays[f"synthetic_reference_exposure_{key}"] = reference_exposure
        arrays[f"posterior_transport_support_{key}"] = transport_support
        arrays[f"posterior_risk_level_{key}"] = risk_levels
        arrays[f"synthetic_reference_risk_level_{key}"] = reference_risk_levels
        arrays[f"posterior_dominant_modality_{key}"] = dominant_codes
        arrays[f"posterior_exposure_component_{key}"] = component_grid
        for modality in MODALITIES:
            arrays[f"posterior_{modality}_severity_{key}"] = np.asarray(getattr(hazard, modality))
        summary["risk_grid"]["counts_by_forecast_time"][key] = level_counts
        summary["synthetic_reference_risk_grid"]["counts_by_forecast_time"][key] = reference_level_counts
        comparable_reference = (risk_levels >= 0) & (reference_risk_levels >= 0)
        level_error = risk_levels.astype(np.int16) - reference_risk_levels.astype(np.int16)
        summary["synthetic_reference_risk_grid"]["posterior_comparison_by_forecast_time"][key] = {
            "exposure_mae": float(np.mean(np.abs(exposure - reference_exposure))),
            "exact_level_agreement_cell_count": int(
                np.count_nonzero(comparable_reference & (level_error == 0))
            ),
            "posterior_higher_level_cell_count": int(
                np.count_nonzero(comparable_reference & (level_error > 0))
            ),
            "posterior_lower_level_cell_count": int(
                np.count_nonzero(comparable_reference & (level_error < 0))
            ),
            "maximum_absolute_level_difference": int(
                np.max(np.abs(level_error[comparable_reference]))
            ),
        }
        summary["risk_component_diagnostics"][key] = component_diagnostics
        summary["transport_only_component_count"] += component_diagnostics[
            "transport_only_component_count_across_thresholds"
        ]
        summary["weakly_supported_component_count"] += component_diagnostics[
            "weakly_supported_component_count_across_thresholds"
        ]
        summary["unexpected_disconnected_component_count"] += component_diagnostics[
            "unexpected_disconnected_component_count"
        ]
        if previous_exposure is None:
            summary["exposure_change_from_previous_time"][key] = None
            summary["risk_level_change_from_previous_time"][key] = None
        else:
            assert previous_risk_levels is not None
            difference = np.abs(exposure - previous_exposure)
            summary["exposure_change_from_previous_time"][key] = {
                "mean_absolute_change": float(np.mean(difference)),
                "max_absolute_change": float(np.max(difference)),
                "changed_cell_count": int(np.count_nonzero(difference > 1e-12)),
            }
            comparable = (risk_levels >= 0) & (previous_risk_levels >= 0)
            level_delta = risk_levels.astype(np.int16) - previous_risk_levels.astype(np.int16)
            summary["risk_level_change_from_previous_time"][key] = {
                "changed_cell_count": int(np.count_nonzero(comparable & (level_delta != 0))),
                "increased_risk_cell_count": int(np.count_nonzero(comparable & (level_delta > 0))),
                "decreased_risk_cell_count": int(np.count_nonzero(comparable & (level_delta < 0))),
                "unchanged_cell_count": int(np.count_nonzero(comparable & (level_delta == 0))),
            }
        previous_exposure = np.array(exposure, copy=True)
        previous_risk_levels = np.array(risk_levels, copy=True)
    summary["model_validation_status"] = (
        "passed"
        if summary["unexpected_disconnected_component_count"] == 0
        else "failed_unexpected_disconnected_transport"
    )
    return P8V2CaseOutput(summary, arrays)


def attach_dense_prediction_evaluation(
    context: P8V2Context,
    cases: dict[str, P8V2CaseOutput],
    primary_run_id: str,
) -> tuple[dict[str, Any], ...]:
    """Batch-evaluate all independent cases every 10 s without causal backfill."""
    if primary_run_id not in cases:
        raise ValueError("Dense prediction primary run is missing")
    case_items = tuple(cases.items())
    candidate_ids = tuple(item.id for item in context.candidates)
    weights = np.asarray(
        [
            [output.summary["posterior_weights"][candidate_id] for candidate_id in candidate_ids]
            for _, output in case_items
        ],
        dtype=np.float64,
    )
    hazard_model = PhysicalHazardModel(
        load_hazard_parameters(context.parameters.hazard_config)
    )
    primary = cases[primary_run_id]
    primary_index = next(
        index for index, (run_id, _) in enumerate(case_items) if run_id == primary_run_id
    )
    shape = (
        context.scene.occupancy.geometry.rows,
        context.scene.occupancy.geometry.cols,
    )
    dense_rows: list[dict[str, Any]] = []
    primary_prior_exposure: list[NDArray[np.float64]] = []
    primary_reference_exposure: list[NDArray[np.float64]] = []
    primary_posterior_exposure: list[NDArray[np.float64]] = []
    primary_prior_levels: list[NDArray[np.int8]] = []
    primary_reference_levels: list[NDArray[np.int8]] = []
    primary_posterior_levels: list[NDArray[np.int8]] = []
    primary_availability: list[bool] = []

    for _, output in case_items:
        output.summary["dense_prediction"] = {
            "start_s": context.parameters.dense_prediction_times_s[0],
            "end_s": context.parameters.dense_prediction_times_s[-1],
            "interval_s": 10.0,
            "posterior_availability_rule": "time_s >= independent_case_observation_time_s",
            "no_retrodictive_backfill": True,
            "records": [],
        }

    uniform = np.full(len(context.candidates), 1.0 / len(context.candidates))
    for time_s in context.parameters.dense_prediction_times_s:
        modality_stacks = {
            modality: np.stack(
                [
                    np.asarray(
                        getattr(context.candidate_bank[candidate_id][time_s], modality),
                        dtype=np.float64,
                    )
                    for candidate_id in candidate_ids
                ]
            )
            for modality in MODALITIES
        }
        posterior_means = {
            modality: np.tensordot(weights, stack, axes=(1, 0))
            for modality, stack in modality_stacks.items()
        }
        prior_mean = BeliefMean(
            **{
                modality: np.tensordot(uniform, stack, axes=(0, 0))
                for modality, stack in modality_stacks.items()
            }
        )
        prior_hazard = hazard_model.compute(prior_mean)
        prior_levels, prior_counts = _risk_level_grid(context, prior_hazard.field)
        reference_cache: dict[str, tuple[HazardResult, NDArray[np.int8], dict[str, int]]] = {}

        for case_index, (run_id, output) in enumerate(case_items):
            summary = output.summary
            scenario_id = str(summary["scenario_id"])
            if scenario_id not in reference_cache:
                reference_hazard = hazard_model.compute(
                    context.reference_banks[scenario_id][time_s]
                )
                reference_levels, reference_counts = _risk_level_grid(
                    context, reference_hazard.field
                )
                reference_cache[scenario_id] = (
                    reference_hazard,
                    reference_levels,
                    reference_counts,
                )
            reference_hazard, reference_levels, reference_counts = reference_cache[
                scenario_id
            ]
            observation_time_s = float(summary["observation_time_s"])
            posterior_available = time_s >= observation_time_s - 1e-12
            record: dict[str, Any] = {
                "time_s": time_s,
                "phase": "pre_entry" if time_s <= context.parameters.handoff_time_s else "post_handoff",
                "posterior_available": posterior_available,
                "posterior_unavailable_reason": None
                if posterior_available
                else "observation_not_yet_available",
                "prior_reference_exposure_mae": float(
                    np.mean(np.abs(prior_hazard.field - reference_hazard.field))
                ),
                "prior_risk_level_counts": prior_counts,
                "synthetic_reference_risk_level_counts": reference_counts,
                "posterior_reference_exposure_mae": None,
                "posterior_reference_exact_level_agreement_cell_count": None,
                "posterior_reference_higher_level_cell_count": None,
                "posterior_reference_lower_level_cell_count": None,
                "posterior_risk_level_counts": None,
            }
            posterior_hazard: HazardResult | None = None
            posterior_levels: NDArray[np.int8] | None = None
            if posterior_available:
                posterior_mean = BeliefMean(
                    **{
                        modality: posterior_means[modality][case_index]
                        for modality in MODALITIES
                    }
                )
                posterior_hazard = hazard_model.compute(posterior_mean)
                posterior_levels, posterior_counts = _risk_level_grid(
                    context, posterior_hazard.field
                )
                comparable = (posterior_levels >= 0) & (reference_levels >= 0)
                delta = (
                    posterior_levels.astype(np.int16)
                    - reference_levels.astype(np.int16)
                )
                record.update(
                    {
                        "posterior_reference_exposure_mae": float(
                            np.mean(
                                np.abs(
                                    posterior_hazard.field
                                    - reference_hazard.field
                                )
                            )
                        ),
                        "posterior_reference_exact_level_agreement_cell_count": int(
                            np.count_nonzero(comparable & (delta == 0))
                        ),
                        "posterior_reference_higher_level_cell_count": int(
                            np.count_nonzero(comparable & (delta > 0))
                        ),
                        "posterior_reference_lower_level_cell_count": int(
                            np.count_nonzero(comparable & (delta < 0))
                        ),
                        "posterior_risk_level_counts": posterior_counts,
                    }
                )
            summary["dense_prediction"]["records"].append(record)
            dense_row = {
                "run_id": run_id,
                "scenario_id": scenario_id,
                "observation_time_s": observation_time_s,
                "observation_id": summary["observation_point"]["id"],
                **{
                    name: value
                    for name, value in record.items()
                    if not name.endswith("_risk_level_counts")
                },
            }
            for source_name, counts_name in (
                ("prior", "prior_risk_level_counts"),
                ("reference", "synthetic_reference_risk_level_counts"),
                ("posterior", "posterior_risk_level_counts"),
            ):
                counts = record[counts_name]
                for level in context.parameters.exposure_levels:
                    dense_row[f"{source_name}_{level.id}_cell_count"] = (
                        None if counts is None else counts[level.id]
                    )
            dense_rows.append(dense_row)

            if case_index == primary_index:
                primary_prior_exposure.append(np.asarray(prior_hazard.field))
                primary_reference_exposure.append(np.asarray(reference_hazard.field))
                primary_prior_levels.append(np.asarray(prior_levels))
                primary_reference_levels.append(np.asarray(reference_levels))
                primary_availability.append(posterior_available)
                if posterior_hazard is None or posterior_levels is None:
                    primary_posterior_exposure.append(
                        np.full(shape, np.nan, dtype=np.float64)
                    )
                    primary_posterior_levels.append(
                        np.full(shape, -3, dtype=np.int8)
                    )
                else:
                    primary_posterior_exposure.append(
                        np.asarray(posterior_hazard.field)
                    )
                    primary_posterior_levels.append(
                        np.asarray(posterior_levels)
                    )

    primary.arrays["dense_prediction_times_s"] = np.asarray(
        context.parameters.dense_prediction_times_s, dtype=np.float64
    )
    primary.arrays["dense_posterior_available"] = np.asarray(
        primary_availability, dtype=np.bool_
    )
    primary.arrays["dense_prior_exposure"] = np.stack(primary_prior_exposure)
    primary.arrays["dense_synthetic_reference_exposure"] = np.stack(
        primary_reference_exposure
    )
    primary.arrays["dense_posterior_exposure"] = np.stack(
        primary_posterior_exposure
    )
    primary.arrays["dense_prior_risk_level"] = np.stack(primary_prior_levels)
    primary.arrays["dense_synthetic_reference_risk_level"] = np.stack(
        primary_reference_levels
    )
    primary.arrays["dense_posterior_risk_level"] = np.stack(
        primary_posterior_levels
    )
    return tuple(dense_rows)


def attach_online_sequential_evaluation(
    context: P8V2Context,
    primary: P8V2CaseOutput,
) -> tuple[dict[str, Any], ...]:
    """Run the primary pre-entry track with a new Bayesian update every 5 s."""
    scenario_id = str(primary.summary["scenario_id"])
    point_id = str(primary.summary["observation_point"]["id"])
    point = next(item for item in context.points if item.id == point_id)
    reference = context.reference_banks[scenario_id]
    candidate_ids = tuple(item.id for item in context.candidates)
    weights = np.full(len(candidate_ids), 1.0 / len(candidate_ids), dtype=np.float64)
    uniform = np.array(weights, copy=True)
    update_times = set(context.parameters.online_update_times_s)
    seed_payload = f"{context.parameters.random_seed}:online:{scenario_id}:{point_id}"
    seed = int(hashlib.sha256(seed_payload.encode()).hexdigest()[:8], 16)
    sensor = UGVSensor(context.scene, context.parameters.sensor, seed)
    hazard_model = PhysicalHazardModel(
        load_hazard_parameters(context.parameters.hazard_config)
    )
    shape = (
        context.scene.occupancy.geometry.rows,
        context.scene.occupancy.geometry.cols,
    )
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    prior_fields: list[NDArray[np.float64]] = []
    reference_fields: list[NDArray[np.float64]] = []
    posterior_fields: list[NDArray[np.float64]] = []
    prior_levels_all: list[NDArray[np.int8]] = []
    reference_levels_all: list[NDArray[np.int8]] = []
    posterior_levels_all: list[NDArray[np.int8]] = []
    availability: list[bool] = []
    update_applied: list[bool] = []

    for time_s in context.parameters.online_evaluation_times_s:
        prior_forecast = weighted_forecast(
            context.candidates, context.candidate_bank, time_s, uniform
        )
        prior_hazard = hazard_model.compute(prior_forecast.mean)
        prior_levels, prior_counts = _risk_level_grid(context, prior_hazard.field)
        reference_hazard = hazard_model.compute(
            BeliefMean.from_fire_field(reference[time_s])
        )
        reference_levels, reference_counts = _risk_level_grid(
            context, reference_hazard.field
        )
        applied = time_s in update_times
        if applied:
            observation = sensor.observe(reference[time_s], point.pose)
            observation = replace(observation, timestamp=time_s)
            likelihood = posterior_weights(
                context.candidates,
                context.candidate_bank,
                observation,
                point.cell,
                context.parameters.sensor,
            )
            weights = weights * np.maximum(likelihood, np.finfo(np.float64).tiny)
            weights /= float(np.sum(weights))

        posterior_available = time_s > 0.0
        posterior_hazard: HazardResult | None = None
        posterior_levels: NDArray[np.int8] | None = None
        posterior_counts: dict[str, int] | None = None
        if posterior_available:
            posterior_forecast = weighted_forecast(
                context.candidates, context.candidate_bank, time_s, weights
            )
            posterior_hazard = hazard_model.compute(posterior_forecast.mean)
            posterior_levels, posterior_counts = _risk_level_grid(
                context, posterior_hazard.field
            )

        posterior_mae = None
        if posterior_hazard is not None:
            posterior_mae = float(
                np.mean(np.abs(posterior_hazard.field - reference_hazard.field))
            )
        record = {
            "time_s": time_s,
            "phase": "pre_entry"
            if time_s <= context.parameters.handoff_time_s
            else "post_handoff_forecast",
            "posterior_available": posterior_available,
            "posterior_unavailable_reason": None
            if posterior_available
            else "no_observation_at_initial_state",
            "sequential_update_applied": applied,
            "cumulative_observation_count": int(
                min(
                    len(context.parameters.online_update_times_s),
                    math.floor(time_s / 5.0 + 1e-12),
                )
            ),
            "posterior_weight_hash": None
            if not posterior_available
            else hashlib.sha256(weights.tobytes()).hexdigest(),
            "posterior_effective_sample_size": None
            if not posterior_available
            else float(1.0 / np.sum(weights**2)),
            "prior_reference_exposure_mae": float(
                np.mean(np.abs(prior_hazard.field - reference_hazard.field))
            ),
            "posterior_reference_exposure_mae": posterior_mae,
            "prior_risk_level_counts": prior_counts,
            "synthetic_reference_risk_level_counts": reference_counts,
            "posterior_risk_level_counts": posterior_counts,
        }
        records.append(record)
        flat = {
            name: value
            for name, value in record.items()
            if not name.endswith("_risk_level_counts")
        }
        flat.update({"scenario_id": scenario_id, "observation_id": point_id})
        for source_name, counts_name in (
            ("prior", "prior_risk_level_counts"),
            ("reference", "synthetic_reference_risk_level_counts"),
            ("posterior", "posterior_risk_level_counts"),
        ):
            counts = record[counts_name]
            for level in context.parameters.exposure_levels:
                flat[f"{source_name}_{level.id}_cell_count"] = (
                    None if counts is None else counts[level.id]
                )
        rows.append(flat)
        prior_fields.append(np.asarray(prior_hazard.field))
        reference_fields.append(np.asarray(reference_hazard.field))
        prior_levels_all.append(np.asarray(prior_levels))
        reference_levels_all.append(np.asarray(reference_levels))
        availability.append(posterior_available)
        update_applied.append(applied)
        if posterior_hazard is None or posterior_levels is None:
            posterior_fields.append(np.full(shape, np.nan, dtype=np.float64))
            posterior_levels_all.append(np.full(shape, -3, dtype=np.int8))
        else:
            posterior_fields.append(np.asarray(posterior_hazard.field))
            posterior_levels_all.append(np.asarray(posterior_levels))

    primary.summary["online_sequential_prediction"] = {
        "update_interval_s": 5.0,
        "first_update_s": 5.0,
        "last_pre_entry_update_s": context.parameters.handoff_time_s,
        "posterior_availability_rule": "unavailable only at t=0; first observation and posterior update occur at t=5 s",
        "post_handoff_semantics": "hold the t=300 s posterior weights fixed for no-intervention forecasting",
        "observation_point_id": point_id,
        "observation_point_provenance": "registered_geometry_only",
        "visualization_keyframes_s": list(
            context.parameters.online_visualization_times_s
        ),
        "records": records,
    }
    primary.arrays["online_evaluation_times_s"] = np.asarray(
        context.parameters.online_evaluation_times_s, dtype=np.float64
    )
    primary.arrays["online_posterior_available"] = np.asarray(
        availability, dtype=np.bool_
    )
    primary.arrays["online_update_applied"] = np.asarray(
        update_applied, dtype=np.bool_
    )
    primary.arrays["online_prior_exposure"] = np.stack(prior_fields)
    primary.arrays["online_synthetic_reference_exposure"] = np.stack(
        reference_fields
    )
    primary.arrays["online_posterior_exposure"] = np.stack(posterior_fields)
    primary.arrays["online_prior_risk_level"] = np.stack(prior_levels_all)
    primary.arrays["online_synthetic_reference_risk_level"] = np.stack(
        reference_levels_all
    )
    primary.arrays["online_posterior_risk_level"] = np.stack(
        posterior_levels_all
    )
    return tuple(rows)

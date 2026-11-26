"""P9 post-arrival action-conditioned World Model experiment."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from indoor_fire_world.fire import FireField, load_fire_parameters
from indoor_fire_world.hazard import (
    PhysicalHazardModel,
    UncertaintyAggregationModel,
    load_hazard_parameters,
)
from indoor_fire_world.intervention import (
    ActionConditionedFireSimulator,
    InterventionAction,
)
from indoor_fire_world.p9_config import P9Parameters, load_p9_parameters
from indoor_fire_world.planning import (
    DirectedRoutePlanner,
    PlanningCostModel,
    PlanningCostParameters,
)
from indoor_fire_world.sensing import UGVSensor
from indoor_fire_world.simulation.p8_v2 import (
    P8V2Context,
    _route_dict,
    build_p8_v2_context,
)
from indoor_fire_world.world_model import (
    BeliefMean,
    posterior_weights,
    regularized_sequential_weights,
    weighted_forecast,
)


@dataclass(frozen=True)
class P9Context:
    parameters: P9Parameters
    p8: P8V2Context
    candidate_banks: dict[str, dict[str, dict[float, FireField]]]
    reference_banks: dict[str, dict[str, dict[float, FireField]]]


@dataclass(frozen=True)
class P9CaseOutput:
    summary: dict[str, Any]
    arrays: dict[str, NDArray[Any]]


def _advance_branch(
    simulator: ActionConditionedFireSimulator,
    requested_times: tuple[float, ...],
) -> dict[float, FireField]:
    snapshots: dict[float, FireField] = {simulator.current_state.timestamp: simulator.current_state}
    for target in sorted(set(requested_times)):
        if target < simulator.current_state.timestamp - 1e-12:
            raise ValueError("Intervention branch cannot move backwards in time")
        while simulator.current_state.timestamp < target - 1e-12:
            simulator.step(
                min(
                    simulator.parameters.time_step_s,
                    target - simulator.current_state.timestamp,
                )
            )
        snapshots[target] = simulator.current_state
    return snapshots


def _branch_candidate_bank(
    p8: P8V2Context,
    action: InterventionAction,
    times: tuple[float, ...],
) -> dict[str, dict[float, FireField]]:
    if action.action_type == "no_intervention":
        return {
            item.id: {time_s: p8.candidate_bank[item.id][time_s] for time_s in times}
            for item in p8.candidates
        }
    bank: dict[str, dict[float, FireField]] = {}
    for candidate in p8.candidates:
        snapshot = p8.candidate_bank[candidate.id][300.0]
        simulator = ActionConditionedFireSimulator.resume_from_snapshot(
            p8.scene, candidate.parameters, action, snapshot
        )
        bank[candidate.id] = _advance_branch(simulator, times)
    return bank


def _branch_reference_bank(
    p8: P8V2Context,
    action: InterventionAction,
    times: tuple[float, ...],
) -> dict[str, dict[float, FireField]]:
    if action.action_type == "no_intervention":
        return {
            scenario.id: {
                time_s: p8.reference_banks[scenario.id][time_s] for time_s in times
            }
            for scenario in p8.parameters.scenarios
        }
    banks: dict[str, dict[float, FireField]] = {}
    for scenario in p8.parameters.scenarios:
        parameters = load_fire_parameters(scenario.reference_config)
        snapshot = p8.reference_banks[scenario.id][300.0]
        simulator = ActionConditionedFireSimulator.resume_from_snapshot(
            p8.scene, parameters, action, snapshot
        )
        banks[scenario.id] = _advance_branch(simulator, times)
    return banks


def build_p9_context(config_path: str | Path) -> P9Context:
    parameters = load_p9_parameters(config_path)
    p8 = build_p8_v2_context(parameters.p8_config)
    times = parameters.evaluation_times_s
    candidate_banks = {
        action.id: _branch_candidate_bank(p8, action, times)
        for action in parameters.actions
    }
    reference_banks = {
        action.id: _branch_reference_bank(p8, action, times)
        for action in parameters.actions
    }
    return P9Context(parameters, p8, candidate_banks, reference_banks)


def _pre_entry_weights(
    context: P9Context,
    scenario_id: str,
    observation_id: str,
    *,
    include_dynamic: bool = False,
) -> tuple[NDArray[np.float64], dict[str, Any], dict[str, NDArray[Any]]]:
    p8 = context.p8
    point = next(item for item in p8.points if item.id == observation_id)
    reference = p8.reference_banks[scenario_id]
    weights = np.full(len(p8.candidates), 1.0 / len(p8.candidates), dtype=np.float64)
    seed_payload = f"{p8.parameters.random_seed}:p9-pre:{scenario_id}:{observation_id}"
    seed = int(hashlib.sha256(seed_payload.encode()).hexdigest()[:8], 16)
    sensor = UGVSensor(p8.scene, p8.parameters.sensor, seed)
    max_regularization = 0.0
    dynamic_times: list[float] = []
    dynamic_exposure: list[NDArray[np.float64]] = []
    hazard_model = (
        PhysicalHazardModel(load_hazard_parameters(p8.parameters.hazard_config))
        if include_dynamic
        else None
    )
    if hazard_model is not None:
        initial = weighted_forecast(
            p8.candidates, p8.candidate_bank, 0.0, weights
        )
        dynamic_times.append(0.0)
        dynamic_exposure.append(hazard_model.compute(initial.mean).field)
    for time_s in p8.parameters.online_update_times_s:
        observation = replace(sensor.observe(reference[time_s], point.pose), timestamp=time_s)
        likelihood = posterior_weights(
            p8.candidates,
            p8.candidate_bank,
            observation,
            point.cell,
            p8.parameters.sensor,
        )
        weights, fraction = regularized_sequential_weights(
            weights,
            likelihood,
            likelihood_power=p8.parameters.online_likelihood_power,
            minimum_effective_sample_size=p8.parameters.online_minimum_effective_sample_size,
        )
        max_regularization = max(max_regularization, fraction)
        if hazard_model is not None:
            posterior = weighted_forecast(
                p8.candidates, p8.candidate_bank, time_s, weights
            )
            dynamic_times.append(time_s)
            dynamic_exposure.append(hazard_model.compute(posterior.mean).field)
    dynamic: dict[str, NDArray[Any]] = {}
    if include_dynamic:
        dynamic = {
            "times": np.asarray(dynamic_times, dtype=np.float64),
            "exposure": np.stack(dynamic_exposure),
        }
    return weights, {
        "effective_sample_size": float(1.0 / np.sum(weights**2)),
        "maximum_regularization_fraction": max_regularization,
        "observation_count": len(p8.parameters.online_update_times_s),
    }, dynamic


def _risk_counts(context: P9Context, field: NDArray[np.floating]) -> dict[str, int]:
    traversable = np.asarray(context.p8.scene.occupancy.cells) == 0
    remaining = traversable.copy()
    counts: dict[str, int] = {}
    for level in context.p8.parameters.exposure_levels:
        selected = remaining if level.upper_bound is None else (
            remaining & (np.asarray(field) <= level.upper_bound + 1e-12)
        )
        counts[level.id] = int(np.count_nonzero(selected))
        remaining[selected] = False
    counts["blocked"] = int(np.count_nonzero(~traversable))
    return counts


def _post_action_weights(
    context: P9Context,
    scenario_id: str,
    observation_id: str,
    action: InterventionAction,
    prior_weights: NDArray[np.floating],
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    p8 = context.p8
    point = next(item for item in p8.points if item.id == observation_id)
    time_s = context.parameters.post_action_observation_time_s
    seed_payload = f"{p8.parameters.random_seed}:p9-post:{scenario_id}:{observation_id}:{action.id}"
    seed = int(hashlib.sha256(seed_payload.encode()).hexdigest()[:8], 16)
    sensor = UGVSensor(p8.scene, p8.parameters.sensor, seed)
    truth = context.reference_banks[action.id][scenario_id][time_s]
    observation = replace(sensor.observe(truth, point.pose), timestamp=time_s)
    likelihood = posterior_weights(
        p8.candidates,
        context.candidate_banks[action.id],
        observation,
        point.cell,
        p8.parameters.sensor,
    )
    updated, fraction = regularized_sequential_weights(
        np.asarray(prior_weights),
        likelihood,
        likelihood_power=p8.parameters.online_likelihood_power,
        minimum_effective_sample_size=p8.parameters.online_minimum_effective_sample_size,
    )
    return updated, {
        "timestamp_s": time_s,
        "action_conditioned": True,
        "regularization_fraction": fraction,
        "effective_sample_size": float(1.0 / np.sum(updated**2)),
    }


def _safe_staging_comparison(
    context: P9Context,
    exposure: NDArray[np.floating],
    uncertainty: NDArray[np.floating],
    *,
    disable_exposure_gate: bool = False,
) -> dict[str, Any]:
    """Choose a safe source-near staging node independently for each entrance."""
    p8 = context.p8
    threshold = 1.0 if disable_exposure_gate else p8.parameters.ingress_exposure_threshold
    parameters = PlanningCostParameters(
        p8.parameters.base_cost,
        p8.parameters.exposure_weight,
        p8.parameters.uncertainty_weight,
        threshold,
        notes="P9 safety-first source-near staging comparison",
    )
    cost_exposure = (
        np.minimum(np.asarray(exposure), np.nextafter(1.0, 0.0))
        if disable_exposure_gate
        else np.asarray(exposure)
    )
    cost = PlanningCostModel(parameters).compute(
        cost_exposure, uncertainty, p8.scene.navigation
    )
    planner = DirectedRoutePlanner(parameters, p8.graph)
    source = p8.scene.semantic.by_id("microwave")
    if source.position is None:
        raise ValueError("Microwave source requires a registered position")
    for target in context.parameters.safe_staging_candidate_nodes:
        if target not in p8.graph.nodes:
            raise ValueError(f"Unknown P9 safe-staging node {target!r}")

    entrances: dict[str, Any] = {}
    selected_per_entrance: list[tuple[str, str, Any]] = []
    for entrance_id in sorted(p8.parameters.entrances):
        target_results: dict[str, Any] = {}
        feasible: list[tuple[str, Any]] = []
        for target_id in context.parameters.safe_staging_candidate_nodes:
            node = p8.graph.nodes[target_id]
            route = planner.plan(
                cost.field,
                cost.forbidden,
                entrance_id,
                target_id,
                p8.scene.navigation,
                np.asarray(exposure),
                np.asarray(uncertainty),
                allow_reverse=True,
            )
            source_distance = math.dist(node.position_m, source.position)
            route_data = _route_dict(route)
            # Candidate audit records retain reproducible metrics and path hashes;
            # only the selected route needs the full cell-by-cell path in JSON.
            route_data.pop("path", None)
            route_data.update(
                {
                    "target_node": target_id,
                    "target_source_distance_m": source_distance,
                    "target_exposure": float(np.asarray(exposure)[node.cell]),
                    "estimated_travel_time_s": (
                        route.geometric_length_m
                        / context.parameters.firefighter_travel_speed_m_s
                        if route.feasible
                        else None
                    ),
                }
            )
            target_results[target_id] = route_data
            if route.feasible:
                feasible.append((target_id, route))
        selected = min(
            feasible,
            key=lambda item: (
                item[1].max_hazard,
                item[1].integrated_hazard,
                item[1].total_cost,
                item[1].geometric_length_m,
                math.dist(p8.graph.nodes[item[0]].position_m, source.position),
                item[0],
            ),
            default=None,
        )
        if selected is None:
            entrances[entrance_id] = {
                "feasible": False,
                "selected_target_node": None,
                "route": _route_dict(
                    planner.plan(
                        cost.field,
                        cost.forbidden,
                        entrance_id,
                        context.parameters.safe_staging_candidate_nodes[0],
                        p8.scene.navigation,
                        np.asarray(exposure),
                        np.asarray(uncertainty),
                        allow_reverse=True,
                    )
                ),
                "estimated_travel_time_s": None,
                "candidate_targets": target_results,
            }
            continue
        target_id, route = selected
        selected_per_entrance.append((entrance_id, target_id, route))
        entrances[entrance_id] = {
            "feasible": True,
            "selected_target_node": target_id,
            "route": _route_dict(route),
            "estimated_travel_time_s": (
                route.geometric_length_m
                / context.parameters.firefighter_travel_speed_m_s
            ),
            "candidate_targets": target_results,
        }

    recommended = min(
        selected_per_entrance,
        key=lambda item: (
            item[2].max_hazard,
            item[2].integrated_hazard,
            item[2].total_cost,
            item[2].geometric_length_m,
            item[0],
            item[1],
        ),
        default=None,
    )
    shortest = min(
        selected_per_entrance,
        key=lambda item: (item[2].geometric_length_m, item[0], item[1]),
        default=None,
    )
    lowest_integrated = min(
        selected_per_entrance,
        key=lambda item: (
            item[2].integrated_hazard,
            item[2].max_hazard,
            item[0],
            item[1],
        ),
        default=None,
    )
    return {
        "feasible": recommended is not None,
        "exposure_gate_disabled": disable_exposure_gate,
        "configured_exposure_threshold": p8.parameters.ingress_exposure_threshold,
        "candidate_target_nodes": list(context.parameters.safe_staging_candidate_nodes),
        "target_semantics": (
            "configured source-near green-corridor staging nodes; the burning "
            "microwave cell is never a route goal"
        ),
        "selection_rule": (
            "minimum (route max exposure, integrated exposure, planning cost, "
            "length, source distance, deterministic IDs)"
        ),
        "selection_semantics": context.parameters.safe_staging_selection,
        "travel_speed_m_s": context.parameters.firefighter_travel_speed_m_s,
        "recommended_entrance": None if recommended is None else recommended[0],
        "recommended_target_node": None if recommended is None else recommended[1],
        "recommended_route": (
            None if recommended is None else _route_dict(recommended[2])
        ),
        "shortest_entrance": None if shortest is None else shortest[0],
        "fastest_entrance": None if shortest is None else shortest[0],
        "lowest_integrated_exposure_entrance": (
            None if lowest_integrated is None else lowest_integrated[0]
        ),
        "entrances": entrances,
    }


def _dynamic_action_exposure(
    context: P9Context,
    action: InterventionAction,
    weights_300: NDArray[np.floating],
    weights_420: NDArray[np.floating],
    hazard_model: PhysicalHazardModel,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Create a true 5 s action-conditioned posterior sequence after handoff."""
    p8 = context.p8
    times = tuple(float(value) for value in np.arange(300.0, 480.0 + 5.0, 5.0))
    bank = _branch_candidate_bank(p8, action, times)
    fields: list[NDArray[np.float64]] = []
    for time_s in times:
        weights = weights_420 if time_s >= 420.0 else weights_300
        forecast = weighted_forecast(p8.candidates, bank, time_s, weights)
        fields.append(hazard_model.compute(forecast.mean).field)
    return np.asarray(times, dtype=np.float64), np.stack(fields)


def _retreat_summary(
    context: P9Context,
    exposure: NDArray[np.floating],
    uncertainty: NDArray[np.floating],
    start_node: str,
) -> dict[str, Any]:
    p8 = context.p8
    parameters = PlanningCostParameters(
        p8.parameters.base_cost,
        p8.parameters.exposure_weight,
        p8.parameters.uncertainty_weight,
        p8.parameters.ingress_exposure_threshold,
        notes="P9 post-arrival action-conditioned planning risk",
    )
    cost = PlanningCostModel(parameters).compute(
        exposure, uncertainty, p8.scene.navigation
    )
    plan = DirectedRoutePlanner(parameters, p8.graph).plan_retreat(
        cost.field,
        cost.forbidden,
        exposure,
        uncertainty,
        p8.scene.navigation,
        start_node,
        p8.parameters.entrances,
    )
    return {
        "feasible": plan.feasible,
        "selected_exit": plan.selected_exit,
        "route": _route_dict(plan.route),
        "candidates": {
            item.exit_id: _route_dict(item.route) for item in plan.candidates
        },
    }


def run_p9_case(
    context: P9Context,
    scenario_id: str,
    observation_id: str,
    *,
    include_dynamic: bool = False,
) -> P9CaseOutput:
    p8 = context.p8
    if scenario_id not in {item.id for item in p8.parameters.scenarios}:
        raise ValueError("Unknown P9 synthetic scenario")
    if observation_id not in {item.id for item in p8.points}:
        raise ValueError("Unknown P9 observation point")
    weights_300, pre_diagnostics, pre_dynamic = _pre_entry_weights(
        context,
        scenario_id,
        observation_id,
        include_dynamic=include_dynamic,
    )
    hazard_model = PhysicalHazardModel(
        load_hazard_parameters(p8.parameters.hazard_config)
    )
    uncertainty_model = UncertaintyAggregationModel(
        load_hazard_parameters(p8.parameters.hazard_config)
    )
    traversable = np.asarray(p8.scene.occupancy.cells) == 0
    action_results: dict[str, Any] = {}
    arrays: dict[str, NDArray[Any]] = {
        "pre_entry_weights_t300": np.asarray(weights_300),
    }
    if include_dynamic:
        arrays["dynamic_pre_entry_times_s"] = pre_dynamic["times"]
        arrays["dynamic_pre_entry_exposure"] = pre_dynamic["exposure"]
    no_action = "no_intervention"
    no_action_forecast = weighted_forecast(
        p8.candidates,
        context.candidate_banks[no_action],
        300.0,
        weights_300,
    )
    exposure_300 = hazard_model.compute(no_action_forecast.mean).field
    uncertainty_300 = uncertainty_model.compute(no_action_forecast.uncertainty).field
    strict_access = _safe_staging_comparison(
        context, exposure_300, uncertainty_300
    )
    diagnostic_access = _safe_staging_comparison(
        context,
        exposure_300,
        uncertainty_300,
        disable_exposure_gate=True,
    )
    planning_target = (
        strict_access["recommended_target_node"]
        or diagnostic_access["recommended_target_node"]
        or context.parameters.intervention_staging_node
    )
    predicted_objectives: dict[str, float] = {}
    oracle_objectives: dict[str, float] = {}
    post_weights_by_action: dict[str, NDArray[np.float64]] = {}
    for action in context.parameters.actions:
        candidate_bank = context.candidate_banks[action.id]
        reference_bank = context.reference_banks[action.id][scenario_id]
        predicted_fields: dict[float, NDArray[np.float64]] = {}
        reference_fields: dict[float, NDArray[np.float64]] = {}
        uncertainties: dict[float, NDArray[np.float64]] = {}
        for time_s in context.parameters.evaluation_times_s:
            forecast = weighted_forecast(
                p8.candidates, candidate_bank, time_s, weights_300
            )
            predicted_fields[time_s] = hazard_model.compute(forecast.mean).field
            uncertainties[time_s] = uncertainty_model.compute(
                forecast.uncertainty
            ).field
            reference_fields[time_s] = hazard_model.compute(
                BeliefMean.from_fire_field(reference_bank[time_s])
            ).field
            key = f"{action.id}_t{time_s:g}"
            arrays[f"predicted_exposure_{key}"] = predicted_fields[time_s]
            arrays[f"reference_exposure_{key}"] = reference_fields[time_s]
        predicted_objectives[action.id] = float(
            np.mean(
                [
                    np.mean(predicted_fields[time_s][traversable])
                    for time_s in (360.0, 420.0, 480.0)
                ]
            )
        )
        oracle_objectives[action.id] = float(
            np.mean(
                [
                    np.mean(reference_fields[time_s][traversable])
                    for time_s in (360.0, 420.0, 480.0)
                ]
            )
        )
        post_weights, post_diagnostics = _post_action_weights(
            context, scenario_id, observation_id, action, weights_300
        )
        post_weights_by_action[action.id] = post_weights
        arrays[f"post_action_weights_{action.id}"] = post_weights
        forecast_480 = weighted_forecast(
            p8.candidates, candidate_bank, 480.0, post_weights
        )
        post_hazard_480 = hazard_model.compute(forecast_480.mean).field
        post_uncertainty_480 = uncertainty_model.compute(
            forecast_480.uncertainty
        ).field
        arrays[f"post_update_exposure_{action.id}_t480"] = post_hazard_480
        before_error = float(
            np.mean(np.abs(predicted_fields[480.0] - reference_fields[480.0]))
        )
        after_error = float(
            np.mean(np.abs(post_hazard_480 - reference_fields[480.0]))
        )
        action_results[action.id] = {
            "action_type": action.action_type,
            "activation_time_s": action.activation_time_s,
            "source_multiplier_t480": action.source_multiplier(480.0),
            "predicted_objective": predicted_objectives[action.id],
            "oracle_objective": oracle_objectives[action.id],
            "forecast_error_t480_before_post_action_update": before_error,
            "forecast_error_t480_after_post_action_update": after_error,
            "post_action_update": post_diagnostics,
            "risk_counts_t480": _risk_counts(context, post_hazard_480),
            "retreat_t480": _retreat_summary(
                context, post_hazard_480, post_uncertainty_480, planning_target
            ),
        }
    predicted_action = min(
        predicted_objectives, key=lambda key: (predicted_objectives[key], key)
    )
    oracle_action = min(
        oracle_objectives, key=lambda key: (oracle_objectives[key], key)
    )
    comparison_action = predicted_action
    if comparison_action == no_action:
        comparison_action = min(
            (
                action.id
                for action in context.parameters.actions
                if action.action_type != "no_intervention"
            ),
            key=lambda key: (predicted_objectives[key], key),
        )
    if include_dynamic:
        no_times, no_fields = _dynamic_action_exposure(
            context,
            next(item for item in context.parameters.actions if item.id == no_action),
            weights_300,
            post_weights_by_action[no_action],
            hazard_model,
        )
        intervention_times, intervention_fields = _dynamic_action_exposure(
            context,
            next(
                item
                for item in context.parameters.actions
                if item.id == comparison_action
            ),
            weights_300,
            post_weights_by_action[comparison_action],
            hazard_model,
        )
        if not np.array_equal(no_times, intervention_times):
            raise RuntimeError("P9 comparison timelines must align")
        pre_times = np.asarray(pre_dynamic["times"])
        pre_fields = np.asarray(pre_dynamic["exposure"])
        arrays["dynamic_evaluation_times_s"] = np.concatenate(
            (pre_times, no_times[1:])
        )
        arrays["dynamic_no_intervention_exposure"] = np.concatenate(
            (pre_fields, no_fields[1:]), axis=0
        )
        arrays["dynamic_intervention_exposure"] = np.concatenate(
            (pre_fields, intervention_fields[1:]), axis=0
        )
    executed_action = predicted_action if strict_access["feasible"] else no_action
    summary = {
        "schema_version": 1,
        "scenario_id": scenario_id,
        "observation_id": observation_id,
        "phase_contract": {
            "pre_entry_observation_s": [0.0, 300.0],
            "firefighter_intervention_s": [300.0, 480.0],
            "firefighter_availability_time_s": 300.0,
            "availability_semantics": "configurable simulation handoff; not measured W1 arrival time",
            "intervention_before_handoff_forbidden": True,
        },
        "pre_entry_posterior": pre_diagnostics,
        "intervention_access": {
            "staging_node": strict_access["recommended_target_node"],
            "staging_semantics": context.parameters.staging_semantics,
            "strict": strict_access,
            "diagnostic_geometry_only": diagnostic_access,
        },
        "dynamic_risk_comparison": {
            "available": include_dynamic,
            "interval_s": 5.0,
            "comparison_action": comparison_action,
            "pre_handoff_semantics": "identical sequential posterior through t=300 s",
            "post_handoff_semantics": (
                "action-conditioned forecasts with branch-matched t=420 observation updates"
            ),
        },
        "predicted_selected_action": predicted_action,
        "oracle_selected_action": oracle_action,
        "executed_action_under_strict_access": executed_action,
        "selection_regret": float(
            oracle_objectives[predicted_action] - oracle_objectives[oracle_action]
        ),
        "predicted_improvement_over_no_intervention": float(
            predicted_objectives[no_action] - predicted_objectives[predicted_action]
        ),
        "reference_improvement_over_no_intervention": float(
            oracle_objectives[no_action] - oracle_objectives[predicted_action]
        ),
        "action_results": action_results,
        "scope_exclusions": list(context.parameters.scope_exclusions),
    }
    return P9CaseOutput(summary, arrays)


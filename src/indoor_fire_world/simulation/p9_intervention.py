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
    _plan_ingress,
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
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    p8 = context.p8
    point = next(item for item in p8.points if item.id == observation_id)
    reference = p8.reference_banks[scenario_id]
    weights = np.full(len(p8.candidates), 1.0 / len(p8.candidates), dtype=np.float64)
    seed_payload = f"{p8.parameters.random_seed}:p9-pre:{scenario_id}:{observation_id}"
    seed = int(hashlib.sha256(seed_payload.encode()).hexdigest()[:8], 16)
    sensor = UGVSensor(p8.scene, p8.parameters.sensor, seed)
    max_regularization = 0.0
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
    return weights, {
        "effective_sample_size": float(1.0 / np.sum(weights**2)),
        "maximum_regularization_fraction": max_regularization,
        "observation_count": len(p8.parameters.online_update_times_s),
    }


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


def _retreat_summary(
    context: P9Context,
    exposure: NDArray[np.floating],
    uncertainty: NDArray[np.floating],
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
        context.parameters.intervention_staging_node,
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
) -> P9CaseOutput:
    p8 = context.p8
    if scenario_id not in {item.id for item in p8.parameters.scenarios}:
        raise ValueError("Unknown P9 synthetic scenario")
    if observation_id not in {item.id for item in p8.points}:
        raise ValueError("Unknown P9 observation point")
    weights_300, pre_diagnostics = _pre_entry_weights(
        context, scenario_id, observation_id
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
    predicted_objectives: dict[str, float] = {}
    oracle_objectives: dict[str, float] = {}
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
                context, post_hazard_480, post_uncertainty_480
            ),
        }
    predicted_action = min(
        predicted_objectives, key=lambda key: (predicted_objectives[key], key)
    )
    oracle_action = min(
        oracle_objectives, key=lambda key: (oracle_objectives[key], key)
    )
    no_action = "no_intervention"
    no_action_forecast = weighted_forecast(
        p8.candidates,
        context.candidate_banks[no_action],
        300.0,
        weights_300,
    )
    exposure_300 = hazard_model.compute(no_action_forecast.mean).field
    uncertainty_300 = uncertainty_model.compute(no_action_forecast.uncertainty).field
    strict_access = _plan_ingress(
        p8,
        exposure_300,
        uncertainty_300,
        target_node=context.parameters.intervention_staging_node,
    )
    diagnostic_access = _plan_ingress(
        p8,
        exposure_300,
        uncertainty_300,
        target_node=context.parameters.intervention_staging_node,
        disable_exposure_gate=True,
    )
    executed_action = predicted_action if strict_access.feasible else no_action
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
            "staging_node": context.parameters.intervention_staging_node,
            "staging_semantics": context.parameters.staging_semantics,
            "strict": {
                "feasible": strict_access.feasible,
                "selected_entrance": strict_access.selected_entrance,
                "route": _route_dict(strict_access.route),
            },
            "diagnostic_geometry_only": {
                "feasible": diagnostic_access.feasible,
                "selected_entrance": diagnostic_access.selected_entrance,
                "route": _route_dict(diagnostic_access.route),
            },
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


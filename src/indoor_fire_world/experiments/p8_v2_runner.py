"""Run and serialize the P8 v2 3 x 4 x 4 experiment matrix."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from indoor_fire_world.simulation.p8_v2 import (
    P8V2CaseOutput,
    attach_dense_prediction_evaluation,
    attach_online_sequential_evaluation,
    build_p8_v2_context,
    run_p8_v2_case,
)
from indoor_fire_world.visualization.dynamic import render_p8_v2_dynamic_risk_html
from indoor_fire_world.visualization.svg import (
    render_p8_v2_categorical_risk_svg,
    render_p8_v2_online_pre_entry_timeseries_svg,
    render_p8_v2_online_risk_map_svg,
    render_p8_v2_exposure_evolution_svg,
    render_p8_v2_forecast_svg,
    render_p8_v2_grid_diagnostic_svg,
    render_p8_v2_planning_envelope_svg,
    render_p8_v2_posterior_reference_comparison_svg,
    render_p8_v2_pre_entry_keyframes_svg,
    render_p8_v2_synthetic_reference_risk_svg,
    render_p8_v2_t300_three_entrance_ingress_svg,
)


@dataclass(frozen=True)
class P8V2SuiteOutput:
    summary: dict[str, Any]
    rows: tuple[dict[str, Any], ...]
    primary: P8V2CaseOutput
    dense_rows: tuple[dict[str, Any], ...]
    online_rows: tuple[dict[str, Any], ...]


def _row(run_id: str, output: P8V2CaseOutput) -> dict[str, Any]:
    summary = output.summary
    error = summary.get("aggregate_normalized_forecast_error", {})
    prior = summary.get("prior_ingress", {})
    updated = summary.get("updated_ingress", {})
    counterfactual = summary.get("posterior_counterfactual", {})
    return {
        "run_id": run_id,
        "scenario_id": summary["scenario_id"],
        "observation_time_s": summary["observation_time_s"],
        "observation_id": summary["observation_point"]["id"],
        "case_status": summary["case_status"],
        "prior_forecast_error": error.get("prior"),
        "posterior_forecast_error": error.get("posterior"),
        "local_gaussian_forecast_error": error.get("local_gaussian_baseline"),
        "hrr_only_forecast_error": error.get("hrr_only_ablation"),
        "forecast_error_reduction": error.get("reduction"),
        "posterior_forecast_better": error.get("posterior_better"),
        "prior_entrance": prior.get("selected_entrance"),
        "updated_entrance": updated.get("selected_entrance"),
        "prior_path_hash": prior.get("route", {}).get("path_hash"),
        "updated_path_hash": updated.get("route", {}).get("path_hash"),
        "route_outcome": summary.get("route_outcome"),
        "old_route_posterior_feasible": counterfactual.get("old_route_posterior_feasible"),
        "delta_j": counterfactual.get("delta_j"),
        "improvement_type": counterfactual.get("improvement_type"),
        "posterior_effective_sample_size": summary.get("posterior_weight_diagnostics", {}).get("effective_sample_size"),
        "reference_semantics": summary.get("reference_semantics"),
        "ignition_model": summary.get("fire_source_semantics", {}).get("ignition_model"),
        "secondary_ignition_mode": summary.get("fire_source_semantics", {}).get("secondary_ignition_mode"),
        "spread_severity_profile": summary.get("spread_severity_profile", {}).get("id"),
        "transport_only_component_count": summary.get("transport_only_component_count"),
        "weakly_supported_component_count": summary.get("weakly_supported_component_count"),
        "unexpected_disconnected_component_count": summary.get("unexpected_disconnected_component_count"),
        "model_validation_status": summary.get("model_validation_status"),
    }


def run_p8_v2_suite(config_path: str | Path) -> P8V2SuiteOutput:
    context = build_p8_v2_context(config_path)
    cases: dict[str, P8V2CaseOutput] = {}
    for scenario in context.parameters.scenarios:
        for time_s in context.parameters.observation_times_s:
            for point in context.parameters.observation_points:
                run_id = f"{scenario.id}-t{time_s:g}-{point.id}"
                cases[run_id] = run_p8_v2_case(context, scenario.id, time_s, point.id)
    primary_id = f"{context.parameters.primary_scenario_id}-t100-{context.parameters.primary_observation_id}"
    dense_rows = attach_dense_prediction_evaluation(context, cases, primary_id)
    primary = cases[primary_id]
    online_rows = attach_online_sequential_evaluation(context, primary)
    rows = tuple(_row(run_id, output) for run_id, output in cases.items())
    valid_rows = [row for row in rows if row["case_status"] == "valid"]
    improved_forecasts = sum(bool(row["posterior_forecast_better"]) for row in valid_rows)
    positive_delta = sum(row["delta_j"] is not None and float(row["delta_j"]) > 1e-12 for row in valid_rows)
    beats_gaussian = sum(
        row["posterior_forecast_error"] < row["local_gaussian_forecast_error"]
        for row in valid_rows
    )
    beats_hrr_only = sum(
        row["posterior_forecast_error"] < row["hrr_only_forecast_error"]
        for row in valid_rows
    )
    route_outcomes: dict[str, int] = {}
    for row in valid_rows:
        outcome = str(row["route_outcome"])
        route_outcomes[outcome] = route_outcomes.get(outcome, 0) + 1
    unexpected_components = sum(int(row["unexpected_disconnected_component_count"] or 0) for row in rows)
    if unexpected_components:
        raise RuntimeError(
            f"P8 v2 transport validation found {unexpected_components} continuous-support components "
            "that cannot be traced to the single microwave source"
        )
    summary = {
        "schema_version": 2,
        "experiment_id": context.parameters.experiment_id,
        "assumption_status": "simulation_assumed",
        "reference_provider": context.parameters.reference_provider,
        "reference_semantics": primary.summary["reference_semantics"],
        "fire_source_semantics": primary.summary["fire_source_semantics"],
        "spread_severity_profile": primary.summary["spread_severity_profile"],
        "planning_field_semantics": primary.summary["planning_field_semantics"],
        "model_validation_status": "passed",
        "matrix_shape": {"scenarios": 3, "observation_times": 4, "observation_points": 4, "normal_cases": len(cases)},
        "primary_run_id": primary_id,
        "success_metrics": {
            "posterior_forecast_better_cases": improved_forecasts,
            "valid_cases": len(valid_rows),
            "positive_delta_j_cases": positive_delta,
            "posterior_beats_local_gaussian_cases": beats_gaussian,
            "posterior_beats_hrr_only_cases": beats_hrr_only,
        },
        "route_outcome_counts": route_outcomes,
        "frozen_observation_point_set_hash": context.point_hash,
        "candidate_ids": [item.id for item in context.candidates],
        "dense_prediction": {
            "start_s": context.parameters.dense_prediction_times_s[0],
            "end_s": context.parameters.dense_prediction_times_s[-1],
            "interval_s": 10.0,
            "time_count": len(context.parameters.dense_prediction_times_s),
            "pre_entry_visualization_keyframes_s": list(
                context.parameters.pre_entry_visualization_times_s
            ),
            "post_handoff_visualization_keyframes_s": list(
                context.parameters.post_handoff_visualization_times_s
            ),
            "posterior_availability_rule": "time_s >= independent_case_observation_time_s",
            "no_retrodictive_backfill": True,
        },
        "online_sequential_update": {
            "first_update_s": context.parameters.online_update_times_s[0],
            "last_pre_entry_update_s": context.parameters.online_update_times_s[-1],
            "update_interval_s": 5.0,
            "evaluation_time_count": len(
                context.parameters.online_evaluation_times_s
            ),
            "visualization_keyframes_s": list(
                context.parameters.online_visualization_times_s
            ),
            "posterior_availability_rule": "unavailable only at t=0; recursively updated every 5 s from t=5 through t=300",
            "post_handoff_semantics": "t=300 posterior held fixed for forecasts through t=480",
        },
        "cases": {run_id: output.summary for run_id, output in cases.items()},
    }
    return P8V2SuiteOutput(summary, rows, primary, dense_rows, online_rows)


def write_p8_v2_artifacts(output: P8V2SuiteOutput, config_path: str | Path, target_directory: str | Path) -> dict[str, Path]:
    target = Path(target_directory)
    target.mkdir(parents=True, exist_ok=True)
    summary_path = target / "summary.json"
    summary_path.write_text(json.dumps(output.summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    matrix_path = target / "experiment_matrix.csv"
    with matrix_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output.rows[0]))
        writer.writeheader()
        writer.writerows(output.rows)
    dense_matrix_path = target / "dense_prediction_timeseries.csv"
    with dense_matrix_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output.dense_rows[0]))
        writer.writeheader()
        writer.writerows(output.dense_rows)
    online_matrix_path = target / "online_sequential_timeseries.csv"
    with online_matrix_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output.online_rows[0]))
        writer.writeheader()
        writer.writerows(output.online_rows)
    arrays_path = target / "arrays.npz"
    np.savez_compressed(arrays_path, **output.primary.arrays)
    figure_path = render_p8_v2_forecast_svg(config_path, output.primary.summary, output.primary.arrays, target / "forecast_exposure_and_ingress.svg")
    categorical_risk_path = render_p8_v2_categorical_risk_svg(
        config_path,
        output.primary.summary,
        output.primary.arrays,
        target / "forecast_risk_level_grid.svg",
    )
    evolution_path = render_p8_v2_exposure_evolution_svg(
        config_path,
        output.primary.summary,
        output.primary.arrays,
        target / "forecast_exposure_evolution.svg",
    )
    planning_path = render_p8_v2_planning_envelope_svg(
        config_path,
        output.primary.summary,
        output.primary.arrays,
        target / "planning_envelope_and_ingress.svg",
    )
    diagnostic_path = render_p8_v2_grid_diagnostic_svg(
        config_path,
        output.primary.summary,
        output.primary.arrays,
        target / "forecast_exposure_grid_diagnostic.svg",
    )
    reference_risk_path = render_p8_v2_synthetic_reference_risk_svg(
        config_path,
        output.primary.summary,
        output.primary.arrays,
        target / "synthetic_reference_risk_level_grid.svg",
    )
    comparison_path = render_p8_v2_posterior_reference_comparison_svg(
        config_path,
        output.primary.summary,
        output.primary.arrays,
        target / "posterior_vs_synthetic_reference_risk.svg",
    )
    pre_entry_keyframes_path = render_p8_v2_pre_entry_keyframes_svg(
        config_path,
        output.primary.summary,
        output.primary.arrays,
        target / "pre_entry_risk_keyframes.svg",
    )
    online_pre_entry_timeseries_path = render_p8_v2_online_pre_entry_timeseries_svg(
        config_path,
        output.primary.summary,
        output.primary.arrays,
        target / "online_pre_entry_risk_timeseries.svg",
    )
    online_risk_map_path = render_p8_v2_online_risk_map_svg(
        config_path,
        output.primary.summary,
        output.primary.arrays,
        target / "online_risk_map_0_480.svg",
    )
    t300_ingress_path = render_p8_v2_t300_three_entrance_ingress_svg(
        config_path,
        output.primary.summary,
        output.primary.arrays,
        target / "t300_three_entrance_ingress.svg",
    )
    online_risk_map_small_path = render_p8_v2_online_risk_map_svg(
        config_path,
        output.primary.summary,
        output.primary.arrays,
        target / "online_risk_map_0_480_small.svg",
        compact=True,
    )
    t300_ingress_small_path = render_p8_v2_t300_three_entrance_ingress_svg(
        config_path,
        output.primary.summary,
        output.primary.arrays,
        target / "t300_three_entrance_ingress_small.svg",
        compact=True,
    )
    dynamic_risk_path = render_p8_v2_dynamic_risk_html(
        config_path,
        output.primary.arrays,
        target / "online_risk_map_0_480_dynamic.html",
    )
    manifest_path = target / "fds_manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": 1,
        "status": "not_generated_in_current_environment",
        "reference_provider_used": output.summary["reference_provider"],
        "reference_semantics": output.summary["reference_semantics"],
        "required_before_fds_claim": ["official_fds_version", "fds_input_sha256", "converted_arrays_sha256"],
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "summary": summary_path.resolve(),
        "matrix": matrix_path.resolve(),
        "dense_prediction_timeseries": dense_matrix_path.resolve(),
        "online_sequential_timeseries": online_matrix_path.resolve(),
        "arrays": arrays_path.resolve(),
        "visualization": figure_path,
        "categorical_risk_map": categorical_risk_path,
        "exposure_evolution": evolution_path,
        "planning_envelope": planning_path,
        "grid_diagnostic": diagnostic_path,
        "synthetic_reference_risk_map": reference_risk_path,
        "posterior_reference_comparison": comparison_path,
        "pre_entry_risk_keyframes": pre_entry_keyframes_path,
        "online_pre_entry_risk_timeseries": online_pre_entry_timeseries_path,
        "online_risk_map_0_480": online_risk_map_path,
        "t300_three_entrance_ingress": t300_ingress_path,
        "online_risk_map_0_480_small": online_risk_map_small_path,
        "t300_three_entrance_ingress_small": t300_ingress_small_path,
        "online_risk_map_0_480_dynamic": dynamic_risk_path,
        "fds_manifest": manifest_path.resolve(),
    }

"""Run and serialize the frozen P8 matrix and primary control cases."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from indoor_fire_world.p8_config import load_p8_parameters
from indoor_fire_world.scene import load_scene
from indoor_fire_world.simulation.p8_snapshot import P8CaseOutput, run_p8_snapshot_case
from indoor_fire_world.visualization.svg import render_p8_registered_overlay_svg


@dataclass(frozen=True)
class P8SuiteOutput:
    summary: dict[str, Any]
    rows: tuple[dict[str, Any], ...]
    primary: P8CaseOutput
    normal_cases: tuple[tuple[str, P8CaseOutput], ...]


def _row(run_id: str, group: str, output: P8CaseOutput) -> dict[str, Any]:
    summary = output.summary
    counterfactual = summary.get("posterior_counterfactual", {})
    observation = summary.get("observation", {})
    prior = summary.get("prior_retreat", {})
    updated = summary.get("updated_retreat", {})
    constraints = summary.get("route_constraints", {})
    observation_risk = summary.get("risk_level_at_observation", {})
    return {
        "run_id": run_id,
        "experiment_group": group,
        "fire_time_s": summary["fire_time_s"],
        "seconds_after_dispatch": summary["seconds_after_dispatch"],
        "observation_id": summary["observation_point"]["id"],
        "case_status": summary["case_status"],
        "invalid_reason": summary.get("invalid_reason"),
        "route_outcome": summary["route_outcome"],
        "observation_enabled": summary["run_controls"]["observation_enabled"],
        "assimilation_enabled": summary["run_controls"]["assimilation_enabled"],
        "oracle_planning": summary["run_controls"]["oracle_planning"],
        "prior_exit": prior.get("selected_exit"),
        "updated_exit": updated.get("selected_exit"),
        "prior_path_hash": prior.get("route", {}).get("path_hash"),
        "updated_path_hash": updated.get("route", {}).get("path_hash"),
        "route_graph_id": constraints.get("graph_id"),
        "ugv_approach_directed_conformant": constraints.get(
            "ugv_approach_directed_conformant"
        ),
        "prior_retreat_directed_conformant": constraints.get(
            "prior_retreat_directed_conformant"
        ),
        "updated_retreat_directed_conformant": constraints.get(
            "updated_retreat_directed_conformant"
        ),
        "prior_risk_level_at_observation": observation_risk.get("prior"),
        "posterior_risk_level_at_observation": observation_risk.get("posterior"),
        "old_route_posterior_feasible": counterfactual.get("old_route_posterior_feasible"),
        "delta_j": counterfactual.get("delta_j"),
        "improvement_type": counterfactual.get("improvement_type"),
        "observation_recorded": observation.get("observed") is not None,
        "ugv_operability_threshold": summary["thresholds"]["ugv_operability_threshold"],
        "retreat_hazard_threshold": summary["thresholds"]["retreat_hazard_threshold"],
    }


def run_p8_suite(config_path: str | Path) -> P8SuiteOutput:
    parameters = load_p8_parameters(config_path)
    normal: list[tuple[str, P8CaseOutput]] = []
    for fire_time_s in parameters.evaluation_fire_times_s:
        for point in parameters.observation_points:
            run_id = f"normal-t{fire_time_s:g}-{point.id}"
            normal.append(
                (
                    run_id,
                    run_p8_snapshot_case(config_path, fire_time_s, point.id),
                )
            )
    primary_run_id = f"normal-t{parameters.primary_fire_time_s:g}-{parameters.primary_observation_id}"
    primary = next(output for run_id, output in normal if run_id == primary_run_id)
    controls = [
        (
            "no_observation",
            run_p8_snapshot_case(
                config_path,
                parameters.primary_fire_time_s,
                parameters.primary_observation_id,
                observation_enabled=False,
                assimilation_enabled=False,
            ),
        ),
        (
            "observation_without_fusion",
            run_p8_snapshot_case(
                config_path,
                parameters.primary_fire_time_s,
                parameters.primary_observation_id,
                observation_enabled=True,
                assimilation_enabled=False,
            ),
        ),
        (
            "truth_oracle",
            run_p8_snapshot_case(
                config_path,
                parameters.primary_fire_time_s,
                parameters.primary_observation_id,
                oracle_planning=True,
            ),
        ),
    ]
    rows = tuple(
        [_row(run_id, "normal_matrix", output) for run_id, output in normal]
        + [_row(name, "primary_control", output) for name, output in controls]
    )
    route_graph = primary.summary["directed_evacuation_route_graph"]

    def compact(case: dict[str, Any]) -> dict[str, Any]:
        # The graph is suite-level immutable provenance; avoid repeating its
        # full rasterized edge paths in every one of the 20 case records.
        result = dict(case)
        result.pop("directed_evacuation_route_graph", None)
        return result

    summary = {
        "schema_version": 1,
        "scenario_id": parameters.scenario_id,
        "matrix_shape": {
            "evaluation_times": len(parameters.evaluation_fire_times_s),
            "observation_points": len(parameters.observation_points),
            "normal_cases": len(normal),
        },
        "primary_run_id": primary_run_id,
        "research_horizon": primary.summary["research_horizon"],
        "snapshot_semantics": parameters.snapshot_semantics,
        "directed_evacuation_route_graph": route_graph,
        "normal_cases": {run_id: compact(output.summary) for run_id, output in normal},
        "primary_controls": {
            "normal_fusion": compact(primary.summary),
            **{name: compact(output.summary) for name, output in controls},
        },
    }
    return P8SuiteOutput(
        summary=summary,
        rows=rows,
        primary=primary,
        normal_cases=tuple(normal),
    )


def write_p8_artifacts(
    output: P8SuiteOutput,
    config_path: str | Path,
    artifact_directory: str | Path,
) -> dict[str, Path]:
    target = Path(artifact_directory)
    target.mkdir(parents=True, exist_ok=True)
    summary_path = target / "summary.json"
    summary_path.write_text(
        json.dumps(output.summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    csv_path = target / "experiment_matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output.rows[0]))
        writer.writeheader()
        writer.writerows(output.rows)
    arrays_path = target / "arrays.npz"
    np.savez_compressed(arrays_path, **output.primary.arrays)
    scene = load_scene(load_p8_parameters(config_path).scene_config)
    svg_path = render_p8_registered_overlay_svg(
        scene,
        output.primary.summary,
        output.primary.arrays,
        target / "registered_snapshot_closed_loop.svg",
    )
    prior_svg = render_p8_registered_overlay_svg(
        scene,
        output.primary.summary,
        output.primary.arrays,
        target / "prior_risk_map.svg",
        panels=("prior",),
    )
    posterior_svg = render_p8_registered_overlay_svg(
        scene,
        output.primary.summary,
        output.primary.arrays,
        target / "posterior_risk_map.svg",
        panels=("posterior",),
    )
    delta_svg = render_p8_registered_overlay_svg(
        scene,
        output.primary.summary,
        output.primary.arrays,
        target / "risk_delta_map.svg",
        panels=("delta",),
    )
    paths = {
        "summary": summary_path.resolve(),
        "matrix": csv_path.resolve(),
        "arrays": arrays_path.resolve(),
        "visualization": svg_path,
        "prior_risk_map": prior_svg,
        "posterior_risk_map": posterior_svg,
        "risk_delta_map": delta_svg,
    }
    parameters = load_p8_parameters(config_path)
    for _, case in output.normal_cases:
        case_summary = case.summary
        if (
            case_summary["case_status"] != "valid"
            or case_summary["observation_point"]["id"]
            != parameters.primary_observation_id
        ):
            continue
        time_label = f"{case_summary['fire_time_s']:g}"
        key = f"posterior_risk_map_t{time_label}"
        paths[key] = render_p8_registered_overlay_svg(
            scene,
            case_summary,
            case.arrays,
            target / f"posterior_risk_map_t{time_label}.svg",
            panels=("posterior",),
        )
    return paths

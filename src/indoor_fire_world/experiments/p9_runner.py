"""P9 experiment matrix and artifact writer."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from indoor_fire_world.simulation.p9_intervention import (
    P9CaseOutput,
    build_p9_context,
    run_p9_case,
)
from indoor_fire_world.visualization.p9_svg import render_p9_action_comparison_svg


@dataclass(frozen=True)
class P9SuiteOutput:
    summary: dict[str, Any]
    rows: tuple[dict[str, Any], ...]
    primary: P9CaseOutput


def run_p9_suite(config_path: str | Path) -> P9SuiteOutput:
    context = build_p9_context(config_path)
    p8_config_sha256 = hashlib.sha256(
        context.parameters.p8_config.read_bytes()
    ).hexdigest()
    cases: dict[str, P9CaseOutput] = {}
    rows: list[dict[str, Any]] = []
    for scenario in context.p8.parameters.scenarios:
        for point in context.p8.parameters.observation_points:
            case_id = f"{scenario.id}-{point.id}"
            output = run_p9_case(context, scenario.id, point.id)
            cases[case_id] = output
            for action in context.parameters.actions:
                result = output.summary["action_results"][action.id]
                rows.append(
                    {
                        "case_id": case_id,
                        "scenario_id": scenario.id,
                        "observation_id": point.id,
                        "action_id": action.id,
                        "action_type": action.action_type,
                        "activation_time_s": action.activation_time_s,
                        "predicted_objective": result["predicted_objective"],
                        "oracle_objective": result["oracle_objective"],
                        "forecast_error_before_post_update": result[
                            "forecast_error_t480_before_post_action_update"
                        ],
                        "forecast_error_after_post_update": result[
                            "forecast_error_t480_after_post_action_update"
                        ],
                        "retreat_feasible_t480": result["retreat_t480"]["feasible"],
                        "predicted_selected": action.id
                        == output.summary["predicted_selected_action"],
                        "oracle_selected": action.id
                        == output.summary["oracle_selected_action"],
                        "strict_access_feasible": output.summary[
                            "intervention_access"
                        ]["strict"]["feasible"],
                    }
                )
    primary_id = "enclosure_breakout-geometry_route_high"
    primary = cases[primary_id]
    summary = {
        "schema_version": 1,
        "experiment_id": context.parameters.experiment_id,
        "assumption_status": context.parameters.assumption_status,
        "phase_contract": primary.summary["phase_contract"],
        "matrix_shape": {
            "scenarios": 3,
            "observation_points": 4,
            "base_cases": len(cases),
            "action_branches": len(rows),
        },
        "p8_frozen_dependency": {
            "config": str(context.parameters.p8_config),
            "config_sha256": p8_config_sha256,
            "modified_by_p9": False,
        },
        "primary_case_id": primary_id,
        "predicted_action_counts": {
            action.id: sum(
                output.summary["predicted_selected_action"] == action.id
                for output in cases.values()
            )
            for action in context.parameters.actions
        },
        "strict_access_feasible_cases": sum(
            output.summary["intervention_access"]["strict"]["feasible"]
            for output in cases.values()
        ),
        "cases": {case_id: output.summary for case_id, output in cases.items()},
    }
    return P9SuiteOutput(summary, tuple(rows), primary)


def write_p9_artifacts(
    output: P9SuiteOutput,
    config_path: str | Path,
    target_directory: str | Path,
) -> dict[str, Path]:
    target = Path(target_directory)
    target.mkdir(parents=True, exist_ok=True)
    summary_path = target / "summary.json"
    summary_path.write_text(
        json.dumps(output.summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    matrix_path = target / "action_matrix.csv"
    with matrix_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output.rows[0]))
        writer.writeheader()
        writer.writerows(output.rows)
    arrays_path = target / "arrays.npz"
    np.savez_compressed(arrays_path, **output.primary.arrays)
    figure_path = render_p9_action_comparison_svg(
        config_path,
        output.primary.summary,
        output.primary.arrays,
        target / "action_conditioned_risk_comparison.svg",
    )
    return {
        "summary": summary_path.resolve(),
        "matrix": matrix_path.resolve(),
        "arrays": arrays_path.resolve(),
        "visualization": figure_path,
    }


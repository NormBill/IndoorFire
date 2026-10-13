"""P7 runner and metrics built on the reusable P6 pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from indoor_fire_world.simulation.p6_demo import MODALITIES, P6DemoOutput, run_p6_closed_loop_demo

from .config import ExperimentSpec, ExperimentSuite


@dataclass(frozen=True)
class ExperimentResult:
    config_snapshot: dict[str, Any]
    summary: dict[str, Any]
    route_metrics: dict[str, Any]
    field_metrics: dict[str, Any]
    path_hashes: dict[str, Any]
    row: dict[str, Any]
    artifact_directory: str


def _snapshot(suite: ExperimentSuite, spec: ExperimentSpec) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "suite_id": suite.suite_id,
        "base_config": suite.base_config.name,
        "experiment": asdict(spec),
    }


def _snapshot_hash(snapshot: dict[str, Any]) -> str:
    encoded = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def artifact_directory_name(suite: ExperimentSuite, spec: ExperimentSpec) -> str:
    snapshot = _snapshot(suite, spec)
    return f"{spec.scenario_id}-{_snapshot_hash(snapshot)[:12]}"


def _region_masks(output: P6DemoOutput) -> tuple[np.ndarray, np.ndarray]:
    free = np.asarray(output.arrays["free_mask"], dtype=np.bool_)
    observation = output.summary["ugv_observation"]["cell"]
    radius = output.summary["scenario_parameters"]["assimilation"]["max_radius_m"]
    resolution = output.summary["grid_resolution_m"]
    rows, cols = np.indices(free.shape)
    distance = np.hypot(rows - observation["row"], cols - observation["col"])
    local = free & (distance * resolution <= radius + 1e-12)
    return free, local


def _mae(values: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(np.abs(values[mask] - truth[mask])))


def _field_metrics(output: P6DemoOutput) -> dict[str, Any]:
    free, local = _region_masks(output)
    estimation: dict[str, Any] = {}
    for name in MODALITIES:
        truth = np.asarray(output.arrays[f"truth_{name}"])
        prior = np.asarray(output.arrays[f"prior_mean_{name}"])
        posterior = np.asarray(output.arrays[f"posterior_mean_{name}"])
        estimation[name] = {
            "global_mae_prior": _mae(prior, truth, free),
            "global_mae_posterior": _mae(posterior, truth, free),
            "local_mae_prior": _mae(prior, truth, local),
            "local_mae_posterior": _mae(posterior, truth, local),
        }
    prior_uncertainty = np.asarray(output.arrays["prior_uncertainty"])
    posterior_uncertainty = np.asarray(
        output.arrays["belief_posterior_uncertainty"]
    )
    uncertainty = {
        "mean_prior": float(np.mean(prior_uncertainty[free])),
        "mean_posterior": float(np.mean(posterior_uncertainty[free])),
        "mean_reduction": float(
            np.mean(prior_uncertainty[free])
            - np.mean(posterior_uncertainty[free])
        ),
        "local_prior": float(np.mean(prior_uncertainty[local])),
        "local_posterior": float(np.mean(posterior_uncertainty[local])),
        "local_reduction": float(
            np.mean(prior_uncertainty[local])
            - np.mean(posterior_uncertainty[local])
        ),
    }
    return {"estimation": estimation, "uncertainty": uncertainty}


def _mission_metrics(mission: dict[str, Any]) -> dict[str, Any]:
    entry = mission["entry_route"]
    retreat = mission["retreat_route"]
    feasible = bool(mission["feasible"])
    if not feasible:
        return {
            "mission_feasible": False,
            "path_hash": mission["mission_path_hash"],
            "selected_exit": mission["selected_exit"],
            "length_m": None,
            "total_cost": None,
            "integrated_hazard": None,
            "integrated_uncertainty": None,
            "max_hazard": None,
        }
    return {
        "mission_feasible": True,
        "path_hash": mission["mission_path_hash"],
        "selected_exit": mission["selected_exit"],
        "length_m": entry["geometric_length_m"] + retreat["geometric_length_m"],
        "total_cost": entry["total_cost"] + retreat["total_cost"],
        "integrated_hazard": (
            entry["integrated_hazard"] + retreat["integrated_hazard"]
        ),
        "integrated_uncertainty": (
            entry["integrated_uncertainty"] + retreat["integrated_uncertainty"]
        ),
        "max_hazard": max(entry["max_hazard"], retreat["max_hazard"]),
    }


def _route_metrics(output: P6DemoOutput) -> dict[str, Any]:
    summary = output.summary
    initial = _mission_metrics(summary["initial_route_a"])
    posterior = _mission_metrics(summary["posterior_route_b"])
    old = summary["counterfactual"]["route_a_under_posterior"]
    new = summary["counterfactual"]["route_b_under_posterior"]
    old_cost = old["aggregate"]["total_cost"] if old["aggregate"] else None
    new_cost = new["aggregate"]["total_cost"] if new["aggregate"] else None
    improvement = (
        old_cost - new_cost
        if old_cost is not None and new_cost is not None
        else None
    )
    return {
        "initial": initial,
        "posterior": posterior,
        "route_changed": summary["route_changed"],
        "old_route_posterior_cost": old_cost,
        "new_route_posterior_cost": new_cost,
        "cost_improvement": improvement,
        "change_reason": summary["change_reason"],
    }


def _result_row(
    spec: ExperimentSpec,
    output: P6DemoOutput,
    fields: dict[str, Any],
    routes: dict[str, Any],
) -> dict[str, Any]:
    planning = output.summary["scenario_parameters"]["planning_weights"]
    assimilation = output.summary["scenario_parameters"]["assimilation"]
    row: dict[str, Any] = {
        "scenario_id": spec.scenario_id,
        "seed": spec.seed,
        "ablation_type": spec.ablation_type,
        "mismatch_level": spec.mismatch_level,
        "sensor_noise": spec.sensor_noise,
        "observation_enabled": spec.observation_enabled,
        "assimilation_enabled": spec.assimilation_enabled,
        "oracle_planning": spec.oracle_planning,
        "assimilation_gain": assimilation["gain"],
        "kernel_sigma_m": assimilation["kernel_sigma_m"],
        "kernel_radius_m": assimilation["max_radius_m"],
        "uncertainty_reduction_fraction": assimilation[
            "uncertainty_reduction_fraction"
        ],
        "hazard_weight": planning["hazard_weight"],
        "uncertainty_weight": planning["uncertainty_weight"],
        "mission_feasible": routes["posterior"]["mission_feasible"],
        "route_changed": routes["route_changed"],
        "path_hash": routes["posterior"]["path_hash"],
        "length_m": routes["posterior"]["length_m"],
        "total_cost": routes["posterior"]["total_cost"],
        "integrated_hazard": routes["posterior"]["integrated_hazard"],
        "integrated_uncertainty": routes["posterior"]["integrated_uncertainty"],
        "max_hazard": routes["posterior"]["max_hazard"],
        "selected_exit": routes["posterior"]["selected_exit"],
        "old_route_posterior_cost": routes["old_route_posterior_cost"],
        "new_route_posterior_cost": routes["new_route_posterior_cost"],
        "cost_improvement": routes["cost_improvement"],
        "mean_uncertainty_prior": fields["uncertainty"]["mean_prior"],
        "mean_uncertainty_posterior": fields["uncertainty"]["mean_posterior"],
        "local_uncertainty_prior": fields["uncertainty"]["local_prior"],
        "local_uncertainty_posterior": fields["uncertainty"]["local_posterior"],
    }
    for name in MODALITIES:
        metrics = fields["estimation"][name]
        for metric_name, value in metrics.items():
            row[f"{name}_{metric_name}"] = value
    return row


def run_experiment(
    suite: ExperimentSuite,
    spec: ExperimentSpec,
    artifact_root: str | Path | None = None,
) -> ExperimentResult:
    overrides = spec.overrides
    output = run_p6_closed_loop_demo(
        suite.base_config,
        observation_enabled=spec.observation_enabled,
        assimilation_enabled=spec.assimilation_enabled,
        oracle_planning=spec.oracle_planning,
        truth_parameter_overrides=overrides.get("truth"),
        sensor_overrides=overrides.get("sensor"),
        assimilation_overrides=overrides.get("assimilation"),
        planning_overrides=overrides.get("planning"),
        random_seed=spec.seed,
    )
    snapshot = _snapshot(suite, spec)
    snapshot["effective_parameters"] = output.summary["scenario_parameters"]
    snapshot["planner_field_source"] = output.summary[
        "planning_field_provenance"
    ]["planner_field_source"]
    fields = _field_metrics(output)
    routes = _route_metrics(output)
    path_hashes = {
        "initial_mission": output.summary["initial_route_a"]["mission_path_hash"],
        "posterior_mission": output.summary["posterior_route_b"]["mission_path_hash"],
        "initial_entry": output.summary["initial_route_a"]["entry_route"]["path_hash"],
        "posterior_entry": output.summary["posterior_route_b"]["entry_route"]["path_hash"],
        "initial_retreat": output.summary["initial_route_a"]["retreat_route"]["path_hash"],
        "posterior_retreat": output.summary["posterior_route_b"]["retreat_route"]["path_hash"],
    }
    directory = artifact_directory_name(suite, spec)
    row = _result_row(spec, output, fields, routes)
    result = ExperimentResult(
        config_snapshot=snapshot,
        summary=output.summary,
        route_metrics=routes,
        field_metrics=fields,
        path_hashes=path_hashes,
        row=row,
        artifact_directory=directory,
    )
    if artifact_root is not None:
        target = Path(artifact_root) / "runs" / directory
        target.mkdir(parents=True, exist_ok=True)
        payloads = {
            "config_snapshot.json": snapshot,
            "summary.json": output.summary,
            "route_metrics.json": routes,
            "field_metrics.json": fields,
            "path_hashes.json": path_hashes,
        }
        for filename, payload in payloads.items():
            (target / filename).write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    return result


def aggregate_results(results: list[ExperimentResult]) -> dict[str, Any]:
    improvements = [
        result.route_metrics["cost_improvement"]
        for result in results
        if result.route_metrics["cost_improvement"] is not None
    ]
    def grouped(field: str, ablation_filter: str | None = None) -> dict[str, Any]:
        selected = [
            result
            for result in results
            if ablation_filter is None
            or result.config_snapshot["experiment"]["ablation_type"]
            == ablation_filter
        ]
        labels = sorted(
            {
                str(result.config_snapshot["experiment"][field])
                for result in selected
            }
        )
        output: dict[str, Any] = {}
        for label in labels:
            group = [
                result
                for result in selected
                if str(result.config_snapshot["experiment"][field]) == label
            ]
            group_improvements = [
                result.route_metrics["cost_improvement"]
                for result in group
                if result.route_metrics["cost_improvement"] is not None
            ]
            output[label] = {
                "run_count": len(group),
                "route_changed_count": sum(
                    bool(result.route_metrics["route_changed"]) for result in group
                ),
                "mean_cost_improvement": (
                    float(np.mean(group_improvements))
                    if group_improvements
                    else None
                ),
                "mean_global_mae_reduction": {
                    name: float(
                        np.mean(
                            [
                                result.field_metrics["estimation"][name][
                                    "global_mae_prior"
                                ]
                                - result.field_metrics["estimation"][name][
                                    "global_mae_posterior"
                                ]
                                for result in group
                            ]
                        )
                    )
                    for name in MODALITIES
                },
            }
        return output

    return {
        "run_count": len(results),
        "feasible_count": sum(
            bool(result.route_metrics["posterior"]["mission_feasible"])
            for result in results
        ),
        "route_changed_count": sum(
            bool(result.route_metrics["route_changed"]) for result in results
        ),
        "mean_cost_improvement": (
            float(np.mean(improvements)) if improvements else None
        ),
        "mean_global_mae_reduction": {
            name: float(
                np.mean(
                    [
                        result.field_metrics["estimation"][name]["global_mae_prior"]
                        - result.field_metrics["estimation"][name][
                            "global_mae_posterior"
                        ]
                        for result in results
                    ]
                )
            )
            for name in MODALITIES
        },
        "by_ablation_type": grouped("ablation_type"),
        "by_mismatch_level": grouped("mismatch_level", "model_mismatch_sweep"),
        "by_sensor_noise": grouped("sensor_noise", "sensor_noise_sweep"),
    }


def run_experiment_suite(
    suite: ExperimentSuite,
    artifact_root: str | Path,
) -> list[ExperimentResult]:
    root = Path(artifact_root)
    results = [run_experiment(suite, spec, root) for spec in suite.experiments]
    rows = [result.row for result in results]
    (root / "results.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "results.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (root / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "aggregate_stats.json").write_text(
        json.dumps(aggregate_results(results), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return results

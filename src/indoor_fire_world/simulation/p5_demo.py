"""P5 entry/retreat planning and weight-sensitivity scenarios."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import math
from pathlib import Path
from typing import Any

from indoor_fire_world.planning import (
    GridAStarPlanner,
    MissionPlan,
    MissionPlanner,
    PlanningCostModel,
    RouteResult,
    load_mission_parameters,
    load_planning_cost_parameters,
    resolve_mission_cells,
)
from indoor_fire_world.scene import load_scene

from .p4_demo import P4DemoOutput, run_p4_demo


@dataclass(frozen=True)
class P5DemoOutput:
    summary: dict[str, Any]


def _route_dict(route: RouteResult, scene) -> dict[str, Any]:
    return {
        "feasible": route.feasible,
        "path": [{"row": row, "col": col} for row, col in route.path],
        "path_metric_m": [
            {"x": x, "y": y}
            for x, y in (scene.navigation.grid_to_metric(*cell) for cell in route.path)
        ],
        "path_hash": route.path_hash,
        "geometric_length_m": route.geometric_length_m,
        "total_cost": route.total_cost if math.isfinite(route.total_cost) else None,
        "cost_decomposition": {
            "base": route.base_contribution,
            "hazard": route.hazard_contribution,
            "uncertainty": route.uncertainty_contribution,
        },
        "integrated_hazard": route.integrated_hazard,
        "integrated_uncertainty": route.integrated_uncertainty,
        "max_hazard": route.max_hazard,
        "rejection_reasons": list(route.rejection_reasons),
    }


def _mission_dict(mission: MissionPlan, scene) -> dict[str, Any]:
    return {
        "feasible": mission.feasible,
        "selected_exit": mission.selected_exit,
        "entry_route": _route_dict(mission.entry_route, scene),
        "retreat_route": _route_dict(mission.retreat_route, scene),
        "retreat_candidates": {
            candidate.exit_id: _route_dict(candidate.route, scene)
            for candidate in mission.retreat_candidates
        },
        "rejection_reasons": list(mission.rejection_reasons),
    }


def _p4_snapshot(
    scene_config: str | Path,
    truth_config: str | Path,
    prior_config: str | Path,
    p3_config: str | Path,
    hazard_config: str | Path,
    planning_config: str | Path,
) -> P4DemoOutput:
    return run_p4_demo(
        scene_config,
        truth_config,
        prior_config,
        p3_config,
        hazard_config,
        planning_config,
    )


def run_p5_demo(
    scene_config: str | Path,
    truth_config: str | Path,
    prior_config: str | Path,
    p3_config: str | Path,
    hazard_config: str | Path,
    planning_config: str | Path,
    mission_config: str | Path,
) -> P5DemoOutput:
    scene = load_scene(scene_config)
    p4 = _p4_snapshot(
        scene_config,
        truth_config,
        prior_config,
        p3_config,
        hazard_config,
        planning_config,
    )
    cost_parameters = load_planning_cost_parameters(planning_config)
    mission_parameters = load_mission_parameters(mission_config)
    entry, target, exits = resolve_mission_cells(scene, mission_parameters)
    mission = MissionPlanner(GridAStarPlanner(cost_parameters)).plan(
        p4.arrays["posterior_planning_cost"],
        p4.arrays["posterior_forbidden"],
        p4.arrays["posterior_hazard"],
        p4.arrays["posterior_uncertainty"],
        scene.navigation,
        entry,
        target,
        exits,
    )
    return P5DemoOutput(
        summary={
            "timestamp": p4.summary["timestamp"],
            "planning_parameters": asdict(cost_parameters),
            "entry_start_id": mission_parameters.entry_start.id,
            "inspection_target_id": mission_parameters.inspection_target.id,
            "mission": _mission_dict(mission, scene),
        }
    )


def run_p5_weight_sensitivity(
    scene_config: str | Path,
    truth_config: str | Path,
    prior_config: str | Path,
    p3_config: str | Path,
    hazard_config: str | Path,
    planning_config: str | Path,
    mission_config: str | Path,
) -> list[dict[str, Any]]:
    scene = load_scene(scene_config)
    p4 = _p4_snapshot(
        scene_config,
        truth_config,
        prior_config,
        p3_config,
        hazard_config,
        planning_config,
    )
    base_parameters = load_planning_cost_parameters(planning_config)
    mission_parameters = load_mission_parameters(mission_config)
    entry, target, exits = resolve_mission_cells(scene, mission_parameters)
    records: list[dict[str, Any]] = []
    for hazard_weight in mission_parameters.hazard_weights:
        for uncertainty_weight in mission_parameters.uncertainty_weights:
            parameters = replace(
                base_parameters,
                hazard_weight=hazard_weight,
                uncertainty_weight=uncertainty_weight,
            )
            cost = PlanningCostModel(parameters).compute(
                p4.arrays["posterior_hazard"],
                p4.arrays["posterior_uncertainty"],
                scene.navigation,
            )
            mission = MissionPlanner(GridAStarPlanner(parameters)).plan(
                cost.field,
                cost.forbidden,
                p4.arrays["posterior_hazard"],
                p4.arrays["posterior_uncertainty"],
                scene.navigation,
                entry,
                target,
                exits,
            )
            routes = (mission.entry_route, mission.retreat_route)
            if mission.feasible:
                combined_identity = hashlib.sha256(
                    f"{routes[0].path_hash}|{routes[1].path_hash}".encode("ascii")
                ).hexdigest()
                total_cost = sum(route.total_cost for route in routes)
                length = sum(route.geometric_length_m for route in routes)
                integrated_hazard = sum(route.integrated_hazard for route in routes)
                integrated_uncertainty = sum(
                    route.integrated_uncertainty for route in routes
                )
                max_hazard = max(route.max_hazard for route in routes)
            else:
                combined_identity = None
                total_cost = None
                length = None
                integrated_hazard = None
                integrated_uncertainty = None
                max_hazard = None
            records.append(
                {
                    "hazard_weight": hazard_weight,
                    "uncertainty_weight": uncertainty_weight,
                    "mission_feasible": mission.feasible,
                    "path_identity": combined_identity,
                    "entry_path_hash": mission.entry_route.path_hash,
                    "retreat_path_hash": mission.retreat_route.path_hash,
                    "selected_exit": mission.selected_exit,
                    "total_cost": total_cost,
                    "geometric_length_m": length,
                    "integrated_hazard": integrated_hazard,
                    "integrated_uncertainty": integrated_uncertainty,
                    "max_hazard": max_hazard,
                    "rejection_reasons": list(mission.rejection_reasons),
                }
            )
    return records

"""Entry and lowest-cost feasible-exit retreat planning."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from indoor_fire_world.scene.models import NavigationLayer

from .astar import GridAStarPlanner
from .result import ExitRouteEvaluation, GridCell, MissionPlan, RouteResult


class MissionPlanner:
    def __init__(self, route_planner: GridAStarPlanner) -> None:
        self.route_planner = route_planner

    def plan(
        self,
        planning_cost_field: NDArray[np.floating],
        forbidden_mask: NDArray[np.bool_],
        hazard_field: NDArray[np.floating],
        uncertainty_field: NDArray[np.floating],
        navigation: NavigationLayer,
        entry_start: GridCell,
        inspection_target: GridCell,
        exits: Mapping[str, GridCell],
    ) -> MissionPlan:
        entry = self.route_planner.plan(
            planning_cost_field,
            forbidden_mask,
            entry_start,
            inspection_target,
            navigation,
            hazard_field,
            uncertainty_field,
        )
        candidates = tuple(
            ExitRouteEvaluation(
                exit_id=exit_id,
                route=self.route_planner.plan(
                    planning_cost_field,
                    forbidden_mask,
                    inspection_target,
                    exit_cell,
                    navigation,
                    hazard_field,
                    uncertainty_field,
                ),
            )
            for exit_id, exit_cell in sorted(exits.items())
        )
        feasible_candidates = [candidate for candidate in candidates if candidate.route.feasible]
        if feasible_candidates:
            selected = min(
                feasible_candidates,
                key=lambda candidate: (candidate.route.total_cost, candidate.exit_id),
            )
            retreat = selected.route
            selected_exit: str | None = selected.exit_id
        else:
            retreat = RouteResult.rejected("no_feasible_exit")
            selected_exit = None

        reasons: list[str] = []
        if not entry.feasible:
            reasons.append("entry_infeasible")
            reasons.extend(f"entry:{reason}" for reason in entry.rejection_reasons)
        if not retreat.feasible:
            reasons.append("retreat_infeasible")
            reasons.extend(f"retreat:{reason}" for reason in retreat.rejection_reasons)
        feasible = entry.feasible and retreat.feasible
        return MissionPlan(
            entry_route=entry,
            retreat_route=retreat,
            selected_exit=selected_exit,
            retreat_candidates=candidates,
            feasible=feasible,
            rejection_reasons=tuple(reasons),
        )

"""Entry and lowest-cost feasible-exit retreat planning."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from indoor_fire_world.scene.models import NavigationLayer

from .astar import GridAStarPlanner
from .result import ExitRouteEvaluation, GridCell, MissionPlan, RetreatPlan, RouteResult


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
        retreat_plan = self.plan_retreat(
            planning_cost_field,
            forbidden_mask,
            hazard_field,
            uncertainty_field,
            navigation,
            inspection_target,
            exits,
        )

        reasons: list[str] = []
        if not entry.feasible:
            reasons.append("entry_infeasible")
            reasons.extend(f"entry:{reason}" for reason in entry.rejection_reasons)
        if not retreat_plan.feasible:
            reasons.extend(retreat_plan.rejection_reasons)
        feasible = entry.feasible and retreat_plan.feasible
        return MissionPlan(
            entry_route=entry,
            retreat_route=retreat_plan.route,
            selected_exit=retreat_plan.selected_exit,
            retreat_candidates=retreat_plan.candidates,
            feasible=feasible,
            rejection_reasons=tuple(reasons),
        )

    def plan_retreat(
        self,
        planning_cost_field: NDArray[np.floating],
        forbidden_mask: NDArray[np.bool_],
        hazard_field: NDArray[np.floating],
        uncertainty_field: NDArray[np.floating],
        navigation: NavigationLayer,
        start: GridCell,
        exits: Mapping[str, GridCell],
    ) -> RetreatPlan:
        """Select the lowest-cost feasible exit from the robot's current cell."""
        candidates = tuple(
            ExitRouteEvaluation(
                exit_id=exit_id,
                route=self.route_planner.plan(
                    planning_cost_field,
                    forbidden_mask,
                    start,
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
        feasible = retreat.feasible
        reasons = () if feasible else (
            "retreat_infeasible",
            *(f"retreat:{reason}" for reason in retreat.rejection_reasons),
        )
        return RetreatPlan(
            route=retreat,
            selected_exit=selected_exit,
            candidates=candidates,
            feasible=feasible,
            rejection_reasons=reasons,
        )

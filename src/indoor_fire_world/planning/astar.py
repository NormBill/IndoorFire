"""Deterministic four-connected grid A* over a precomputed planning-cost field."""

from __future__ import annotations

import heapq
import math

import numpy as np
from numpy.typing import NDArray

from indoor_fire_world.scene.models import NavigationLayer

from .cost_field import PlanningCostParameters
from .result import GridCell, RouteResult


class GridAStarPlanner:
    def __init__(self, parameters: PlanningCostParameters) -> None:
        self.parameters = parameters

    def plan(
        self,
        planning_cost_field: NDArray[np.floating],
        forbidden_mask: NDArray[np.bool_],
        start: GridCell,
        goal: GridCell,
        navigation: NavigationLayer,
        hazard_field: NDArray[np.floating],
        uncertainty_field: NDArray[np.floating],
    ) -> RouteResult:
        cost = np.asarray(planning_cost_field, dtype=np.float64)
        forbidden = np.asarray(forbidden_mask, dtype=np.bool_)
        hazard = np.asarray(hazard_field, dtype=np.float64)
        uncertainty = np.asarray(uncertainty_field, dtype=np.float64)
        geometry = navigation.occupancy.geometry
        shape = (geometry.rows, geometry.cols)
        if any(array.shape != shape for array in (cost, forbidden, hazard, uncertainty)):
            raise ValueError("All planning fields must match navigation geometry")
        if np.any(np.isnan(cost)) or not np.all(np.isfinite(hazard)) or not np.all(
            np.isfinite(uncertainty)
        ):
            raise ValueError("Planning inputs must not contain NaN or non-finite metrics")
        finite_allowed = (~forbidden) & np.isfinite(cost)
        if np.any(cost[finite_allowed] < 0.0):
            raise ValueError("Finite traversal costs must be non-negative")
        if np.any((hazard < 0.0) | (hazard > 1.0)) or np.any(
            (uncertainty < 0.0) | (uncertainty > 1.0)
        ):
            raise ValueError("Hazard and uncertainty inputs must remain in [0, 1]")

        rejected = self._validate_endpoint("start", start, navigation, forbidden, cost)
        rejected += self._validate_endpoint("goal", goal, navigation, forbidden, cost)
        if rejected:
            return RouteResult.rejected(*rejected)
        if start == goal:
            return self._route_result((start,), cost, hazard, uncertainty, geometry.resolution_m)

        g_score: dict[GridCell, float] = {start: 0.0}
        came_from: dict[GridCell, GridCell] = {}
        minimum_density = min(
            self.parameters.base_cost,
            float(np.min(cost[finite_allowed])),
        )
        start_h = self._heuristic(start, goal, geometry.resolution_m, minimum_density)
        frontier: list[tuple[float, float, int, int]] = [
            (start_h, start_h, start[0], start[1])
        ]
        closed: set[GridCell] = set()
        while frontier:
            _, _, row, col = heapq.heappop(frontier)
            current = (row, col)
            if current in closed:
                continue
            if current == goal:
                path = self._reconstruct(came_from, current)
                return self._route_result(
                    path,
                    cost,
                    hazard,
                    uncertainty,
                    geometry.resolution_m,
                )
            closed.add(current)
            for neighbor in navigation.neighbors(current):
                if forbidden[neighbor] or not math.isfinite(float(cost[neighbor])):
                    continue
                step_cost = float(cost[neighbor]) * geometry.resolution_m
                tentative = g_score[current] + step_cost
                if tentative + 1e-12 >= g_score.get(neighbor, math.inf):
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                heuristic = self._heuristic(
                    neighbor,
                    goal,
                    geometry.resolution_m,
                    minimum_density,
                )
                heapq.heappush(
                    frontier,
                    (tentative + heuristic, heuristic, neighbor[0], neighbor[1]),
                )
        return RouteResult.rejected("goal_unreachable")

    def evaluate_path(
        self,
        path: tuple[GridCell, ...],
        planning_cost_field: NDArray[np.floating],
        forbidden_mask: NDArray[np.bool_],
        navigation: NavigationLayer,
        hazard_field: NDArray[np.floating],
        uncertainty_field: NDArray[np.floating],
    ) -> RouteResult:
        """Evaluate a fixed path without searching for or altering a route."""
        cost = np.asarray(planning_cost_field, dtype=np.float64)
        forbidden = np.asarray(forbidden_mask, dtype=np.bool_)
        hazard = np.asarray(hazard_field, dtype=np.float64)
        uncertainty = np.asarray(uncertainty_field, dtype=np.float64)
        geometry = navigation.occupancy.geometry
        shape = (geometry.rows, geometry.cols)
        if any(array.shape != shape for array in (cost, forbidden, hazard, uncertainty)):
            raise ValueError("All planning fields must match navigation geometry")
        if not path:
            return RouteResult.rejected("path_empty")
        for index, cell in enumerate(path):
            if not geometry.contains_cell(cell):
                return RouteResult.rejected(f"path_cell_{index}_out_of_bounds")
            if not navigation.is_traversable(cell):
                return RouteResult.rejected(f"path_cell_{index}_not_traversable")
            if forbidden[cell] or not math.isfinite(float(cost[cell])):
                return RouteResult.rejected(f"path_cell_{index}_forbidden")
            if index > 0:
                previous = path[index - 1]
                if abs(cell[0] - previous[0]) + abs(cell[1] - previous[1]) != 1:
                    return RouteResult.rejected(f"path_step_{index}_not_four_connected")
        return self._route_result(
            path,
            cost,
            hazard,
            uncertainty,
            geometry.resolution_m,
        )

    def _validate_endpoint(
        self,
        name: str,
        cell: GridCell,
        navigation: NavigationLayer,
        forbidden: NDArray[np.bool_],
        cost: NDArray[np.float64],
    ) -> tuple[str, ...]:
        geometry = navigation.occupancy.geometry
        if not geometry.contains_cell(cell):
            return (f"{name}_out_of_bounds",)
        if not navigation.is_traversable(cell):
            return (f"{name}_not_traversable",)
        if forbidden[cell] or not math.isfinite(float(cost[cell])):
            return (f"{name}_forbidden",)
        return ()

    def _heuristic(
        self,
        cell: GridCell,
        goal: GridCell,
        resolution_m: float,
        minimum_density: float,
    ) -> float:
        manhattan_steps = abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])
        return manhattan_steps * resolution_m * minimum_density

    @staticmethod
    def _reconstruct(came_from: dict[GridCell, GridCell], current: GridCell) -> tuple[GridCell, ...]:
        reversed_path = [current]
        while current in came_from:
            current = came_from[current]
            reversed_path.append(current)
        return tuple(reversed(reversed_path))

    def _route_result(
        self,
        path: tuple[GridCell, ...],
        cost: NDArray[np.float64],
        hazard: NDArray[np.float64],
        uncertainty: NDArray[np.float64],
        resolution_m: float,
    ) -> RouteResult:
        entered = path[1:]
        geometric_length = len(entered) * resolution_m
        integrated_hazard = sum(float(hazard[cell]) for cell in entered) * resolution_m
        integrated_uncertainty = (
            sum(float(uncertainty[cell]) for cell in entered) * resolution_m
        )
        total_cost = sum(float(cost[cell]) for cell in entered) * resolution_m
        max_hazard = max(float(hazard[cell]) for cell in path)
        return RouteResult(
            path=path,
            feasible=True,
            geometric_length_m=geometric_length,
            total_cost=total_cost,
            base_contribution=self.parameters.base_cost * geometric_length,
            hazard_contribution=self.parameters.hazard_weight * integrated_hazard,
            uncertainty_contribution=(
                self.parameters.uncertainty_weight * integrated_uncertainty
            ),
            integrated_hazard=integrated_hazard,
            integrated_uncertainty=integrated_uncertainty,
            max_hazard=max_hazard,
        )

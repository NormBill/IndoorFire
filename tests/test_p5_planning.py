from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from indoor_fire_world.hazard import PhysicalHazardModel, load_hazard_parameters
from indoor_fire_world.planning import (
    GridAStarPlanner,
    MissionPlanner,
    PlanningCostModel,
    PlanningCostParameters,
)
from indoor_fire_world.scene.models import (
    GridGeometry,
    NavigationLayer,
    Occupancy,
    OccupancyLayer,
)
from indoor_fire_world.world_model import BeliefMean


HAZARD_CONFIG = REPO_ROOT / "configs" / "hazard.yaml"


class P5PlanningTests(unittest.TestCase):
    @staticmethod
    def _navigation(
        rows: int,
        cols: int,
        blocked: set[tuple[int, int]] | None = None,
    ) -> NavigationLayer:
        blocked = blocked or set()
        geometry = GridGeometry(float(cols), float(rows), 1.0)
        cells = tuple(
            tuple(
                Occupancy.OCCUPIED if (row, col) in blocked else Occupancy.FREE
                for col in range(cols)
            )
            for row in range(rows)
        )
        return NavigationLayer(OccupancyLayer(geometry, cells))

    @staticmethod
    def _parameters(hazard_weight: float = 2.0, uncertainty_weight: float = 3.0):
        return PlanningCostParameters(
            base_cost=1.0,
            hazard_weight=hazard_weight,
            uncertainty_weight=uncertainty_weight,
            untenable_hazard_threshold=0.95,
        )

    def _fields(
        self,
        navigation: NavigationLayer,
        parameters: PlanningCostParameters,
        hazard: np.ndarray | None = None,
        uncertainty: np.ndarray | None = None,
    ):
        shape = (navigation.occupancy.geometry.rows, navigation.occupancy.geometry.cols)
        hazard = np.zeros(shape) if hazard is None else hazard
        uncertainty = np.zeros(shape) if uncertainty is None else uncertainty
        cost = PlanningCostModel(parameters).compute(hazard, uncertainty, navigation)
        return cost, hazard, uncertainty

    def _plan(
        self,
        navigation: NavigationLayer,
        start: tuple[int, int],
        goal: tuple[int, int],
        parameters: PlanningCostParameters | None = None,
        hazard: np.ndarray | None = None,
        uncertainty: np.ndarray | None = None,
    ):
        parameters = parameters or self._parameters()
        cost, hazard, uncertainty = self._fields(
            navigation,
            parameters,
            hazard,
            uncertainty,
        )
        route = GridAStarPlanner(parameters).plan(
            cost.field,
            cost.forbidden,
            start,
            goal,
            navigation,
            hazard,
            uncertainty,
        )
        return route, cost

    def test_zero_hazard_has_no_dominant_modality(self) -> None:
        model = PhysicalHazardModel(load_hazard_parameters(HAZARD_CONFIG))
        mean = BeliefMean(
            temperature=np.array([[22.0]]),
            co=np.array([[0.0]]),
            smoke=np.array([[0.0]]),
            visibility=np.array([[30.0]]),
        )
        result = model.compute(mean)
        self.assertEqual(result.field[0, 0], 0.0)
        self.assertEqual(result.dominant_modality[0, 0], "none")

    def test_free_map_route_is_found(self) -> None:
        route, _ = self._plan(self._navigation(3, 3), (0, 0), (2, 2))
        self.assertTrue(route.feasible)
        self.assertEqual(route.path[0], (0, 0))
        self.assertEqual(route.path[-1], (2, 2))
        self.assertEqual(route.geometric_length_m, 4.0)

    def test_obstacles_are_never_crossed(self) -> None:
        blocked = {(1, 1), (1, 2), (1, 3)}
        route, _ = self._plan(self._navigation(3, 5, blocked), (1, 0), (1, 4))
        self.assertTrue(route.feasible)
        self.assertTrue(blocked.isdisjoint(route.path))

    def test_untenable_cells_are_never_crossed(self) -> None:
        navigation = self._navigation(3, 5)
        hazard = np.zeros((3, 5))
        hazard[1, 1:4] = 0.95
        route, _ = self._plan(navigation, (1, 0), (1, 4), hazard=hazard)
        self.assertTrue(route.feasible)
        self.assertTrue(all(hazard[cell] < 0.95 for cell in route.path))

    def test_invalid_start_and_goal_are_rejected(self) -> None:
        navigation = self._navigation(3, 3, {(2, 2)})
        invalid_start, _ = self._plan(navigation, (-1, 0), (0, 0))
        invalid_goal, _ = self._plan(navigation, (0, 0), (2, 2))
        self.assertFalse(invalid_start.feasible)
        self.assertIn("start_out_of_bounds", invalid_start.rejection_reasons)
        self.assertFalse(invalid_goal.feasible)
        self.assertIn("goal_not_traversable", invalid_goal.rejection_reasons)

    def test_unreachable_goal_is_handled(self) -> None:
        navigation = self._navigation(3, 5, {(0, 2), (1, 2), (2, 2)})
        route, _ = self._plan(navigation, (1, 0), (1, 4))
        self.assertFalse(route.feasible)
        self.assertIn("goal_unreachable", route.rejection_reasons)

    def test_lower_cost_corridor_is_preferred(self) -> None:
        navigation = self._navigation(3, 5)
        uncertainty = np.zeros((3, 5))
        uncertainty[1, 1:4] = 1.0
        route, _ = self._plan(
            navigation,
            (1, 0),
            (1, 4),
            parameters=self._parameters(uncertainty_weight=20.0),
            uncertainty=uncertainty,
        )
        self.assertTrue(route.feasible)
        self.assertTrue(any(row != 1 for row, _ in route.path[1:-1]))

    def test_total_cost_matches_accumulated_field_cost(self) -> None:
        navigation = self._navigation(3, 4)
        hazard = np.full((3, 4), 0.2)
        uncertainty = np.full((3, 4), 0.4)
        route, cost = self._plan(
            navigation,
            (1, 0),
            (1, 3),
            hazard=hazard,
            uncertainty=uncertainty,
        )
        expected = sum(float(cost.field[cell]) for cell in route.path[1:])
        self.assertAlmostEqual(route.total_cost, expected)
        self.assertAlmostEqual(
            route.total_cost,
            route.base_contribution
            + route.hazard_contribution
            + route.uncertainty_contribution,
        )

    def test_route_hazard_and_uncertainty_metrics_are_correct(self) -> None:
        navigation = self._navigation(1, 4)
        hazard = np.full((1, 4), 0.2)
        uncertainty = np.full((1, 4), 0.5)
        route, _ = self._plan(
            navigation,
            (0, 0),
            (0, 3),
            hazard=hazard,
            uncertainty=uncertainty,
        )
        self.assertEqual(route.geometric_length_m, 3.0)
        self.assertAlmostEqual(route.integrated_hazard, 0.6)
        self.assertAlmostEqual(route.integrated_uncertainty, 1.5)
        self.assertAlmostEqual(route.max_hazard, 0.2)

    def test_tie_handling_is_deterministic(self) -> None:
        navigation = self._navigation(3, 3, {(1, 1)})
        first, _ = self._plan(navigation, (1, 0), (1, 2))
        second, _ = self._plan(navigation, (1, 0), (1, 2))
        self.assertEqual(first.path, second.path)
        self.assertEqual(first.path_hash, second.path_hash)
        self.assertIn((0, 1), first.path)

    def test_lowest_cost_feasible_exit_is_selected(self) -> None:
        navigation = self._navigation(3, 5)
        parameters = self._parameters()
        cost, hazard, uncertainty = self._fields(navigation, parameters)
        mission = MissionPlanner(GridAStarPlanner(parameters)).plan(
            cost.field,
            cost.forbidden,
            hazard,
            uncertainty,
            navigation,
            (1, 0),
            (1, 2),
            {"exit_far": (0, 0), "exit_near": (1, 4)},
        )
        self.assertTrue(mission.feasible)
        self.assertEqual(mission.selected_exit, "exit_near")

    def test_blocked_exits_are_ignored(self) -> None:
        navigation = self._navigation(3, 5, {(1, 4)})
        parameters = self._parameters()
        cost, hazard, uncertainty = self._fields(navigation, parameters)
        mission = MissionPlanner(GridAStarPlanner(parameters)).plan(
            cost.field,
            cost.forbidden,
            hazard,
            uncertainty,
            navigation,
            (1, 0),
            (1, 2),
            {"blocked": (1, 4), "open": (0, 0)},
        )
        self.assertTrue(mission.feasible)
        self.assertEqual(mission.selected_exit, "open")

    def test_no_retreat_makes_mission_infeasible(self) -> None:
        navigation = self._navigation(2, 3)
        parameters = self._parameters()
        cost, hazard, uncertainty = self._fields(navigation, parameters)
        mission = MissionPlanner(GridAStarPlanner(parameters)).plan(
            cost.field,
            cost.forbidden,
            hazard,
            uncertainty,
            navigation,
            (0, 0),
            (0, 2),
            {},
        )
        self.assertFalse(mission.feasible)
        self.assertFalse(mission.retreat_route.feasible)
        self.assertIn("retreat_infeasible", mission.rejection_reasons)

    def test_entry_feasible_and_retreat_infeasible_makes_mission_infeasible(self) -> None:
        barrier = {(0, 3), (1, 3), (2, 3)}
        navigation = self._navigation(3, 5, barrier)
        parameters = self._parameters()
        cost, hazard, uncertainty = self._fields(navigation, parameters)
        mission = MissionPlanner(GridAStarPlanner(parameters)).plan(
            cost.field,
            cost.forbidden,
            hazard,
            uncertainty,
            navigation,
            (1, 0),
            (1, 2),
            {"isolated_exit": (1, 4)},
        )
        self.assertTrue(mission.entry_route.feasible)
        self.assertFalse(mission.retreat_route.feasible)
        self.assertFalse(mission.feasible)

    def test_increasing_hazard_weight_changes_hazard_contribution(self) -> None:
        navigation = self._navigation(1, 3)
        hazard = np.full((1, 3), 0.2)
        low, _ = self._plan(
            navigation,
            (0, 0),
            (0, 2),
            parameters=self._parameters(hazard_weight=1.0, uncertainty_weight=0.0),
            hazard=hazard,
        )
        high, _ = self._plan(
            navigation,
            (0, 0),
            (0, 2),
            parameters=self._parameters(hazard_weight=3.0, uncertainty_weight=0.0),
            hazard=hazard,
        )
        self.assertAlmostEqual(high.hazard_contribution, 3.0 * low.hazard_contribution)
        self.assertGreater(high.total_cost, low.total_cost)

    def test_increasing_uncertainty_weight_changes_uncertainty_contribution(self) -> None:
        navigation = self._navigation(1, 3)
        uncertainty = np.full((1, 3), 0.4)
        low, _ = self._plan(
            navigation,
            (0, 0),
            (0, 2),
            parameters=self._parameters(hazard_weight=0.0, uncertainty_weight=1.0),
            uncertainty=uncertainty,
        )
        high, _ = self._plan(
            navigation,
            (0, 0),
            (0, 2),
            parameters=self._parameters(hazard_weight=0.0, uncertainty_weight=4.0),
            uncertainty=uncertainty,
        )
        self.assertAlmostEqual(
            high.uncertainty_contribution,
            4.0 * low.uncertainty_contribution,
        )
        self.assertGreater(high.total_cost, low.total_cost)


if __name__ == "__main__":
    unittest.main()

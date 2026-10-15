from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from indoor_fire_world.planning import (
    DirectedRoutePlanner,
    PlanningCostParameters,
    load_directed_evacuation_graph,
)
from indoor_fire_world.scene import load_scene


REPO_ROOT = Path(__file__).resolve().parents[1]
SCENE_CONFIG = REPO_ROOT / "configs" / "scene_w1_4f_registered_v1.yaml"
ROUTE_CONFIG = REPO_ROOT / "configs" / "evacuation_routes_w1_4f_registered_v1.yaml"


class Gate3DirectedRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scene = load_scene(SCENE_CONFIG)
        cls.graph = load_directed_evacuation_graph(ROUTE_CONFIG, cls.scene)
        cls.parameters = PlanningCostParameters(
            base_cost=1.0,
            hazard_weight=1.0,
            uncertainty_weight=0.0,
            untenable_hazard_threshold=0.95,
        )
        cls.planner = DirectedRoutePlanner(cls.parameters, cls.graph)
        shape = (cls.scene.occupancy.geometry.rows, cls.scene.occupancy.geometry.cols)
        cls.cost = np.ones(shape, dtype=np.float64)
        cls.forbidden = np.zeros(shape, dtype=np.bool_)
        cls.hazard = np.zeros(shape, dtype=np.float64)
        cls.uncertainty = np.zeros(shape, dtype=np.float64)

    def plan(self, start: str, goal: str):
        return self.planner.plan(
            self.cost,
            self.forbidden,
            start,
            goal,
            self.scene.navigation,
            self.hazard,
            self.uncertainty,
        )

    def test_all_registered_edges_are_traversable_and_directed(self) -> None:
        for edge in self.graph.edges:
            self.assertTrue(all(self.scene.navigation.is_traversable(cell) for cell in edge.path))
            self.assertTrue(self.graph.path_is_directed(edge.path))

    def test_ugv_start_reaches_all_declared_exits_on_green_routes(self) -> None:
        for exit_id in ("exit_west", "exit_center_stair", "exit_southeast"):
            route = self.plan("ugv_start", exit_id)
            self.assertTrue(route.feasible, exit_id)
            self.assertTrue(self.graph.path_is_directed(route.path))

    def test_reverse_motion_against_green_arrow_is_rejected(self) -> None:
        reverse = self.plan("geometry_route_high", "ugv_start")
        self.assertFalse(reverse.feasible)
        self.assertIn("goal_unreachable_on_directed_route_graph", reverse.rejection_reasons)

    def test_observation_nodes_are_on_registered_route(self) -> None:
        for node_id in (
            "geometry_route_high",
            "geometry_route_mid",
            "geometry_route_near",
            "geometry_route_low_control",
        ):
            self.assertIn(node_id, self.graph.nodes)
            self.assertTrue(self.scene.navigation.is_traversable(self.graph.nodes[node_id].cell))


if __name__ == "__main__":
    unittest.main()

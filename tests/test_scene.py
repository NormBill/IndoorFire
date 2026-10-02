from __future__ import annotations

from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from indoor_fire_world.scene import Occupancy, load_scene


CONFIG = REPO_ROOT / "configs" / "scene_w1_4f.yaml"


class W14FSceneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = load_scene(CONFIG)

    def test_scene_config_loads(self) -> None:
        self.assertEqual(self.scene.scene_id, "w1_4f_makerspace")

    def test_dimensions_and_layers_are_consistent(self) -> None:
        geometry = self.scene.occupancy.geometry
        self.assertEqual((geometry.rows, geometry.cols), (56, 80))
        self.assertIs(self.scene.navigation.occupancy, self.scene.occupancy)

    def test_metric_grid_round_trip_within_one_cell(self) -> None:
        geometry = self.scene.occupancy.geometry
        for point in ((0.26, 0.26), (1.5, 1.5), (7.99, 13.49), (19.74, 6.0)):
            cell = geometry.metric_to_grid(*point)
            result = geometry.grid_to_metric(*cell)
            self.assertLessEqual(abs(result[0] - point[0]), geometry.resolution_m)
            self.assertLessEqual(abs(result[1] - point[1]), geometry.resolution_m)

    def test_known_obstacles_are_blocked(self) -> None:
        for point in ((0.1, 4.0), (3.0, 4.5), (16.0, 10.0)):
            cell = self.scene.navigation.metric_to_grid(*point)
            self.assertEqual(self.scene.occupancy.at(cell), Occupancy.OCCUPIED)
            self.assertFalse(self.scene.navigation.is_traversable(cell))

    def test_known_free_cells_are_traversable(self) -> None:
        for point in ((1.0, 1.0), (7.0, 6.0), (10.0, 6.0)):
            cell = self.scene.navigation.metric_to_grid(*point)
            self.assertTrue(self.scene.navigation.is_traversable(cell))

    def test_required_semantics_exist(self) -> None:
        self.assertEqual(len(self.scene.semantic.by_kind("microwave")), 1)
        self.assertGreaterEqual(len(self.scene.semantic.by_kind("exit")), 1)
        self.assertGreaterEqual(len(self.scene.semantic.by_kind("fixed_sensor")), 1)

    def test_ugv_start_is_valid_and_traversable(self) -> None:
        pose = self.scene.ugv_initial_pose
        cell = self.scene.navigation.metric_to_grid(pose.x, pose.y)
        self.assertTrue(self.scene.navigation.is_traversable(cell))

    def test_repeated_loading_is_deterministic(self) -> None:
        again = load_scene(CONFIG)
        self.assertEqual(self.scene, again)


if __name__ == "__main__":
    unittest.main()


from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from indoor_fire_world.fire import GroundTruthSimulator, load_fire_parameters
from indoor_fire_world.scene import Occupancy, load_scene


SCENE_CONFIG = REPO_ROOT / "configs" / "scene_w1_4f_registered_v1.yaml"
TRUTH_CONFIG = REPO_ROOT / "configs" / "fire_microwave_electrical_truth_v1.yaml"
PRIOR_CONFIG = REPO_ROOT / "configs" / "fire_microwave_electrical_prior_v1.yaml"
SCENARIO_CONFIG = REPO_ROOT / "configs" / "microwave_electrical_scenario_v1.yaml"


class Gate1RegisteredMapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = load_scene(SCENE_CONFIG)

    def test_registered_map_keeps_source_image_and_provisional_status(self) -> None:
        registration = self.scene.map_registration
        self.assertIsNotNone(registration)
        assert registration is not None
        self.assertTrue(registration.source_image.is_file())
        self.assertEqual(
            (registration.source_width_px, registration.source_height_px),
            (1713, 849),
        )
        self.assertIn("scale_pending", registration.calibration_status)
        self.assertEqual(self.scene.geometry_status, "image_registered_scale_pending")

    def test_pixel_metric_registration_round_trip(self) -> None:
        registration = self.scene.map_registration
        assert registration is not None
        for pixel in ((40.0, 165.0), (840.0, 390.0), (1690.0, 680.0)):
            metric = registration.pixel_to_metric(*pixel)
            round_trip = registration.metric_to_pixel(*metric)
            self.assertAlmostEqual(round_trip[0], pixel[0])
            self.assertAlmostEqual(round_trip[1], pixel[1])

    def test_registered_grid_and_three_exits_are_explicit(self) -> None:
        geometry = self.scene.occupancy.geometry
        self.assertEqual((geometry.rows, geometry.cols), (50, 160))
        exits = self.scene.semantic.by_kind("exit")
        self.assertEqual(
            {item.id for item in exits},
            {"exit_west", "exit_center_stair", "exit_southeast"},
        )

    def test_critical_semantics_are_traversable(self) -> None:
        ids = ("microwave", "exit_west", "exit_center_stair", "exit_southeast")
        for object_id in ids:
            item = self.scene.semantic.by_id(object_id)
            assert item.position is not None
            cell = self.scene.navigation.metric_to_grid(*item.position)
            self.assertTrue(self.scene.navigation.is_traversable(cell), object_id)
        pose = self.scene.ugv_initial_pose
        self.assertTrue(
            self.scene.navigation.is_traversable(
                self.scene.navigation.metric_to_grid(pose.x, pose.y)
            )
        )

    def test_ugv_microwave_and_exits_share_a_connected_component(self) -> None:
        pose = self.scene.ugv_initial_pose
        start = self.scene.navigation.metric_to_grid(pose.x, pose.y)
        reachable = {start}
        frontier = [start]
        while frontier:
            cell = frontier.pop()
            for neighbor in self.scene.navigation.neighbors(cell):
                if neighbor not in reachable:
                    reachable.add(neighbor)
                    frontier.append(neighbor)
        for object_id in (
            "microwave",
            "exit_west",
            "exit_center_stair",
            "exit_southeast",
        ):
            item = self.scene.semantic.by_id(object_id)
            assert item.position is not None
            self.assertIn(
                self.scene.navigation.metric_to_grid(*item.position),
                reachable,
                object_id,
            )

    def test_polygon_occupancy_distinguishes_floor_void_and_furniture(self) -> None:
        free = self.scene.navigation.metric_to_grid(10.0, 10.0)
        outside = self.scene.navigation.metric_to_grid(1.0, 0.25)
        counter = self.scene.navigation.metric_to_grid(8.5, 6.0)
        self.assertEqual(self.scene.occupancy.at(free), Occupancy.FREE)
        self.assertEqual(self.scene.occupancy.at(outside), Occupancy.UNKNOWN)
        self.assertEqual(self.scene.occupancy.at(counter), Occupancy.OCCUPIED)


class Gate2ElectricalFireScenarioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = load_scene(SCENE_CONFIG)
        self.truth = load_fire_parameters(TRUTH_CONFIG)
        self.prior = load_fire_parameters(PRIOR_CONFIG)

    def test_phase_schedule_is_contiguous_and_explicit(self) -> None:
        self.assertEqual(
            tuple(phase.phase_id for phase in self.truth.source_phases),
            (
                "overheating_pre_ignition",
                "concealed_smoldering",
                "internal_flaming",
                "enclosure_breakout",
            ),
        )
        self.assertEqual(self.truth.source_phases[-1].end_s, None)
        self.assertIn("not_w1_calibrated", self.truth.assumption_status)

    def test_truth_hrr_is_continuous_at_declared_phase_boundaries(self) -> None:
        source = GroundTruthSimulator(self.scene, self.truth).source
        self.assertEqual(source.heat_release_rate_kw(29.9), 0.0)
        self.assertEqual(source.heat_release_rate_kw(30.0), 0.0)
        self.assertAlmostEqual(source.heat_release_rate_kw(60.0), 9.0)
        self.assertAlmostEqual(source.heat_release_rate_kw(90.0), 36.0)
        self.assertAlmostEqual(source.heat_release_rate_kw(180.0), 250.0)

    def test_truth_and_prior_encode_hidden_fire_mismatch(self) -> None:
        truth_source = GroundTruthSimulator(self.scene, self.truth).source
        prior_source = GroundTruthSimulator(self.scene, self.prior).source
        self.assertGreater(
            truth_source.heat_release_rate_kw(100.0),
            prior_source.heat_release_rate_kw(100.0),
        )
        self.assertNotEqual(
            truth_source.phase_at(100.0).phase_id,
            prior_source.phase_at(100.0).phase_id,
        )

    def test_pre_ignition_phase_produces_no_room_scale_source(self) -> None:
        simulator = GroundTruthSimulator(self.scene, self.truth)
        state = simulator.step()
        source_cell = simulator.source.cell
        self.assertEqual(state.temperature[source_cell], self.truth.ambient_temperature_c)
        self.assertEqual(state.smoke[source_cell], 0.0)
        self.assertEqual(state.co[source_cell], 0.0)

    def test_secondary_cable_spread_is_explicitly_out_of_scope(self) -> None:
        with SCENARIO_CONFIG.open("r", encoding="utf-8") as stream:
            scenario = json.load(stream)
        self.assertFalse(scenario["secondary_cable_spread"]["enabled"])
        self.assertFalse(scenario["gate_status"]["metric_scale_surveyed"])
        self.assertFalse(scenario["gate_status"]["w1_fire_parameters_calibrated"])


if __name__ == "__main__":
    unittest.main()

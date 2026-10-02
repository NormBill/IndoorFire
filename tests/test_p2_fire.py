from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from indoor_fire_world.fire import GroundTruthSimulator, load_fire_parameters
from indoor_fire_world.fire.transport import advance_scalar
from indoor_fire_world.scene import Occupancy, load_scene
from indoor_fire_world.world_model import PhysicsPrior


SCENE_CONFIG = REPO_ROOT / "configs" / "scene_w1_4f.yaml"
TRUTH_CONFIG = REPO_ROOT / "configs" / "fire_truth.yaml"
PRIOR_CONFIG = REPO_ROOT / "configs" / "fire_prior.yaml"


class P2FireSimulatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = load_scene(SCENE_CONFIG)
        self.truth_parameters = load_fire_parameters(TRUTH_CONFIG)
        self.prior_parameters = load_fire_parameters(PRIOR_CONFIG)

    def test_deterministic_initialization(self) -> None:
        first = GroundTruthSimulator(self.scene, self.truth_parameters).current_state
        second = GroundTruthSimulator(self.scene, self.truth_parameters).current_state
        self.assertEqual(first.timestamp, 0.0)
        np.testing.assert_array_equal(first.temperature, second.temperature)
        np.testing.assert_array_equal(first.co, second.co)
        np.testing.assert_array_equal(first.smoke, second.smoke)
        np.testing.assert_array_equal(first.visibility, second.visibility)

    def test_truth_and_prior_states_are_independent(self) -> None:
        truth = GroundTruthSimulator(self.scene, self.truth_parameters)
        prior = PhysicsPrior(self.scene, self.prior_parameters)
        prior_before = prior.current_state
        truth.step()
        prior_after = prior.current_state
        self.assertFalse(np.shares_memory(truth.current_state.smoke, prior_after.smoke))
        np.testing.assert_array_equal(prior_before.smoke, prior_after.smoke)
        self.assertEqual(prior_after.timestamp, 0.0)

    def test_ignition_occurs_at_microwave_cell(self) -> None:
        simulator = GroundTruthSimulator(self.scene, self.truth_parameters)
        state = simulator.step()
        source_cell = simulator.source.cell
        microwave = self.scene.semantic.by_id(self.truth_parameters.source_semantic_id)
        self.assertEqual(source_cell, self.scene.navigation.metric_to_grid(*microwave.position))
        self.assertGreater(state.temperature[source_cell], self.truth_parameters.ambient_temperature_c)
        self.assertGreater(state.smoke[source_cell], 0.0)
        self.assertGreater(state.co[source_cell], 0.0)

    def test_zero_source_produces_no_artificial_growth(self) -> None:
        parameters = replace(self.truth_parameters, source_enabled=False)
        simulator = GroundTruthSimulator(self.scene, parameters)
        for _ in range(10):
            state = simulator.step()
        np.testing.assert_array_equal(state.smoke, np.zeros(state.shape))
        np.testing.assert_array_equal(state.co, np.zeros(state.shape))
        np.testing.assert_array_equal(
            state.temperature,
            np.full(state.shape, parameters.ambient_temperature_c),
        )

    def test_smoke_and_co_spread_spatially(self) -> None:
        simulator = GroundTruthSimulator(self.scene, self.truth_parameters)
        for _ in range(5):
            state = simulator.step()
        row, col = simulator.source.cell
        free_neighbors = list(self.scene.navigation.neighbors((row, col)))
        self.assertTrue(free_neighbors)
        self.assertTrue(any(state.smoke[cell] > 0.0 for cell in free_neighbors))
        self.assertTrue(any(state.co[cell] > 0.0 for cell in free_neighbors))

    def test_occupied_wall_blocks_direct_transport(self) -> None:
        shape = (self.scene.occupancy.geometry.rows, self.scene.occupancy.geometry.cols)
        fluid = np.array(
            [[cell == Occupancy.FREE for cell in row] for row in self.scene.occupancy.cells],
            dtype=np.bool_,
        )
        field = np.zeros(shape, dtype=np.float64)
        wall_row, wall_col = self.scene.navigation.metric_to_grid(8.1, 1.0)
        left_cell = (wall_row, wall_col - 1)
        right_cell = (wall_row, wall_col + 1)
        self.assertFalse(fluid[wall_row, wall_col])
        self.assertTrue(fluid[left_cell])
        self.assertTrue(fluid[right_cell])
        field[left_cell] = 1.0
        updated = advance_scalar(
            field,
            fluid,
            np.zeros_like(field),
            dt=0.1,
            cell_size_m=1.0,
            diffusion_m2_per_s=0.1,
            velocity_mps=(0.5, 0.0),
            decay_per_s=0.0,
            solid_value=0.0,
        )
        self.assertEqual(updated[wall_row, wall_col], 0.0)
        self.assertEqual(updated[right_cell], 0.0)

    def test_visibility_decreases_as_smoke_increases(self) -> None:
        simulator = GroundTruthSimulator(self.scene, self.truth_parameters)
        clear_visibility = simulator.current_state.visibility[simulator.source.cell]
        smoky_state = simulator.step()
        self.assertLess(smoky_state.visibility[simulator.source.cell], clear_visibility)

    def test_invalid_timestep_fails_explicitly(self) -> None:
        unstable = replace(self.truth_parameters, time_step_s=10.0)
        with self.assertRaisesRegex(ValueError, "Unstable explicit timestep"):
            GroundTruthSimulator(self.scene, unstable)
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            replace(self.truth_parameters, smoke_diffusion_m2_per_s=-0.1)

    def test_truth_and_prior_diverge_with_mismatched_parameters(self) -> None:
        truth = GroundTruthSimulator(self.scene, self.truth_parameters)
        prior = PhysicsPrior(self.scene, self.prior_parameters)
        for _ in range(10):
            truth_state = truth.step()
            prior_state = prior.step()
        self.assertFalse(np.array_equal(truth_state.temperature, prior_state.temperature))
        self.assertFalse(np.array_equal(truth_state.smoke, prior_state.smoke))
        self.assertFalse(np.array_equal(truth_state.co, prior_state.co))

    def test_repeated_runs_are_deterministic(self) -> None:
        first = GroundTruthSimulator(self.scene, self.truth_parameters)
        second = GroundTruthSimulator(self.scene, self.truth_parameters)
        for _ in range(20):
            first_state = first.step()
            second_state = second.step()
        np.testing.assert_array_equal(first_state.temperature, second_state.temperature)
        np.testing.assert_array_equal(first_state.smoke, second_state.smoke)
        np.testing.assert_array_equal(first_state.co, second_state.co)
        np.testing.assert_array_equal(first_state.visibility, second_state.visibility)


if __name__ == "__main__":
    unittest.main()

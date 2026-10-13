from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from indoor_fire_world.fire import FireField, GroundTruthSimulator
from indoor_fire_world.p3_config import SensorNoiseParameters, load_p3_parameters
from indoor_fire_world.robotics import Pose2D, UGV
from indoor_fire_world.scene import load_scene
from indoor_fire_world.sensing import Observation, UGVSensor
from indoor_fire_world.simulation import run_assimilation_demo
from indoor_fire_world.world_model import PhysicsPrior, SpatialAssimilator, initialize_belief


SCENE_CONFIG = REPO_ROOT / "configs" / "scene_w1_4f.yaml"
TRUTH_CONFIG = REPO_ROOT / "configs" / "fire_truth.yaml"
PRIOR_CONFIG = REPO_ROOT / "configs" / "fire_prior.yaml"
P3_CONFIG = REPO_ROOT / "configs" / "p3_assimilation.yaml"
MODALITIES = ("temperature", "co", "smoke", "visibility")


class P3AssimilationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = load_scene(SCENE_CONFIG)
        self.parameters = load_p3_parameters(P3_CONFIG)
        self.waypoint = self.parameters.waypoint(self.parameters.demo_waypoint_id)

    def _prior_belief(self):
        prior = PhysicsPrior(self.scene, PRIOR_CONFIG)
        return initialize_belief(prior.current_state, self.parameters.initial_uncertainty)

    def _observation(self, belief, **overrides) -> Observation:
        cell = self.scene.navigation.metric_to_grid(self.waypoint.pose.x, self.waypoint.pose.y)
        values = {name: float(getattr(belief.mean, name)[cell]) for name in MODALITIES}
        values.update(overrides)
        return Observation(
            timestamp=belief.timestamp,
            position=(self.waypoint.pose.x, self.waypoint.pose.y),
            source="test_ugv",
            **values,
        )

    def test_sensor_samples_correct_grid_location(self) -> None:
        shape = (self.scene.occupancy.geometry.rows, self.scene.occupancy.geometry.cols)
        pose = self.scene.ugv_initial_pose
        cell = self.scene.navigation.metric_to_grid(pose.x, pose.y)
        temperature = np.full(shape, 22.0)
        co = np.zeros(shape)
        smoke = np.zeros(shape)
        temperature[cell] = 47.5
        co[cell] = 0.25
        smoke[cell] = 0.5
        truth = FireField(temperature, co, smoke, 3.0, 30.0, 1.6)
        sensor = UGVSensor(self.scene, self.parameters.sensor, self.parameters.random_seed)
        observation = sensor.observe(truth, pose)
        self.assertEqual(observation.temperature, 47.5)
        self.assertEqual(observation.co, 0.25)
        self.assertEqual(observation.smoke, 0.5)
        self.assertEqual(observation.visibility, float(truth.visibility[cell]))

    def test_sensor_does_not_mutate_truth(self) -> None:
        truth = GroundTruthSimulator(self.scene, TRUTH_CONFIG).current_state
        before = {name: np.array(getattr(truth, name), copy=True) for name in MODALITIES}
        sensor = UGVSensor(self.scene, self.parameters.sensor, self.parameters.random_seed)
        sensor.observe(truth, self.scene.ugv_initial_pose)
        for name in MODALITIES:
            np.testing.assert_array_equal(getattr(truth, name), before[name])

    def test_seeded_noise_is_deterministic(self) -> None:
        noise = SensorNoiseParameters(0.5, 0.01, 0.02, 0.3)
        truth = GroundTruthSimulator(self.scene, TRUTH_CONFIG).current_state
        first = UGVSensor(self.scene, noise, seed=71).observe(truth, self.scene.ugv_initial_pose)
        second = UGVSensor(self.scene, noise, seed=71).observe(truth, self.scene.ugv_initial_pose)
        self.assertEqual(first, second)

    def test_invalid_ugv_waypoint_is_rejected(self) -> None:
        ugv = UGV(self.scene)
        with self.assertRaisesRegex(ValueError, "not traversable"):
            ugv.move_to(Pose2D(0.1, 4.0, 0.0))

    def test_prior_and_posterior_are_separate_objects(self) -> None:
        prior = self._prior_belief()
        cell = self.scene.navigation.metric_to_grid(self.waypoint.pose.x, self.waypoint.pose.y)
        observation = self._observation(
            prior,
            temperature=float(prior.mean.temperature[cell] + 5.0),
        )
        posterior = SpatialAssimilator(self.scene, self.parameters.assimilation).assimilate(prior, observation)
        self.assertIsNot(prior, posterior)
        for name in MODALITIES:
            self.assertFalse(np.shares_memory(getattr(prior.mean, name), getattr(posterior.mean, name)))
            self.assertFalse(
                np.shares_memory(
                    getattr(prior.uncertainty, name),
                    getattr(posterior.uncertainty, name),
                )
            )

    def test_zero_residual_causes_no_mean_correction(self) -> None:
        prior = self._prior_belief()
        posterior = SpatialAssimilator(self.scene, self.parameters.assimilation).assimilate(
            prior,
            self._observation(prior),
        )
        for name in MODALITIES:
            np.testing.assert_array_equal(getattr(prior.mean, name), getattr(posterior.mean, name))

    def test_positive_residual_changes_posterior_in_correct_direction(self) -> None:
        prior = self._prior_belief()
        cell = self.scene.navigation.metric_to_grid(self.waypoint.pose.x, self.waypoint.pose.y)
        observation = Observation(
            timestamp=prior.timestamp,
            position=(self.waypoint.pose.x, self.waypoint.pose.y),
            temperature=float(prior.mean.temperature[cell] + 10.0),
            source="test_ugv",
        )
        posterior = SpatialAssimilator(self.scene, self.parameters.assimilation).assimilate(prior, observation)
        self.assertGreater(posterior.mean.temperature[cell], prior.mean.temperature[cell])
        for name in ("co", "smoke", "visibility"):
            np.testing.assert_array_equal(getattr(prior.mean, name), getattr(posterior.mean, name))

    def test_correction_decays_with_distance(self) -> None:
        prior = self._prior_belief()
        cell = self.scene.navigation.metric_to_grid(self.waypoint.pose.x, self.waypoint.pose.y)
        neighbor = (cell[0], cell[1] + 1)
        observation = self._observation(
            prior,
            temperature=float(prior.mean.temperature[cell] + 10.0),
        )
        posterior = SpatialAssimilator(self.scene, self.parameters.assimilation).assimilate(prior, observation)
        local_change = posterior.mean.temperature[cell] - prior.mean.temperature[cell]
        neighbor_change = posterior.mean.temperature[neighbor] - prior.mean.temperature[neighbor]
        self.assertGreater(local_change, neighbor_change)
        self.assertGreater(neighbor_change, 0.0)

    def test_far_cells_remain_unchanged(self) -> None:
        prior = self._prior_belief()
        cell = self.scene.navigation.metric_to_grid(self.waypoint.pose.x, self.waypoint.pose.y)
        far_cell = self.scene.navigation.metric_to_grid(
            self.scene.ugv_initial_pose.x,
            self.scene.ugv_initial_pose.y,
        )
        posterior = SpatialAssimilator(self.scene, self.parameters.assimilation).assimilate(
            prior,
            self._observation(prior, smoke=float(prior.mean.smoke[cell] + 1.0)),
        )
        self.assertEqual(posterior.mean.smoke[far_cell], prior.mean.smoke[far_cell])

    def test_uncertainty_decreases_near_observation(self) -> None:
        prior = self._prior_belief()
        cell = self.scene.navigation.metric_to_grid(self.waypoint.pose.x, self.waypoint.pose.y)
        posterior = SpatialAssimilator(self.scene, self.parameters.assimilation).assimilate(
            prior,
            self._observation(prior),
        )
        for name in MODALITIES:
            self.assertLess(
                getattr(posterior.uncertainty, name)[cell],
                getattr(prior.uncertainty, name)[cell],
            )

    def test_uncertainty_remains_non_negative(self) -> None:
        prior = self._prior_belief()
        posterior = SpatialAssimilator(self.scene, self.parameters.assimilation).assimilate(
            prior,
            self._observation(prior),
        )
        for name in MODALITIES:
            self.assertTrue(np.all(getattr(posterior.uncertainty, name) >= 0.0))

    def test_posterior_observation_point_error_is_lower(self) -> None:
        result = run_assimilation_demo(SCENE_CONFIG, TRUTH_CONFIG, PRIOR_CONFIG, P3_CONFIG)
        for name in MODALITIES:
            self.assertLess(result["posterior_error"][name], result["prior_error"][name])
            self.assertLess(
                result["uncertainty_after"][name],
                result["uncertainty_before"][name],
            )

    def test_repeated_assimilation_demo_is_deterministic(self) -> None:
        first = run_assimilation_demo(SCENE_CONFIG, TRUTH_CONFIG, PRIOR_CONFIG, P3_CONFIG)
        second = run_assimilation_demo(SCENE_CONFIG, TRUTH_CONFIG, PRIOR_CONFIG, P3_CONFIG)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()

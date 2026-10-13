from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from indoor_fire_world.hazard import (
    HazardResult,
    PhysicalHazardModel,
    UncertaintyAggregationModel,
    UncertaintyAggregationResult,
    load_hazard_parameters,
)
from indoor_fire_world.p3_config import load_p3_parameters
from indoor_fire_world.planning import PlanningCostModel, load_planning_cost_parameters
from indoor_fire_world.scene import load_scene
from indoor_fire_world.sensing import Observation
from indoor_fire_world.simulation import run_p4_demo
from indoor_fire_world.world_model import (
    BeliefMean,
    PhysicsPrior,
    SpatialAssimilator,
    UncertaintyField,
    initialize_belief,
)


SCENE_CONFIG = REPO_ROOT / "configs" / "scene_w1_4f.yaml"
TRUTH_CONFIG = REPO_ROOT / "configs" / "fire_truth.yaml"
PRIOR_CONFIG = REPO_ROOT / "configs" / "fire_prior.yaml"
P3_CONFIG = REPO_ROOT / "configs" / "p3_assimilation.yaml"
HAZARD_CONFIG = REPO_ROOT / "configs" / "hazard.yaml"
PLANNING_CONFIG = REPO_ROOT / "configs" / "planning_cost.yaml"


class P4HazardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = load_scene(SCENE_CONFIG)
        self.hazard_parameters = load_hazard_parameters(HAZARD_CONFIG)
        self.hazard_model = PhysicalHazardModel(self.hazard_parameters)
        self.uncertainty_model = UncertaintyAggregationModel(self.hazard_parameters)
        self.cost_model = PlanningCostModel(load_planning_cost_parameters(PLANNING_CONFIG))

    @staticmethod
    def _mean(
        temperature: np.ndarray,
        co: np.ndarray | None = None,
        smoke: np.ndarray | None = None,
        visibility: np.ndarray | None = None,
    ) -> BeliefMean:
        shape = temperature.shape
        return BeliefMean(
            temperature=temperature,
            co=np.zeros(shape) if co is None else co,
            smoke=np.zeros(shape) if smoke is None else smoke,
            visibility=np.full(shape, 30.0) if visibility is None else visibility,
        )

    def _constant_results(
        self,
        hazard_value: float,
        uncertainty_value: float,
    ) -> tuple[HazardResult, UncertaintyAggregationResult]:
        shape = (self.scene.occupancy.geometry.rows, self.scene.occupancy.geometry.cols)
        hazard = np.full(shape, hazard_value)
        uncertainty = np.full(shape, uncertainty_value)
        hazard_result = HazardResult(
            field=hazard,
            dominant_modality=np.full(shape, "temperature"),
            temperature=hazard,
            co=np.zeros(shape),
            smoke=np.zeros(shape),
            visibility=np.zeros(shape),
        )
        uncertainty_result = UncertaintyAggregationResult(
            field=uncertainty,
            temperature=uncertainty,
            co=np.zeros(shape),
            smoke=np.zeros(shape),
            visibility=np.zeros(shape),
        )
        return hazard_result, uncertainty_result

    def test_temperature_hazard_monotonicity(self) -> None:
        values = np.array([[20.0, 30.0, 55.0, 80.0, 100.0]])
        severity = self.hazard_model.compute(self._mean(values)).temperature[0]
        self.assertTrue(np.all(np.diff(severity) >= 0.0))
        self.assertEqual(severity[0], 0.0)
        self.assertEqual(severity[-1], 1.0)

    def test_co_hazard_monotonicity(self) -> None:
        values = np.array([[0.0, 0.05, 0.175, 0.30, 0.50]])
        severity = self.hazard_model.compute(
            self._mean(np.full(values.shape, 22.0), co=values)
        ).co[0]
        self.assertTrue(np.all(np.diff(severity) >= 0.0))
        self.assertEqual(severity[0], 0.0)
        self.assertEqual(severity[-1], 1.0)

    def test_smoke_hazard_monotonicity(self) -> None:
        values = np.array([[0.0, 0.10, 0.55, 1.0, 2.0]])
        severity = self.hazard_model.compute(
            self._mean(np.full(values.shape, 22.0), smoke=values)
        ).smoke[0]
        self.assertTrue(np.all(np.diff(severity) >= 0.0))
        self.assertEqual(severity[0], 0.0)
        self.assertEqual(severity[-1], 1.0)

    def test_visibility_hazard_reverse_monotonicity(self) -> None:
        values = np.array([[2.0, 5.0, 12.5, 20.0, 30.0]])
        severity = self.hazard_model.compute(
            self._mean(np.full(values.shape, 22.0), visibility=values)
        ).visibility[0]
        self.assertTrue(np.all(np.diff(severity) <= 0.0))
        self.assertEqual(severity[0], 1.0)
        self.assertEqual(severity[-1], 0.0)

    def test_all_normalized_hazards_remain_in_unit_interval(self) -> None:
        mean = BeliefMean(
            temperature=np.array([[-100.0, 1000.0]]),
            co=np.array([[-1.0, 10.0]]),
            smoke=np.array([[-1.0, 10.0]]),
            visibility=np.array([[-10.0, 1000.0]]),
        )
        result = self.hazard_model.compute(mean)
        for name in ("field", "temperature", "co", "smoke", "visibility"):
            values = getattr(result, name)
            self.assertTrue(np.all((values >= 0.0) & (values <= 1.0)))

    def test_dominant_modality_is_correct(self) -> None:
        result = self.hazard_model.compute(
            BeliefMean(
                temperature=np.array([[22.0]]),
                co=np.array([[0.0]]),
                smoke=np.array([[1.0]]),
                visibility=np.array([[30.0]]),
            )
        )
        self.assertEqual(result.dominant_modality[0, 0], "smoke")
        self.assertEqual(result.field[0, 0], 1.0)

    def test_uncertainty_modalities_are_normalized_before_aggregation(self) -> None:
        refs = self.hazard_parameters
        uncertainty = UncertaintyField(
            temperature=np.array([[12.5, 12.5]]),
            co=np.array([[0.0025, 0.005]]),
            smoke=np.array([[0.04, 0.02]]),
            visibility=np.array([[0.0, 12.5]]),
            meaning="test variance proxies",
        )
        result = self.uncertainty_model.compute(uncertainty)
        self.assertEqual(result.field[0, 0], 1.0)
        self.assertEqual(result.field[0, 1], 0.5)
        self.assertEqual(result.temperature[0, 1], 0.5)
        self.assertEqual(result.co[0, 1], 0.5)
        self.assertEqual(refs.smoke_uncertainty.value, 0.04)

    def test_uncertainty_field_remains_in_unit_interval(self) -> None:
        uncertainty = UncertaintyField(
            temperature=np.array([[0.0, 250.0]]),
            co=np.array([[0.0, 1.0]]),
            smoke=np.array([[0.0, 4.0]]),
            visibility=np.array([[0.0, 250.0]]),
            meaning="test variance proxies",
        )
        result = self.uncertainty_model.compute(uncertainty)
        for name in ("field", "temperature", "co", "smoke", "visibility"):
            values = getattr(result, name)
            self.assertTrue(np.all((values >= 0.0) & (values <= 1.0)))

    def test_p3_local_reduction_propagates_to_p4_uncertainty(self) -> None:
        p3 = load_p3_parameters(P3_CONFIG)
        prior_model = PhysicsPrior(self.scene, PRIOR_CONFIG)
        prior = initialize_belief(prior_model.current_state, p3.initial_uncertainty)
        waypoint = p3.waypoint(p3.demo_waypoint_id)
        cell = self.scene.navigation.metric_to_grid(waypoint.pose.x, waypoint.pose.y)
        observation = Observation(
            timestamp=prior.timestamp,
            position=(waypoint.pose.x, waypoint.pose.y),
            temperature=float(prior.mean.temperature[cell]),
            co=float(prior.mean.co[cell]),
            smoke=float(prior.mean.smoke[cell]),
            visibility=float(prior.mean.visibility[cell]),
            source="test_ugv",
        )
        posterior = SpatialAssimilator(self.scene, p3.assimilation).assimilate(prior, observation)
        before = self.uncertainty_model.compute(prior.uncertainty)
        after = self.uncertainty_model.compute(posterior.uncertainty)
        self.assertLess(after.field[cell], before.field[cell])

    def test_planning_cost_increases_with_hazard(self) -> None:
        low_hazard, uncertainty = self._constant_results(0.1, 0.2)
        high_hazard, _ = self._constant_results(0.2, 0.2)
        low = self.cost_model.compute(
            low_hazard.field,
            uncertainty.field,
            self.scene.navigation,
        )
        high = self.cost_model.compute(
            high_hazard.field,
            uncertainty.field,
            self.scene.navigation,
        )
        cell = self.scene.navigation.metric_to_grid(1.0, 1.0)
        self.assertGreater(high.field[cell], low.field[cell])

    def test_planning_cost_increases_with_uncertainty(self) -> None:
        hazard, low_uncertainty = self._constant_results(0.1, 0.2)
        _, high_uncertainty = self._constant_results(0.1, 0.6)
        low = self.cost_model.compute(hazard, low_uncertainty, self.scene.navigation)
        high = self.cost_model.compute(hazard, high_uncertainty, self.scene.navigation)
        cell = self.scene.navigation.metric_to_grid(1.0, 1.0)
        self.assertGreater(high.field[cell], low.field[cell])

    def test_obstacles_are_planning_forbidden(self) -> None:
        hazard, uncertainty = self._constant_results(0.0, 0.0)
        result = self.cost_model.compute(hazard, uncertainty, self.scene.navigation)
        wall_cell = self.scene.navigation.metric_to_grid(0.1, 4.0)
        self.assertTrue(result.obstacle_forbidden[wall_cell])
        self.assertTrue(result.forbidden[wall_cell])
        self.assertTrue(np.isinf(result.field[wall_cell]))

    def test_untenable_hazard_cells_are_planning_forbidden(self) -> None:
        hazard, uncertainty = self._constant_results(0.0, 0.0)
        cell = self.scene.navigation.metric_to_grid(1.0, 1.0)
        field = np.array(hazard.field, copy=True)
        field[cell] = self.cost_model.parameters.untenable_hazard_threshold
        hazardous = HazardResult(
            field=field,
            dominant_modality=hazard.dominant_modality,
            temperature=field,
            co=hazard.co,
            smoke=hazard.smoke,
            visibility=hazard.visibility,
        )
        result = self.cost_model.compute(hazardous, uncertainty, self.scene.navigation)
        self.assertTrue(result.hazard_forbidden[cell])
        self.assertTrue(result.forbidden[cell])
        self.assertEqual(hazardous.field[cell], self.cost_model.parameters.untenable_hazard_threshold)

    def test_hazard_and_uncertainty_remain_separate_fields(self) -> None:
        mean = self._mean(np.array([[40.0]]))
        uncertainty = UncertaintyField(
            temperature=np.array([[25.0]]),
            co=np.array([[0.01]]),
            smoke=np.array([[0.04]]),
            visibility=np.array([[25.0]]),
            meaning="test variance proxies",
        )
        hazard_result = self.hazard_model.compute(mean)
        uncertainty_result = self.uncertainty_model.compute(uncertainty)
        self.assertFalse(np.shares_memory(hazard_result.field, uncertainty_result.field))
        self.assertNotEqual(hazard_result.field[0, 0], uncertainty_result.field[0, 0])

    def test_repeated_p4_demo_output_is_deterministic(self) -> None:
        args = (
            SCENE_CONFIG,
            TRUTH_CONFIG,
            PRIOR_CONFIG,
            P3_CONFIG,
            HAZARD_CONFIG,
            PLANNING_CONFIG,
        )
        first = run_p4_demo(*args)
        second = run_p4_demo(*args)
        self.assertEqual(first.summary, second.summary)
        self.assertEqual(first.arrays.keys(), second.arrays.keys())
        for name in first.arrays:
            np.testing.assert_array_equal(first.arrays[name], second.arrays[name])


if __name__ == "__main__":
    unittest.main()

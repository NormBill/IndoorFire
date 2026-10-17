from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from indoor_fire_world.p8_v2_config import load_p8_v2_parameters
from indoor_fire_world.planning import DirectedRoutePlanner, PlanningCostParameters
from indoor_fire_world.simulation.p8_v2 import build_p8_v2_context, run_p8_v2_case


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "p8_v2_preentry_ingress.yaml"


class P8V2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.parameters = load_p8_v2_parameters(CONFIG)
        cls.context = build_p8_v2_context(CONFIG)
        cls.case = run_p8_v2_case(
            cls.context,
            "enclosure_breakout",
            100.0,
            "geometry_route_high",
        )

    def test_fixed_experiment_matrix_contract(self) -> None:
        self.assertEqual(len(self.parameters.scenarios), 3)
        self.assertEqual(len(self.parameters.observation_times_s), 4)
        self.assertEqual(len(self.parameters.observation_points), 4)
        self.assertEqual(self.parameters.forecast_times_s, (300.0, 360.0, 420.0, 480.0))
        self.assertEqual([item.id for item in self.parameters.exposure_levels], ["minimal", "elevated", "high", "critical", "extreme"])

    def test_candidate_ensemble_is_frozen_and_small(self) -> None:
        self.assertEqual(len(self.context.candidates), 12)
        self.assertEqual(len({item.id for item in self.context.candidates}), 12)

    def test_firefighter_ingress_can_reverse_on_physical_corridor(self) -> None:
        shape = (self.context.scene.occupancy.geometry.rows, self.context.scene.occupancy.geometry.cols)
        zeros = np.zeros(shape)
        forbidden = np.asarray(self.context.scene.occupancy.cells) != 0
        cost = np.ones(shape)
        cost[forbidden] = np.inf
        planner = DirectedRoutePlanner(PlanningCostParameters(1.0, 0.0, 0.0, 0.95), self.context.graph)
        route = planner.plan(cost, forbidden, "exit_center_stair", "center_lower_left", self.context.scene.navigation, zeros, zeros, allow_reverse=True)
        self.assertTrue(route.feasible)
        self.assertTrue(self.context.graph.path_is_corridor_conformant(route.path))
        self.assertFalse(self.context.graph.path_is_directed(route.path))

    def test_reference_is_not_mislabeled_as_fds(self) -> None:
        self.assertEqual(self.case.summary["reference_semantics"], "reduced_order_synthetic_fixture_not_fds")

    def test_posterior_weights_are_normalized(self) -> None:
        weights = self.case.summary["posterior_weights"]
        self.assertEqual(len(weights), 12)
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_forecast_metrics_include_required_baselines_and_ablation(self) -> None:
        metrics = self.case.summary["aggregate_normalized_forecast_error"]
        self.assertIn("prior", metrics)
        self.assertIn("posterior", metrics)
        self.assertIn("local_gaussian_baseline", metrics)
        self.assertIn("hrr_only_ablation", metrics)
        self.assertIn("planning_ablation_without_uncertainty_penalty", self.case.summary)

    def test_risk_map_classifies_every_registered_grid_cell(self) -> None:
        expected_shape = (50, 160)
        for time_s in self.parameters.forecast_times_s:
            key = f"t{time_s:g}"
            grid = self.case.arrays[f"posterior_risk_level_{key}"]
            self.assertEqual(grid.shape, expected_shape)
            self.assertTrue(np.all((grid >= -1) & (grid < len(self.parameters.exposure_levels))))
            counts = self.case.summary["risk_grid"]["counts_by_forecast_time"][key]
            self.assertEqual(sum(counts.values()), expected_shape[0] * expected_shape[1])

    def test_counterfactual_uses_one_posterior_forecast_sequence(self) -> None:
        counterfactual = self.case.summary["posterior_counterfactual"]
        self.assertEqual(counterfactual["definition"], "delta_j = J_post(prior_route) - J_post(updated_route)")
        self.assertEqual(counterfactual["field_identity"], "same_posterior_forecast_exposure_sequence")

    def test_case_repetition_is_deterministic(self) -> None:
        repeated = run_p8_v2_case(self.context, "enclosure_breakout", 100.0, "geometry_route_high")
        self.assertEqual(self.case.summary, repeated.summary)
        for name in self.case.arrays:
            np.testing.assert_array_equal(self.case.arrays[name], repeated.arrays[name])


if __name__ == "__main__":
    unittest.main()

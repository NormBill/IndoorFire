from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from indoor_fire_world.simulation import run_p6_closed_loop_demo


P6_CONFIG = REPO_ROOT / "configs" / "p6_closed_loop.yaml"
MODALITIES = ("temperature", "co", "smoke", "visibility")


class P6ClosedLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.first = run_p6_closed_loop_demo(P6_CONFIG)
        cls.second = run_p6_closed_loop_demo(P6_CONFIG)

    def test_controlled_scenario_is_deterministic(self) -> None:
        self.assertEqual(self.first.summary, self.second.summary)
        for name in self.first.arrays:
            np.testing.assert_array_equal(
                self.first.arrays[name],
                self.second.arrays[name],
            )

    def test_parameters_remain_frozen_during_experiment(self) -> None:
        summary = self.first.summary
        self.assertEqual(
            summary["parameter_fingerprint_before"],
            summary["parameter_fingerprint_after"],
        )
        self.assertTrue(summary["acceptance"]["parameters_frozen"])

    def test_prior_route_is_deterministic(self) -> None:
        first = self.first.summary["initial_route_a"]["mission_path_hash"]
        second = self.second.summary["initial_route_a"]["mission_path_hash"]
        self.assertIsNotNone(first)
        self.assertEqual(first, second)

    def test_truth_observation_differs_from_prior_near_route_a(self) -> None:
        summary = self.first.summary
        self.assertTrue(summary["acceptance"]["observation_on_initial_route_a"])
        self.assertTrue(summary["acceptance"]["observation_differs_from_prior"])
        self.assertTrue(
            any(
                summary["ugv_observation"]["prior_error"][name] > 0.0
                for name in MODALITIES
            )
        )

    def test_assimilation_reduces_local_prediction_error(self) -> None:
        observation = self.first.summary["ugv_observation"]
        for name in MODALITIES:
            self.assertLess(
                observation["posterior_error"][name],
                observation["prior_error"][name],
            )

    def test_uncertainty_decreases_locally(self) -> None:
        observation = self.first.summary["ugv_observation"]
        for name in MODALITIES:
            self.assertLess(
                observation["uncertainty_after"][name],
                observation["uncertainty_before"][name],
            )
        self.assertTrue(
            self.first.summary["acceptance"]["local_uncertainty_decreases"]
        )

    def test_posterior_hazard_increases_consistently(self) -> None:
        observation_cell = next(
            record
            for record in self.first.summary["prior_posterior_local_fields"]
            if record["is_observation_cell"]
        )
        self.assertTrue(observation_cell["on_original_route_a"])
        self.assertGreater(
            observation_cell["posterior"]["hazard"],
            observation_cell["prior"]["hazard"],
        )

    def test_replanning_uses_posterior_cost_field(self) -> None:
        route = self.first.summary["posterior_route_b"]["entry_route"]
        path = [(cell["row"], cell["col"]) for cell in route["path"]]
        accumulated = sum(
            float(self.first.arrays["posterior_planning_cost"][cell])
            for cell in path[1:]
        ) * self.first.summary["grid_resolution_m"]
        self.assertAlmostEqual(route["total_cost"], accumulated)
        self.assertTrue(self.first.summary["acceptance"]["planning_field_changes"])

    def test_original_route_a_is_reevaluated_under_posterior(self) -> None:
        counterfactual = self.first.summary["counterfactual"]
        self.assertIn("route_a_under_prior", counterfactual)
        self.assertIn("route_a_under_posterior", counterfactual)
        self.assertEqual(
            counterfactual["route_a_under_prior"]["mission_path_hash"],
            counterfactual["route_a_under_posterior"]["mission_path_hash"],
        )

    def test_controlled_scenario_changes_route(self) -> None:
        summary = self.first.summary
        self.assertTrue(summary["initial_route_a"]["feasible"])
        self.assertTrue(summary["posterior_route_b"]["feasible"])
        self.assertTrue(summary["route_changed"])
        self.assertTrue(summary["acceptance"]["controlled_scenario_pass"])
        self.assertNotEqual(
            summary["initial_route_a"]["mission_path_hash"],
            summary["posterior_route_b"]["mission_path_hash"],
        )

    def test_route_change_is_reproducible(self) -> None:
        self.assertTrue(self.first.summary["route_changed"])
        self.assertEqual(
            self.first.summary["posterior_route_b"]["mission_path_hash"],
            self.second.summary["posterior_route_b"]["mission_path_hash"],
        )
        self.assertEqual(
            self.first.summary["change_reason"],
            self.second.summary["change_reason"],
        )

    def test_no_manual_route_penalty_exists(self) -> None:
        summary = self.first.summary
        provenance = summary["planning_field_provenance"]
        self.assertFalse(provenance["manual_route_penalty"])
        weights = summary["scenario_parameters"]["planning_weights"]
        expected = (
            weights["base_cost"]
            + weights["hazard_weight"] * self.first.arrays["posterior_hazard"]
            + weights["uncertainty_weight"]
            * self.first.arrays["posterior_uncertainty"]
        )
        allowed = ~self.first.arrays["posterior_forbidden"]
        np.testing.assert_allclose(
            self.first.arrays["posterior_planning_cost"][allowed],
            expected[allowed],
        )


if __name__ == "__main__":
    unittest.main()

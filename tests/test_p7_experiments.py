from __future__ import annotations

from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from indoor_fire_world.experiments import (
    artifact_directory_name,
    load_experiment_suite,
    run_experiment,
)


SUITE_CONFIG = REPO_ROOT / "configs" / "experiments" / "p7_suite.yaml"


class P7ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.suite = load_experiment_suite(SUITE_CONFIG)
        cls.by_id = {
            experiment.scenario_id: experiment
            for experiment in cls.suite.experiments
        }
        cls.baseline = run_experiment(cls.suite, cls.by_id["baseline_v1"])
        cls.baseline_repeat = run_experiment(cls.suite, cls.by_id["baseline_v1"])
        cls.no_assimilation = run_experiment(
            cls.suite,
            cls.by_id["no_assimilation"],
        )
        cls.no_observation = run_experiment(cls.suite, cls.by_id["no_observation"])
        cls.oracle = run_experiment(cls.suite, cls.by_id["perfect_information"])
        cls.sensor_low = run_experiment(cls.suite, cls.by_id["sensor_low"])
        cls.sensor_low_repeat = run_experiment(cls.suite, cls.by_id["sensor_low"])

    def test_experiment_config_is_reproducible(self) -> None:
        repeated = load_experiment_suite(SUITE_CONFIG)
        self.assertEqual(self.suite, repeated)
        self.assertEqual(
            self.baseline.config_snapshot,
            self.baseline_repeat.config_snapshot,
        )

    def test_artifact_output_names_are_unique(self) -> None:
        names = [
            artifact_directory_name(self.suite, experiment)
            for experiment in self.suite.experiments
        ]
        self.assertEqual(len(names), len(set(names)))

    def test_no_assimilation_leaves_belief_unchanged(self) -> None:
        observation = self.no_assimilation.summary["ugv_observation"]
        self.assertIsNotNone(observation["observed"])
        for name in ("temperature", "co", "smoke", "visibility"):
            self.assertEqual(
                observation["prior_prediction"][name],
                observation["posterior_prediction"][name],
            )
        self.assertEqual(
            self.no_assimilation.route_metrics["initial"]["path_hash"],
            self.no_assimilation.route_metrics["posterior"]["path_hash"],
        )

    def test_no_observation_uses_physics_prior_only(self) -> None:
        self.assertIsNone(self.no_observation.summary["ugv_observation"]["observed"])
        self.assertEqual(
            self.no_observation.route_metrics["initial"]["path_hash"],
            self.no_observation.route_metrics["posterior"]["path_hash"],
        )
        self.assertFalse(self.no_observation.route_metrics["route_changed"])

    def test_ground_truth_never_leaks_into_normal_planner(self) -> None:
        provenance = self.baseline.summary["planning_field_provenance"]
        self.assertEqual(provenance["planner_field_source"], "posterior_belief")
        self.assertFalse(provenance["manual_route_penalty"])

    def test_oracle_mode_explicitly_enables_truth(self) -> None:
        provenance = self.oracle.summary["planning_field_provenance"]
        self.assertTrue(self.oracle.config_snapshot["experiment"]["oracle_planning"])
        self.assertEqual(provenance["planner_field_source"], "ground_truth_oracle")

    def test_metrics_are_computed_from_fields(self) -> None:
        output = self.baseline
        summary_route = output.summary["posterior_route_b"]
        expected_length = (
            summary_route["entry_route"]["geometric_length_m"]
            + summary_route["retreat_route"]["geometric_length_m"]
        )
        self.assertAlmostEqual(
            output.route_metrics["posterior"]["length_m"],
            expected_length,
        )
        old_cost = output.route_metrics["old_route_posterior_cost"]
        new_cost = output.route_metrics["new_route_posterior_cost"]
        self.assertAlmostEqual(
            output.route_metrics["cost_improvement"],
            old_cost - new_cost,
        )

    def test_same_seed_and_config_produce_identical_results(self) -> None:
        self.assertEqual(self.baseline.row, self.baseline_repeat.row)
        self.assertEqual(self.baseline.path_hashes, self.baseline_repeat.path_hashes)
        self.assertEqual(
            self.baseline.field_metrics,
            self.baseline_repeat.field_metrics,
        )

    def test_mismatch_sweep_changes_only_declared_truth_parameters(self) -> None:
        sweep = [
            self.by_id["mismatch_low"],
            self.by_id["mismatch_medium"],
            self.by_id["mismatch_high"],
        ]
        for experiment in sweep:
            self.assertEqual(experiment.ablation_type, "model_mismatch_sweep")
            self.assertEqual(experiment.seed, 20261013)
            self.assertEqual(set(experiment.overrides) - {"truth"}, set())
            self.assertTrue(experiment.observation_enabled)
            self.assertTrue(experiment.assimilation_enabled)
            self.assertFalse(experiment.oracle_planning)

    def test_sensor_noise_sweep_is_deterministic_by_seed(self) -> None:
        self.assertEqual(self.sensor_low.row, self.sensor_low_repeat.row)
        self.assertEqual(
            self.sensor_low.path_hashes,
            self.sensor_low_repeat.path_hashes,
        )

    def test_p6_controlled_result_is_preserved(self) -> None:
        summary = self.baseline.summary
        counterfactual = summary["counterfactual"]
        a_prior = counterfactual["route_a_under_prior"]["aggregate"]["total_cost"]
        b_prior = counterfactual["route_b_under_prior"]["aggregate"]["total_cost"]
        a_posterior = counterfactual["route_a_under_posterior"]["aggregate"][
            "total_cost"
        ]
        b_posterior = counterfactual["route_b_under_posterior"]["aggregate"][
            "total_cost"
        ]
        self.assertAlmostEqual(a_prior, b_prior)
        self.assertLess(b_posterior, a_posterior)
        self.assertTrue(summary["route_changed"])
        self.assertTrue(summary["acceptance"]["controlled_scenario_pass"])


if __name__ == "__main__":
    unittest.main()

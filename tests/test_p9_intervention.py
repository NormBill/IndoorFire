from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from indoor_fire_world.fire import ReducedOrderFireSimulator, load_fire_parameters
from indoor_fire_world.intervention import (
    ActionConditionedFireSimulator,
    InterventionAction,
)
from indoor_fire_world.p9_config import load_p9_parameters
from indoor_fire_world.scene import load_scene
from indoor_fire_world.simulation.p9_intervention import (
    build_p9_context,
    run_p9_case,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "p9_postarrival_intervention.yaml"


class P9InterventionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.parameters = load_p9_parameters(CONFIG)
        cls.context = build_p9_context(CONFIG)
        cls.case = run_p9_case(
            cls.context, "enclosure_breakout", "geometry_route_high"
        )

    def test_phase_contract_forbids_pre_handoff_intervention(self) -> None:
        self.assertEqual(self.parameters.firefighter_availability_time_s, 300.0)
        self.assertEqual(self.parameters.evaluation_times_s, (300.0, 360.0, 420.0, 480.0))
        for action in self.parameters.actions:
            if action.activation_time_s is not None:
                self.assertGreaterEqual(action.activation_time_s, 300.0)
            self.assertEqual(action.source_multiplier(299.999), 1.0)
        phase = self.case.summary["phase_contract"]
        self.assertEqual(phase["pre_entry_observation_s"], [0.0, 300.0])
        self.assertEqual(phase["firefighter_intervention_s"], [300.0, 480.0])
        self.assertTrue(phase["intervention_before_handoff_forbidden"])
        self.assertIn("not measured W1 arrival", phase["availability_semantics"])

    def test_no_intervention_is_numerically_identical_to_frozen_p8_solver(self) -> None:
        scene = load_scene(self.context.p8.parameters.scene_config)
        fire = load_fire_parameters(self.context.p8.parameters.prior_config)
        action = next(
            item for item in self.parameters.actions if item.id == "no_intervention"
        )
        baseline = ReducedOrderFireSimulator(scene, fire)
        conditioned = ActionConditionedFireSimulator(scene, fire, action)
        for target in (300.0, 480.0):
            while baseline.current_state.timestamp < target - 1e-12:
                dt = min(fire.time_step_s, target - baseline.current_state.timestamp)
                baseline.step(dt)
                conditioned.step(dt)
            np.testing.assert_array_equal(
                baseline.current_state.temperature, conditioned.current_state.temperature
            )
            np.testing.assert_array_equal(
                baseline.current_state.smoke, conditioned.current_state.smoke
            )
            np.testing.assert_array_equal(
                baseline.current_state.co, conditioned.current_state.co
            )

    def test_all_counterfactual_branches_share_the_t300_state(self) -> None:
        reference = self.case.arrays["predicted_exposure_no_intervention_t300"]
        for action in self.parameters.actions:
            np.testing.assert_array_equal(
                reference,
                self.case.arrays[f"predicted_exposure_{action.id}_t300"],
            )

    def test_suppression_reduces_future_source_and_reference_risk(self) -> None:
        no_action = self.case.summary["action_results"]["no_intervention"]
        early = self.case.summary["action_results"]["localized_suppression_t315"]
        self.assertEqual(no_action["source_multiplier_t480"], 1.0)
        self.assertLess(early["source_multiplier_t480"], 0.16)
        self.assertLess(early["oracle_objective"], no_action["oracle_objective"])
        self.assertGreater(
            self.case.summary["reference_improvement_over_no_intervention"], 0.0
        )

    def test_post_action_update_uses_the_executed_action_bank(self) -> None:
        for result in self.case.summary["action_results"].values():
            update = result["post_action_update"]
            self.assertTrue(update["action_conditioned"])
            self.assertEqual(update["timestamp_s"], 420.0)
            self.assertGreaterEqual(update["effective_sample_size"], 8.0 - 1e-9)

    def test_intervention_access_is_separate_from_counterfactual_effect(self) -> None:
        access = self.case.summary["intervention_access"]
        self.assertEqual(access["staging_node"], "center_lower_left")
        self.assertIn("not the burning microwave cell", access["staging_semantics"].lower())
        self.assertTrue(access["diagnostic_geometry_only"]["feasible"])
        if not access["strict"]["feasible"]:
            self.assertEqual(
                self.case.summary["executed_action_under_strict_access"],
                "no_intervention",
            )

    def test_repeated_case_is_deterministic(self) -> None:
        repeated = run_p9_case(
            self.context, "enclosure_breakout", "geometry_route_high"
        )
        self.assertEqual(self.case.summary, repeated.summary)
        for key in self.case.arrays:
            np.testing.assert_array_equal(self.case.arrays[key], repeated.arrays[key])


if __name__ == "__main__":
    unittest.main()


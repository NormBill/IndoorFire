from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from indoor_fire_world.hazard import HazardResult, UncertaintyAggregationResult
from indoor_fire_world.p8_config import ObservationOffset, load_p8_parameters
from indoor_fire_world.planning import (
    ExitRouteEvaluation,
    GridAStarPlanner,
    PlanningCostModel,
    PlanningCostParameters,
    RetreatPlan,
    RouteResult,
)
from indoor_fire_world.scene import Pose2D, load_scene
from indoor_fire_world.scene.models import (
    GridGeometry,
    NavigationLayer,
    Occupancy,
    OccupancyLayer,
    Scene,
    SemanticLayer,
)
from indoor_fire_world.simulation.p8_snapshot import (
    DecisionFields,
    _counterfactual,
    freeze_observation_points,
    run_p8_snapshot_case,
)
from indoor_fire_world.experiments.p8_runner import run_p8_suite


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "p8_registered_closed_loop.yaml"


class P8SnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.parameters = load_p8_parameters(CONFIG)
        cls.scene = load_scene(cls.parameters.scene_config)
        cls.suite = run_p8_suite(CONFIG)
        cls.primary = cls.suite.primary.summary

    def test_research_horizon_is_explicitly_not_an_arrival_claim(self) -> None:
        horizon = self.primary["research_horizon"]
        self.assertEqual(horizon["pre_response_research_horizon_s"], 300.0)
        self.assertIn("5-min fire-station layout principle", horizon["definition"])
        self.assertEqual(
            set(horizon["explicit_exclusions"]),
            {
                "not_w1_measured_response_time",
                "not_actual_fire_engine_arrival_time",
                "not_firefighter_interior_deployment_completion_time",
            },
        )

    def test_observation_points_are_frozen_geometry_offsets(self) -> None:
        points = freeze_observation_points(self.scene, self.parameters)
        expected = {
            "geometry_offset_west_far": (18.875, 7.625),
            "geometry_offset_west_mid": (19.125, 7.625),
            "geometry_offset_west_near": (19.375, 7.625),
            "geometry_offset_east_control": (21.0, 8.0),
        }
        self.assertEqual({point.id for point in points}, set(expected))
        for point in points:
            self.assertTrue(point.valid)
            self.assertEqual((point.pose.x, point.pose.y), expected[point.id])
        self.assertFalse(
            self.primary["provenance"]["truth_used_for_observation_point_selection"]
        )
        self.assertEqual(
            self.primary["observation_point"]["selection_provenance"],
            "registered_geometry_only",
        )

    def test_matrix_and_primary_controls_are_complete(self) -> None:
        self.assertEqual(self.suite.summary["matrix_shape"]["normal_cases"], 16)
        self.assertEqual(len(self.suite.rows), 19)
        self.assertEqual(
            set(self.suite.summary["primary_controls"]),
            {
                "normal_fusion",
                "no_observation",
                "observation_without_fusion",
                "truth_oracle",
            },
        )

    def test_primary_case_is_retreat_only_posterior_improvement(self) -> None:
        self.assertEqual(self.primary["route_outcome"], "same_exit_reroute")
        prior = self.primary["prior_retreat"]
        updated = self.primary["updated_retreat"]
        self.assertEqual(prior["selected_exit"], updated["selected_exit"])
        self.assertNotEqual(prior["route"]["path_hash"], updated["route"]["path_hash"])
        comparison = self.primary["posterior_counterfactual"]
        self.assertEqual(
            comparison["field_identity"],
            "same_effective_posterior_planning_field",
        )
        old_cost = comparison["old_route_under_posterior"]["total_cost"]
        new_cost = comparison["updated_route_under_posterior"]["total_cost"]
        self.assertAlmostEqual(comparison["delta_j"], old_cost - new_cost)
        self.assertGreater(comparison["delta_j"], 0.0)

    def test_snapshot_clock_does_not_advance_during_approach_or_replanning(self) -> None:
        clocks = self.primary["simulator_timestamps"]
        self.assertEqual(clocks["prior_snapshot_s"], 100.0)
        self.assertEqual(clocks["truth_before_observation_s"], 100.0)
        self.assertEqual(clocks["truth_after_replanning_s"], 100.0)
        self.assertTrue(self.primary["fire_frozen_during_approach_and_comparison"])
        self.assertIn("not a fully time-coupled", self.primary["snapshot_semantics"])

    def test_thresholds_have_independent_configuration_semantics(self) -> None:
        changed = replace(
            self.parameters,
            ugv_operability_threshold=0.70,
            retreat_hazard_threshold=0.85,
        )
        approach = changed.planning_parameters(
            changed.ugv_operability_threshold,
            "UGV approach operability",
        )
        retreat = changed.planning_parameters(
            changed.retreat_hazard_threshold,
            "retreat safety",
        )
        self.assertEqual(approach.untenable_hazard_threshold, 0.70)
        self.assertEqual(retreat.untenable_hazard_threshold, 0.85)
        self.assertEqual(
            set(self.primary["thresholds"]),
            {"ugv_operability_threshold", "retreat_hazard_threshold"},
        )

    def test_no_observation_and_no_fusion_do_not_create_route_change(self) -> None:
        controls = self.suite.summary["primary_controls"]
        for name in ("no_observation", "observation_without_fusion"):
            case = controls[name]
            self.assertEqual(case["route_outcome"], "unchanged")
            self.assertEqual(
                case["prior_retreat"]["route"]["path_hash"],
                case["updated_retreat"]["route"]["path_hash"],
            )
        self.assertIsNone(controls["no_observation"]["observation"]["observed"])
        self.assertIsNotNone(
            controls["observation_without_fusion"]["observation"]["observed"]
        )

    def test_oracle_truth_access_is_explicit(self) -> None:
        oracle = self.suite.summary["primary_controls"]["truth_oracle"]
        self.assertEqual(
            self.primary["provenance"]["planner_field_source"],
            "posterior_belief",
        )
        self.assertEqual(
            oracle["provenance"]["planner_field_source"],
            "ground_truth_oracle",
        )

    def test_infeasible_counterfactual_has_no_finite_delta(self) -> None:
        case = self.suite.summary["normal_cases"][
            "normal-t180-geometry_offset_west_far"
        ]
        comparison = case["posterior_counterfactual"]
        self.assertFalse(comparison["updated_route_posterior_feasible"])
        self.assertIsNone(comparison["delta_j"])
        self.assertIn(
            comparison["improvement_type"],
            {"posterior_replanning_failure", "no_feasible_updated_retreat"},
        )

    def test_old_route_infeasible_reports_feasibility_improvement(self) -> None:
        geometry = GridGeometry(3.0, 3.0, 1.0)
        occupancy = OccupancyLayer(
            geometry,
            tuple(tuple(Occupancy.FREE for _ in range(3)) for _ in range(3)),
        )
        navigation = NavigationLayer(occupancy)
        scene = Scene(
            scene_id="counterfactual",
            name="counterfactual",
            geometry_status="test",
            notes="",
            semantic=SemanticLayer(()),
            occupancy=occupancy,
            navigation=navigation,
            ugv_initial_pose=Pose2D(0.5, 0.5, 0.0),
            raw_config={},
        )
        hazard = np.zeros((3, 3), dtype=np.float64)
        hazard[0, 1] = 1.0
        zeros = np.zeros_like(hazard)
        dominant = np.full((3, 3), "none", dtype="<U11")
        dominant[0, 1] = "temperature"
        hazard_result = HazardResult(
            field=hazard,
            dominant_modality=dominant,
            temperature=hazard,
            co=zeros,
            smoke=zeros,
            visibility=zeros,
        )
        uncertainty = UncertaintyAggregationResult(
            field=zeros,
            temperature=zeros,
            co=zeros,
            smoke=zeros,
            visibility=zeros,
        )
        parameters = PlanningCostParameters(
            base_cost=1.0,
            hazard_weight=1.0,
            uncertainty_weight=0.0,
            untenable_hazard_threshold=0.95,
        )
        cost = PlanningCostModel(parameters).compute(
            hazard_result,
            uncertainty,
            navigation,
        )
        fields = DecisionFields(hazard_result, uncertainty, cost)

        def route(path):
            return RouteResult(
                path=path,
                feasible=True,
                geometric_length_m=float(len(path) - 1),
                total_cost=float(len(path) - 1),
                base_contribution=float(len(path) - 1),
                hazard_contribution=0.0,
                uncertainty_contribution=0.0,
                integrated_hazard=0.0,
                integrated_uncertainty=0.0,
                max_hazard=0.0,
            )

        old_route = route(((0, 0), (0, 1), (0, 2)))
        new_route = route(((0, 0), (1, 0), (1, 1), (1, 2), (0, 2)))
        prior = RetreatPlan(
            old_route,
            "exit",
            (ExitRouteEvaluation("exit", old_route),),
            True,
        )
        updated = RetreatPlan(
            new_route,
            "exit",
            (ExitRouteEvaluation("exit", new_route),),
            True,
        )
        result = _counterfactual(
            prior,
            updated,
            fields,
            GridAStarPlanner(parameters),
            scene,
        )
        self.assertFalse(result["old_route_posterior_feasible"])
        self.assertTrue(result["updated_route_posterior_feasible"])
        self.assertIsNone(result["delta_j"])
        self.assertEqual(result["improvement_type"], "feasibility_improvement")

    def test_repeat_is_deterministic(self) -> None:
        repeated = run_p8_snapshot_case(
            CONFIG,
            self.parameters.primary_fire_time_s,
            self.parameters.primary_observation_id,
        ).summary
        self.assertEqual(
            repeated["updated_retreat"]["route"]["path_hash"],
            self.primary["updated_retreat"]["route"]["path_hash"],
        )
        self.assertEqual(
            repeated["posterior_counterfactual"],
            self.primary["posterior_counterfactual"],
        )

    def test_occupied_observation_is_an_invalid_case(self) -> None:
        parameters = replace(
            self.parameters,
            observation_points=(ObservationOffset("occupied", 1.6, -1.5),),
            primary_observation_id="occupied",
        )
        with patch(
            "indoor_fire_world.simulation.p8_snapshot.load_p8_parameters",
            return_value=parameters,
        ):
            result = run_p8_snapshot_case(CONFIG, 60.0, "occupied")
        self.assertEqual(result.summary["case_status"], "invalid_observation_case")
        self.assertEqual(result.summary["invalid_reason"], "observation_point_occupied")
        self.assertFalse(result.summary["truth_simulator_constructed"])

    def test_invalid_start_and_exit_are_not_route_failures(self) -> None:
        invalid_start_scene = replace(
            self.scene,
            ugv_initial_pose=Pose2D(-1.0, -1.0, 0.0),
        )
        with patch(
            "indoor_fire_world.simulation.p8_snapshot.load_scene",
            return_value=invalid_start_scene,
        ):
            start_result = run_p8_snapshot_case(
                CONFIG,
                60.0,
                self.parameters.primary_observation_id,
            )
        self.assertEqual(start_result.summary["invalid_reason"], "invalid_ugv_start")

        invalid_exit_parameters = replace(
            self.parameters,
            retreat_exit_ids=("missing_exit",),
        )
        with patch(
            "indoor_fire_world.simulation.p8_snapshot.load_p8_parameters",
            return_value=invalid_exit_parameters,
        ):
            exit_result = run_p8_snapshot_case(
                CONFIG,
                60.0,
                self.parameters.primary_observation_id,
            )
        self.assertEqual(exit_result.summary["invalid_reason"], "invalid_configured_exit")

    def test_unreachable_approach_is_an_invalid_case(self) -> None:
        with patch(
            "indoor_fire_world.simulation.p8_snapshot.GridAStarPlanner.plan",
            return_value=RouteResult.rejected("goal_unreachable"),
        ):
            result = run_p8_snapshot_case(
                CONFIG,
                60.0,
                self.parameters.primary_observation_id,
            )
        self.assertEqual(result.summary["case_status"], "invalid_observation_case")
        self.assertEqual(result.summary["invalid_reason"], "ugv_approach_unreachable")
        self.assertFalse(result.summary["truth_simulator_constructed"])


if __name__ == "__main__":
    unittest.main()

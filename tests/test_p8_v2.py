from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from indoor_fire_world.fire import load_fire_parameters
from indoor_fire_world.p8_v2_config import load_p8_v2_parameters
from indoor_fire_world.planning import DirectedRoutePlanner, PlanningCostParameters
from indoor_fire_world.simulation.p8_v2 import (
    attach_dense_prediction_evaluation,
    attach_online_sequential_evaluation,
    build_p8_v2_context,
    run_p8_v2_case,
)
from indoor_fire_world.visualization.dynamic import render_p8_v2_dynamic_risk_html
from indoor_fire_world.visualization.risk_surface import complete_blocked_risk_levels
from indoor_fire_world.visualization.svg import (
    render_p8_v2_categorical_risk_svg,
    render_p8_v2_online_pre_entry_timeseries_svg,
    render_p8_v2_online_risk_map_svg,
    render_p8_v2_exposure_evolution_svg,
    render_p8_v2_grid_diagnostic_svg,
    render_p8_v2_planning_envelope_svg,
    render_p8_v2_posterior_reference_comparison_svg,
    render_p8_v2_pre_entry_keyframes_svg,
    render_p8_v2_synthetic_reference_risk_svg,
    render_p8_v2_t300_three_entrance_ingress_svg,
)


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
        cls.dense_case = run_p8_v2_case(
            cls.context,
            "enclosure_breakout",
            100.0,
            "geometry_route_high",
        )
        cls.dense_rows = attach_dense_prediction_evaluation(
            cls.context,
            {"primary": cls.dense_case},
            "primary",
        )
        cls.online_rows = attach_online_sequential_evaluation(
            cls.context,
            cls.dense_case,
        )

    def test_fixed_experiment_matrix_contract(self) -> None:
        self.assertEqual(len(self.parameters.scenarios), 3)
        self.assertEqual(len(self.parameters.observation_times_s), 4)
        self.assertEqual(len(self.parameters.observation_points), 4)
        self.assertEqual(self.parameters.forecast_times_s, (300.0, 360.0, 420.0, 480.0))
        self.assertEqual(len(self.parameters.dense_prediction_times_s), 43)
        self.assertEqual(self.parameters.dense_prediction_times_s[0], 60.0)
        self.assertEqual(self.parameters.dense_prediction_times_s[-1], 480.0)
        self.assertTrue(
            all(
                b - a == 10.0
                for a, b in zip(
                    self.parameters.dense_prediction_times_s,
                    self.parameters.dense_prediction_times_s[1:],
                )
            )
        )
        self.assertEqual(
            self.parameters.pre_entry_visualization_times_s,
            (60.0, 100.0, 180.0, 300.0),
        )
        self.assertEqual(len(self.parameters.online_update_times_s), 60)
        self.assertEqual(self.parameters.online_update_times_s[:2], (5.0, 10.0))
        self.assertEqual(self.parameters.online_update_times_s[-1], 300.0)
        self.assertEqual(len(self.parameters.online_evaluation_times_s), 97)
        self.assertEqual(
            self.parameters.online_visualization_times_s,
            (0.0, 30.0, 60.0, 100.0),
        )
        self.assertEqual([item.id for item in self.parameters.exposure_levels], ["minimal", "elevated", "high", "critical", "extreme"])
        self.assertEqual(
            [item.id for item in self.parameters.scenarios],
            ["appliance_contained", "enclosure_breakout", "high_intensity_appliance_fire"],
        )

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

    def test_fire_semantics_are_single_source_without_secondary_ignition(self) -> None:
        semantics = self.case.summary["fire_source_semantics"]
        self.assertEqual(semantics["ignition_model"], "single_primary_source")
        self.assertEqual(semantics["primary_source"], "microwave")
        self.assertEqual(semantics["primary_source_count"], 1)
        self.assertEqual(semantics["secondary_ignition_mode"], "disabled")
        self.assertEqual(semantics["secondary_ignition_count"], 0)
        self.assertEqual(int(np.sum(self.case.arrays["primary_source_mask"])), 1)
        self.assertFalse(np.any(self.case.arrays["secondary_ignition_mask"]))
        profile = self.case.summary["spread_severity_profile"]
        self.assertEqual(profile["id"], "enhanced_persistent_smoke_v1")
        self.assertFalse(profile["thresholds_or_palette_modified"])
        self.assertFalse(profile["secondary_ignition_added"])
        for scenario in self.parameters.scenarios:
            reference = load_fire_parameters(scenario.reference_config)
            self.assertEqual(reference.source_semantic_id, "microwave")
            fds_text = scenario.fds_input.read_text(encoding="utf-8")
            self.assertEqual(fds_text.count("SURF_ID='SYNTHETIC_FIRE'"), 1)
            self.assertNotIn("cable", scenario.id.lower())

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

    def test_risk_levels_are_derived_from_posterior_forecast_only(self) -> None:
        provenance = self.case.summary["risk_map_provenance"]
        self.assertEqual(
            provenance["field_source"],
            "posterior_weighted_candidate_ensemble_forecast",
        )
        self.assertFalse(provenance["ground_truth_used_to_construct_map"])
        bounds = [level.upper_bound for level in self.parameters.exposure_levels]
        for time_s in self.parameters.forecast_times_s:
            key = f"t{time_s:g}"
            exposure = self.case.arrays[f"posterior_exposure_{key}"]
            levels = self.case.arrays[f"posterior_risk_level_{key}"]
            traversable = levels >= 0
            expected = np.full(levels.shape, -1, dtype=np.int8)
            remaining = traversable.copy()
            for index, upper_bound in enumerate(bounds):
                selected = remaining if upper_bound is None else remaining & (exposure <= upper_bound + 1e-12)
                expected[selected] = index
                remaining[selected] = False
            np.testing.assert_array_equal(levels, expected)

    def test_synthetic_reference_risk_uses_same_grid_and_thresholds(self) -> None:
        metadata = self.case.summary["synthetic_reference_risk_grid"]
        self.assertIn("not real W1 ground truth", metadata["role"])
        bounds = [level.upper_bound for level in self.parameters.exposure_levels]
        for time_s in self.parameters.forecast_times_s:
            key = f"t{time_s:g}"
            exposure = self.case.arrays[f"synthetic_reference_exposure_{key}"]
            levels = self.case.arrays[f"synthetic_reference_risk_level_{key}"]
            self.assertEqual(exposure.shape, (50, 160))
            self.assertEqual(levels.shape, (50, 160))
            expected = np.full(levels.shape, -1, dtype=np.int8)
            remaining = levels >= 0
            for index, upper_bound in enumerate(bounds):
                selected = remaining if upper_bound is None else remaining & (exposure <= upper_bound + 1e-12)
                expected[selected] = index
                remaining[selected] = False
            np.testing.assert_array_equal(levels, expected)
            comparison = metadata["posterior_comparison_by_forecast_time"][key]
            self.assertEqual(
                comparison["exact_level_agreement_cell_count"]
                + comparison["posterior_higher_level_cell_count"]
                + comparison["posterior_lower_level_cell_count"],
                5800,
            )

    def test_continuous_fields_and_component_provenance_are_auditable(self) -> None:
        allowed = {
            "source_connected_exposure",
            "threshold_disconnected_transport_region",
            "weakly_supported_transport_region",
        }
        self.assertEqual(self.case.summary["model_validation_status"], "passed")
        self.assertEqual(self.case.summary["unexpected_disconnected_component_count"], 0)
        for time_s in self.parameters.forecast_times_s:
            key = f"t{time_s:g}"
            for modality in ("temperature", "co", "smoke", "visibility"):
                self.assertEqual(self.case.arrays[f"posterior_{modality}_severity_{key}"].shape, (50, 160))
            self.assertEqual(self.case.arrays[f"posterior_dominant_modality_{key}"].shape, (50, 160))
            self.assertEqual(self.case.arrays[f"posterior_exposure_component_{key}"].shape, (50, 160))
            diagnostics = self.case.summary["risk_component_diagnostics"][key]
            self.assertGreater(diagnostics["continuous_support_epsilon"], 0.0)
            self.assertGreater(
                diagnostics["meaningful_support_threshold"],
                diagnostics["continuous_support_epsilon"],
            )
            for threshold in diagnostics["thresholds"].values():
                for component in threshold["components"]:
                    self.assertIn(component["classification"], allowed)
                    if component["classification"] == "threshold_disconnected_transport_region":
                        self.assertTrue(component["source_connected_in_meaningful_support"])
                    if component["classification"] == "weakly_supported_transport_region":
                        self.assertTrue(component["source_connected_in_continuous_support"])
                        self.assertFalse(component["source_connected_in_meaningful_support"])

    def test_planning_field_is_cellwise_worst_case_envelope(self) -> None:
        snapshots = np.stack([
            self.case.arrays[f"posterior_exposure_t{time_s:g}"]
            for time_s in self.parameters.forecast_times_s
        ])
        np.testing.assert_array_equal(
            self.case.arrays["planning_exposure_envelope"],
            np.max(snapshots, axis=0),
        )
        self.assertEqual(
            self.case.summary["planning_field_semantics"]["aggregation"],
            "cellwise_maximum_across_forecast_times",
        )

    def test_exposure_change_is_measured_without_panel_normalization(self) -> None:
        changes = self.case.summary["exposure_change_from_previous_time"]
        self.assertIsNone(changes["t300"])
        for key in ("t360", "t420", "t480"):
            self.assertGreater(changes[key]["changed_cell_count"], 0)
            self.assertGreater(changes[key]["max_absolute_change"], 0.0)
            transitions = self.case.summary["risk_level_change_from_previous_time"][key]
            self.assertGreater(transitions["changed_cell_count"], 0)
            self.assertEqual(
                transitions["changed_cell_count"],
                transitions["increased_risk_cell_count"] + transitions["decreased_risk_cell_count"],
            )

    def test_dense_prediction_is_causal_and_does_not_backfill_posterior(self) -> None:
        records = self.dense_case.summary["dense_prediction"]["records"]
        self.assertEqual(len(records), 43)
        self.assertEqual(len(self.dense_rows), 43)
        self.assertEqual(sum(record["time_s"] <= 300.0 for record in records), 25)
        self.assertTrue(all(not record["posterior_available"] for record in records[:4]))
        self.assertTrue(all(record["posterior_available"] for record in records[4:]))
        self.assertTrue(
            all(record["posterior_reference_exposure_mae"] is None for record in records[:4])
        )
        self.assertTrue(
            all(record["posterior_unavailable_reason"] == "observation_not_yet_available" for record in records[:4])
        )
        np.testing.assert_array_equal(
            self.dense_case.arrays["dense_posterior_available"][:4],
            np.zeros(4, dtype=np.bool_),
        )
        self.assertTrue(np.all(np.isnan(self.dense_case.arrays["dense_posterior_exposure"][:4])))
        self.assertTrue(np.all(self.dense_case.arrays["dense_posterior_risk_level"][:4] == -3))
        self.assertTrue(np.all(np.isfinite(self.dense_case.arrays["dense_posterior_exposure"][4:])))

    def test_pre_entry_visuals_are_sparse_views_of_dense_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            keyframes = render_p8_v2_pre_entry_keyframes_svg(
                CONFIG,
                self.dense_case.summary,
                self.dense_case.arrays,
                root / "keyframes.svg",
            )
            time_series = render_p8_v2_online_pre_entry_timeseries_svg(
                CONFIG,
                self.dense_case.summary,
                self.dense_case.arrays,
                root / "timeseries.svg",
            )
            keyframe_text = keyframes.read_text(encoding="utf-8")
            time_series_text = time_series.read_text(encoding="utf-8")
            self.assertEqual(keyframe_text.count('class="risk-level-cell"'), 11 * 50 * 160)
            self.assertIn("Unavailable at t=0", keyframe_text)
            self.assertIn("updates posterior weights every 5 s", keyframe_text)
            self.assertIn('data-online-update-interval-s="5"', time_series_text)
            self.assertIn("First UGV update; then every 5 s", time_series_text)
            self.assertIn("Sequential posterior (available from t=5 s)", time_series_text)

    def test_online_track_updates_every_five_seconds_after_prior_only_t0(self) -> None:
        records = self.dense_case.summary["online_sequential_prediction"]["records"]
        self.assertEqual(len(records), 97)
        self.assertEqual(len(self.online_rows), 97)
        self.assertFalse(records[0]["posterior_available"])
        self.assertFalse(records[0]["sequential_update_applied"])
        self.assertIsNone(records[0]["posterior_weight_hash"])
        self.assertTrue(all(record["posterior_available"] for record in records[1:]))
        self.assertTrue(all(record["sequential_update_applied"] for record in records[1:61]))
        self.assertTrue(all(not record["sequential_update_applied"] for record in records[61:]))
        self.assertEqual(records[6]["time_s"], 30.0)
        self.assertEqual(records[6]["cumulative_observation_count"], 6)
        self.assertEqual(records[12]["time_s"], 60.0)
        self.assertEqual(records[12]["cumulative_observation_count"], 12)
        self.assertEqual(records[20]["time_s"], 100.0)
        self.assertEqual(records[20]["cumulative_observation_count"], 20)
        self.assertTrue(np.all(np.isnan(self.dense_case.arrays["online_posterior_exposure"][0])))
        self.assertTrue(np.all(np.isfinite(self.dense_case.arrays["online_posterior_exposure"][1:])))
        self.assertTrue(np.all(self.dense_case.arrays["online_posterior_risk_level"][0] == -3))

    def test_dynamic_risk_player_uses_all_online_frames_without_reference_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = render_p8_v2_dynamic_risk_html(
                CONFIG,
                self.dense_case.arrays,
                Path(directory) / "dynamic.html",
            )
            html = output.read_text(encoding="utf-8")
            self.assertIn('data-frame-count="97"', html)
            self.assertIn('data-interval-s="5"', html)
            self.assertIn('data-source="online-sequential-posterior"', html)
            self.assertIn('data-display-completion="conservative-nearest-air"', html)
            self.assertIn('"times":[0,5,10', html)
            self.assertIn(',470,475,480]', html)
            self.assertIn('"planningBlocked"', html)
            self.assertNotIn("online_synthetic_reference", html)
            self.assertLess(output.stat().st_size, 2_000_000)

    def test_display_completion_fills_holes_without_changing_air_cells(self) -> None:
        raw = np.asarray([[0, -1, 4], [0, -1, 2]], dtype=np.int8)
        completed = complete_blocked_risk_levels(raw)
        self.assertTrue(np.all(completed >= 0))
        np.testing.assert_array_equal(completed[raw >= 0], raw[raw >= 0])
        self.assertEqual(int(completed[0, 1]), 4)

    def test_t300_compares_all_entrances_on_one_posterior_snapshot(self) -> None:
        result = self.dense_case.summary["online_ingress_at_t300"]
        self.assertEqual(result["field_time_s"], 300.0)
        self.assertIn("not the 300/360/420/480", result["field_semantics"])
        self.assertEqual(result["target_node"], self.parameters.source_approach_target_node)
        self.assertEqual(
            result["strict_operational_status"],
            "no_route_meets_configured_exposure_threshold",
        )
        self.assertFalse(result["strict_operational_plan"]["feasible"])
        candidates = result["comparative_routes"]["candidates"]
        self.assertEqual(set(candidates), set(self.parameters.entrances))
        feasible = {key: value for key, value in candidates.items() if value["feasible"]}
        self.assertTrue(feasible)
        for route in feasible.values():
            path = tuple((item["row"], item["col"]) for item in route["path"])
            self.assertTrue(self.context.graph.path_is_corridor_conformant(path))
        expected_fastest = min(
            feasible,
            key=lambda key: (
                feasible[key]["geometric_length_m"],
                feasible[key]["total_cost"],
                key,
            ),
        )
        expected_safest = min(
            feasible,
            key=lambda key: (
                feasible[key]["integrated_exposure"],
                feasible[key]["max_exposure"],
                feasible[key]["total_cost"],
                key,
            ),
        )
        self.assertEqual(result["fastest_entrance"], expected_fastest)
        self.assertEqual(result["safest_entrance"], expected_safest)

    def test_online_risk_map_and_t300_ingress_render_complete_grid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            risk_map = render_p8_v2_online_risk_map_svg(
                CONFIG,
                self.dense_case.summary,
                self.dense_case.arrays,
                root / "risk.svg",
            )
            ingress = render_p8_v2_t300_three_entrance_ingress_svg(
                CONFIG,
                self.dense_case.summary,
                self.dense_case.arrays,
                root / "ingress.svg",
            )
            compact_risk = render_p8_v2_online_risk_map_svg(
                CONFIG,
                self.dense_case.summary,
                self.dense_case.arrays,
                root / "risk-small.svg",
                compact=True,
            )
            compact_ingress = render_p8_v2_t300_three_entrance_ingress_svg(
                CONFIG,
                self.dense_case.summary,
                self.dense_case.arrays,
                root / "ingress-small.svg",
                compact=True,
            )
            risk_text = risk_map.read_text(encoding="utf-8")
            ingress_text = ingress.read_text(encoding="utf-8")
            compact_risk_text = compact_risk.read_text(encoding="utf-8")
            compact_ingress_text = compact_ingress.read_text(encoding="utf-8")
            self.assertEqual(risk_text.count('class="risk-level-cell"'), 9 * 50 * 160)
            self.assertIn('data-fixed-risk-thresholds="true"', risk_text)
            self.assertNotIn('data-risk-level="blocked"', risk_text)
            self.assertIn('data-planning-blocked="true"', risk_text)
            self.assertIn('data-planning-time-s="300"', ingress_text)
            self.assertEqual(ingress_text.count('class="risk-level-cell"'), 50 * 160)
            feasible_count = sum(
                route["feasible"]
                for route in self.dense_case.summary["online_ingress_at_t300"]["comparative_routes"]["candidates"].values()
            )
            self.assertEqual(ingress_text.count('class="t300-ingress-route"'), feasible_count)
            self.assertEqual(compact_risk_text.count('data-grid-cell-count="8000"'), 9)
            self.assertEqual(compact_ingress_text.count('data-grid-cell-count="8000"'), 1)
            self.assertIn('data-encoding="row-run-length"', compact_risk_text)
            self.assertIn('data-encoding="row-run-length"', compact_ingress_text)
            self.assertIn('class="planning-blocked-overlay"', compact_risk_text)
            self.assertIn('class="planning-blocked-overlay"', compact_ingress_text)
            self.assertLess(compact_risk.stat().st_size, risk_map.stat().st_size / 4)
            self.assertLess(compact_ingress.stat().st_size, ingress.stat().st_size / 2)

    def test_enhanced_spread_profile_has_material_temporal_growth(self) -> None:
        counts = self.case.summary["risk_grid"]["counts_by_forecast_time"]
        nonminimal_300 = 5800 - counts["t300"]["minimal"]
        nonminimal_480 = 5800 - counts["t480"]["minimal"]
        self.assertGreater(nonminimal_480, 2 * nonminimal_300)
        self.assertGreater(counts["t480"]["extreme"], counts["t300"]["extreme"])
        for key in ("t360", "t420", "t480"):
            self.assertGreaterEqual(
                self.case.summary["risk_level_change_from_previous_time"][key]["changed_cell_count"],
                500,
            )

    def test_visualizations_use_fixed_scale_and_render_every_cell(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            categorical = render_p8_v2_categorical_risk_svg(CONFIG, self.case.summary, self.case.arrays, root / "categorical.svg")
            evolution = render_p8_v2_exposure_evolution_svg(CONFIG, self.case.summary, self.case.arrays, root / "evolution.svg")
            planning = render_p8_v2_planning_envelope_svg(CONFIG, self.case.summary, self.case.arrays, root / "planning.svg")
            diagnostic = render_p8_v2_grid_diagnostic_svg(CONFIG, self.case.summary, self.case.arrays, root / "diagnostic.svg")
            reference = render_p8_v2_synthetic_reference_risk_svg(CONFIG, self.case.summary, self.case.arrays, root / "reference.svg")
            comparison = render_p8_v2_posterior_reference_comparison_svg(CONFIG, self.case.summary, self.case.arrays, root / "comparison.svg")
            categorical_text = categorical.read_text(encoding="utf-8")
            evolution_text = evolution.read_text(encoding="utf-8")
            planning_text = planning.read_text(encoding="utf-8")
            diagnostic_text = diagnostic.read_text(encoding="utf-8")
            reference_text = reference.read_text(encoding="utf-8")
            comparison_text = comparison.read_text(encoding="utf-8")
            self.assertIn('data-fixed-color-scale="0:1"', evolution_text)
            self.assertEqual(categorical_text.count('class="risk-level-cell"'), 4 * 50 * 160)
            self.assertEqual(categorical_text.count('data-burning="false"'), 4 * 50 * 160)
            self.assertIn('data-risk-map-source="posterior_model_forecast"', categorical_text)
            self.assertIn("not used to construct this posterior risk map", categorical_text)
            self.assertEqual(evolution_text.count('class="registered-cell"'), 4 * 50 * 160)
            self.assertEqual(planning_text.count('class="registered-cell"'), 50 * 160)
            self.assertEqual(diagnostic_text.count('class="registered-cell"'), 50 * 160)
            self.assertIn("Colored cells indicate forecast exposure, not burning or secondary ignition", evolution_text)
            self.assertIn('data-ignition-source="microwave"', planning_text)
            self.assertEqual(reference_text.count('class="risk-level-cell"'), 4 * 50 * 160)
            self.assertIn('data-risk-map-source="synthetic_reference_model"', reference_text)
            self.assertIn("not measured W1 ground truth", reference_text)
            self.assertEqual(comparison_text.count('class="risk-level-cell"'), 8 * 50 * 160)
            self.assertIn('data-comparison="posterior_vs_synthetic_reference"', comparison_text)

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

"""Independent finite examples and CLI checks for Semantic Echo CPU metrics."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from research_gym.analysis.semantic_echo import (
    aggregate_graphs,
    aggregate_trajectories,
    analyze_trajectory,
    cycle_length,
    cycle_nodes,
    normalized_phase_residual,
    oracle_trajectory,
    phase_telescoping,
    validate_permutation,
)
from research_gym.scripts.semantic_echo_controls import run_controls


class ValidationAndOracleTests(unittest.TestCase):
    def test_full_permutation_validation_including_unvisited_component(self):
        for bad in ([1, 0, 2, 2], [1, 2], [-1, 0], [0, 2], [], [[0]], [0.0, 1.0], [False, True]):
            with self.subTest(permutation=bad), self.assertRaises(ValueError):
                validate_permutation(bad)

    def test_large_unsigned_rejected_without_integer_wraparound(self):
        with self.assertRaises(ValueError):
            validate_permutation(np.array([0, 2 ** 64 - 1], dtype=np.uint64))

    def test_validation_returns_independent_copy(self):
        original = np.array([1, 0])
        result = validate_permutation(original)
        result[0] = 0
        self.assertEqual(original.tolist(), [1, 0])

    def test_nontrivial_cycle_indices_and_post_loop_oracle(self):
        permutation = [2, 4, 1, 0, 3, 6, 5]
        self.assertEqual(cycle_nodes(permutation, 0), [0, 2, 1, 4, 3])
        self.assertEqual(cycle_length(permutation, 0), 5)
        self.assertEqual(cycle_length(permutation, 5), 2)
        self.assertEqual(oracle_trajectory(permutation, 0, 7).tolist(), [2, 1, 4, 3, 0, 2, 1])
        self.assertEqual(oracle_trajectory(permutation, 0, 0).tolist(), [])

    def test_invalid_starts_labels_and_bounds(self):
        cases = [dict(start=-1), dict(start=3), dict(start=True),
                 dict(predictions=[1.0, 0.0]), dict(predictions=[1, 3]),
                 dict(predictions=[]), dict(predictions=[[1]]), dict(predictions=[True]),
                 dict(horizon=0), dict(horizon=1.5), dict(min_visit=0),
                 dict(min_visit=4, max_visit=3), dict(max_visit=False)]
        for changes in cases:
            kwargs = {"permutation": [1, 2, 0], "start": 0, "predictions": [1, 2, 0]}
            kwargs.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                analyze_trajectory(**kwargs)


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.permutation = [1, 2, 0]
        self.correct = [1, 2, 0, 1, 2, 0, 1, 2]

    def test_endpoint_indexing_and_exact_denominators(self):
        result = analyze_trajectory(self.permutation, 0, self.correct, horizon=2, max_visit=4)
        c = result["continuation"]
        self.assertEqual(c["eligible_pairs"], 1)
        self.assertEqual(c["endpoint_gated_pairs"], 1)
        self.assertEqual(c["per_step_comparisons"], 2)
        self.assertEqual(c["oracle_comparisons"], 4)
        self.assertEqual(c["oracle_correct_count"], 4)
        self.assertEqual(c["oracle_continuation_accuracy"], 1.0)
        self.assertEqual(c["raw_rows"][0]["t"], 1)
        self.assertEqual(c["raw_rows"][0]["u"], 4)
        self.assertEqual(c["raw_rows"][0]["expected_future"], [2, 0])

    def test_horizon_availability_excludes_late_endpoints(self):
        result = analyze_trajectory(self.permutation, 0, self.correct, horizon=2, max_visit=999)
        self.assertEqual(result["last_available_visit"], 6)
        self.assertEqual(result["continuation"]["eligible_pairs"], 3)
        self.assertTrue(all(row["u"] + 2 <= len(self.correct) for row in result["continuation"]["raw_rows"]))

    def test_future_corruption_preserves_endpoint_selection(self):
        damaged = self.correct.copy()
        damaged[4] = 0
        original = analyze_trajectory(self.permutation, 0, self.correct, horizon=2, max_visit=4)
        result = analyze_trajectory(self.permutation, 0, damaged, horizon=2, max_visit=4)
        c = result["continuation"]
        self.assertEqual(c["eligible_pairs"], original["continuation"]["eligible_pairs"])
        self.assertEqual(c["endpoint_gated_pairs"], original["continuation"]["endpoint_gated_pairs"])
        self.assertEqual(c["pair_any_disagreement_count"], 1)
        self.assertEqual(c["per_step_mismatch_count"], 1)
        self.assertEqual(c["per_step_mismatch_rate"], 0.5)
        self.assertEqual(c["oracle_continuation_accuracy"], 0.75)
        self.assertEqual(c["first_witness"], {"kind": "continuation_disagreement", "t": 1, "u": 4,
                                             "offset": 1, "semantic_label": 1, "expected": 2,
                                             "prediction_from_t": 2, "prediction_from_u": 0})

    def test_endpoint_failure_reports_coverage_and_undefined_future_rate(self):
        damaged = self.correct.copy()
        damaged[3] = 0
        result = analyze_trajectory(self.permutation, 0, damaged, horizon=2, max_visit=4)
        c = result["continuation"]
        self.assertEqual(c["eligible_pairs"], 1)
        self.assertEqual(c["endpoint_gated_pairs"], 0)
        self.assertEqual(c["endpoint_gate_coverage"], 0.0)
        self.assertIsNone(c["pair_any_disagreement_rate"])
        self.assertIsNone(c["oracle_continuation_accuracy"])
        self.assertIsNone(c["raw_rows"][0]["future_predictions_t"])

    def test_agreeing_futures_can_both_be_wrong(self):
        result = analyze_trajectory(self.permutation, 0, [1] * 8, horizon=2, max_visit=4)
        c = result["continuation"]
        self.assertEqual(c["endpoint_gate_coverage"], 1.0)
        self.assertEqual(c["pair_any_disagreement_rate"], 0.0)
        self.assertEqual(c["oracle_continuation_accuracy"], 0.0)
        self.assertIsNone(c["first_witness"])
        self.assertEqual(c["first_oracle_failure"]["offset"], 1)

    def test_zero_structural_pairs_are_uninformative(self):
        for horizon, minimum in [(4, 1), (1, 50)]:
            result = analyze_trajectory(self.permutation, 0, [1, 2, 0], horizon=horizon, min_visit=minimum)
            c = result["continuation"]
            self.assertEqual(c["eligible_pairs"], 0)
            self.assertIsNone(c["endpoint_gate_coverage"])
            self.assertEqual(c["oracle_comparisons"], 0)
            self.assertIsNone(c["oracle_continuation_accuracy"])

    def test_graph_aggregation_uses_graph_means_and_exposes_coverage(self):
        correct = analyze_trajectory(self.permutation, 0, self.correct, horizon=2, max_visit=4)
        wrong = analyze_trajectory(self.permutation, 0, [1] * 8, horizon=2, max_visit=4)
        empty = analyze_trajectory(self.permutation, 0, [1], horizon=2)
        agg = aggregate_graphs([correct, wrong, empty])
        metric = agg["graph_means"]["oracle_continuation_accuracy"]
        self.assertEqual(metric, {"mean": 0.5, "graphs_with_metric": 2, "graphs_total": 3})
        self.assertEqual(aggregate_graphs([])["n_graphs"], 0)
        self.assertIsNone(aggregate_graphs([])["graph_means"]["label_accuracy"]["mean"])

    def test_item_aggregation_preserves_repeated_sampling_values(self):
        result = analyze_trajectory(self.permutation, 0, self.correct, horizon=2, max_visit=4)
        summary = aggregate_trajectories([result, result], unit="item")
        self.assertEqual(summary["aggregation_unit"], "item")
        self.assertEqual(summary["n_units"], 2)
        self.assertEqual(summary["unit_means"]["oracle_continuation_accuracy"],
                         {"mean": 1.0, "units_with_metric": 2, "units_total": 2})
        with self.assertRaises(ValueError):
            aggregate_trajectories([result], unit="visit_pair")


class PhaseTests(unittest.TestCase):
    def test_true_cycle_order_is_used_instead_of_numeric_node_order(self):
        result = analyze_trajectory([2, 0, 1], 0, [2, 1, 0, 2, 1, 0], horizon=1)
        self.assertEqual(result["phase"]["cycle_nodes"], [0, 2, 1])
        self.assertLess(result["phase"]["mean_normalized_residual"], 1e-24)

    def test_frozen_unit_phase_scores_one_at_several_cycle_lengths(self):
        for ell in (2, 3, 5, 17, 1000):
            with self.subTest(cycle_length=ell):
                score = normalized_phase_residual(np.ones(7), ell)
                self.assertAlmostEqual(score["mean_normalized_residual"], 1.0, places=11)

    def test_uniform_wrong_phase_passes_law_and_fails_label_anchor(self):
        result = analyze_trajectory([1, 2, 0], 0, [2, 0, 1, 2, 0, 1, 2, 0], horizon=2, max_visit=4)
        self.assertLess(result["phase"]["mean_normalized_residual"], 1e-24)
        self.assertEqual(result["transition"]["accuracy"], 1.0)
        self.assertEqual(result["label_accuracy"], 0.0)
        self.assertEqual(result["continuation"]["endpoint_gate_coverage"], 0.0)

    def test_off_cycle_predictions_are_invalid_for_phase(self):
        result = analyze_trajectory([1, 0, 3, 2], 0, [1, 0, 2, 0, 1], horizon=1)
        phase = result["phase"]
        self.assertEqual(phase["off_cycle_prediction_count"], 1)
        self.assertEqual(phase["off_cycle_predictions"], [{"loop": 3, "label": 2}])
        self.assertEqual(phase["n_valid_transitions"], 2)
        self.assertEqual(phase["n_invalid_transitions"], 2)
        self.assertEqual(phase["valid_transition_coverage"], 0.5)
        self.assertIsNone(phase["raw_rows"][1]["normalized_residual"])

    def test_fixed_points_are_explicitly_uninformative(self):
        result = analyze_trajectory([0, 2, 1], 0, [0, 0, 0, 0], horizon=1)
        self.assertFalse(result["phase"]["informative"])
        self.assertEqual(result["phase"]["reason"], "fixed_point")
        self.assertIsNone(result["phase"]["mean_normalized_residual"])
        self.assertFalse(normalized_phase_residual([1, 1], 1)["informative"])

    def test_one_loop_has_no_transition_denominator(self):
        result = analyze_trajectory([1, 0], 0, [1], horizon=1)
        self.assertEqual(result["transition"]["comparisons"], 0)
        self.assertIsNone(result["transition"]["accuracy"])
        self.assertEqual(result["phase"]["n_scored_transitions"], 0)
        self.assertIsNone(result["phase"]["mean_normalized_residual"])

    def test_phase_validation(self):
        for phases, ell in [([], 2), ([0, 0], 2), ([1, np.nan], 2), ([1], 0), ([[1]], 2), ([1], True)]:
            with self.subTest(phases=phases, ell=ell), self.assertRaises(ValueError):
                normalized_phase_residual(phases, ell)

    def test_telescoping_with_canceling_transition_errors(self):
        omega = np.exp(2j * np.pi / 5)
        result = phase_telescoping([1, omega ** 2, omega ** 2], 5)
        self.assertLess(result["final_deviation"], 1e-12)
        self.assertLess(result["identity_error"], 1e-12)
        self.assertGreater(result["sum_absolute_residuals"], 1)
        self.assertFalse(result["certifies_phase_relative_to_initial"])

    def test_telescoping_error_bound_and_anchor_scope(self):
        omega = np.exp(2j * np.pi / 7)
        z = omega ** np.arange(9)
        exact = phase_telescoping(z, 7)
        self.assertTrue(exact["certifies_phase_relative_to_initial"])
        # A wrong but valid initial root also passes this *relative* certificate.
        shifted = phase_telescoping(z * omega, 7)
        self.assertTrue(shifted["certifies_phase_relative_to_initial"])
        perturbed = z * np.exp(1j * np.linspace(0, 0.1, len(z)))
        noisy = phase_telescoping(perturbed, 7)
        self.assertLess(noisy["identity_error"], 1e-12)
        self.assertLessEqual(noisy["final_deviation"], noisy["sum_absolute_residuals"] + 1e-12)
        self.assertLessEqual(noisy["sum_absolute_residuals"], noisy["rms_bound"] + 1e-12)


class RunnerTests(unittest.TestCase):
    def test_controls_pass_and_are_deterministic(self):
        report = run_controls(731)
        self.assertEqual(report["status"], "CONTROL_ONLY")
        self.assertTrue(report["all_controls_passed"])
        self.assertEqual(report["controls_total"], 7)
        self.assertEqual(json.dumps(report, sort_keys=True, allow_nan=False),
                         json.dumps(run_controls(731), sort_keys=True, allow_nan=False))

    def test_cli_writes_control_only_json_and_markdown(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="semantic-echo-test-") as output:
            completed = subprocess.run([sys.executable, "-m", "research_gym.scripts.semantic_echo_controls",
                                        "--out", output, "--seed", "17"],
                                       cwd=root, capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads((Path(output) / "controls.json").read_text())
            self.assertEqual(report["seed"], 17)
            self.assertEqual(report["status"], "CONTROL_ONLY")
            self.assertIn("CONTROL_ONLY", (Path(output) / "report.md").read_text())


if __name__ == "__main__":
    unittest.main()

"""Focused independent-score and corrupted-receipt checks on old seed 469."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.special import ndtr

import audit16 as audited
import search15
import search16


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.initial = audited.reconstruct_initial(469)
        codes, scales = np.asarray(cls.initial["codes"]), np.asarray(cls.initial["scales"])
        scales, offset, _ = search16.parent.fit(codes, scales, cls.initial["offset"])
        cls.first = search16.state(codes, scales, offset, cls.initial["down_bias"], "example:ls", "s0")
        refined, refined_offset, cls.lp = search16.lp_fit(codes, scales, offset, cls.initial["down_bias"])
        cls.second = search16.state(codes, refined, refined_offset, cls.initial["down_bias"], "example:lp", "s0")

    def test_fp32_decode_matches_inherited_receipt(self):
        self.assertEqual(audited.decode(self.initial["codes"], self.initial["scales"], self.initial["offset"], self.initial["down_bias"]), self.initial["weights"])
        for state in (self.first, self.second):
            self.assertEqual(audited.decode(state["codes"], state["scales"], state["offset"], self.initial["down_bias"]), search16.parent.decoded(np.asarray(state["codes"]), np.asarray(state["scales"]), state["offset"], self.initial["down_bias"]))

    def test_independent_segment_score_matches_endpoint_implementation(self):
        for state in (self.first, self.second):
            weights = audited.check_state(state, self.initial["down_bias"])
            actual, inherited = audited.metric(weights), search15.metric(weights)
            for key in actual:
                self.assertAlmostEqual(actual[key], inherited[key], places=10)

    def test_segment_bound_contains_dense_diagnostic_samples(self):
        weights = audited.check_state(self.second, self.initial["down_bias"])
        dense = np.empty((32, 1025, 4))
        dense[:, :, 0] = audited.CASES[:, 0, None] - 2 + np.linspace(-1 / 8, 1 / 8, 1025)
        dense[:, :, 1:] = audited.CASES[:, None, 1:]
        pre = dense @ np.asarray(weights["up"]).T + weights["up_bias"]
        output = dense[:, :, 0] + dense @ np.asarray(weights["skip"]) + weights["skip_bias"][0] + ((pre * ndtr(pre)) @ np.asarray(weights["down"]) + weights["down_bias"][0]) / 5
        observed = float(np.max(np.abs(output - audited.TARGETS)))
        self.assertLessEqual(observed, audited.metric(weights)["bound"] + 1e-10)

    def test_independent_lp_matrix_matches_fixed_problem(self):
        matrix, rhs = audited.lp_matrix(self.first, self.initial["down_bias"])
        expected, expected_rhs = search15.lp_matrix(np.asarray(self.first["codes"]), np.asarray(self.first["scales"]), self.first["offset"], self.initial["down_bias"], True)
        np.testing.assert_allclose(matrix, expected, atol=1e-12, rtol=1e-12)
        np.testing.assert_array_equal(rhs, expected_rhs)

    def test_valid_sparse_dual_receipt(self):
        result = audited.check_lp(self.first, self.second, self.lp, self.initial["down_bias"])
        self.assertTrue(result["success"])
        self.assertLessEqual(result["stationarity"], 1e-6)

    def test_corrupt_dual_rejected(self):
        bad = copy.deepcopy(self.lp)
        bad["lower_duals"][0] += 1
        with self.assertRaises(AssertionError):
            audited.check_lp(self.first, self.second, bad, self.initial["down_bias"])

    def test_duplicate_dual_row_rejected(self):
        bad = copy.deepcopy(self.lp)
        self.assertTrue(bad["inequality_duals"])
        bad["inequality_duals"].append(bad["inequality_duals"][0])
        with self.assertRaises(AssertionError):
            audited.check_lp(self.first, self.second, bad, self.initial["down_bias"])

    def test_corrupt_primal_rejected(self):
        bad = copy.deepcopy(self.lp)
        bad["x"][3] = -1
        with self.assertRaises(AssertionError):
            audited.check_lp(self.first, self.second, bad, self.initial["down_bias"])

    def test_lp_failure_keeps_first_parameters(self):
        second = copy.deepcopy(self.first)
        second["id"] = "failed:lp"
        result = audited.check_lp(self.first, second, {"success": False, "status": 1, "nit": 1000}, self.initial["down_bias"])
        self.assertFalse(result["success"])

    def test_lp_iteration_budget_enforced(self):
        bad = copy.deepcopy(self.lp)
        bad["nit"] = 1001
        with self.assertRaises(AssertionError):
            audited.check_lp(self.first, self.second, bad, self.initial["down_bias"])

    def test_inherited_rollout_horizons_are_8_and_128(self):
        result = audited.evaluate(self.initial["weights"], 469)
        self.assertEqual(set(result), {"8", "128"})
        self.assertEqual(audited.HORIZONS, (8, 128))

    def test_corrupt_score_rejected(self):
        bad = copy.deepcopy(self.first)
        bad["scores"]["bound"] += 1e-3
        with self.assertRaises(AssertionError):
            audited.check_state(bad, self.initial["down_bias"])

    def test_fractional_or_out_of_range_code_rejected(self):
        for invalid in (0.5, 8):
            bad = copy.deepcopy(self.first)
            bad["codes"][0] = invalid
            with self.assertRaises(AssertionError):
                audited.check_state(bad, self.initial["down_bias"])

    def test_selection_ties_dedup_and_sub_tolerance_improvement(self):
        def state(identity, code, bound):
            return {"id": identity, "codes": [code] + [0] * 51, "scores": {"bound": bound}}
        nodes = [state("parent", 0, 1), state("tie", 0, 1), state("small_improvement", 0, 1 - 1e-13), state("different", 1, 1), state("third", 2, 2)]
        self.assertEqual([item["id"] for item in audited.keep(nodes, 3)], ["small_improvement", "different", "third"])
        self.assertEqual([item["id"] for item in audited.keep(nodes[:2] + nodes[3:], 1)], ["parent"])

    def test_shortlist_excludes_padded_duplicate_noops(self):
        candidates, names = audited.proposals(self.initial["codes"])
        rank = np.arange(177, dtype=float)
        for i, name in enumerate(names):
            if name == "padding":
                rank[i] = -1
        chosen = audited.shortlist(candidates, rank)
        self.assertEqual(chosen[0], 0)
        self.assertEqual(len({tuple(candidates[i]) for i in chosen}), 3)
        self.assertFalse(any(names[i] == "padding" for i in chosen))

    def test_missing_and_extra_endpoint_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("initial", "common", "results"):
                (root / name).mkdir()
            for name in ("initial", "common"):
                (root / name / "469.json").touch()
            (root / "results" / "greedy-469.json").touch()
            with self.assertRaises(AssertionError):
                audited.check_coverage(root, (469,))
            (root / "results" / "beam-469.json").touch()
            audited.check_coverage(root, (469,))
            (root / "results" / "beam-470.json").touch()
            with self.assertRaises(AssertionError):
                audited.check_coverage(root, (469,))


if __name__ == "__main__":
    unittest.main()

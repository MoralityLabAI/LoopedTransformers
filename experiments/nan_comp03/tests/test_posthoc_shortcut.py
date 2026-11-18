"""Source-only regression on frozen NAN-COMP-03 development receipts."""
import unittest

from experiments.nan_comp03.posthoc_shortcut import (
    audit, first_two_classify, predicted,
)


class TestComp03Posthoc(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = audit()

    def test_shortcut_on_frozen_sourced_worlds(self):
        r = self.report["source_shortcut"]
        self.assertEqual(r["development_worlds"], 16)
        self.assertEqual(r["correct_with_b1_b2"], 16)
        self.assertEqual(first_two_classify([1, 1, 0], [1, -1, 1]), "R")
        self.assertEqual(first_two_classify([1, 1, 0], [1, -1, 0]), "N")
        self.assertEqual(first_two_classify([1, 1, 0], [1, 0, 1]), "S")
        self.assertEqual(first_two_classify([1, 1, 0], [1, 1, 0]), "I")
        with self.assertRaises(ValueError):
            first_two_classify([1, 0, 1], [1, 0, 1])

    def test_source_only_original_gates_still_fail(self):
        self.assertFalse(self.report["sealed_168_case_test_accessed"])
        self.assertTrue(self.report["original_COMP03_gates_unchanged"])
        b = self.report["backends"]["bf16"]["bias_scan"]["original"]
        n = self.report["backends"]["nf4"]["bias_scan"]["original"]
        self.assertEqual((b["correct"], b["both_pairs"], b["reciprocal"]), (12, 4, 0))
        self.assertEqual((n["correct"], n["both_pairs"], n["reciprocal"]), (10, 2, 0))
        self.assertEqual(b["zero_bias_agrees_with_generated"], 16)
        self.assertEqual(n["zero_bias_agrees_with_generated"], 16)

    def test_exposed_data_score_bias_is_not_a_new_model_run(self):
        b = self.report["backends"]["bf16"]["bias_scan"]
        n = self.report["backends"]["nf4"]["bias_scan"]
        self.assertEqual(b["passing_bias_count"], 25)
        self.assertEqual((b["passing_bias_min"], b["passing_bias_max"]), (1.25, 1.49))
        self.assertEqual(n["passing_bias_count"], 0)
        self.assertEqual((b["at_R_class_score_bias_1p25"]["world_correct"],
                          b["at_R_class_score_bias_1p25"]["both_pairs"],
                          b["at_R_class_score_bias_1p25"]["reciprocal"]), (14, 6, 3))
        self.assertEqual((n["at_R_class_score_bias_1p25"]["world_correct"],
                          n["at_R_class_score_bias_1p25"]["both_pairs"],
                          n["at_R_class_score_bias_1p25"]["reciprocal"]), (12, 4, 3))
        self.assertFalse(b["free_generation_under_bias_executed"])
        self.assertFalse(n["free_generation_under_bias_executed"])

    def test_reciprocal_pair_source_margin_moves_without_erasure_effect(self):
        for name, expected in (("bf16", 2.5312441895), ("nf4", 1.9999948507)):
            entry = self.report["backends"][name]["reciprocal_pair_R_margin"]
            self.assertEqual(entry["n_R_counterfactual_pairs"], 4)
            self.assertTrue(entry["curriculum_all_positive"])
            self.assertAlmostEqual(entry["curriculum_mean_R_margin_change"], expected, places=8)
            self.assertEqual(entry["erased_max_abs_pair_change"], 0)

    def test_class_prior_can_flip_answer_and_harm_other_classes(self):
        score = {"I": -5, "N": -0.5, "S": -3, "R": -1.9}
        self.assertEqual(predicted(score, 0), "N")
        self.assertEqual(predicted(score, 1.5), "R")


if __name__ == "__main__":
    unittest.main()

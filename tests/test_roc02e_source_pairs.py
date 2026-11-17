"""Source-only and optional torch checks for ROC-02E preregistered controls."""
from __future__ import annotations
import copy
import json
import math
import unittest
from collections import Counter
from pathlib import Path

from research_gym.analysis.roc02_delayed_rule_source import rule_table, symbolic_execute
from research_gym.analysis.roc02e_source_pairs import (
    ALLOWED_FAMILIES, CALIBRATION_SPLIT_SHA256, RADII,
    calibration_source, make_counterfactual, make_pair, raw_feature_l2,
    source_pair_audit, interpolate_to_midpoint_radius,
)


class ROC02ESourcePairs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = calibration_source()

    def test_prereg_and_provenance(self):
        cfg = json.loads(Path("configs/roc02e_rule_vs_cue_causal_dev_v1.json").read_text())
        self.assertEqual(cfg["provenance"]["parent_head"],
                         "004301995d44388bf7ceefccdc89c845c8452d00")
        self.assertEqual(cfg["data"]["calibration_families"], [4, 5])
        self.assertEqual(cfg["data"]["sealed_families"], [6, 7])
        self.assertEqual(cfg["physical_matching"]["radii"], list(RADII))
        self.assertEqual(len(self.cases), 256)
        self.assertEqual(Counter(x["family"] for x in self.cases), {4: 117, 5: 139})
        self.assertEqual(len(CALIBRATION_SPLIT_SHA256), 64)

    def test_all_256_rule_cue_pairs_have_same_exact_target(self):
        ids = set()
        for record in self.cases:
            rule, cue, metadata = make_pair(record)
            y = record["label"]
            self.assertEqual(rule["label"], (y + 4) % 8)
            self.assertEqual(cue["label"], (y + 4) % 8)
            self.assertEqual(symbolic_execute(rule["events"])["label"], rule["label"])
            self.assertEqual(symbolic_execute(cue["events"])["label"], cue["label"])
            self.assertEqual(rule["events"][1:], record["events"][1:])
            self.assertEqual(cue["events"][:-1], record["events"][:-1])
            self.assertEqual(rule["events"][0]["table"], rule_table(5 if record["family"] == 4 else 4))
            self.assertEqual(rule["delay"], cue["delay"])
            self.assertEqual(metadata["counterfactual_label"], rule["label"])
            ids.add(record["id"])
        self.assertEqual(len(ids), 256)

    def test_feature_doses_and_unaffected_distractors(self):
        for record in self.cases:
            rule = make_counterfactual(record, "rule")
            cue = make_counterfactual(record, "cue")
            self.assertAlmostEqual(raw_feature_l2(record, rule), 4.0, places=12)
            self.assertAlmostEqual(raw_feature_l2(record, cue), math.sqrt(2.0), places=12)
            for before, edited in zip(record["events"][1:-1], rule["events"][1:-1]):
                self.assertEqual(before, edited)

    def test_noncalibration_family_and_wrong_edit_rejected(self):
        row = copy.deepcopy(self.cases[0])
        row["family"] = 6
        with self.assertRaises(ValueError):
            make_counterfactual(row, "rule")
        with self.assertRaises(ValueError):
            make_counterfactual(self.cases[0], "invalid")

    def test_source_audit_receipt(self):
        cases, receipt = source_pair_audit()
        self.assertEqual(len(cases), 256)
        self.assertEqual(receipt["paired_cases"], 256)
        self.assertFalse(receipt["sealed_test_evaluated"])
        self.assertTrue(receipt["both_oracle_targets_equal"])


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "optional torch physical-midpoint geometry")
class ROC02EDoseMatcher(unittest.TestCase):
    @staticmethod
    def fake_midpoint(model, x, mask):
        h = torch.zeros((1, 128), dtype=torch.float64)
        h[:, 0] = x[:, 0, 0].double()
        return h

    def test_first_crossing_at_registered_radius(self):
        x0 = torch.zeros(1, 1, 79)
        x1 = x0.clone()
        x1[:, 0, 0] = 1.0
        mask = torch.ones(1, 1)
        for radius in RADII:
            out = interpolate_to_midpoint_radius(None, x0, x1, mask,
                                                  self.fake_midpoint, radius)
            self.assertTrue(out["available"])
            self.assertLess(abs(out["t"] - radius), 2e-5)
            self.assertLess(abs(out["actual_midpoint_l2"] - radius), 2e-5)
            self.assertFalse(out["soft_input_has_symbolic_label"])

    def test_abstain_if_full_legal_edit_is_too_small(self):
        x0 = torch.zeros(1, 1, 79)
        x1 = x0.clone()
        x1[:, 0, 0] = 0.03
        out = interpolate_to_midpoint_radius(None, x0, x1, torch.ones(1, 1),
                                              self.fake_midpoint, 0.20)
        self.assertFalse(out["available"])
        self.assertIn("below radius", out["reason"])


if __name__ == "__main__":
    unittest.main()

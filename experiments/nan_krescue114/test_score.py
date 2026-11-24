"""Deterministic endpoint, sign, failure and prospective gate tests."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from .score import bind_parent_orientation, gate, margin_footprint

REG = json.loads(Path(__file__).with_name("REGISTRATION.json").read_bytes())
CRITERIA = REG["outcome"] | {"registered_roots": REG["source"]["roots"]}


def panel(intact=.25, restored=.0625, sham=.03125, closed=16):
    return [
        dict(root=f"r{i:02d}", status="closed", intact_advantage=intact,
             restored_advantage=restored, removed_advantage=intact-restored, sham_only_removal=sham)
        if i < closed else {"root": f"r{i:02d}", "status": "geometry_unqualified"}
        for i in range(16)
    ]


def observations(intact_active=.5, intact_sham=.125, rescued_active=.1875, rescued_sham=.0625):
    result = []
    for own, peer in (("E", "H"), ("H", "E")):
        for sign in (1, -1):
            for raw, restored in (
                (0., 0.), (intact_active, rescued_active), (0., 0.), (intact_sham, rescued_sham),
            ):
                for rescue, value in ((False, raw), (True, restored)):
                    result.append({
                        "rescued": rescue, "phases": [
                            {"labels": {own: 8., peer: -4.}},
                            {"labels": {own: 2., peer: -1. + 2 * sign * value}},
                        ],
                    })
    return result


class KRescueTest(unittest.TestCase):
    def decision(self, rows):
        return gate(rows, {f"r{i:02d}" for i in range(16)}, CRITERIA)

    def test_donor_orientation_sham_and_full_root(self):
        r = margin_footprint(observations(), "E", "H")
        self.assertEqual(r, {"intact_advantage": .375, "restored_advantage": .125,
                             "removed_advantage": .25, "sham_only_removal": .0625})
        swap = margin_footprint(observations(), "H", "E")
        self.assertEqual(swap, {key: -value for key, value in r.items()})
        bind_parent_orientation(r, {"second_pass": -.375, "first_pass": .125, "delayed_with_clean_k": -.25,
                                    "k_rescue_removed": -.25})
        with self.assertRaisesRegex(ValueError, "orientation|audit"):
            bind_parent_orientation(swap, {"second_pass": -.375, "first_pass": .125, "delayed_with_clean_k": -.25,
                                          "k_rescue_removed": -.25})

    def test_unchanged_k_response_has_zero_removal(self):
        r = margin_footprint(observations(rescued_active=.5, rescued_sham=.125), "E", "H")
        self.assertEqual(r["intact_advantage"], r["restored_advantage"])
        self.assertEqual(r["removed_advantage"], 0.)

    def test_only_sham_restoration_moves_the_controlled_contrast(self):
        r = margin_footprint(observations(rescued_active=.5), "E", "H")
        self.assertEqual(r["removed_advantage"], -.0625)
        self.assertEqual(r["sham_only_removal"], .0625)

    def test_omission_phase_mutation_reordering_and_nan_fail(self):
        source = observations()
        with self.assertRaisesRegex(ValueError, "needs every"):
            margin_footprint(source[:-1], "E", "H")
        altered = deepcopy(source)
        altered[1]["phases"][0]["labels"]["E"] += 1
        with self.assertRaisesRegex(ValueError, "pass-1 head"):
            margin_footprint(altered, "E", "H")
        with self.assertRaisesRegex(ValueError, "reordered"):
            margin_footprint([source[1], source[0], *source[2:]], "E", "H")
        altered = deepcopy(source)
        altered[1]["phases"][1]["labels"]["H"] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            margin_footprint(altered, "E", "H")

    def test_gate_has_a_reachable_go_with_unmeasured_roots_accounted_for(self):
        verdict = self.decision(panel(closed=12))
        self.assertEqual(verdict["decision"], "GO_TO_LARGER_SHAM_CALIBRATION")
        self.assertEqual(verdict["gate_mean_removed_logits"], .140625)
        self.assertEqual(verdict["measured_only_mean_removed_logits"], .1875)
        self.assertEqual(verdict["unmeasured_roots"], 4)
        self.assertIsNone(verdict["sham_tolerance_bound"])
        self.assertEqual(self.decision(panel(closed=11))["decision"], "NO_GO")

    def test_gate_rejects_sham_dominance_or_no_intact_advantage(self):
        high_sham = self.decision(panel(sham=.1875))
        self.assertIn("removal_does_not_exceed_observed_sham_maximum", high_sham["reasons"])
        no_active = self.decision(panel(intact=0., restored=-.25))
        self.assertIn("intact_donor_advantage_below_minimum", no_active["reasons"])
        null = self.decision(panel(restored=.25))
        self.assertIn("removed_advantage_below_minimum", null["reasons"])
        negative = self.decision(panel(restored=.5))
        self.assertEqual(negative["gate_mean_removed_logits"], -.25)
        self.assertEqual(negative["decision"], "NO_GO")

    def test_gate_fails_closed_on_incompatible_or_duplicated_evidence(self):
        rows = panel()
        with self.assertRaisesRegex(ValueError, "entire frozen root"):
            self.decision([rows[0], *rows[:-1]])
        rows[0]["removed_advantage"] = 1.
        with self.assertRaisesRegex(ValueError, "intact minus restored"):
            self.decision(rows)
        rows = panel(closed=12)
        rows[-1]["removed_advantage"] = 0.
        with self.assertRaisesRegex(ValueError, "unmeasured"):
            self.decision(rows)


if __name__ == "__main__":
    unittest.main()

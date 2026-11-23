from unittest import TestCase

import torch

from ppt_gym.failed_prefix import CHARGE, PREFIX_COST
from ppt_gym.live_quote_policy import FEATURES
from ppt_gym.residue_acquisition import COMMIT, execute, forecast, queries
from ppt_gym.residue_acquisition_policy import FeedbackSlotActor
from ppt_gym.residue_recovery import cases


class ResidueAcquisitionTests(TestCase):
    def setUp(self):
        self.menu = [{"members": [0, 1, 2]}, {"members": [6, 7, 8]}]
        self.prior = [.5, 0., 0., .5, 0., 0., 0., 0., 0.]
        self.support = [bool(mass) for mass in self.prior]
        self.target = {"routes": [
            {"residues": [0, 1, 2], "pass_cost": 3, "fail_cost": 2},
            {"residues": [3, 4, 5], "pass_cost": 3, "fail_cost": 2},
            {"residues": [6, 7, 8], "pass_cost": 4, "fail_cost": 2},
        ]}

    def test_paid_check_charges_both_outcomes_and_updates_the_next_route(self):
        scores = forecast(self.target, 4, self.menu, self.prior, self.support)
        self.assertEqual(set(scores), {COMMIT, 1})
        self.assertEqual(set(forecast(self.target, 3, self.menu, self.prior, self.support)), {COMMIT})
        for bit in (False, True):
            target = {**self.target, "truth": (bit, not bit, False)}
            row = execute(target, 4, self.menu, self.prior, 1, [bit, False])
            self.assertEqual(row["second"]["allowance"], 3)
            self.assertEqual(row["second"]["trace"][0]["arm"], 0 if bit else 1)
            self.assertEqual(row["query"], {"action": 1, "passed": bit, "charge": 1})
            self.assertAlmostEqual(row["utility"], 1 - CHARGE * (PREFIX_COST + 4))
            self.assertAlmostEqual(scores[1], row["utility"] + CHARGE * PREFIX_COST)
        self.assertGreater(scores[1], scores[COMMIT])

    def test_query_menu_is_public_and_uses_five_distinct_predicates(self):
        for pair in cases("development", 40):
            menu = queries(pair)
            carrier, _ = pair["certificate"].rsplit(" = ", 1)
            self.assertEqual(len({tuple(row["members"]) for row in menu}), 5)
            self.assertTrue(all(carrier in row["tactic"] for row in menu))
            self.assertEqual(menu, queries({**pair, "residue": -1, "certificate": carrier + " = -1"}))

    def test_slot_reset_removes_source_features_only_at_the_boundary(self):
        actor = FeedbackSlotActor().eval()
        original = torch.zeros(1, 3, FEATURES)
        changed = original.clone()
        changed[:, 0] = 1.
        with torch.no_grad():
            self.assertTrue(torch.equal(actor(original, reset=True), actor(changed, reset=True)))
            self.assertFalse(torch.equal(actor(original), actor(changed)))

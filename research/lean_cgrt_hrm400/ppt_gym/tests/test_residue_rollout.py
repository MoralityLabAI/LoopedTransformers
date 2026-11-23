from unittest import TestCase

import torch

from configs.run_ppt_residue_online import EDITS, perturb
from ppt_gym.failed_prefix import CHARGE, PREFIX_COST
from ppt_gym.live_quote_policy import FEATURES
from ppt_gym.residue_rollout import FIRST_QUERY, FIRST_ROUTE, Recovery, State, greedy, mask
from ppt_gym.residue_rollout_policy import PersistentRecoverySlots


class ResidueRolloutTests(TestCase):
    def setUp(self):
        groups = ([0], [1], [2])
        target = {"routes": [{"residues": group, "pass_cost": 3, "fail_cost": 2} for group in groups]}
        self.env = Recovery(target, [{"members": group} for group in groups])

    def test_two_adaptive_checks_improve_exact_value_at_same_budget(self):
        plan = self.env.planner([1 / 3] * 3)
        starts = [State(5, mask(range(3)), depth) for depth in range(3)]
        values = [max(plan(state).values()) for state in starts]
        self.assertGreater(values[2], max(values[:2]))
        actual = []
        for truth in ([True, False, False], [False, True, False], [False, False, True]):
            row = self.env.rollout(starts[2], truth, truth, lambda state, _: greedy(plan(state)))
            self.assertTrue(row["proved"])
            self.assertLessEqual(row["checks"], PREFIX_COST + 5)
            actual.append(row["utility"])
        self.assertAlmostEqual(sum(actual) / 3, values[2] - CHARGE * PREFIX_COST)

    def test_query_can_follow_rejected_route_and_is_never_rebought(self):
        truth = [False, False, True]
        steps = iter((FIRST_ROUTE, FIRST_QUERY + 1, FIRST_ROUTE + 2))
        row = self.env.rollout(State(6, 0b111, 2), truth, truth, lambda _, __: next(steps))
        self.assertTrue(row["proved"])
        self.assertEqual(row["checks"], PREFIX_COST + 6)
        self.assertEqual([step["support"] for step in row["trace"]], [0b111, 0b110, 0b100])
        steps = iter((FIRST_QUERY, FIRST_QUERY))
        with self.assertRaisesRegex(ValueError, "unavailable"):
            self.env.rollout(State(6, 0b111, 2), truth, truth, lambda _, __: next(steps))

    def test_feedback_counterfactual_and_matched_sham_edit_only_the_feedback_slot(self):
        model = PersistentRecoverySlots().eval()
        with torch.no_grad():
            initial = model.initialize(torch.zeros(1, 2, FEATURES))
            event = torch.zeros(1, FEATURES)
            actual = model.observe(initial, event)
            swapped, dose = perturb(model, initial, actual, event, EDITS[0], 1)
            sham, sham_dose = perturb(model, initial, actual, event, EDITS[1], 1)
        self.assertGreater(dose, 0.)
        self.assertEqual(dose, sham_dose)
        self.assertTrue(torch.equal(swapped[:, 1], actual[:, 1]))
        self.assertTrue(torch.equal(sham[:, 1], actual[:, 1]))
        self.assertAlmostEqual(float(((swapped - actual) * (sham - actual)).sum()), 0., places=5)

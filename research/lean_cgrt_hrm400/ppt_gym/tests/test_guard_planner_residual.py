import unittest

import torch

from ppt_gym.guard_planner_residual import RADIUS, BoundedCorrection, packed_action
from ppt_gym.guard_recovery import feasible_menu, initial
from ppt_gym.guard_recovery_learning import baked_policy, baked_scores
from ppt_gym.live_quote_policy import FEATURES, TwoSlotQuoteActor


class GuardPlannerResidualTests(unittest.TestCase):
    def test_initial_actor_leaves_the_supervised_scores_intact(self):
        model = BoundedCorrection(TwoSlotQuoteActor())
        x = torch.randn(2, 3, FEATURES)
        base = torch.tensor([.5, .3])
        torch.testing.assert_close(model(x, base), base)
        with torch.no_grad():
            model.actor.head.bias.fill_(100.)
        self.assertTrue((model.residual(x).abs() <= RADIUS).all())
        self.assertEqual(int(model(x, base).argmax()), 0)

    def test_control_and_explicit_base_values_agree(self):
        state = initial(8)
        probability = [.9, .2, .9, .9]
        scores = baked_scores(probability, action_space=feasible_menu)(state)
        best = max(scores, key=scores.__getitem__)
        self.assertEqual(baked_policy(probability, action_space=feasible_menu)(state), best)
        fields = packed_action([[0.] * 20] * 4, probability, [[0.] * 8] * 2, state, best, scores[best], scores[best])
        self.assertEqual(len(fields[2]), FEATURES)
        self.assertEqual(fields[2][-2:], [scores[best], 0.])

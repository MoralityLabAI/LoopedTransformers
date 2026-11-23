from functools import lru_cache
from math import exp
import unittest

import torch

from ppt_gym.guard_recovery import ACTIONS, ALLOWANCES, CHARGE, feasible_menu, initial, transition
from ppt_gym.guard_recovery_learning import GUARD_WORLDS
from ppt_gym.guard_return import CHANNELS, TEMPERATURE, ReturnPlanner


class GuardReturnTests(unittest.TestCase):
    def test_stochastic_rollout_equals_independent_lean_action_enumeration(self):
        planner = ReturnPlanner()
        weights = torch.tensor([[.6, .1, .2, -.4], [-.3, .7, .2, .1]])
        public = planner.scores(weights).requires_grad_()
        indices = torch.tensor([2, 0])
        actual = planner.rollout(indices, public)
        for row, index in enumerate(indices.tolist()):
            @lru_cache(maxsize=None)
            def enumerate_return(state):
                if state is None or not feasible_menu(state):
                    return 0.
                numerator, denominator = 0., 1.
                for action in feasible_menu(state):
                    following, observation = transition(state, action, GUARD_WORLDS[index])
                    reward = float(action == "close" and observation is True) - CHARGE
                    weight = exp(float(public[row, planner.index[state], ACTIONS.index(action)]) / TEMPERATURE)
                    numerator += weight * (reward + enumerate_return(following))
                    denominator += weight
                return numerator / denominator

            for col, budget in enumerate(ALLOWANCES):
                self.assertAlmostEqual(actual[row, col].item(), enumerate_return(initial(budget)), places=6)
        gradient = torch.autograd.grad(actual.sum(), public)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), .1)

    def test_correction_channels_follow_tactic_and_paid_evidence(self):
        planner = ReturnPlanner()
        heads = torch.arange(4 * CHANNELS, dtype=torch.float32).reshape(1, 4, CHANNELS)
        addends = planner.addends(heads)[0]
        tight = planner.index[initial(6)]
        spare = planner.index[initial(7)]
        self.assertEqual(addends[tight, ACTIONS.index("enter_0")].item(), 2)
        self.assertEqual(addends[spare, ACTIONS.index("enter_0")].item(), 3)
        self.assertEqual(addends[spare, ACTIONS.index("probe_2")].item(), 10)
        entered, _ = transition(initial(8), "enter_1", GUARD_WORLDS[0])
        self.assertEqual(addends[planner.index[entered], ACTIONS.index("verify_0")].item(), 11)
        known, _ = transition(initial(8), "probe_0", GUARD_WORLDS[0])
        self.assertEqual(addends[planner.index[known]].count_nonzero().item(), 0)

import unittest

import torch

from configs.run_ppt_guard_belief import policy_scores
from ppt_gym.guard_belief import BeliefCorrection, BeliefPlanner, belief_inputs
from ppt_gym.guard_recovery import ACTIONS, PROBE, feasible_menu, initial, transition
from ppt_gym.guard_recovery_learning import GUARD_WORLDS, baked_scores, false_guard_weights
from ppt_gym.live_quote_policy import TwoSlotQuoteActor


class GuardBeliefTests(unittest.TestCase):
    def test_batched_values_agree_with_scalar_planner_at_every_reachable_state(self):
        planner = BeliefPlanner()
        priors = ([.91, .77, .45, .33], [.8] * 4, [.02, .98, .6, .98])
        weights = torch.tensor([false_guard_weights(p) for p in priors], dtype=torch.float64)
        dense = planner.scores(weights.log())
        for index, probability in enumerate(priors):
            scalar = baked_scores(probability, action_space=feasible_menu)
            zero = policy_scores(weights[index].tolist(), [0.] * 4)
            for state in planner.states:
                expected = scalar(state)
                self.assertEqual(expected, zero(state))
                for action in ACTIONS:
                    actual = dense[index, planner.index[state], ACTIONS.index(action)].item()
                    if action in expected:
                        self.assertAlmostEqual(actual, expected[action], places=7)
                    else:
                        self.assertEqual(actual, -float("inf"))

    def test_paid_false_guard_eliminates_correction_gradient(self):
        planner = BeliefPlanner()
        opening = initial(8)
        proved_false, _ = transition(opening, PROBE[0], GUARD_WORLDS[0])
        logits = torch.tensor([[.1, .2, -.5, .8]], requires_grad=True)
        scores = planner.scores(logits)
        score = scores[0, planner.index[opening], ACTIONS.index("enter_0")]
        self.assertGreater(torch.autograd.grad(score, logits, retain_graph=True)[0].abs().sum().item(), .01)
        certain = scores[0, planner.index[proved_false], ACTIONS.index("enter_1")]
        torch.testing.assert_close(torch.autograd.grad(certain, logits)[0], torch.zeros_like(logits), atol=1e-7, rtol=0)

    def test_tokens_follow_swapped_witnesses_and_correction_starts_zero(self):
        descriptors = [[float(i)] * 20 for i in range(4)]
        probabilities = [.96, .2, .7, .9]
        warm = [[.1] * 8, [.2] * 8]
        tokens = belief_inputs(descriptors, probabilities, warm)
        swapped = belief_inputs(descriptors[2:] + descriptors[:2], probabilities[2:] + probabilities[:2], warm[::-1])
        self.assertEqual(tokens[2:] + tokens[:2], swapped)
        model = BeliefCorrection(TwoSlotQuoteActor())
        torch.testing.assert_close(model.correction(torch.tensor([tokens])), torch.zeros(1, 4))

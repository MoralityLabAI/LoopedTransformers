import unittest

import torch

from ppt_gym.failed_prefix import trace_case
from ppt_gym.failed_prefix_learning import COUNT
from ppt_gym.failed_prefix_menu import BOUND, MenuActor, corrections, group_candidates, policy, rank_oracle


class FailedPrefixMenuTests(unittest.TestCase):
    def test_grouping_attaches_affine_value_to_actual_candidate(self):
        inputs = torch.randn(3, 3, 30)
        tokens, states, base = group_candidates(inputs, [120, 122, 168], [.5, .3, .7])
        self.assertEqual(states, [40, 56])
        self.assertFalse(base[0, 1].isfinite())
        torch.testing.assert_close(tokens[0, 2, :2], inputs[1, :2])
        self.assertAlmostEqual(tokens[1, 0, 2, 27].item(), .7)

    def test_recurrent_menu_matches_joint_permutation_and_starts_at_anchor(self):
        model = MenuActor()
        tokens = torch.randn(2, 3, 3, 30)
        base = torch.tensor([[.3, .2, -torch.inf], [.1, .5, .6]])
        torch.testing.assert_close(corrections(model, tokens[None], base[None]), torch.zeros(1, 2, 3))
        with torch.no_grad():
            model.head.weight.fill_(.2)
        order = [2, 0, 1]
        delta = corrections(model, tokens[None], base[None])
        self.assertTrue((delta.abs() <= BOUND).all())
        permuted = corrections(model, tokens[None][:, :, order], base[None][:, :, order])
        torch.testing.assert_close(permuted, delta[:, :, order], atol=1e-6, rtol=1e-6)

    def test_policy_does_not_learn_a_different_stop_gate(self):
        base = torch.tensor([[[.02, -.2, -torch.inf], [-.03, -.1, -torch.inf],
                              [-torch.inf, -torch.inf, -torch.inf]]])
        delta = torch.tensor([[[-.1, .1, 0.], [.1, .1, 0.], [0., 0., 0.]]])
        result = policy(base, delta, torch.tensor([[56, 40, COUNT]]), torch.tensor([[True, False, False]]))
        torch.testing.assert_close(result[0, 56].sum(), torch.tensor(1.))
        self.assertEqual(result[0, 56, 2].item(), 0.)
        self.assertEqual(result[0, 40].count_nonzero().item(), 0)

    def test_truth_oracle_cannot_escape_fixed_stop_or_rank_envelope(self):
        routes = [{"pass_cost": 2, "fail_cost": 2}, {"pass_cost": 4, "fail_cost": 3},
                  {"pass_cost": 5, "fail_cost": 3}]
        truth = [False, True, True]
        # Route 1 would prove the goal but cannot outrank route 0 within the bound.
        anchor = {40: {0: .6, 1: .3, 2: .2}, 25: {1: -.1}}
        oracle = rank_oracle(routes, truth, anchor)
        row = trace_case({"routes": routes, "truth": truth}, 5, oracle)
        self.assertFalse(row["proved"])
        self.assertEqual([event["arm"] for event in row["trace"]], [0])

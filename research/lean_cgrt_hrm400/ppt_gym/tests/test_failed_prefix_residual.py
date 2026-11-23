import unittest

import torch

from ppt_gym.failed_prefix import cases, planned_values, trace_case
from ppt_gym.failed_prefix_learning import observations
from ppt_gym.failed_prefix_residual import BOUND, RecoveryCorrection, augment, bounded_oracle, possible_choices, redistribute
from ppt_gym.live_quote_policy import TwoSlotQuoteActor


class FailedPrefixResidualTests(unittest.TestCase):
    def test_recurrent_readout_starts_at_the_frozen_values_and_stays_bounded(self):
        actor = RecoveryCorrection(TwoSlotQuoteActor())
        tokens = torch.randn(3, 3, 30)
        torch.testing.assert_close(actor(tokens), torch.zeros(3))
        with torch.no_grad():
            actor.actor.head.bias.fill_(100)
        self.assertTrue((actor(tokens).abs() <= BOUND).all())

    def test_observation_uses_conditional_values_after_paid_rejection(self):
        case = next(cases("source", 1))
        routes = case["routes"]
        tokens, ids, _, _ = observations(routes, [[0.] * 8] * 3)
        tokens = torch.tensor(tokens)
        weights = [1.] * 6
        augmented, values = augment(tokens, ids, routes, weights)
        expected = planned_values(routes, weights)
        for row, code in enumerate(ids):
            state, arm = divmod(code, 3)
            remaining, failed = divmod(state, 8)
            self.assertEqual(values[row], expected(remaining, failed)[arm])
            if failed & 1:
                self.assertEqual(augmented[row, 0, 25].item(), 0.)
        self.assertTrue((augmented[:, :, :17] == tokens[:, :, :17]).all())

    def test_correction_envelope_preserves_the_frozen_choice(self):
        self.assertEqual(possible_choices({0: .8, 1: .485}), (0,))
        self.assertEqual(possible_choices({0: .1, 1: .01, 2: -.2}), (0, 1, None))
        routes = [{"pass_cost": 2, "fail_cost": 2}, {"pass_cost": 4, "fail_cost": 3},
                  {"pass_cost": 5, "fail_cost": 3}]
        oracle = bounded_oracle(routes, [1.] * 6, [True, False, True])
        result = trace_case({"routes": routes, "truth": [True, False, True]}, 5, oracle)
        self.assertTrue(result["proved"])
        self.assertEqual(result["trace"][0]["arm"], 0)

    def test_swap_and_orthogonal_sham_have_identical_dose_and_zero_sum(self):
        original = {0: .02, 1: -.05, 2: .04}
        swap, dose = redistribute(original, 0, 1)
        sham, sham_dose = redistribute(original, 0, 1, sham_sign=-1)
        swap_diff = [swap[arm] - original[arm] for arm in original]
        sham_diff = [sham[arm] - original[arm] for arm in original]
        self.assertEqual(dose, sham_dose)
        self.assertAlmostEqual(sum(swap_diff), 0.)
        self.assertAlmostEqual(sum(sham_diff), 0.)
        self.assertAlmostEqual(sum(a * b for a, b in zip(swap_diff, sham_diff, strict=True)), 0.)
        self.assertAlmostEqual(sum(a * a for a in swap_diff), dose * dose)
        self.assertAlmostEqual(sum(a * a for a in sham_diff), dose * dose)

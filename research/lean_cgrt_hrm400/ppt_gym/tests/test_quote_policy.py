from unittest import TestCase

import torch

from ppt_gym.quote_policy import ACTIONS, BeliefQuotePolicy, FullHistoryQuoteTRM, history_index


class QuotePolicyTests(TestCase):
    def setUp(self):
        self.plans = torch.zeros(2, 2, 4)
        self.context = torch.zeros(2, 3)
        self.fees = torch.tensor([[1., 1., 3.], [1., 1., 3.]])
        self.slack = torch.tensor([[2.], [2.]])
        self.history = torch.tensor([history_index(()), history_index((("ledger_u", 1),))])

    def test_belief_backup_charges_the_quote_before_choosing_a_continuation(self):
        model = BeliefQuotePolicy(4)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            scores, forecast = model(self.plans, self.context, self.fees, self.slack, self.history)
        torch.testing.assert_close(scores[0, :4], torch.tensor([.5, .5, .4375, .4375]))
        torch.testing.assert_close(scores[1, [0, 1, 3]], torch.tensor([.5, .5, .4375]))
        self.assertEqual(forecast[1, 0].item(), 1.)
        self.assertEqual(forecast[0, 0].item(), .5)
        self.assertTrue(torch.isneginf(scores[1, ACTIONS.index("ledger_u")]))
        self.assertTrue(torch.isneginf(scores[:, ACTIONS.index("reconcile")]).all())

    def test_models_share_the_action_mask_and_have_finite_fit_gradients(self):
        masks = []
        for model in (BeliefQuotePolicy(4), FullHistoryQuoteTRM(4)):
            score, forecast = model(self.plans, self.context, self.fees, self.slack, self.history)
            legal = score.isfinite()
            masks.append(legal)
            (score[legal].square().mean() + forecast[:, 1].mean()).backward()
            self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters()))
        torch.testing.assert_close(masks[0], masks[1])

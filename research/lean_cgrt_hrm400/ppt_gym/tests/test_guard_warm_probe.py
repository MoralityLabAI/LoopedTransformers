import unittest

import torch

from configs.freeze_ppt_guard_curriculum import NAMESPACE, PAIRS, audit_cohort
from configs.run_ppt_guard_warm_probe import BLEND, blended, probe_policy
from ppt_gym.guard_belief import LOGIT_BOUND
from ppt_gym.guard_probe import ProbeGraph
from ppt_gym.guard_recovery import cases


class WarmProbeTests(unittest.TestCase):
    def test_deep_curriculum_has_new_paired_roots_and_native_depth(self):
        original = {"source": list(cases("source", 48)), "development": list(cases("development", 40))}
        deep = list(cases("development", PAIRS, namespace=NAMESPACE))
        self.assertEqual(deep, list(cases("development", PAIRS, namespace=NAMESPACE)))
        audit = audit_cohort(deep, original)
        self.assertEqual(audit["source_overlap"], 0)
        self.assertGreater(audit["development_overlap"], 0)
        with self.assertRaisesRegex(ValueError, "overlaps"):
            audit_cohort(deep, {**original, "development": deep})

    def test_adaptation_starts_at_anchor_and_stays_inside_bound(self):
        original = torch.tensor([[-LOGIT_BOUND, LOGIT_BOUND, .1, .2]])
        torch.testing.assert_close(blended(original, original), original)
        proposed = torch.tensor([[LOGIT_BOUND, -LOGIT_BOUND, .1, .2]], requires_grad=True)
        adapted = blended(original, proposed)
        self.assertLessEqual((adapted - original).abs().max().item(), 2 * LOGIT_BOUND * BLEND + 1e-6)
        self.assertLessEqual(adapted.abs().max().item(), LOGIT_BOUND)
        torch.testing.assert_close(torch.autograd.grad(adapted.sum(), proposed)[0],
                                   torch.full_like(proposed, BLEND))

    def test_probe_kl_is_finite_with_unaffordable_actions(self):
        graph = ProbeGraph()
        encoded = graph.encode([[0.] * 20] * 4, [.42, .88, .76, .97], [[0.] * 8] * 2)
        _, prior, _, offsets, payoffs, _, _, _ = (torch.tensor([part]) for part in encoded)
        anchor = torch.zeros(1, len(graph.states), 4)
        adjusted = torch.full_like(anchor, .2).requires_grad_()
        policy, log_policy = probe_policy(graph, prior, adjusted, offsets, payoffs)
        _, log_anchor = probe_policy(graph, prior, anchor, offsets, payoffs)
        self.assertTrue((policy[:, ~graph.available] == 0.).all())
        loss = (policy * (log_policy - log_anchor)).sum()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(torch.autograd.grad(loss, adjusted)[0]).all())

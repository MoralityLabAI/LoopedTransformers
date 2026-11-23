"""Contracts for the downstream expert-selection intervention."""

import unittest
from types import SimpleNamespace

from research_gym.inkling_routing_01.hooks import reader_call
from research_gym.inkling_routing_02.k_factorial import VECTOR, FactorialTrace
from research_gym.inkling_routing_02.k_only import PASS, KOnlyTrace
from research_gym.inkling_routing_02.k_trace import DownstreamTrace, router_weights


class KTraceTests(unittest.TestCase):
    def test_swap_changes_only_two_last_token_sets_and_uses_recipient_scores(self):
        import torch

        reader = KOnlyTrace(None, "synthetic-root", clamp_beta=True)
        reader.visits = reader_call(PASS) + 1
        trace = DownstreamTrace(reader, 2, swap_layer=12)
        module = SimpleNamespace(score_fn=torch.sigmoid, e_score_correction_bias=torch.zeros(4),
                                 routed_scaling_factor=1.5)
        logits = torch.arange(4, dtype=torch.float32).repeat(16, 1)
        tokens = [reader.lanes[PASS, arm] * 2 + 1 for arm in ("zero_k", "zero_k_unclamped")]
        logits[tokens[0]] = logits[tokens[0]].flip(0)
        indices = logits.topk(2).indices
        weights = router_weights(module, logits, indices)
        delivered_logits, delivered_weights, delivered_indices = trace.save_router(12, module, (), (logits, weights, indices))
        self.assertIs(delivered_logits, logits)
        torch.testing.assert_close(delivered_indices[tokens], indices[tokens[::-1]])
        torch.testing.assert_close(delivered_weights[tokens], router_weights(module, logits[tokens], indices[tokens[::-1]]))
        others = [token for token in range(len(logits)) if token not in tokens]
        torch.testing.assert_close(delivered_indices[others], indices[others])
        torch.testing.assert_close(delivered_weights[others], weights[others])

    def test_fixed_set_targets_only_the_last_vector_token(self):
        import torch

        reader = KOnlyTrace(None, "synthetic-root", clamp_beta=True)
        reader.visits = reader_call(PASS) + 1
        trace = FactorialTrace(reader, 2, {14: [0, 1], 16: [1, 2]})
        module = SimpleNamespace(score_fn=torch.sigmoid, e_score_correction_bias=torch.zeros(4),
                                 routed_scaling_factor=1.5)
        token = VECTOR * 2 + 1
        for layer, chosen, shift in ((14, [0, 1], 0.0), (16, [1, 2], 0.5)):
            logits = torch.arange(4, dtype=torch.float32).repeat(16, 1)
            logits[token, 1] += shift
            indices = logits.topk(2).indices
            weights = router_weights(module, logits, indices)
            delivered_logits, delivered_weights, delivered_indices = trace.save_router(  # native hook contract
                layer, module, (), (logits, weights, indices))
            self.assertIs(delivered_logits, logits)
            self.assertEqual(delivered_indices[token].tolist(), chosen)
            expected = router_weights(module, logits[token:token + 1], torch.tensor([chosen]))
            torch.testing.assert_close(delivered_weights[token:token + 1], expected)
            others = [i for i in range(len(logits)) if i != token]
            torch.testing.assert_close(delivered_indices[others], indices[others])
            torch.testing.assert_close(delivered_weights[others], weights[others])

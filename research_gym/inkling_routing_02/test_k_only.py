"""Offline contracts for the K-only treatment and development gate."""

import unittest

import numpy as np

from research_gym.inkling_routing_01.hooks import reader_call
from research_gym.inkling_routing_02.k_only import ARMS, K, PASS, KOnlyTrace, clamp_k_messages, summarize, total_variation


class KOnlyTests(unittest.TestCase):
    def test_delivers_only_final_token_k_and_rescues_both_edits(self):
        import torch

        trace = KOnlyTrace(None, "synthetic-root")
        n = len(trace.lanes)
        bank = torch.zeros(n, 2, 4)
        raw = torch.zeros(n, 2, 4, 4)
        raw[:, 0] = 7
        zero = trace.lanes[PASS, "zero_k"]
        vector = trace.lanes[PASS, "vector_k"]
        raw[[zero, trace.lanes[PASS, "zero_rescue"]], 1] = 2
        raw[[vector, trace.lanes[PASS, "vector_rescue"]], 1] = 3
        trace.visits = reader_call(PASS) + 1
        trace.readers[PASS] = {}
        delivered = trace.after_reader(None, (None, bank, 0), raw)
        self.assertTrue(torch.equal(delivered[:, 0], raw[:, 0]))
        expected = torch.zeros_like(delivered[:, 1])
        expected[zero, K], expected[vector, K] = 2, 3
        self.assertTrue(torch.equal(delivered[:, 1], expected))
        self.assertEqual(raw[zero, 1, 0, 0].item(), 2)

    def test_tv_measures_probability_mass_moved(self):
        first = np.log([0.2, 0.5, 0.3])
        second = np.log([0.5, 0.2, 0.3])
        self.assertAlmostEqual(total_variation(first, second), 0.3)
        self.assertEqual(total_variation(first, first), 0)

    def test_clamp_uses_shared_weight_with_each_arms_own_bank(self):
        import torch
        from types import SimpleNamespace

        query = torch.zeros(1, 4, 2)
        query[0, K, 0] = 1
        reader = SimpleNamespace(query=query, norm=torch.nn.Identity())
        bank2 = torch.zeros(2, 1, 1, 2)
        bank3 = torch.tensor([[[1., 2.]], [[3., 4.]]])
        messages, beta = clamp_k_messages(reader, (bank2, bank3, 0), [0, 1])
        native_betas = torch.sigmoid(bank3[:, 0, 0] / 2**0.5)
        torch.testing.assert_close(beta, native_betas.mean())
        torch.testing.assert_close(messages, beta * bank3[:, 0])

    def test_expansion_requires_both_replays_to_clear_noise_adjusted_floor(self):
        rows = [
            {"root_id": "dev", "replay": replay, "arm": arm,
             "tv_same_arm_cross_replay": 0.002, "tv_from_clean": 0,
             "tv_zero_vs_vector": tv, "matched_beta_gap": 0, "k_message_l2_zero_vs_vector": 1,
             "top_token_id": 0, "top_token_flip": False, "reference_logprob_delta": 0,
             "reader_scores": {"reconstruction_relative_error": [0]}}
            for replay, tv in enumerate((0.021, 0.019)) for arm in ("clean", *ARMS)
        ]
        result = summarize(rows)
        self.assertAlmostEqual(result["minimum_meaningful_tv"], 0.02)
        self.assertAlmostEqual(result["mean_paired_root_tv"], 0.02)
        self.assertFalse(result["expand"])
        frozen = summarize(rows, frozen_floor=0.01)
        self.assertEqual(frozen["minimum_meaningful_tv"], 0.01)
        self.assertFalse(frozen["noise_resolves_floor"])
        self.assertFalse(frozen["expand"])

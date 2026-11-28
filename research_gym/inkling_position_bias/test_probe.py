"""CPU numerical and provenance tests. All weights here are synthetic."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from research_gym.inkling_position_bias.probe import (
    attention_delta, bias_at_distances, checkpoint_tensors, conv_taps,
    layer0_inputs, relative_curves, rms_norm,
)


class InklingBiasTests(unittest.TestCase):
    def test_layer0_input_two_weighted_rmsnorms(self) -> None:
        emb = np.array([[3.0, 4.0], [1.0, -2.0]], dtype=np.float32)
        e_gain = np.array([2.0, 0.75], dtype=np.float32)
        a_gain = np.array([0.5, 3.0], dtype=np.float32)
        self.assertTrue(np.allclose(layer0_inputs(emb, e_gain, a_gain),
                                    rms_norm(rms_norm(emb, e_gain), a_gain)))
        self.assertFalse(np.allclose(layer0_inputs(emb, e_gain, a_gain), emb))

    def test_projection_distances_and_head_specific_scales(self) -> None:
        states = np.array([[2.0, 1.0]], dtype=np.float32)
        wr = np.zeros((4, 2), dtype=np.float32)  # heads=2, d_rel=2
        wr[0, 0] = 1.0
        wr[2, 1] = 1.0
        scale = np.ones((2, 2), dtype=np.float32)
        scale[1, 0] = 3.0
        basis = np.array([[1.0, 2.0, 4.0, 8.0], [0.0, 0.0, 0.0, 0.0]])
        curves = relative_curves(states, wr, scale, basis, clip=1e4)
        self.assertEqual(curves.shape, (1, 2, 4))
        self.assertTrue(np.allclose(curves[0, 0], [2, 4, 8, 16]))
        self.assertTrue(np.allclose(curves[0, 1], [3, 6, 12, 24]))

    def test_l2_clip_and_logscale(self) -> None:
        wr = np.array([[2, 0]], dtype=np.float32)
        x = np.array([[2, 0], [2, 0]], dtype=np.float32)
        scale = np.ones((1, 1), dtype=np.float32)
        basis = np.ones((1, 3), dtype=np.float32)
        rel = relative_curves(x, wr, scale, basis, clip=2, position_ids=np.array([3, 7]),
                              log_scaling=True, log_floor=4, log_alpha=1)
        self.assertAlmostEqual(float(rel[0, 0, 0]), 2, places=5)
        self.assertAlmostEqual(float(rel[1, 0, 0]), 2 * (1 + np.log(2)), places=5)
        with self.assertRaises(ValueError):
            relative_curves(x, wr, scale, basis, log_scaling=True)

    def test_bias_masks_distance_bounds(self) -> None:
        curves = np.array([[[5, 4, 3, 2], [1, 2, 3, 4]]], dtype=np.float32)
        distances = np.array([[-1, 0, 1, 3, 4, 7]], dtype=np.int64)
        values, visible = bias_at_distances(curves, distances, window=4)
        self.assertTrue(np.array_equal(values[0, 0], [0, 5, 4, 2, 0, 0]))
        self.assertTrue(np.array_equal(visible[0], [False, True, True, True, False, False]))
        _, global_visible = bias_at_distances(curves, distances, window=None)
        self.assertTrue(global_visible[0, 4] and global_visible[0, 5])

    def test_frozen_attention_ablation_includes_sink(self) -> None:
        curves = np.array([[[3, 0]]], dtype=np.float32)
        qk = np.zeros((1, 1, 2), dtype=np.float32)
        distance = np.array([[0, 1]], dtype=np.int64)
        sink = np.zeros((1, 1), dtype=np.float32)
        result = attention_delta(qk, curves, distance, sink, window=2)
        self.assertEqual(result["kind"], "frozen_qk_and_sink_logits_bias_zero_ablation")
        self.assertEqual(result["n_query_head_cells"], 1)
        self.assertGreater(result["mean_l1_probability_change"], 0.5)
        self.assertGreater(result["mean_absolute_sink_probability_change"], 0)
        self.assertEqual(result["top_attention_destination_changes"], 0)
        zero = attention_delta(qk, curves * 0, distance, sink, window=2)
        self.assertAlmostEqual(zero["mean_l1_probability_change"], 0)

    def test_four_tap_convolution_descriptive_only(self) -> None:
        result = conv_taps({"k_sconv": np.array([[1, 0, 0, 0], [0, -2, 0, 0]])})
        self.assertEqual(result["k_sconv"]["n_channels"], 2)
        self.assertAlmostEqual(sum(result["k_sconv"]["normalized_abs_share_oldest_to_current"]), 1)

    def test_safetensors_selective_read_exact_checkpoint_key_names(self) -> None:
        try:
            import torch
            from safetensors.torch import save_file
        except ImportError:
            self.skipTest("optional torch/safetensors missing")
        hidden = 12
        tensors = {
            "model.llm.embed.weight": torch.arange(96, dtype=torch.float).reshape(8, hidden),
            "model.llm.embed_norm.weight": torch.ones(hidden),
            "model.llm.layers.0.attn_norm.weight": torch.ones(hidden),
            "model.llm.layers.0.attn.wr_du.weight": torch.ones(4, hidden),
            "model.llm.layers.0.attn.rel_l2_clip.scale": torch.ones(2, 2),
            "model.llm.layers.0.attn.rel_logits_proj.proj": torch.ones(2, 5),
            "model.llm.layers.0.attn.k_sconv.weight": torch.ones(6, 4),
            "model.llm.layers.0.attn.v_sconv.weight": torch.ones(6, 4),
        }
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "mini.safetensors"
            save_file(tensors, str(target))
            loaded, digests = checkpoint_tensors(target, 0, ids=[2, 5], require_embeddings=True)
            self.assertTrue(np.array_equal(loaded["embed"], tensors["model.llm.embed.weight"][[2, 5]].numpy()))
            self.assertEqual(loaded["basis"].shape, (2, 5))
            self.assertIn("model.llm.layers.0.attn.wr_du.weight", digests)

    def test_reject_mismatched_weights(self) -> None:
        with self.assertRaises(ValueError):
            relative_curves(np.ones((2, 4)), np.ones((3, 4)), np.ones((2, 2)), np.ones((2, 8)))


if __name__ == "__main__":
    unittest.main()

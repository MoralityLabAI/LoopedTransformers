import unittest
from importlib import metadata
from types import SimpleNamespace

import numpy as np
import torch

from research_gym.inkling_routing_01.core import (
    ARMS, WIDTH, TARGET_COORD, patch_vector, sham_coordinate,
)
from research_gym.inkling_routing_01.checkpoint import TRANSFORMERS_VERSION
from research_gym.inkling_routing_01.hooks import ReaderTrace, apply_native_patch, reader_call
from research_gym.inkling_routing_01.local import plan, score_reader


class StubReader(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Parameter(torch.zeros(12, 4, WIDTH))

    def forward(self, block2, block3, layer_index):
        bank = torch.cat([block2, block3.unsqueeze(-2)], dim=-2).float()
        bank = bank * torch.rsqrt(bank.square().mean(-1, keepdim=True) + 1e-6)
        scores = self.query[layer_index] @ bank.transpose(-2, -1) / WIDTH**0.5
        return scores.softmax(-1) @ bank


class RepeatedDecoder(torch.nn.Module):
    """Synthetic driver for the local positional signature and repeated visits."""

    def __init__(self, reader):
        super().__init__()
        self.layers = torch.nn.ModuleList(torch.nn.Identity() for _ in range(18))
        self.message_input = reader

    def forward(self, value):
        writer = self.layers[17](value)
        block2 = value.roll(1, dims=-1).unsqueeze(-2)
        for _ in range(2):
            for layer_index in range(12):
                self.message_input(block2, writer, layer_index)
            writer = self.layers[17](writer * 0.97)


class NativeHookContractTests(unittest.TestCase):
    def test_reader_call_index(self):
        self.assertEqual(reader_call(2), 6)
        self.assertEqual(reader_call(3), 18)
        with self.assertRaises(ValueError):
            reader_call(1)

    def test_plan_manifest_and_no_secret(self):
        manifest = plan(2, 2)
        self.assertEqual(len(manifest["root_prompts"]), 2)
        self.assertEqual(manifest["restore_call"], 7)
        self.assertEqual(len(manifest["manifest_sha256"]), 64)
        self.assertNotIn("API_KEY", str(manifest))
        self.assertEqual(manifest, plan(2, 2))
        self.assertEqual(plan(None, 2)["reader_passes"], [2, 3])

    def test_torch_patch_matches_numpy_and_is_restorable(self):
        rng = np.random.default_rng(13)
        value = rng.normal(size=WIDTH).astype(np.float32)
        value[TARGET_COORD] = 32.4
        sham = sham_coordinate("capital")
        for arm in ARMS:
            with self.subTest(arm=arm):
                bank = torch.from_numpy(np.tile(value, (2, 3, 1)).copy())
                expected = patch_vector(value, arm, sham)
                original = apply_native_patch(bank, arm, sham)
                self.assertTrue(np.allclose(bank[:, -1].numpy(), expected, rtol=1e-5, atol=1e-5))
                bank[:, -1] = original
                self.assertTrue(np.array_equal(bank[0, -1].numpy(), value))
                self.assertTrue(np.array_equal(bank[:, 0].numpy(), np.tile(value, (2, 1))))

    def check_repeated_trace(self, reader):
        with torch.no_grad():
            reader.query[:, :, TARGET_COORD] = torch.arange(4) + 4
            decoder = RepeatedDecoder(reader)
            sham = sham_coordinate("capital")
            trace = ReaderTrace(decoder, [2, 3], sham)
            source = torch.rand((1, 3, WIDTH), generator=torch.Generator().manual_seed(42))
            source[..., TARGET_COORD] = 32
            value = source.expand(len(trace.lanes), -1, -1).clone()
            with trace:
                decoder(value)
            trace.verify()
            torch.testing.assert_close(value, source.expand_as(value), rtol=0, atol=0)
            self.assertEqual(len(reader._forward_pre_hooks), 0)
            for reader_pass, captured in trace.readers.items():
                zero, rescue = (trace.lanes[reader_pass, arm] for arm in ("zero_2027", "reader_rescue"))
                np.testing.assert_array_equal(captured["original"], captured["restored"])
                np.testing.assert_array_equal(captured["message"][zero], captured["message"][rescue])
                np.testing.assert_array_equal(captured["delivered"][rescue], captured["message"][0])
                clean = score_reader(captured["bank2"][0], captured["bank3"][0], captured["message"][0], captured["query"])
                patched = score_reader(
                    captured["bank2"][zero], captured["bank3"][zero], captured["message"][zero], captured["query"],
                )
                self.assertGreater(min(clean["beta3_qkvm"]), max(patched["beta3_qkvm"]))

    def test_repeated_reader_restoration_and_rescue(self):
        self.check_repeated_trace(StubReader())

    def test_installed_transformers_reader_smoke(self):
        try:
            version = metadata.version("transformers")
        except metadata.PackageNotFoundError:
            self.skipTest("optional Transformers native module smoke test")
        self.assertEqual(version, TRANSFORMERS_VERSION)
        from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopMessageAttentionResidual

        config = SimpleNamespace(
            loop_layer_index_start=6, loop_layer_index_end=18, hidden_size=WIDTH, _attn_implementation="eager",
        )
        self.check_repeated_trace(Inkling2LoopMessageAttentionResidual(config))


if __name__ == "__main__":
    unittest.main()

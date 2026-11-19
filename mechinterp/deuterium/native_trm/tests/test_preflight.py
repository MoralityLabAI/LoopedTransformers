"""Source-only numerical regression tests: no protected calibration/eval cohorts."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from mechinterp.loopscope04.gradient_flow import TinyRecursiveModel, fixed_data
from mechinterp.deuterium.native_trm import preflight


class TestNativePreflight(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(61073)
        self.model = TinyRecursiveModel(48, 4)
        self.model.eval().requires_grad_(False)
        # Explicitly excluded untrained smoke-only root; no original checkpoint.
        self.x, self.y = fixed_data(4, 61073)

    def test_original_git_blob_bytes(self):
        self.assertEqual(preflight.check_source_bytes(), preflight.SOURCE_BLOBS)

    def test_native_four_pass_closure(self):
        with torch.no_grad():
            logits = self.model(self.x)
            replay = preflight.four_pass_manual(self.model, self.x)
            self.assertTrue(torch.equal(logits, replay))
            states = self.model.reason_states(self.x)
            self.assertEqual(len(states), 5)
            self.assertTrue(torch.equal(logits, self.model.head(states[4])))

    def test_real_hidden_and_parameter_edits_and_restoration(self):
        with torch.no_grad():
            original = self.model.block.chan_mix[2].weight.detach().clone()
            baseline = self.model(self.x)
            metrics = preflight.closure_gate(self.model, self.x, 103)
            self.assertTrue(torch.equal(original, self.model.block.chan_mix[2].weight))
            self.assertTrue(torch.equal(baseline, self.model(self.x)))
        self.assertGreater(metrics["hidden_edit_native_max_logit_change"], 1e-8)
        self.assertGreater(metrics["weight_edit_native_max_logit_change"], 1e-8)
        self.assertLessEqual(metrics["weight_edit_actual_relative_dose_error"], 1e-4)
        self.assertEqual(metrics["weight_edit_conceptual_rank"], 1)
        self.assertEqual(metrics["weight_shape"], [48, 128])

    def test_historical_only_and_reserved_unopened(self):
        self.assertEqual(preflight.HISTORICAL_SEED, 31043)
        self.assertEqual(preflight.RESERVED_CALIBRATION_SEED, 731055)
        self.assertEqual(preflight.RESERVED_HELDOUT_SEED, 731057)
        self.assertEqual(preflight.TEACHERS, (103, 211, 307))
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 44001)

    def test_tampered_artifact_rejected_before_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bad_bundle = root / "bundle.pt"
            bad_meta = root / "train.json"
            bad_bundle.write_bytes(b"forged")
            bad_meta.write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                preflight.run(bad_bundle, bad_meta, root / "never_created.json")
            self.assertFalse((root / "never_created.json").exists())


if __name__ == "__main__":
    unittest.main()

"""Simulated native-order protocol checks; FakeLean is never research evidence."""
from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from ppt_gym.core import Config, PlasticProofTransformer
from ppt_gym.data import sha256_file
from ppt_gym.live_suite import create_suite
from ppt_gym.native_order import FrozenOrder, _interval, replay, run
from ppt_gym.tests.test_live_lean import FakeLean, mock_certify


class NativeOrderTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(37)

    def _inputs(self, root):
        suite = root / "suite"
        create_suite(suite, seed=20261123, per_split=3)
        model = PlasticProofTransformer(Config(plan="b", width=16, heads=4, max_bytes=32)).eval()
        checkpoint = root / "checkpoint.pt"
        torch.save({
            "schema": "ppt_gym.checkpoint.v1",
            "config": asdict(model.cfg), "weights": model.state_dict(),
            "source_manifest_sha256": "synthetic-example-only",
            "source_evidence_class": "synthetic_fixture",
            "train_order": "ab", "train_steps": 1,
        }, checkpoint)
        return suite, checkpoint, model

    def test_no_model_mutation_and_fake_lean_cannot_certify(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite, checkpoint, model = self._inputs(root)
            before = sha256_file(checkpoint)
            outcome = run(
                FakeLean(), suite, checkpoint, root / "session",
                roots=2, depth=1, max_seconds=40,
                certified_backend=False, certifier=mock_certify,
            )
            self.assertEqual(outcome["paired_roots"], 2)
            self.assertEqual(outcome["status"], "engineering_only_or_partial")
            self.assertFalse(outcome["scientific_eligible"])
            self.assertFalse(outcome["architecture_promotion_eligible"])
            self.assertFalse(outcome["backend"]["actual_lean_process"])
            self.assertEqual(outcome["query_label_training_updates"], 0)
            self.assertEqual(outcome["scores"]["ab"]["standalone_verified_closures"], 0)
            self.assertEqual(outcome["scores"]["ba"]["standalone_verified_closures"], 0)
            self.assertEqual(sha256_file(checkpoint), before)
            self.assertTrue((root / "session" / "standalone.lean").is_file())
            trace = (root / "session" / "paired_episodes.jsonl").read_text()
            self.assertEqual(len(trace.splitlines()), 6)
            self.assertEqual({e["order"] for e in map(json.loads, trace.splitlines())},
                             {"ab", "ba", "auto8"})
            self.assertTrue(all(e["replay_passed"] for e in map(json.loads, trace.splitlines())))
            receipt = replay(FakeLean(), suite, root / "session", root / "audit")
            self.assertTrue(receipt["replay_passed"])
            self.assertFalse(receipt["scientific_eligible"])
            self.assertEqual(receipt["independent_replay_episodes"], 6)
            with self.assertRaisesRegex(ValueError, "fresh"):
                replay(FakeLean(), suite, root / "session", root / "audit")
            with (root / "session" / "paired_episodes.jsonl").open("a") as handle:
                handle.write('{"tamper":true}\n')
            with self.assertRaisesRegex(ValueError, "hash changed"):
                replay(FakeLean(), suite, root / "session", root / "audit2")

    def test_frozen_order_calls_same_model(self):
        _, _, model = self._inputs(Path(tempfile.mkdtemp()))
        menu = ("apply h0", "apply r0")
        text = "goal: P1"
        with torch.no_grad():
            reference = model(text, menu, depth=2, middle_order="ab")[-1]["logits"]
            through = FrozenOrder(model, "ab")(text, menu, depth=2)[-1]["logits"]
            other = FrozenOrder(model, "ba")(text, menu, depth=2)[-1]["logits"]
        self.assertTrue(torch.equal(reference, through))
        self.assertGreater(float((reference - other).norm()), 1e-8)
        with self.assertRaisesRegex(ValueError, "PPT-B"):
            FrozenOrder(PlasticProofTransformer(Config(plan="a", width=16, heads=4,
                                                       max_bytes=32)), "ba")

    def test_gates_refuse_locked_test_and_non_plan_b(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite, checkpoint, _ = self._inputs(root)
            with self.assertRaisesRegex(ValueError, "development-only"):
                run(FakeLean(), suite, checkpoint, root / "test",
                    split="ordinary_test", certified_backend=False)
            with self.assertRaisesRegex(ValueError, "actual pinned Lean"):
                run(FakeLean(), suite, checkpoint, root / "pretend",
                    certified_backend=True)
            with self.assertRaisesRegex(ValueError, "fresh"):
                run(FakeLean(), suite, checkpoint, root / "existing",
                    roots=0)
            self.assertIsNone(_interval([1]))
            self.assertEqual(_interval([0, 0]), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()

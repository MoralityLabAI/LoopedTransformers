"""LSA-02D field-recovery and deterministic-weight tests, all synthetic."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from ppt_gym.data import canonical
from ppt_gym.determinism_audit import _compare_checkpoints
from ppt_gym.field_audit import (
    _canonical_bytes_and_spans, _fully_visible, inspect_public_frame,
)
from ppt_gym.live_lean import candidate_menu, state_input
from ppt_gym.tests.test_live_lean import FakeLean


class PublicFieldTests(unittest.TestCase):
    def setUp(self):
        repl = FakeLean()
        state = repl.init("synthetic")
        self.packet, _ = state_input(repl, state, (), 8)
        self.menu = list(candidate_menu(repl, state))
        self.example = {
            "root": "fake_root", "lineage": "fake_lineage",
            "step": 0, "goal": self.packet, "candidates": self.menu,
        }

    def test_canonical_paths_are_byte_exact_even_with_unicode(self):
        original = json.loads(self.packet)
        original["state"]["goals"][0]["pp"] = "⊢ P1"
        literal = canonical(original)
        serialized, spans = _canonical_bytes_and_spans(original)
        self.assertEqual(serialized, literal.encode("utf-8"))
        self.assertEqual(serialized[spans[("state", "goals", 0, "pp")][0]:
                                    spans[("state", "goals", 0, "pp")][1]].decode("utf-8"),
                         '"⊢ P1"')
        self.assertTrue(_fully_visible(spans, ("state", "goals", 0, "target"), 0))
        self.assertFalse(_fully_visible(spans, ("state", "goals", 0, "target"), len(serialized)))

    def test_typed_first_candidate_counts_as_a_complete_public_type(self):
        inspected = inspect_public_frame(self.example, max_bytes=384)
        self.assertEqual(inspected["candidate_types"], 1)
        self.assertEqual(inspected["typed_complete_candidate_types"], 1)
        self.assertEqual(inspected["typed_full_current_goal"], 1)
        self.assertEqual(inspected["typed_full_budget"], 1)
        self.assertLessEqual(inspected["typed_visible_bytes"], 384)
        self.assertEqual(inspected["typed_full_pretty"], 0)
        self.assertEqual(inspected["lineage"], "fake_lineage")

    def test_long_raw_json_loses_early_rule_and_budget_while_typed_retains_them(self):
        state = json.loads(self.packet)
        goal = state["state"]["goals"][0]
        for i in range(1, 45):
            goal["hypotheses"].append({
                "name": f"r{i}", "type": {
                    "imp": [{"a": "P0"}, {"a": "P1"}],
                },
            })
        self.example["goal"] = canonical(state)
        inspected = inspect_public_frame(self.example, max_bytes=384)
        self.assertGreater(inspected["source_bytes"], 384)
        self.assertEqual(inspected["raw_full_budget"], 0)
        self.assertEqual(inspected["typed_full_budget"], 1)
        self.assertEqual(inspected["raw_complete_candidate_types"], 0)
        self.assertEqual(inspected["typed_complete_candidate_types"], 1)
        self.assertEqual(inspected["raw_full_current_goal"], 1)

    def test_source_packet_modification_or_bad_budget_is_rejected(self):
        row = json.loads(self.packet)
        row["undisclosed_label"] = 1
        self.example["goal"] = canonical(row)
        with self.assertRaisesRegex(ValueError, "packet shape"):
            inspect_public_frame(self.example)
        self.example["goal"] = self.packet
        with self.assertRaisesRegex(ValueError, "model byte budget"):
            inspect_public_frame(self.example, max_bytes=16)


class ExactCPUWeightTests(unittest.TestCase):
    def test_identical_model_weights_and_mutations_are_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = root / "a.pt", root / "b.pt"
            torch.save({
                "schema": "ppt_gym.checkpoint.v1",
                "weights": {"layer.weight": torch.tensor([[1., 2.], [3., 4.]])},
            }, a)
            torch.save({
                "schema": "ppt_gym.checkpoint.v1",
                "weights": {"layer.weight": torch.tensor([[1., 2.], [3., 4.]])},
            }, b)
            result = _compare_checkpoints(a, b)
            self.assertTrue(result["all_tensors_bitwise_equal"])
            self.assertEqual(result["different_tensor_count"], 0)
            torch.save({
                "schema": "ppt_gym.checkpoint.v1",
                "weights": {"layer.weight": torch.tensor([[1., 2.], [3., 5.]])},
            }, b)
            result = _compare_checkpoints(a, b)
            self.assertFalse(result["all_tensors_bitwise_equal"])
            self.assertEqual(result["different_tensor_count"], 1)
            self.assertAlmostEqual(result["largest_tensor_abs_displacement"], 1.0)


if __name__ == "__main__":
    unittest.main()

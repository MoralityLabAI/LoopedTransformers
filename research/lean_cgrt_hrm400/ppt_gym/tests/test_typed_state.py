"""LSA-02C public-only typed state and matched-input regressions."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from ppt_gym.cli import checkpoint_load
from ppt_gym.data import canonical, sha256_file
from ppt_gym.live_lean import candidate_menu, state_input
from ppt_gym.native_order import FrozenOrder
from ppt_gym.native_train import fit
from ppt_gym.tests.test_live_lean import FakeLean
from ppt_gym.tests.test_native_train import fake_teacher
from ppt_gym.typed_state import SCHEMA, encode_public_packet


class TypedStateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.lean = FakeLean()
        state = self.lean.init("simulated")
        self.raw, _ = state_input(self.lean, state, (), 8)
        self.menu = candidate_menu(self.lean, state)

    def test_typed_context_uses_only_public_goal_and_hypotheses(self):
        text = encode_public_packet(
            self.raw, self.menu, view="typed", max_bytes=384,
        )
        self.assertTrue(text.startswith("B8|G:P1|"))
        self.assertIn("r0:P0>P1;", text)
        self.assertIn("h0:P0;", text)
        self.assertLessEqual(len(text.encode("utf-8")), 384)
        self.assertEqual(
            encode_public_packet(self.raw, self.menu, view="raw", max_bytes=384),
            self.raw,
        )
        self.assertNotIn("SIMULATED", text)
        self.assertEqual(
            text,
            encode_public_packet(self.raw, self.menu, view="typed", max_bytes=384),
        )
        self.assertNotIn("label", text)

    def test_public_reorder_and_pretty_text_leave_typed_semantics(self):
        state = json.loads(self.raw)
        state["state"]["goals"][0]["hypotheses"].reverse()
        state["state"]["goals"][0]["pp"] = "IGNORE_THIS_HUMAN_PRINTED_PROOF_TEXT"
        text = encode_public_packet(
            canonical(state), self.menu, view="typed", max_bytes=384,
        )
        self.assertEqual(text, encode_public_packet(
            self.raw, self.menu, view="typed", max_bytes=384,
        ))
        self.assertNotIn("IGNORE_THIS", text)

    def test_budget_and_fail_closed_on_nonpublic_inputs(self):
        for budget in (64, 160, 256, 384):
            text = encode_public_packet(
                self.raw, self.menu, view="typed", max_bytes=budget,
            )
            self.assertLessEqual(len(text.encode("utf-8")), budget)
            self.assertIn("P1", text)
        with self.assertRaisesRegex(ValueError, "public Lean packet shape"):
            row = json.loads(self.raw)
            row["oracle_label"] = 1
            encode_public_packet(canonical(row), self.menu, view="typed")
        with self.assertRaisesRegex(ValueError, "candidate missing"):
            encode_public_packet(self.raw, ("apply r999",), view="typed")
        with self.assertRaisesRegex(ValueError, "invalid"):
            encode_public_packet(self.raw, self.menu, view="hieroglyphic")
        with self.assertRaisesRegex(ValueError, "budget"):
            encode_public_packet(self.raw, self.menu, view="typed", max_bytes=32)

    def test_typed_goal_input_from_multigoal_and_conjunction(self):
        packet = {
            "state": {"goals": [
                {"target": {"and": [{"a": "P1"}, {"a": "P2"}]},
                 "hypotheses": [
                     {"name": "P1", "type": {"sort": "Prop"}},
                     {"name": "P2", "type": {"sort": "Prop"}},
                     {"name": "h0", "type": {"a": "P1"}},
                     {"name": "r0", "type": {"imp": [{"a": "P1"}, {"a": "P2"}]}}]},
                {"target": {"a": "P0"}, "hypotheses": []},
            ]},
            "failed_choices": ["apply r0"],
            "checks_remaining": 3,
        }
        text = encode_public_packet(
            canonical(packet), ("constructor",), view="typed", max_bytes=384,
        )
        self.assertIn("G:(P1&P2),P0", text)
        self.assertIn("A:constructor", text)
        self.assertIn("F:r0", text)

    def test_equal_initial_weights_for_raw_and_typed_teacher_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher = fake_teacher(root)
            train_file = teacher / "teacher.jsonl"
            rows = [json.loads(s) for s in train_file.read_text().splitlines()]
            for row in rows:
                row["goal"] = self.raw
                row["candidates"] = ["apply r0"]
                row["labels"] = [1]
            train_file.write_text(
                "".join(canonical(row) + "\n" for row in rows), encoding="utf-8",
            )
            manifest_file = teacher / "manifest.json"
            manifest = json.loads(manifest_file.read_text())
            manifest["artifacts"]["teacher.jsonl"] = sha256_file(train_file)
            manifest_file.write_text(canonical(manifest) + "\n", encoding="utf-8")
            results = []
            for view in ("raw", "typed"):
                record = fit(
                    teacher, root / view, order="ab", seed=107,
                    steps=3, depth=2, width=16, max_bytes=384,
                    threads=1, max_seconds=30,
                    input_view=view, allow_synthetic=True,
                )
                self.assertEqual(record["optimizer_steps"], 3)
                self.assertEqual(record["input_view"], view)
                self.assertEqual(record["view_input_over_budget"], 0 if view == "typed"
                                 else record["raw_input_over_budget"])
                model, checkpoint = checkpoint_load(root / view / "checkpoint.pt")
                self.assertEqual(checkpoint["input_view"], view)
                self.assertEqual(
                    checkpoint["input_view_schema"],
                    SCHEMA if view == "typed" else "raw-legacy-json",
                )
                self.assertTrue(torch.isfinite(
                    FrozenOrder(model.eval(), "ab", view=view)(
                        self.raw, self.menu, depth=1,
                    )[-1]["logits"]).all())
                results.append(record)
            self.assertEqual(results[0]["initial_weights_sha256"],
                             results[1]["initial_weights_sha256"])
            self.assertEqual(results[0]["unique_parameter_count"],
                             results[1]["unique_parameter_count"])
            self.assertFalse(results[1]["training_source_native"])


if __name__ == "__main__":
    unittest.main()

"""LSA-02E strict public typed+pretty input invariants."""
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
from ppt_gym.typed_state import (
    PRETTY_SCHEMA, encode_public_packet, schema_for_view,
)


class TypedPrettyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        lean = FakeLean()
        s = lean.init("fake")
        self.raw, _ = state_input(lean, s, (), 8)
        self.menu = candidate_menu(lean, s)

    def test_goal_and_candidate_types_survive_pretty_tail(self):
        row = json.loads(self.raw)
        row["state"]["goals"][0]["pp"] = (
            "h0 : P0\nr0 : P0 → P1\n" + ("other : P17\n" * 120) + "⊢ P1"
        )
        packet = canonical(row)
        text = encode_public_packet(packet, self.menu, view="typed_pp", max_bytes=384)
        self.assertIn("B8|G:P1|", text)
        self.assertIn("r0:P0>P1;", text)
        self.assertIn("P:", text)
        self.assertIn("⊢ P1", text)
        self.assertLessEqual(len(text.encode("utf-8")), 384)
        self.assertNotIn("\n", text)
        self.assertEqual(text, encode_public_packet(packet, self.menu, view="typed_pp", max_bytes=384))
        self.assertEqual(schema_for_view("typed_pp"), PRETTY_SCHEMA)

    def test_pretty_is_public_and_changes_only_the_new_view(self):
        a = json.loads(self.raw)
        b = json.loads(self.raw)
        a["state"]["goals"][0]["pp"] = "⊢ P1"
        b["state"]["goals"][0]["pp"] = "h0 : P0\n⊢ P1"
        first = canonical(a)
        second = canonical(b)
        self.assertNotEqual(
            encode_public_packet(first, self.menu, view="typed_pp"),
            encode_public_packet(second, self.menu, view="typed_pp"),
        )
        self.assertEqual(
            encode_public_packet(first, self.menu, view="typed"),
            encode_public_packet(second, self.menu, view="typed"),
        )
        b["state"]["goals"][0]["private_oracle"] = {"label": 1}
        with self.assertRaisesRegex(ValueError, "unrecognized public"):
            encode_public_packet(canonical(b), self.menu, view="typed_pp")

    def test_goal_candidate_and_size_are_preserved_under_budget_pressure(self):
        row = json.loads(self.raw)
        goal = row["state"]["goals"][0]
        goal["pp"] = "context " * 900 + "⊢ P1"
        for j in range(1, 45):
            goal["hypotheses"].append({
                "name": f"r{j}", "type": {
                    "imp": [{"a": "P0"}, {"a": "P1"}],
                },
            })
        packet = canonical(row)
        text = encode_public_packet(packet, self.menu, view="typed_pp", max_bytes=160)
        self.assertIn("B8|G:P1|", text)
        self.assertIn("r0:P0>P1;", text)
        self.assertIn("⊢ P1", text)
        self.assertLessEqual(len(text.encode("utf-8")), 160)
        self.assertNotIn("r43", text)
        self.assertTrue(text.count("P:") == 1)

    def test_hard_fail_if_required_public_evidence_exceeds_byte_budget(self):
        row = json.loads(self.raw)
        packet = canonical(row)
        with self.assertRaisesRegex(ValueError, "cannot fit"):
            encode_public_packet(packet, self.menu, view="typed_pp", max_bytes=32)
        with self.assertRaisesRegex(ValueError, "candidate absent"):
            encode_public_packet(packet, ("apply r44",), view="typed_pp")
        with self.assertRaisesRegex(ValueError, "requires public"):
            bad = json.loads(packet)
            bad["state"]["goals"][0]["pp"] = 17
            encode_public_packet(canonical(bad), self.menu, view="typed_pp")

    def test_source_only_cpu_fit_checkpoint_view_matches_native_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher = fake_teacher(root)
            path = teacher / "teacher.jsonl"
            data = [json.loads(x) for x in path.read_text().splitlines()]
            for row in data:
                row["goal"] = self.raw
                row["candidates"] = list(self.menu)
                row["labels"] = [1]
            path.write_text("".join(canonical(x) + "\n" for x in data), encoding="utf-8")
            manifest_path = teacher / "manifest.json"
            meta = json.loads(manifest_path.read_text())
            meta["artifacts"]["teacher.jsonl"] = sha256_file(path)
            manifest_path.write_text(canonical(meta) + "\n", encoding="utf-8")
            output = root / "fit"
            result = fit(
                teacher, output, order="ab", seed=127, steps=3,
                depth=2, width=16, max_bytes=384, threads=1,
                max_seconds=30, input_view="typed_pp", allow_synthetic=True,
            )
            self.assertEqual(result["optimizer_steps"], 3)
            self.assertEqual(result["input_view_schema"], PRETTY_SCHEMA)
            self.assertEqual(result["view_input_over_budget"], 0)
            model, checkpoint = checkpoint_load(output / "checkpoint.pt")
            self.assertEqual(checkpoint["input_view"], "typed_pp")
            self.assertEqual(checkpoint["input_view_schema"], PRETTY_SCHEMA)
            with torch.no_grad():
                output = FrozenOrder(model.eval(), "ab", view="typed_pp")(
                    self.raw, self.menu, depth=1,
                )
            self.assertTrue(bool(torch.isfinite(output[-1]["logits"]).all()))


if __name__ == "__main__":
    unittest.main()

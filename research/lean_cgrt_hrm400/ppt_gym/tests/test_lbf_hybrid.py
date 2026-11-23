"""LBF-05 exact-budget hybrid public input and native-train compatibility."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from ppt_gym.data import canonical, sha256_file
from ppt_gym.live_lean import candidate_menu, state_input
from ppt_gym.native_train import fit
from ppt_gym.native_order import FrozenOrder
from ppt_gym.tests.test_live_lean import FakeLean
from ppt_gym.tests.test_native_train import fake_teacher
from ppt_gym.typed_state import VIEWS, encode_public_packet
from ppt_gym.lbf_hybrid import SCHEMA, _utf8_suffix, hybrid_public_packet


class HybridInputTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        fake = FakeLean()
        state = fake.init("dummy")
        self.raw, _ = state_input(fake, state, (), 8)
        self.menu = candidate_menu(fake, state)

    def test_hybrid_keeps_structural_and_readable_channels(self):
        src = json.loads(self.raw)
        src["state"]["goals"][0]["pp"] = "h0 : P0\nr0 : P0 → P1\n⊢ P1"
        packet = canonical(src)
        x = encode_public_packet(packet, self.menu, view="hybrid", max_bytes=384)
        self.assertTrue(x.startswith("B8|G:P1|"))
        self.assertIn("r0:P0>P1;", x)
        self.assertIn("P:", x)
        self.assertIn("⊢ P1", x)
        self.assertEqual(x, hybrid_public_packet(packet, self.menu))
        self.assertLessEqual(len(x.encode("utf-8")), 384)
        self.assertIn("hybrid", VIEWS)
        self.assertEqual(SCHEMA, "lbf05.hybrid_public_state.v1")

    def test_bounded_utf8_complete_codepoints(self):
        self.assertEqual(_utf8_suffix("aaaa⊢π", 5), "⊢π")
        for budget in (32, 64, 160, 256, 384):
            raw = json.loads(self.raw)
            raw["state"]["goals"][0]["pp"] = "π⊢" * 500
            text = encode_public_packet(canonical(raw), self.menu,
                                        view="hybrid", max_bytes=budget)
            self.assertLessEqual(len(text.encode("utf-8")), budget)
            self.assertIn("G:", text)

    def test_rejects_private_and_malformed_data(self):
        row = json.loads(self.raw)
        row["future_verdict"] = {"winner":"r0"}
        with self.assertRaises(ValueError):
            encode_public_packet(canonical(row), self.menu, view="hybrid")
        row = json.loads(self.raw)
        row["state"]["goals"][0]["oracle_label"] = 1
        with self.assertRaisesRegex(ValueError, "non-public goal"):
            encode_public_packet(canonical(row), self.menu, view="hybrid")
        with self.assertRaisesRegex(ValueError, "candidate missing"):
            encode_public_packet(self.raw, ("apply r999",), view="hybrid")
        with self.assertRaises(ValueError):
            encode_public_packet(self.raw, ("constructor",), view="hybrid")

    def test_truncated_pretty_is_deterministic_and_never_partially_utf8(self):
        row = json.loads(self.raw)
        row["state"]["goals"][0]["pp"] = "X" * 300 + "①⑤☉⌁" * 80
        packet = canonical(row)
        first = encode_public_packet(packet, self.menu, view="hybrid", max_bytes=384)
        second = encode_public_packet(packet, self.menu, view="hybrid", max_bytes=384)
        self.assertEqual(first, second)
        self.assertLessEqual(len(first.encode("utf-8")), 384)
        first.encode("utf-8").decode("utf-8")

    def test_native_training_checkpoint_binds_hybrid_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher = fake_teacher(root)
            source_path = teacher/"teacher.jsonl"
            rows = [json.loads(x) for x in source_path.read_text().splitlines() if x]
            for row in rows:
                row["goal"] = self.raw
                row["candidates"] = list(self.menu)
                row["labels"] = [1 if i==0 else 0 for i in range(len(self.menu))]
            source_path.write_text("".join(canonical(x) + "\n" for x in rows))
            meta_path = teacher / "manifest.json"
            meta = json.loads(meta_path.read_text())
            meta["artifacts"]["teacher.jsonl"] = sha256_file(source_path)
            meta_path.write_text(canonical(meta)+"\n")

            records = {}
            for view in ("raw", "typed", "hybrid"):
                m = fit(teacher, root / view, order="ab", seed=41,
                        steps=2, depth=2, width=16, max_bytes=384,
                        threads=1, max_seconds=40, input_view=view,
                        allow_synthetic=True)
                self.assertEqual(m["optimizer_steps"], 2)
                self.assertEqual(m["input_view"], view)
                ckpt = torch.load(root/view/"checkpoint.pt",
                                  map_location="cpu", weights_only=True)
                self.assertEqual(ckpt["input_view"], view)
                if view == "hybrid":
                    self.assertEqual(ckpt["input_view_schema"], SCHEMA)
                records[view] = m
            self.assertEqual(len({x["initial_weights_sha256"] for x in records.values()}),1)
            self.assertEqual(len({x["unique_parameter_count"] for x in records.values()}),1)


if __name__=="__main__":
    unittest.main()

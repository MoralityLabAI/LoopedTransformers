"""LSA-02G deterministic source-fork emphasis and read-only native calibration tests."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from ppt_gym.data import canonical, sha256_file
from ppt_gym.fork_eval import _bootstrap, _fork_scores, _hash_files
from ppt_gym.native_train import fit
from ppt_gym.tests.test_native_train import fake_teacher
from ppt_gym.typed_state import schema_for_view


def mixed_teacher(root: Path) -> Path:
    folder = fake_teacher(root)
    data = folder / "teacher.jsonl"
    rows = [json.loads(x) for x in data.read_text().splitlines()]
    state = {
        "state": {"goals": [{
            "target": {"a": "P1"},
            "hypotheses": [
                {"name": "P0", "type": {"sort": "Prop"}},
                {"name": "P1", "type": {"sort": "Prop"}},
                {"name": "h0", "type": {"a": "P0"}},
                {"name": "r0", "type": {"imp": [{"a": "P0"}, {"a": "P1"}]}},
                {"name": "r1", "type": {"imp": [{"a": "P2"}, {"a": "P1"}]}},
            ],
            "pp": "h0 : P0\nr0 : P0 → P1\nr1 : P2 → P1\n⊢ P1",
        }]},
        "failed_choices": [],
        "checks_remaining": 8,
    }
    for i, row in enumerate(rows):
        row["goal"] = canonical(state)
        if i % 2 == 0:
            row["candidates"] = ["apply r0", "apply r1"]
            row["labels"] = [1, 0]
        else:
            row["candidates"] = ["apply r0"]
            row["labels"] = [1]
        row["chosen_reference_tactic"] = "apply r0"
    data.write_text("".join(canonical(row) + "\n" for row in rows), encoding="utf-8")
    meta_path = folder / "manifest.json"
    meta = json.loads(meta_path.read_text())
    meta["artifacts"]["teacher.jsonl"] = sha256_file(data)
    meta_path.write_text(canonical(meta) + "\n", encoding="utf-8")
    return folder


class NativeForkWeightingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_sham_and_fork_have_equal_source_mass_and_source_draws(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = mixed_teacher(root)
            seen = []
            trace_indexes = []
            for mode in ("plain", "fork4", "sham4"):
                folder = root / mode
                receipt = fit(
                    source, folder, order="ab", seed=149,
                    steps=6, depth=2, width=16, max_bytes=384,
                    threads=1, max_seconds=30,
                    input_view="pp96", weight_mode=mode,
                    allow_synthetic=True,
                )
                self.assertEqual(receipt["optimizer_steps"], 6)
                self.assertEqual(receipt["weight_mode"], mode)
                self.assertEqual(receipt["native_train_fork_frames"], 4)
                self.assertEqual(receipt["source_boosted_frames"], 4 if mode != "plain" else 0)
                if mode != "plain":
                    self.assertEqual(receipt["source_loss_mass_normalizer"], 2.5)
                self.assertEqual(receipt["input_view_schema"], schema_for_view("pp96"))
                trace = [json.loads(r) for r in
                         (folder / "optimizer_steps.jsonl").read_text().splitlines()]
                trace_indexes.append([
                    (r["step"], r["root"], r["teacher_frame_index"],
                     r["discriminating_fork"]) for r in trace
                ])
                seen.append(receipt["initial_weights_sha256"])
            self.assertEqual(len(set(seen)), 1)
            self.assertEqual(trace_indexes[0], trace_indexes[1])
            self.assertEqual(trace_indexes[0], trace_indexes[2])

    def test_invalid_weight_mode_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = mixed_teacher(root)
            with self.assertRaisesRegex(ValueError, "emphasis"):
                fit(
                    source, root / "bad", order="ab", seed=3,
                    weight_mode="use_eval_labels", allow_synthetic=True,
                )
            self.assertFalse((root / "bad").exists())

    def test_native_fork_accuracy_is_root_weighted_and_only_on_choices(self):
        class Preference:
            class Config:
                max_bytes = 384
            cfg = Config()

            def __call__(self, goal, menu, *, depth, checks_remaining, middle_order):
                self.assert_legal(goal)
                return [{"logits": torch.tensor([0.0, 1.0])}]

            @staticmethod
            def assert_legal(goal):
                assert "P:" in goal and goal.startswith("B8|G:P1|")

        with tempfile.TemporaryDirectory() as tmp:
            source = mixed_teacher(Path(tmp))
            rows = [json.loads(line) for line in
                    (source / "teacher.jsonl").read_text().splitlines()]
            forks = [r for r in rows if len(r["labels"]) == 2]
            result = _fork_scores(Preference(), forks)
            self.assertEqual(len(result), 4)
            self.assertEqual({x["accuracy"] for x in result.values()}, {0.0})
            self.assertEqual({x["random_menu_optimal_accuracy"] for x in result.values()}, {0.5})
            with self.assertRaisesRegex(ValueError, "forced or ambiguous"):
                _fork_scores(Preference(), rows)

    def test_hash_and_root_bootstrap_contracts(self):
        self.assertEqual(_bootstrap([0, 0, 0]), [0, 0])
        self.assertIsNone(_bootstrap([0]))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / "receipt.txt"
            p.write_text("verified")
            expected = sha256_file(p)
            _hash_files(root, {"receipt.txt": expected})
            p.write_text("edited")
            with self.assertRaisesRegex(ValueError, "hash"):
                _hash_files(root, {"receipt.txt": expected})


if __name__ == "__main__":
    unittest.main()

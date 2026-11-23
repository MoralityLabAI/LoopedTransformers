"""CPU teacher/fit contract tests. Synthetic teacher is never scientific evidence."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import torch

from ppt_gym.core import Config, PlasticProofTransformer
from ppt_gym.data import canonical, sha256_file
from ppt_gym.native_train import (
    EVIDENCE, FRAME_SCHEMA, SCHEMA, _load_teacher,
    _reference_optimal_actions, fit,
)
from ppt_gym.native_order import _checkpoint_provenance


def fake_teacher(root: Path) -> Path:
    folder = root / "teacher"
    folder.mkdir()
    frames = []
    for index in range(4):
        for step in range(2):
            frames.append({
                "schema": FRAME_SCHEMA,
                "root": f"root_{index}", "lineage": f"lineage_{index}",
                "family": "fixture",
                "step": step,
                "goal": f"FAKE TRAIN PACKET P{index} at {step}",
                "public_state_sha256": "fake",
                "candidates": ["apply r0", "apply r1"],
                "labels": [int(index % 2 == 0), int(index % 2 == 1)],
                "chosen_reference_tactic": "apply r0" if index % 2 == 0 else "apply r1",
                "checks_remaining": 8 - step,
                "reference_proof_size": 2,
                "evidence": "synthetic_teacher",
            })
    rows = folder / "teacher.jsonl"
    rows.write_text("".join(canonical(entry) + "\n" for entry in frames), encoding="utf-8")
    receipt = {
        "schema": SCHEMA, "split": "train",
        "source": {"suite_manifest_sha256": "synthetic-standalone"},
        "evidence_class": "synthetic_teacher",
        "backend": {"actual_lean_process": False},
        "source_compiler_passed": False,
        "query_roots_exposed_to_fit": 0,
        "independent_lineages": 4,
        "train_examples": len(frames),
        "artifacts": {"teacher.jsonl": sha256_file(rows)},
    }
    (folder / "manifest.json").write_text(canonical(receipt) + "\n", encoding="utf-8")
    return folder


class NativeTeacherTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_native_shortest_action_selection_counts_all_minima(self):
        size = {"P0": 2, "P1": 1, "P3": 1, "P2": None}
        table = {
            "P0": {
                "apply h0": [],
                "apply r0": ["P1"],
                "apply r1": ["P3"],
                "apply r2": ["P2"],
            },
        }
        result = _reference_optimal_actions(
            "P0", ("apply h0", "apply r0", "apply r1", "apply r2", "apply r3"),
            table, size,
        )
        self.assertEqual(result, (0, 1, 1, 0, 0))
        with self.assertRaisesRegex(ValueError, "no finite shortest"):
            _reference_optimal_actions("P2", ("apply r0",), table, size)
        with self.assertRaisesRegex(ValueError, "loses all"):
            _reference_optimal_actions("P0", ("apply r3",), table, size)

    def test_refuse_synthetic_source_by_default_and_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            teacher = fake_teacher(Path(temp))
            with self.assertRaisesRegex(ValueError, "native compiled Lean"):
                _load_teacher(teacher)
            frames, meta = _load_teacher(teacher, allow_synthetic=True)
            self.assertEqual(len(frames), 8)
            self.assertEqual(meta["independent_lineages"], 4)
            with (teacher / "teacher.jsonl").open("a") as handle:
                handle.write('{"tamper":true}\n')
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                _load_teacher(teacher, allow_synthetic=True)

    def test_matched_initialization_both_orders_and_empty_test_access(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            teacher = fake_teacher(root)
            scores = []
            for order in ("ab", "ba"):
                result = fit(
                    teacher, root / order,
                    order=order, seed=103, steps=3, depth=2,
                    width=16, max_bytes=64, learning_rate=0.001,
                    threads=1, max_seconds=30, allow_synthetic=True,
                )
                self.assertEqual(result["optimizer_steps"], 3)
                self.assertEqual(result["train_order"], order)
                self.assertFalse(result["training_source_native"])
                self.assertEqual(result["optimizer_calls_on_calibration_or_test"], 0)
                self.assertFalse(result["scientific_eligible"])
                self.assertTrue(result["initial_training_loss"] > 0)
                self.assertTrue(result["last_training_loss"] > 0)
                weights = torch.load(root / order / "checkpoint.pt",
                                     map_location="cpu", weights_only=True)
                self.assertEqual(weights["train_steps"], 3)
                self.assertEqual(weights["train_order"], order)
                self.assertEqual(weights["native_teacher_sha256"],
                                 sha256_file(teacher / "manifest.json"))
                scores.append(result)
            self.assertEqual(scores[0]["initial_weights_sha256"],
                             scores[1]["initial_weights_sha256"])
            self.assertEqual(scores[0]["unique_parameter_count"],
                             scores[1]["unique_parameter_count"])
            self.assertNotEqual(scores[0]["final_weights_sha256"],
                                scores[1]["final_weights_sha256"])
            with self.assertRaisesRegex(ValueError, "native teacher provenance"):
                _checkpoint_provenance(
                    torch.load(root / "ab" / "checkpoint.pt",
                               map_location="cpu", weights_only=True),
                    None, [],
                    teacher / "manifest.json",
                )


if __name__ == "__main__":
    unittest.main()

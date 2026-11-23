"""LSA-02B paired order smoke and source/evidence invariants."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from ppt_gym.core import Config, PlasticProofTransformer
from ppt_gym.data import load_fixture, make_fixture, sha256_file
from ppt_gym.order_probe import _commutator, _pearson, _ranks, main, run_probe


class OrderProbeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(103)
        torch.set_num_threads(1)
        self.goal = "x : Nat\n⊢ x + 0 = x"
        self.menu = ("rfl", "simp", "omega", "exact Nat.add_zero x")

    def test_default_ab_bitwise_and_ba_changes_function_with_same_weights(self):
        model = PlasticProofTransformer(
            Config(plan="b", width=16, heads=4, max_bytes=32)
        ).eval()
        params_before = copy.deepcopy(model.state_dict())
        with torch.no_grad():
            default = model(self.goal, self.menu, depth=2)[-1]["logits"]
            explicit = model(self.goal, self.menu, depth=2, middle_order="ab")[-1]["logits"]
            reversed_ = model(self.goal, self.menu, depth=2, middle_order="ba")[-1]["logits"]
            neutral = model(self.goal, self.menu, depth=2, middle_order="ab",
                            message_mode="zero")[-1]["logits"]
            reversed_neutral = model(self.goal, self.menu, depth=2,
                                     middle_order="ba", message_mode="zero")[-1]["logits"]
        self.assertTrue(torch.equal(default, explicit))
        self.assertTrue(bool(torch.isfinite(reversed_).all()))
        self.assertGreater(float((default - reversed_).norm()), 1e-7)
        self.assertGreater(float((neutral - reversed_neutral).norm()), 1e-7)
        self.assertTrue(all(torch.equal(value, model.state_dict()[key])
                            for key, value in params_before.items()))
        self.assertEqual(model.parameter_count(), sum(p.numel() for p in model.parameters()))

    def test_rejects_untyped_order_and_plan_a_swap(self):
        plan_b = PlasticProofTransformer(
            Config(plan="b", width=16, heads=4, max_bytes=32)
        )
        plan_a = PlasticProofTransformer(
            Config(plan="a", width=16, heads=4, max_bytes=32)
        )
        with self.assertRaisesRegex(ValueError, "middle_order"):
            plan_b(self.goal, self.menu, middle_order="abc")
        with self.assertRaisesRegex(ValueError, "Plan A"):
            plan_a(self.goal, self.menu, middle_order="ba")

    def test_fixture_retains_synthetic_label_and_checkpoint_custody(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = root / "fixture"
            make_fixture(fixture, per_family=9, seed=31)
            source_cases, source = load_fixture(fixture, "calibration", 256)
            model = PlasticProofTransformer(
                Config(plan="b", width=16, heads=4, max_bytes=32)
            ).eval()
            checkpoint = root / "checkpoint.pt"
            torch.save({
                "schema": "ppt_gym.checkpoint.v1",
                "config": asdict(model.cfg),
                "weights": model.state_dict(),
                "source_manifest_sha256": source["manifest_sha256"],
                "source_evidence_class": "synthetic_fixture",
                "train_steps": 2,
                "train_order": "ab",
            }, checkpoint)
            original_sha = sha256_file(checkpoint)
            result = run_probe(
                fixture, checkpoint, root / "probe",
                synthetic=True, depth=2, max_cases=256,
            )
            self.assertEqual(sha256_file(checkpoint), original_sha)
            self.assertEqual(result["cases"], len(source_cases))
            self.assertEqual(result["updates_from_query_labels"], 0)
            self.assertEqual(result["new_lean_kernel_calls"], 0)
            self.assertFalse(result["scientific_eligible"])
            self.assertFalse(result["competence_gate_passed"])
            self.assertEqual(result["source"]["evidence_class"], "synthetic_fixture")
            self.assertEqual(
                result["ab_wins"] + result["ba_wins"] + result["ties"],
                result["eligible_witness_roots"],
            )
            self.assertGreater(result["mean_commutator_norm"], 0)
            self.assertGreater(result["mean_normal_logit_gap"], 0)
            records = [
                json.loads(line)
                for line in (root / "probe" / "paired_roots.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len({row["lineage"] for row in records}), len(records))
            self.assertTrue(all(row["evidence_class"] == "synthetic_fixture" for row in records))
            again = run_probe(fixture, checkpoint, root / "replay", synthetic=True, depth=2)
            self.assertEqual(result["rows_sha256"], again["rows_sha256"])
            with self.assertRaisesRegex(ValueError, "fresh directory"):
                run_probe(fixture, checkpoint, root / "probe", synthetic=True)
            with self.assertRaisesRegex(ValueError, "locked test requires"):
                run_probe(fixture, checkpoint, root / "test", split="test", synthetic=True)
            with self.assertRaisesRegex(ValueError, "allow-synthetic"):
                main([
                    "--fixture", str(fixture), "--checkpoint", str(checkpoint),
                    "--out", str(root / "refused"),
                ])

    def test_ba_training_checkpoint_records_mirrored_schedule(self):
        from ppt_gym.cli import main as train_cli
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = root / "fixture"
            make_fixture(fixture, per_family=3, seed=43)
            parameters = []
            for order in ("ab", "ba"):
                folder = root / order
                train_cli([
                    "train", "--fixture", str(fixture), "--allow-synthetic",
                    "--plan", "b", "--middle-order", order,
                    "--width", "16", "--heads", "4", "--max-bytes", "32",
                    "--steps", "1", "--depths", "1", "--max-seconds", "60",
                    "--out", str(folder),
                ])
                item = torch.load(folder / "checkpoint.pt", map_location="cpu",
                                  weights_only=True)
                report = json.loads((folder / "manifest.json").read_text())
                self.assertEqual(item["train_order"], order)
                self.assertEqual(report["train_order"], order)
                self.assertEqual(item["train_steps"], 1)
                self.assertFalse(report["scientific_eligible"])
                parameters.append(item)
            self.assertEqual(parameters[0]["source_manifest_sha256"],
                             parameters[1]["source_manifest_sha256"])
            self.assertEqual(len(parameters[0]["weights"]), len(parameters[1]["weights"]))
            self.assertTrue(any(
                not torch.equal(parameters[0]["weights"][name],
                                parameters[1]["weights"][name])
                for name in parameters[0]["weights"]
            ))

    def test_statistics_handle_constant_and_tied_ranks(self):
        self.assertEqual(_ranks([3.0, 1.0, 1.0]), [3.0, 1.5, 1.5])
        self.assertIsNone(_pearson([1.0, 1.0, 1.0], [2.0, 3.0, 4.0]))
        self.assertAlmostEqual(_pearson(_ranks([1, 2, 3]), _ranks([3, 2, 1])), -1.0)


if __name__ == "__main__":
    unittest.main()

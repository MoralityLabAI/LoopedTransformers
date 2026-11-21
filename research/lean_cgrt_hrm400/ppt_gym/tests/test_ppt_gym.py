"""CPU-only correctness checks for PPT-A/B, custody and online plasticity."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from ppt_gym.cli import main
from ppt_gym.core import Config, FastHead, PlasticProofTransformer, RetrievalControl, supervised_loss
from ppt_gym.data import (Case, PACK_SCHEMA, canonical, ensure_no_overlap, load_checked_pack,
                          load_fixture, make_fixture, sha256_file)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        torch.set_num_threads(1)
        self.goal = "n : Nat\n⊢ n + 0 = n"
        self.tactics = ("rfl", "exact Nat.add_zero n", "simp", "omega")

    def test_both_core_types_differentiable_and_depth_observable(self):
        for plan in ("a", "b"):
            model = PlasticProofTransformer(Config(plan=plan, width=16, heads=4, max_bytes=32))
            self.assertGreater(model.parameter_count(), 0)
            outputs = model(self.goal, self.tactics, depth=3, return_all=True)
            self.assertEqual([int(o["pass"]) for o in outputs], [1, 2, 3])
            self.assertTrue(all(o["logits"].shape == (4,) for o in outputs))
            loss = supervised_loss(outputs, (0, 1, 0, 0))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(any(p.grad is not None and bool(torch.isfinite(p.grad).all())
                                for p in model.parameters()))
            self.assertNotEqual(float(outputs[0]["logits"].sum()),
                                float(outputs[-1]["logits"].sum()))

    def test_previous_pass_message_changes_plan_b_output(self):
        model = PlasticProofTransformer(Config(plan="b", width=16, heads=4, max_bytes=32)).eval()
        with torch.no_grad():
            model.gate_a.gate.bias.fill_(3.0)
            model.gate_b.gate.bias.fill_(3.0)
            normal = model(self.goal, self.tactics, depth=3)[-1]["logits"]
            zero = model(self.goal, self.tactics, depth=3, message_mode="zero")[-1]["logits"]
            rolled = model(self.goal, self.tactics, depth=3, message_mode="roll")[-1]["logits"]
        self.assertFalse(torch.allclose(normal, zero, atol=1e-7))
        self.assertFalse(torch.allclose(normal, rolled, atol=1e-7))

    def test_fast_head_updates_without_backprop_through_backbone(self):
        head = FastHead(16, learning_rate=2.0)
        control = RetrievalControl(16)
        f = torch.randn(16)
        p0 = float(head.predict(f[None])[0])
        head.update(f, 1)
        control.update(f, 1)
        self.assertGreater(float(head.predict(f[None])[0]), p0)
        self.assertGreater(float(control.predict(f[None])[0]), 0.0)
        self.assertEqual(head.updates, 1)
        with self.assertRaises(ValueError):
            head.update(f, -1)

    def test_invalid_depth_and_censored_loss(self):
        model = PlasticProofTransformer(Config(plan="a", width=16, heads=4, max_bytes=32))
        with self.assertRaises(ValueError):
            model(self.goal, self.tactics, depth=17)
        with self.assertRaises(ValueError):
            supervised_loss(model(self.goal, self.tactics), (-1, -1, -1, -1))


class ImportTests(unittest.TestCase):
    def test_fixture_is_explicitly_not_kernel_certification(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "synthetic"
            report = make_fixture(p, per_family=9, seed=7)
            self.assertFalse(report["scientific_eligible"])
            self.assertFalse(report["student_training_eligible"])
            train, train_source = load_fixture(p, "train")
            query, _ = load_fixture(p, "calibration")
            self.assertEqual(len(train), 27)
            self.assertTrue(all(c.evidence == "synthetic_fixture" for c in train))
            self.assertFalse(train_source["scientific_eligible"])
            ensure_no_overlap(train, query)
            with self.assertRaises(ValueError):
                ensure_no_overlap(train, train[:1])
            (p / "train.jsonl").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "bytes have changed"):
                load_fixture(p, "train")

    def _checked_pack(self, root):
        root.mkdir()
        public = {
            "id": "rootA", "epoch": "e", "lineage": "family-root-1", "family": "natural",
            "task": "lean_candidate_proof", "statement": "∀ n : Nat, n + 0 = n",
            "context": "n : Nat\n⊢ n + 0 = n",
            "candidates": [{"id": "a1", "action": "lean_tactic", "tactic": "exact Nat.add_zero n"},
                           {"id": "a2", "action": "lean_tactic", "tactic": "omega"}],
        }
        receipts = {
            "id": "rootA", "epoch": "e",
            "outcomes": {
                "a1": {"label": 1, "closed": True, "kernel_ok": True,
                       "kernel_calls": 1, "axioms": []},
                "a2": {"label": 0, "closed": False, "kernel_ok": False,
                       "kernel_calls": 1, "axioms": []},
            },
        }
        a, b = root / "train.public.jsonl", root / "train.receipts.jsonl"
        a.write_text(canonical(public) + "\n", encoding="utf-8")
        b.write_text(canonical(receipts) + "\n", encoding="utf-8")
        manifest = {"schema": PACK_SCHEMA, "splits": {
            "train": {"public": a.name, "receipts": b.name, "goals": 1,
                      "public_sha256": sha256_file(a), "receipts_sha256": sha256_file(b)}
        }, "environment": {"lean_sha256": "pinned"}, "source_evidence_sha256": "receipt-hash"}
        (root / "manifest.json").write_text(canonical(manifest), encoding="utf-8")
        return receipts

    def test_checked_pack_requires_kernel_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pack"
            receipts = self._checked_pack(root)
            cases, source = load_checked_pack(root, "train")
            self.assertEqual(cases[0].labels, (1, 0))
            self.assertEqual(cases[0].evidence, "lean_kernel_receipt_replay")
            self.assertTrue(source["scientific_eligible"])

            # Even when file hashes are updated, a fake positive certificate must fail.
            receipts["outcomes"]["a1"]["axioms"] = ["sorryAx"]
            b = root / "train.receipts.jsonl"
            b.write_text(canonical(receipts) + "\n", encoding="utf-8")
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["splits"]["train"]["receipts_sha256"] = sha256_file(b)
            manifest_path.write_text(canonical(manifest))
            with self.assertRaisesRegex(ValueError, "inconsistent Lean closure"):
                load_checked_pack(root, "train")

    def test_rejects_manifest_tampering_and_filename_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pack"
            self._checked_pack(root)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["splits"]["train"]["public"] = "../escape.jsonl"
            manifest_path.write_text(canonical(manifest))
            with self.assertRaisesRegex(ValueError, "within its directory"):
                load_checked_pack(root, "train")


class CLITests(unittest.TestCase):
    def test_a_b_training_online_comparison_depth_and_exit_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = root / "fixture"
            main(["fixture", "--out", str(fixture), "--per-family", "6", "--seed", "31"])
            for plan in ("a", "b"):
                out = root / plan
                main(["train", "--fixture", str(fixture), "--allow-synthetic",
                      "--out", str(out), "--plan", plan, "--width", "16",
                      "--heads", "4", "--max-bytes", "32", "--max-cases", "18",
                      "--steps", "2", "--depths", "1", "2", "--max-seconds", "45"])
                meta = json.loads((out / "manifest.json").read_text())
                self.assertEqual(meta["updates_completed"], 2)
                self.assertFalse(meta["scientific_eligible"])
                compare_dir = root / f"{plan}_compare"
                main(["compare", "--fixture", str(fixture), "--allow-synthetic",
                      "--out", str(compare_dir), "--checkpoint", str(out / "checkpoint.pt"),
                      "--max-cases", "18", "--support", "2", "--depth", "2"])
                result = json.loads((compare_dir / "manifest.json").read_text())
                self.assertGreater(result["query_roots"], 0)
                self.assertEqual(result["updates_from_query_labels"], 0)
                self.assertEqual(set(result["arms"]), {"frozen", "dram_fast", "retrieval",
                                                         "inverted_feedback"})
                self.assertFalse(result["scientific_eligible"])
                sweep_dir = root / f"{plan}_sweep"
                main(["sweep", "--fixture", str(fixture), "--allow-synthetic",
                      "--out", str(sweep_dir), "--checkpoint", str(out / "checkpoint.pt"),
                      "--max-cases", "9", "--depths", "1", "2"])
                sweep = json.loads((sweep_dir / "manifest.json").read_text())
                self.assertEqual(len(sweep["results"]), 6 if plan == "b" else 2)
            fit_dir = root / "exit"
            main(["fit-exit", "--fixture", str(fixture), "--allow-synthetic",
                  "--out", str(fit_dir), "--checkpoint", str(root / "b" / "checkpoint.pt"),
                  "--max-cases", "18", "--depth", "2", "--steps", "1", "--max-seconds", "45"])
            fit = json.loads((fit_dir / "manifest.json").read_text())
            self.assertEqual(fit["updates"], 1)
            adaptive_dir = root / "adaptive"
            main(["sweep", "--fixture", str(fixture), "--allow-synthetic",
                  "--checkpoint", str(fit_dir / "checkpoint.pt"),
                  "--max-cases", "9", "--depths", "1", "2",
                  "--adaptive-threshold", "0.5", "--out", str(adaptive_dir)])
            adaptive = json.loads((adaptive_dir / "manifest.json").read_text())
            early = [row for row in adaptive["results"] if row.get("depth_policy") == "trained_exit_head"]
            self.assertEqual(len(early), 3)
            self.assertTrue(all(0 < row["mean_passes"] <= 2 for row in early))

            # Persistent train-only DRAM state and replayed ALEPH data exports.
            accrued = root / "accretion"
            main(["accrete", "--fixture", str(fixture), "--allow-synthetic",
                  "--checkpoint", str(root / "b" / "checkpoint.pt"),
                  "--max-cases", "18", "--depth", "2", "--out", str(accrued)])
            manifest = json.loads((accrued / "manifest.json").read_text())
            self.assertEqual(manifest["new_lean_kernel_calls"], 0)
            self.assertFalse(manifest["scientific_eligible"])
            self.assertTrue((accrued / "fast_memory.json").is_file())
            self.assertTrue((accrued / "sft.jsonl").is_file())
            self.assertTrue((accrued / "outcomes.jsonl").is_file())
            replay_dir = root / "replay"
            main(["replay-accretion", "--fixture", str(fixture), "--allow-synthetic",
                  "--checkpoint", str(root / "b" / "checkpoint.pt"),
                  "--max-cases", "18", "--session", str(accrued), "--out", str(replay_dir)])
            self.assertTrue(json.loads((replay_dir / "replay.json").read_text())["replay_passed"])
            with (accrued / "ledger.jsonl").open("a") as stream:
                stream.write("{\"tampered\":true}\n")
            with self.assertRaisesRegex(ValueError, "artifact differs"):
                main(["replay-accretion", "--fixture", str(fixture), "--allow-synthetic",
                      "--checkpoint", str(root / "b" / "checkpoint.pt"),
                      "--max-cases", "18", "--session", str(accrued),
                      "--out", str(root / "replay-bad")])

    def test_fixture_requires_explicit_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp) / "fixture"
            make_fixture(fixture, per_family=3)
            with self.assertRaisesRegex(ValueError, "allow-synthetic"):
                main(["train", "--fixture", str(fixture), "--plan", "a",
                      "--out", str(Path(tmp) / "refused"), "--steps", "1"])


if __name__ == "__main__":
    unittest.main()

"""Unit tests for Q4/Q5 receipt and checkpoint provenance."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from research_gym.scripts.loopscope_mg_cohort import (
    COHORTS, checksum, execute, inventory, selected_models,
)


class CohortPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = SimpleNamespace(
            receipt_root=self.root / "experiment",
            q4_dir=self.root / "weights" / "q4",
            q5_dir=self.root / "weights" / "q5",
            cohorts=["q4", "q5"],
        )
        for cohort, (subfolder, expected) in COHORTS.items():
            weights = self.args.q4_dir if cohort == "q4" else self.args.q5_dir
            weights.mkdir(parents=True)
            for index in range(expected):
                seed = index + 1
                lam = 0.0 if index < expected // 2 else 0.01
                arm = f"lam{lam:g}_s{seed}"
                filename = arm + ".final.pt"
                checkpoint = weights / filename
                if index == 0:
                    checkpoint.write_bytes((cohort + arm).encode())
                expected_sha = checksum(checkpoint) if checkpoint.is_file() else "0" * 64
                receipt = self.args.receipt_root / subfolder / "runs" / arm / "receipt.json"
                receipt.parent.mkdir(parents=True)
                receipt.write_text(json.dumps({
                    "arm": arm, "checkpoint": filename, "checkpoint_sha256": expected_sha,
                    "seed": seed, "lam": lam,
                    "config": {"seed": seed, "lam": lam, "fixed_depth": True},
                }))

    def test_all_receipts_and_hashes(self):
        rows = inventory(self.args)
        self.assertEqual(len(rows), 24)
        self.assertEqual(sum(r["status"] == "READY" for r in rows), 2)
        self.assertEqual(sum(r["status"] == "MISSING" for r in rows), 22)
        self.assertEqual(len(selected_models(rows, 0)), 24)
        self.assertEqual(len(selected_models(rows, 2)), 4)
        chosen = selected_models(rows, 2)
        self.assertEqual([r["lambda"] for r in chosen if r["cohort"] == "q4"], [0.0, 0.01])

    def test_reject_corrupted_checkpoint(self):
        path = self.args.q4_dir / "lam0_s1.final.pt"
        path.write_bytes(b"modified")
        row = next(r for r in inventory(self.args)
                   if r["cohort"] == "q4" and r["arm"] == "lam0_s1")
        self.assertEqual(row["status"], "HASH_MISMATCH")

    def test_reject_invalid_receipt(self):
        p = self.args.receipt_root / "loopscope_toy" / "runs" / "lam0_s1" / "receipt.json"
        receipt = json.loads(p.read_text())
        receipt["checkpoint"] = "../unsafe.pt"
        p.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "conflicts"):
            inventory(self.args)

    def test_reject_missing_receipt(self):
        p = self.args.receipt_root / "loopscope_toy_q5" / "runs" / "lam0_s1" / "receipt.json"
        p.unlink()
        with self.assertRaisesRegex(ValueError, "expected 12"):
            inventory(self.args)

    def test_resume_requires_argv_and_checkpoint(self):
        target = self.root / "out" / "model" / "probes"
        target.mkdir(parents=True)
        receipt = target / "results.json"
        receipt.write_text(json.dumps({
            "checkpoint_sha256": "a" * 64, "status": "FROZEN_CHECKPOINT_DEVELOPMENT",
        }))
        log = target.parent / "probes.process.json"
        log.write_text(json.dumps({"argv": ["python", "probe"], "returncode": 0}))
        r = execute("probes", target, ["python", "probe"], "a" * 64, resume=True)
        self.assertEqual(r["status"], "REUSED_VERIFIED")
        with self.assertRaisesRegex(ValueError, "provenance changed"):
            execute("probes", target, ["python", "changed"], "a" * 64, resume=True)
        with self.assertRaisesRegex(ValueError, "provenance changed"):
            execute("probes", target, ["python", "probe"], "b" * 64, resume=True)


if __name__ == "__main__":
    unittest.main()

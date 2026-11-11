"""End-to-end CPU-trained fixture; expressly not an original Q4/Q5 checkpoint."""
from __future__ import annotations
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class NativeCohortFixtureTest(unittest.TestCase):
    def test_frozen_cpu_fixture_probes_causal_and_resume(self):
        import torch
        import torch.nn.functional as F
        from research_gym.scripts.loopscope_toy import LoopedChase, make_batch

        torch.set_num_threads(1)
        torch.manual_seed(8128)
        cfg = {"n_nodes": 4, "d": 16, "layers": 1, "heads": 2,
               "fixed_depth": True, "max_train_depth": 8, "loops": 8,
               "seed": 1, "lam": 0.0, "steps": 30,
               "training_status": "NEWLY_TRAINED_CPU_FIXTURE"}
        model = LoopedChase(4, 16, 1, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
        for step in range(30):
            tokens, hops = make_batch(4, 32, seed=5000 + step, max_depth=8)
            loss = F.cross_entropy(model.readout(model(tokens, 8)[-1]), hops[:, -1])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            receipt_root = base / "receipts"
            checkpoint_dir = base / "weights"
            checkpoint_dir.mkdir()
            checkpoint = checkpoint_dir / "lam0_s1.final.pt"
            torch.save({"cfg": cfg, "model": model.state_dict()}, checkpoint)
            sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            for index in range(12):
                seed = index + 1
                lam = 0.0 if index < 6 else 0.01
                arm = f"lam{lam:g}_s{seed}"
                path = receipt_root / "loopscope_toy" / "runs" / arm / "receipt.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({
                    "arm": arm, "seed": seed, "lam": lam,
                    "checkpoint": arm + ".final.pt",
                    "checkpoint_sha256": sha if seed == 1 else "0" * 64,
                    "config": {**cfg, "seed": seed, "lam": lam},
                }))
            output = base / "results"
            cmd = [sys.executable, "-m", "research_gym.scripts.loopscope_mg_cohort",
                   "--receipt-root", str(receipt_root), "--q4-dir", str(checkpoint_dir),
                   "--cohorts", "q4", "--out", str(output), "--run",
                   "--pilot-per-cohort", "1", "--assay", "both",
                   "--targets", "8", "--group-counts", "3", "1", "1",
                   "--items-per-group", "4", "--probe-steps", "12",
                   "--batch-size", "16", "--hidden", "8", "--threads", "1",
                   "--causal-items", "1", "--horizon", "2",
                   "--device", "cpu", "--dtype", "float64"]
            run = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr[-5000:])
            inv = json.loads((output / "inventory.json").read_text())
            self.assertEqual((inv["ready_count"], inv["blocked_count"]), (1, 0))
            receipt = json.loads((output / "execution_receipts.json").read_text())
            self.assertEqual([r["status"] for r in receipt["runs"][0]["jobs"]],
                             ["SUCCEEDED", "SUCCEEDED"])
            for job in receipt["runs"][0]["jobs"]:
                result = json.loads(Path(job["result"]).read_text())
                self.assertEqual(result["checkpoint_sha256"], sha)
                self.assertNotEqual(result["status"], "UNTRAINED_SMOKE")
                self.assertFalse(result["natural_language_monitor_evaluated"])
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["completed_count"], 1)
            self.assertEqual(summary["status"], "TRAINED_COHORT_DEVELOPMENT")
            resumed = subprocess.run(cmd + ["--resume"], capture_output=True, text=True)
            self.assertEqual(resumed.returncode, 0, resumed.stderr[-5000:])
            after = json.loads((output / "execution_receipts.json").read_text())
            self.assertEqual([r["status"] for r in after["runs"][0]["jobs"]],
                             ["REUSED_VERIFIED", "REUSED_VERIFIED"])


if __name__ == "__main__":
    unittest.main()

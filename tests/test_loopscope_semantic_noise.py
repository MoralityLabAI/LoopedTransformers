"""SUD-02 native integration. Untrained smoke results are never research evidence."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from research_gym.scripts import loopscope_semantic_noise as noise

HAS_TORCH = importlib.util.find_spec("torch") is not None
ROOT = Path(__file__).resolve().parents[1]


class TestSemanticNoiseCommand(unittest.TestCase):
    def test_controls_are_exact_and_runnable_without_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "control"
            proc = subprocess.run([
                sys.executable, "-m", "research_gym.scripts.loopscope_semantic_noise",
                "--controls", "--out", str(out),
            ], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            record = json.loads((out / "manifest.json").read_text())
            self.assertEqual(record["status"], "CONTROL_ONLY")
            self.assertTrue(record["complete"])
            depths = record["results"][0]["depths"]
            self.assertLess(depths[1]["source_information_bits"],
                            depths[2]["source_information_bits"])
            rerun = subprocess.run([
                sys.executable, "-m", "research_gym.scripts.loopscope_semantic_noise",
                "--controls", "--out", str(out),
            ], cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(rerun.returncode, 0)
            self.assertIn("nonempty", rerun.stderr)

    def test_help_lazy_native_dependencies(self):
        proc = subprocess.run([
            sys.executable, "-m", "research_gym.scripts.loopscope_semantic_noise",
            "--help",
        ], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--q5-dir", proc.stdout)

    def test_q5_missing_weights_block_without_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "q5"
            empty = Path(tmp) / "empty"
            empty.mkdir()
            proc = subprocess.run([
                sys.executable, "-m", "research_gym.scripts.loopscope_semantic_noise",
                "--q5-dir", str(empty), "--graphs", "1", "--depths", "1",
                "--out", str(out),
            ], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 2, proc.stderr)
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertFalse(manifest["complete"])
            self.assertEqual(manifest["status"], "BLOCKED_CHECKPOINTS")
            invent = json.loads((out / "checkpoint_inventory.json").read_text())
            self.assertEqual(len(invent), 12)
            self.assertTrue(all(entry["status"] == "MISSING" for entry in invent))

    def test_all_start_tokens_show_same_edges_and_distinct_corrupt_starts(self):
        if not HAS_TORCH:
            self.skipTest("PyTorch unavailable")
        import torch
        permutation = [0, 3, 1, 2]
        order = [2, 0, 3, 1]
        inputs = noise._tokens_for_all_starts(4, permutation, order, torch, "cpu")
        self.assertEqual(tuple(inputs.shape), (4, 9))
        self.assertEqual(inputs[:, -1].tolist(), [0, 1, 2, 3])
        self.assertTrue(bool(torch.all(inputs[:, :-1] == inputs[:1, :-1])))

    @unittest.skipUnless(HAS_TORCH, "requires PyTorch")
    def test_untrained_native_smoke_verifies_trace_parity_and_retains_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "smoke"
            proc = subprocess.run([
                sys.executable, "-m", "research_gym.scripts.loopscope_semantic_noise",
                "--smoke", "--graphs", "2", "--depths", "1", "2", "4",
                "--threads", "1", "--out", str(out),
            ], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["status"], "UNTRAINED_SMOKE")
            report = json.loads((out / "00_untrained_smoke" / "report.json").read_text())
            self.assertTrue(report["native_trace_parity"]["passed"])
            self.assertEqual(report["status"], "UNTRAINED_SMOKE")
            self.assertIsNone(report["checkpoint_sha256"])
            self.assertEqual(report["by_depth"][0]["graph_units"], 2)
            self.assertTrue((out / "00_untrained_smoke" / "graphs.jsonl").is_file())
            for d in report["by_depth"]:
                self.assertGreaterEqual(d["mean_native_head_information_loss_bits"], -1e-9)
                memory = d["independent_graph_episode_memory_bound"]
                self.assertEqual(memory["episode_count"], 2)
                self.assertGreaterEqual(
                    memory["native_head_side_memory_lower_bound_bits"] + 1e-9,
                    memory["source_side_memory_lower_bound_bits"])


if __name__ == "__main__":
    unittest.main()

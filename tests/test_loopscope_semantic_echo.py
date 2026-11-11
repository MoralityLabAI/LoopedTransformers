"""Adapter integration checks. Torch-dependent tests skip in numpy-only CI."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_gym.scripts import loopscope_semantic_echo as adapter

HAS_TORCH = importlib.util.find_spec("torch") is not None


class CommandLineTests(unittest.TestCase):
    def test_help_does_not_require_torch(self):
        # The runner must reach argparse's help exit before any torch import.
        program = """
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('help attempted to import torch')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = [sys.argv[1], '--help']
runpy.run_path(sys.argv[0], run_name='__main__')
"""
        result = subprocess.run([sys.executable, "-c", program, str(Path(adapter.__file__))],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--checkpoint", result.stdout)

    def test_empty_checkpoint_directory_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            command = [sys.executable, str(Path(adapter.__file__)), "--checkpoint-dir", tmp,
                       "--outdir", str(Path(tmp) / "output")]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("no *.final.pt checkpoints", result.stderr)

    def test_selection_ignores_future_outcomes(self):
        rows = [{"t": 1, "u": 5, "semantic_label": 2, "endpoint_gate": True,
                 "pair_any_disagreement": False},
                {"t": 2, "u": 6, "semantic_label": 3, "endpoint_gate": True,
                 "pair_any_disagreement": True},
                {"t": 3, "u": 7, "semantic_label": 0, "endpoint_gate": False,
                 "pair_any_disagreement": False}]
        data = [{"continuation": {"raw_rows": rows}}]
        selected = adapter.select_pairs(data, 1, 94)
        for row in rows:
            row["pair_any_disagreement"] = not row["pair_any_disagreement"]
        self.assertEqual(selected, adapter.select_pairs(data, 1, 94))

    def test_future_diagnostics_exhaust_four_cells_and_offsets(self):
        expected = np.asarray([0, 1])
        observed = [([0, 1], [0, 1]), ([2, 1], [2, 1]),
                    ([0, 1], [2, 1]), ([2, 1], [0, 2])]
        rows = []
        for donor, receiver in observed:
            rows.append({"endpoint_gate": True, "future_predictions_t": donor,
                         "future_predictions_u": receiver,
                         "oracle_correct_t": (np.asarray(donor) == expected).tolist(),
                         "oracle_correct_u": (np.asarray(receiver) == expected).tolist()})
        summary = adapter.future_diagnostics(rows, 2)
        self.assertEqual(set(summary["pair_comparison_cells"].values()), {1})
        self.assertEqual(summary["all_h_both_correct_pair_count"], 1)
        self.assertEqual(summary["all_h_donor_correct_pair_count"], 2)
        self.assertEqual(summary["all_h_receiver_correct_pair_count"], 1)
        self.assertEqual([r["disagreement_count"] for r in summary["per_future_offset"]], [2, 1])
        self.assertEqual([r["donor_oracle_correct_count"] for r in summary["per_future_offset"]], [2, 4])
        self.assertEqual([r["receiver_oracle_correct_count"] for r in summary["per_future_offset"]], [2, 3])
        self.assertEqual(sum(summary["step_comparison_cells"].values()), 8)
        empty = adapter.future_diagnostics([], 2)
        self.assertIsNone(empty["all_h_both_correct_pair_rate"])
        self.assertIsNone(empty["per_future_offset"][0]["disagreement_rate"])


@unittest.skipUnless(HAS_TORCH, "PyTorch is optional; install it to run checkpoint integration tests")
class NativeAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from research_gym.scripts.loopscope_toy import LoopedChase, make_batch
        cls.torch = torch
        cls.make_batch = staticmethod(make_batch)
        torch.set_num_threads(1)
        torch.manual_seed(940901)
        cls.model = LoopedChase(n_nodes=4, d=16, layers=1, heads=2).eval()

    def test_native_trace_matches_source_forward_and_own_head(self):
        torch = self.torch
        tokens, _ = self.make_batch(4, 3, 940902, 6)
        with torch.inference_mode():
            _, states = adapter.native_trace(self.model, tokens, 6)
            parity = adapter.verify_native_trace(self.model, tokens, states, rtol=2e-5, atol=2e-6)
        self.assertEqual(tuple(states.shape), (7, 3, 5, 16))
        self.assertTrue(parity["passed"])
        self.assertEqual(parity["own_head_label_mismatches"], 0)
        self.assertTrue(parity["logit_parity"]["passed"])
        self.assertEqual(parity["logit_parity"]["max_abs_error"], 0.0)

    def test_parity_rejects_corrupted_cursor(self):
        torch = self.torch
        tokens, _ = self.make_batch(4, 2, 940903, 5)
        with torch.inference_mode():
            _, states = adapter.native_trace(self.model, tokens, 5)
            states[3, 1, -1, 0] += 0.5
            with self.assertRaisesRegex(RuntimeError, "native parity failed"):
                adapter.verify_native_trace(self.model, tokens, states, rtol=2e-5, atol=2e-6)

    def test_swaps_respect_cursor_and_edge_scopes(self):
        torch = self.torch
        base = torch.arange(80, dtype=torch.float32).reshape(1, 5, 16)
        donor = base + 10
        cursor, meta = adapter.patch_state(base, donor, "cursor_swap", 4, np.random.default_rng(1))
        self.assertTrue(torch.equal(cursor[:, :-1], base[:, :-1]))
        self.assertTrue(torch.equal(cursor[:, -1], donor[:, -1]))
        self.assertEqual(meta["scope"], "cursor")
        edges, _ = adapter.patch_state(base, donor, "edge_swap", 4, np.random.default_rng(1))
        self.assertTrue(torch.equal(edges[:, -1], base[:, -1]))
        self.assertTrue(torch.equal(edges[:, :-1], donor[:, :-1]))
        for kind, scope in (("random_cursor", slice(-1, None)), ("random_edges", slice(0, 4)),
                            ("random_full", slice(None))):
            changed, metadata = adapter.patch_state(base, donor, kind, 4, np.random.default_rng(2))
            self.assertAlmostEqual(metadata["applied_l2"], metadata["donor_difference_l2_in_scope"], places=4)
            if kind == "random_cursor":
                self.assertTrue(torch.equal(changed[:, :-1], base[:, :-1]))
            if kind == "random_edges":
                self.assertTrue(torch.equal(changed[:, -1], base[:, -1]))

    def test_full_donor_replay_matches_known_future(self):
        torch = self.torch
        tokens, hops = self.make_batch(4, 2, 940904, 8)
        with torch.inference_mode():
            embedding, states = adapter.native_trace(self.model, tokens, 8)
            result = adapter.replay_pair(self.model, states, embedding,
                                         {"item": 0, "t": 1, "u": 4},
                                         hops[0, 4:7].numpy(), horizon=3, seed=940905,
                                         rtol=2e-5, atol=2e-6)
        self.assertTrue(result["full_donor_replay_parity"]["passed"])
        self.assertTrue(result["full_donor_logit_parity"]["passed"])
        self.assertEqual(result["interventions"][0]["future_predictions"], result["donor_future"])
        self.assertLess(result["layernorm_shift_max_relative_state_error"], 2e-5)

    def test_variable_depth_checkpoint_is_rejected(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "variable.final.pt"
            cfg = {"n_nodes": 4, "d": 16, "layers": 1, "heads": 2, "fixed_depth": False}
            torch.save({"cfg": cfg, "model": self.model.state_dict()}, path)
            with self.assertRaisesRegex(ValueError, "fixed_depth=True"):
                adapter.load_model(path, smoke=False, seed=1, device="cpu", dtype="float32")

    def test_final_checkpoint_roundtrip_preserves_native_outputs(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixed.final.pt"
            cfg = {"n_nodes": 4, "d": 16, "layers": 1, "heads": 2,
                   "fixed_depth": True, "loops": 8, "max_train_depth": 8}
            torch.save({"cfg": cfg, "model": self.model.state_dict()}, path)
            loaded, recovered_cfg = adapter.load_model(path, smoke=False, seed=1,
                                                       device="cpu", dtype="float32")
            tokens, _ = self.make_batch(4, 2, 940906, 4)
            with torch.inference_mode():
                reference = self.model(tokens, 4)
                observed = loaded(tokens, 4)
            self.assertTrue(torch.equal(observed, reference))
            self.assertEqual(recovered_cfg, cfg)

    def test_smoke_cli_writes_complete_receipts_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "smoke"
            command = [sys.executable, str(Path(adapter.__file__)), "--smoke", "--out", str(output),
                       "--items", "2", "--max-loops", "6", "--horizon", "2",
                       "--max-pairs", "1", "--cpu-threads", "1"]
            result = subprocess.run(command, capture_output=True, text=True, env=os.environ.copy())
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertTrue(manifest["complete"])
            run_dir = output / manifest["runs"][0]["run_directory"]
            metrics = json.loads((run_dir / "metrics.json").read_text())
            self.assertEqual(metrics["receipt"]["status"], "UNTRAINED_SMOKE")
            self.assertEqual(metrics["receipt"]["scope"], "own_head_only")
            self.assertTrue(metrics["receipt"]["native_parity"]["passed"])
            self.assertTrue(metrics["receipt"]["fixed_depth_compatible"])
            self.assertEqual(metrics["receipt"]["evidence_stage"], "untrained_execution_smoke")
            self.assertIn("--out", metrics["receipt"]["argv"])
            self.assertEqual(metrics["primary"]["unit"], "item")
            self.assertEqual(metrics["primary"]["units_total"], 2)
            rates = [r["continuation"]["per_step_mismatch_rate"] for r in metrics["items"]
                     if r["continuation"]["per_step_mismatch_rate"] is not None]
            self.assertEqual(metrics["primary"]["units_with_metric"], len(rates))
            self.assertEqual(metrics["primary"]["mean"], float(np.mean(rates)) if rates else None)
            self.assertEqual(len(metrics["per_depth_oracle_accuracy"]), 6)
            self.assertEqual(len(metrics["pooled_future_diagnostics"]["per_future_offset"]), 2)
            self.assertIn("## Primary", (run_dir / "report.md").read_text())
            with np.load(run_dir / "full_trace.npz", allow_pickle=False) as trace:
                self.assertEqual(trace["full_states"].shape[0], 7)
                self.assertEqual(trace["predictions"].shape, (6, 2))
            for filename in ("pairs.jsonl", "report.md"):
                self.assertTrue((run_dir / filename).is_file())
            again = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(again.returncode, 0)
            self.assertIn("nonempty", again.stderr)


if __name__ == "__main__":
    unittest.main()

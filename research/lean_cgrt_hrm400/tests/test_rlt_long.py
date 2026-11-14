"""Continuation integrity, chunk-resume equivalence and frozen-cohort boundaries."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lcgrt import tactic_lm as tlm
from lcgrt.mathlib_eval import frozen_states
from lcgrt.rlt_long import common_milestone
from lcgrt.train import load_verified


class FakeStore:
    seen = []
    n = 4

    def __init__(self, path):
        self.split = path

    def batch(self, idx, device):
        if self.split == "train": self.seen.append(np.array(idx))
        b = len(idx)
        enc = np.zeros((b, tlm.CTX), np.uint8); enc[:, :4] = [65, 32, 61, 65]
        dec = np.zeros((b, tlm.TGT), np.uint8); dec[:, :3] = [114, 102, 108]
        return tlm.make_batch(enc, np.full(b, 4, np.int16), device, dec, np.full(b, 3, np.int16))


class LongRLTTests(unittest.TestCase):
    def test_optimizer_continuation_and_chunk_resume_match_uninterrupted(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            base = Path(d)
            common = ["--device", "cpu", "--threads", "1", "--preset", "tlm-tiny", "--cfg", "arch=loop",
                      "fast_steps=2", "--data", "fixture", "--seed", "131", "--batch", "2", "--warmup", "1",
                      "--val-items", "2", "--val-batch", "2", "--log-every", "1", "--eval-every", "2", "--save-every", "1"]
            manifest = {"split_sha256": {"train": "fixed-train", "val": "fixed-val"}}
            with mock.patch.object(tlm, "ensure_dataset", return_value=manifest), \
                 mock.patch.object(tlm, "pack", side_effect=lambda data, split: split), \
                 mock.patch.object(tlm, "Store", FakeStore), \
                 mock.patch.object(tlm.hw, "check", return_value=(True, "ok", {})), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(tlm.train(tlm.parser().parse_args(common + ["--out", str(base / "parent"), "--steps", "2"])), 0)
                original = (base / "parent" / "latest.pt").read_bytes()
                extension = common + ["--steps", "4", "--continue-from", str(base / "parent")]
                FakeStore.seen = []
                self.assertEqual(tlm.train(tlm.parser().parse_args(extension + ["--out", str(base / "chunked"), "--stop-at", "2"])), 0)
                np.testing.assert_array_equal(FakeStore.seen[0], np.random.default_rng([131, 2]).integers(0, 4, 2))
                self.assertFalse((base / "chunked" / "complete.json").exists())
                self.assertEqual(tlm.train(tlm.parser().parse_args(extension + ["--out", str(base / "chunked")])), 0)
                self.assertEqual(tlm.train(tlm.parser().parse_args(extension + ["--out", str(base / "whole")])), 0)
                _, chunked = load_verified(base / "chunked", "cpu")
                _, whole = load_verified(base / "whole", "cpu")
                self.assertEqual(chunked["step"], 4)
                self.assertEqual(chunked["data_step_offset"], 2)
                for key in whole["model"]:
                    torch.testing.assert_close(chunked["model"][key], whole["model"][key], rtol=0, atol=0)
                self.assertTrue(all(int(s["step"]) == 6 for s in chunked["opt"]["state"].values()))
                self.assertEqual((base / "parent" / "latest.pt").read_bytes(), original)
                bad = json.loads(json.dumps(chunked["identity"])); bad["seed"] = 999
                with self.assertRaisesRegex(Exception, "seed"):
                    tlm.continuation(base / "parent", bad)

    def test_frozen_cohort_rejects_duplicates_and_unexpected_labels(self):
        row = {"file": "Mathlib/Fixture.lean", "line": 1, "col": 0, "split": "fresh",
               "checker": "certified", "goals_before": ["⊢ True"], "goals_after": [],
               "kind": "Lean.Parser.Tactic.rfl", "tactic": "rfl"}
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            path = Path(d) / "cohort.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            self.assertEqual(frozen_states(path, ["fresh"]), [row])
            with self.assertRaisesRegex(ValueError, "labels"):
                frozen_states(path, ["calibration"])
            path.write_text((json.dumps(row) + "\n") * 2, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                frozen_states(path, ["fresh"])

    def test_selection_uses_only_milestones_complete_for_every_arm(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            runs = [Path(d) / n for n in ("direct", "rlt", "damping")]
            for run in runs:
                for target in (500, 1000):
                    p = run / "snapshots" / f"step{target:05d}"; p.mkdir(parents=True)
                    if target == 500 or run != runs[1]:
                        (p / "snapshot.json").write_text("{}")
            self.assertEqual(common_milestone(runs), 500)
            (runs[1] / "snapshots" / "step01000" / "snapshot.json").write_text("{}")
            self.assertEqual(common_milestone(runs), 1000)


if __name__ == "__main__":
    unittest.main()

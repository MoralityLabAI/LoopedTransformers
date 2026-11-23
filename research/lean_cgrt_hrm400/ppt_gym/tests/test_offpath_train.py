"""LSA-02H frozen source-sampler and same-root control tests, synthetic only."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from ppt_gym.data import canonical, sha256_file
from ppt_gym.offpath_train import _fit, _sample_roster, run
from ppt_gym.tests.test_offpath_source import CONTEXT, SmallLean
from ppt_gym.native_train import _snapshot_trajectory
from ppt_gym.live_lean import state_input
from ppt_gym.tests.test_offpath_source import FakeCase, ORACLE


class OffpathCPUFixtures(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def _rows(self):
        repl, case = SmallLean(), FakeCase()
        reference = _snapshot_trajectory(repl, case, ORACLE)
        repl.reset()
        state = repl.init(case.statement, pp=True)
        next_state = repl.run(state["sid"], "apply r1")["state"]
        goal, sha = state_input(repl, next_state, (), 7)
        off = {
            "root": case.root, "lineage": case.lineage,
            "step": 1000, "goal": goal, "candidates": ["apply r2"],
            "labels": [1], "checks_remaining": 7,
            "public_state_sha256": sha,
            "source_kind": "off_reference",
            "remaining_proof_steps": 2,
        }
        return reference, {case.root: [off]}

    def test_seeded_source_root_and_switch_draws_are_matched(self):
        reference, supplement = self._rows()
        schedule = _sample_roster(reference, supplement, 157, 60)
        self.assertEqual(schedule, _sample_roster(reference, supplement, 157, 60))
        self.assertNotEqual(schedule, _sample_roster(reference, supplement, 163, 60))
        self.assertTrue(any(d["switched"] for d in schedule))
        self.assertEqual({r["root"] for r in schedule}, {"dummy_root"})
        self.assertTrue(all(d["supplement_index"] in (-1, 0) for d in schedule))

    def test_full_one_worker_receipt_writer_and_source_roster(self):
        # This is a synthetic orchestration fixture with all native loaders
        # mocked; it certifies no theorem and performs no real optimizer fit.
        reference, supplement = self._rows()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            teacher = base / "lsa02h-teacher"
            teacher.mkdir()
            (teacher / "manifest.json").write_text('{"synthetic":true}')
            offpath = base / "offreference"
            offpath.mkdir()
            (offpath / "manifest.json").write_text('{"synthetic":true}')
            source = {
                "source": {"suite_manifest_sha256": "synthetic_suite"},
                "independent_lineages": 1,
            }
            def fake_fit(teacher, supplements, refs, metadata, out, *,
                         seed, arm, steps, max_seconds):
                out.mkdir(parents=True)
                (out / "checkpoint.pt").write_text("synthetic-only")
                (out / "optimizer_steps.jsonl").write_text(
                    canonical({
                        "step": 1, "root": "dummy_root", "ref_index": 0,
                        "switched": True, "supplement_index": 0,
                    }) + "\n"
                )
                return {
                    "seed": seed, "arm": arm,
                    "optimizer_steps": steps,
                    "initial_weights_sha256": "fake-start",
                    "final_weights_sha256": "fake-end",
                    "optimizer_steps_sha256": "fake-trace",
                }
            with (
                patch("ppt_gym.offpath_train._load_teacher",
                      return_value=(reference, source)),
                patch("ppt_gym.offpath_train._load_offpath",
                      return_value=(supplement, {
                          "suite_manifest_sha256": "synthetic_suite",
                      })),
                patch("ppt_gym.offpath_train._fit", side_effect=fake_fit),
                patch("ppt_gym.offpath_train._compare_checkpoints",
                      return_value={
                          "all_tensors_bitwise_equal": True,
                          "full_checkpoint_bytes_equal": True,
                      }),
            ):
                report = run(
                    teacher, offpath, base / "fitted", seeds=(157,),
                    steps=720, max_seconds_per_fit=20,
                )
            self.assertEqual(report["schema"],
                             "lsa02h.matched_recoverable_offpath_cpu.v1")
            self.assertTrue(report["same_seed_identical_source_draws"])
            self.assertTrue((base / "fitted" / "manifest.json").is_file())
            manifest = json.loads((base / "fitted" / "manifest.json").read_text())
            self.assertEqual(len(manifest["cells"]), 3)
            self.assertTrue(manifest["all_exact_duplicate_fits_passed"])

    def test_three_arms_use_same_start_and_switch_opportunities(self):
        reference, supplement = self._rows()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            teacher = base / "teacher"
            teacher.mkdir()
            (teacher / "manifest.json").write_text('{"synthetic":true}')
            offpath = base / "offreference"
            offpath.mkdir()
            (offpath / "manifest.json").write_text('{"synthetic":true}')
            receipts=[]
            for arm in ("reference", "offpath", "sham"):
                output = base / arm
                data = _fit(
                    teacher, supplement, reference,
                    {"suite_manifest_sha256": "synthetic_fixture"},
                    output, seed=157, arm=arm, steps=4,
                    max_seconds=40,
                )
                receipts.append(data)
                self.assertEqual(data["optimizer_steps"], 4)
                self.assertEqual(data["query_label_optimizer_updates"], 0)
                self.assertEqual(data["parameter_count"], 57475)
                self.assertTrue((output / "checkpoint.pt").is_file())
            self.assertEqual(len({x["initial_weights_sha256"] for x in receipts}),1)
            self.assertEqual(len({x["matched_switch_opportunities"] for x in receipts}),1)
            self.assertEqual(receipts[1]["actual_offreference_optimizer_draws"],
                             receipts[1]["matched_switch_opportunities"])
            self.assertEqual(receipts[2]["actual_sham_optimizer_draws"],
                             receipts[2]["matched_switch_opportunities"])
            def core(p):
                return [tuple((r[k] for k in (
                    "step", "root", "ref_index",
                    "switched", "supplement_index",
                ))) for r in (
                    json.loads(x) for x in
                    (p / "optimizer_steps.jsonl").read_text().splitlines()
                )]
            self.assertEqual(core(base / "reference"), core(base / "offpath"))
            self.assertEqual(core(base / "reference"), core(base / "sham"))


if __name__ == "__main__":
    unittest.main()

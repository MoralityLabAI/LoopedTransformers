"""LSA-02F engineering tests; FakeLean is never scientific proof evidence."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from ppt_gym.cli import checkpoint_load
from ppt_gym.data import canonical, sha256_file
from ppt_gym.dose_compare import _bootstrap, _coverage
from ppt_gym.dose_train import VIEWS
from ppt_gym.live_lean import candidate_menu, choose, replay_episode, run_episode, state_input
from ppt_gym.live_suite import create_suite, load_suite
from ppt_gym.native_train import fit
from ppt_gym.tests.test_live_lean import FakeLean
from ppt_gym.tests.test_native_train import fake_teacher
from ppt_gym.typed_state import PRETTY_DOSES, encode_public_packet, schema_for_view
from ppt_gym.uniform_baseline import _root_seed, run as run_random


class DoseAndUniformTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.lean = FakeLean()
        current = self.lean.init("synthetic")
        self.packet, _ = state_input(self.lean, current, (), 8)
        self.menu = candidate_menu(self.lean, current)

    def test_pretty_doses_are_byte_bounded_and_candidate_complete(self):
        row = json.loads(self.packet)
        row["state"]["goals"][0]["pp"] = "H: P0\n" + "∧ premise " * 60 + "⊢ P1"
        packet = canonical(row)
        observed = []
        for view in ("pp0", "pp48", "pp96", "pp160"):
            text = encode_public_packet(packet, self.menu, view=view, max_bytes=384)
            self.assertLessEqual(len(text.encode("utf-8")), 384)
            self.assertIn("B8|G:P1|", text)
            self.assertIn("r0:P0>P1;", text)
            self.assertIn("P:", text)
            excerpt = text.split("P:", 1)[1].split("|", 1)[0]
            n = len(excerpt.encode("utf-8"))
            self.assertLessEqual(n, PRETTY_DOSES[view])
            observed.append(n)
            self.assertEqual(
                encode_public_packet(packet, self.menu, view=view, max_bytes=384),
                text,
            )
            self.assertIn("lsa02f", schema_for_view(view))
        self.assertEqual(observed[0], 0)
        self.assertGreater(observed[1], 0)
        self.assertLessEqual(observed[1], observed[2])
        self.assertLessEqual(observed[2], observed[3])

    def test_view_schema_and_private_field_rejection(self):
        self.assertEqual(VIEWS, ("raw", "pp0", "pp48", "pp96", "pp160"))
        self.assertEqual(schema_for_view("typed_pp"), "lsa02e.typed_pretty_public_state.v1")
        row = json.loads(self.packet)
        row["state"]["goals"][0]["oracle_label"] = 1
        for view in PRETTY_DOSES:
            with self.assertRaisesRegex(ValueError, "unrecognized public"):
                encode_public_packet(canonical(row), self.menu, view=view)
        self.assertRaisesRegex(ValueError, "unknown", schema_for_view, "pp80")
        small = encode_public_packet(self.packet, self.menu, view="pp48", max_bytes=64)
        self.assertLessEqual(len(small.encode("utf-8")), 64)
        self.assertIn("r0:P0>P1;", small)

    def test_root_seeded_public_uniform_choices_replay_without_model(self):
        case = "public state only"
        menu = ("apply h0", "apply r0", "apply r1", "apply r2")
        for seed in (17, 31, 47):
            a = choose(None, case, menu, policy="uniform", depth=2, remaining=8,
                       uniform_seed=_root_seed("root_a", seed))
            b = choose(None, case, menu, policy="uniform", depth=2, remaining=8,
                       uniform_seed=_root_seed("root_a", seed))
            self.assertEqual(a, b)
            self.assertTrue(0 <= a[0] < len(menu))
            self.assertIsNone(a[1])
            self.assertFalse(a[2])
        with self.assertRaisesRegex(ValueError, "unsigned"):
            choose(None, case, menu, policy="uniform", depth=2, remaining=8,
                   uniform_seed=-1)
        with tempfile.TemporaryDirectory() as tmp:
            suite = Path(tmp) / "suite"
            create_suite(suite, per_split=2, seed=188)
            root = load_suite(suite, "calibration")[0][0]
            ep = run_episode(self.lean, root, None, policy="uniform",
                             checks=8, depth=1, uniform_seed=17)
            ep["max_checks"] = 8
            self.assertEqual(ep["model_calls"], 0)
            self.assertTrue(ep["live_kernel"])
            self.assertTrue(replay_episode(self.lean, root, ep))

    def test_provenance_rejects_synthetic_teacher_for_native_random_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = root / "suite"
            create_suite(suite, per_split=2)
            teacher = fake_teacher(root)
            with self.assertRaisesRegex(ValueError, "native compiled Lean"):
                run_random(
                    self.lean, suite, teacher, root / "wrong", certified_backend=False,
                )
            self.assertFalse((root / "wrong").exists())

    def test_small_synthetic_fit_roster_and_source_only_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher = fake_teacher(root)
            rows_file = teacher / "teacher.jsonl"
            rows = [json.loads(line) for line in rows_file.read_text().splitlines()]
            for row in rows:
                row["goal"] = self.packet
                row["candidates"] = list(self.menu)
                row["labels"] = [1]
            rows_file.write_text(
                "".join(canonical(row) + "\n" for row in rows), encoding="utf-8",
            )
            meta_file = teacher / "manifest.json"
            meta = json.loads(meta_file.read_text())
            meta["artifacts"]["teacher.jsonl"] = sha256_file(rows_file)
            meta_file.write_text(canonical(meta) + "\n")
            starts = []
            for view in ("raw", "pp0", "pp48", "pp96", "pp160"):
                fit_manifest = fit(
                    teacher, root / view, order="ab", seed=41,
                    steps=2, depth=2, width=16, max_bytes=384,
                    threads=1, max_seconds=20, allow_synthetic=True,
                    input_view=view,
                )
                starts.append(fit_manifest["initial_weights_sha256"])
                model, checkpoint = checkpoint_load(root / view / "checkpoint.pt")
                self.assertEqual(checkpoint["input_view_schema"], schema_for_view(view))
                self.assertEqual(model.cfg.plan, "b")
                self.assertEqual(fit_manifest["optimizer_steps"], 2)
                self.assertEqual(fit_manifest["view_input_over_budget"], 0
                                 if view != "raw" else fit_manifest["raw_input_over_budget"])
            self.assertEqual(len(set(starts)), 1)
            summary = _coverage(rows)
            for view in PRETTY_DOSES:
                self.assertEqual(summary[view]["candidate_types_visible"],
                                 summary[view]["candidate_types_total"])
                self.assertEqual(summary[view]["raw_over_384"], 0)
            self.assertEqual(_bootstrap([0.0, 0.0]), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()

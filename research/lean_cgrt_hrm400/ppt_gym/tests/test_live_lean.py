"""Protocol and custody tests for PPT-G03. FakeLean is *never* scientific evidence."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from lcgrt.certify import decl_name
from ppt_gym.core import Config, PlasticProofTransformer
from ppt_gym.data import canonical
from ppt_gym.live_campaign import audit_campaign, run_campaign
from ppt_gym.live_lean import candidate_menu, replay_episode, run_episode
from ppt_gym.live_suite import create_suite, load_suite


class FakeLean:
    """Simulates the same JSON shape as the real server for tests only."""
    version = "4.19.0 (FAKE)"
    scientific_eligible = False

    def __init__(self, bad_axiom=False):
        self.bad_axiom = bad_axiom
        self.contexts = {}
        self.calls = 0
        self.reset()

    def reset(self):
        self.contexts = {"ctx1": [
            {"name": "P0", "type": {"sort": "Prop"}},
            {"name": "P1", "type": {"sort": "Prop"}},
            {"name": "h0", "type": {"a": "P0"}},
            {"name": "r0", "type": {"imp": [{"a": "P0"}, {"a": "P1"}]}},
        ]}
        self.calls += 1
        return {"ok": True}

    def init(self, stmt, pp=False):
        self.calls += 1
        return {"sid": 1, "goals": [{"ctx": "ctx1", "pp": "⊢ P1",
                                     "target": {"a": "P1"}}]}

    def run(self, sid, tactic):
        self.calls += 1
        if sid == 1 and tactic == "apply r0":
            return {"ok": True, "state": {"sid": 2, "goals": [
                {"ctx": "ctx1", "pp": "⊢ P0", "target": {"a": "P0"}}
            ]}}
        if sid == 2 and tactic == "apply h0":
            return {"ok": True, "state": {"sid": 3, "goals": []}}
        return {"ok": False, "error": "tactic failed"}

    def kernel(self, sid):
        self.calls += 1
        if sid != 3:
            return {"ok": False, "error": "open goals"}
        return {"ok": True, "axioms": ["sorryAx"] if self.bad_axiom else [],
                "term": "r0 h0"}


def mock_certify(items, *, timeout, keep):
    keep.write_text("SIMULATED CERTIFIER: no Lean process\n", encoding="utf-8")
    return {"schema": "synthetic_certifier", "source_sha256": "synthetic",
            "status": {decl_name(tid): "passed" for tid, _, _ in items}}


class LiveSuiteTests(unittest.TestCase):
    def test_generator_lineage_disjoint_and_file_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "suite"
            meta = create_suite(p, seed=831, per_split=3)
            self.assertIn("unverified_until_live_execution", meta["evidence"])
            cases = []
            for split in ("train", "calibration", "ordinary_test", "ood_family"):
                rows, source = load_suite(p, split)
                self.assertEqual(len(rows), 3)
                self.assertEqual(source["selected_roots"], 3)
                cases.extend(rows)
            self.assertEqual(len({c.lineage for c in cases}), 12)
            self.assertTrue(all(c.theorem() for c in cases))
            with (p / "ood_family.jsonl").open("a", encoding="utf-8") as file:
                file.write(canonical({"tampered": True}) + "\n")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                load_suite(p, "train")

    def test_frozen_statement_and_family_must_remain_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "suite"
            create_suite(p, seed=91, per_split=2)
            rows, _ = load_suite(p, "train")
            case = rows[0]
            from dataclasses import replace
            with self.assertRaisesRegex(ValueError, "generator"):
                replace(case, statement="True").theorem()


class LiveEpisodeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(33)
        torch.set_num_threads(1)
        self.model = PlasticProofTransformer(
            Config(plan="a", width=16, heads=4, max_bytes=32)
        ).eval()

    def _case(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "suite"
            create_suite(root, seed=31, per_split=2)
            return load_suite(root, "train")[0][0]

    def test_live_nonterminal_then_kernel_closed_and_replay(self):
        case = self._case()
        lean = FakeLean()
        ep = run_episode(lean, case, self.model, policy="auto8", checks=8, depth=2)
        ep["max_checks"] = 8
        self.assertEqual(ep["status"], "kernel_closed")
        self.assertEqual(ep["accepted_script"], ["apply r0", "apply h0"])
        self.assertEqual(ep["kernel_calls"], 1)
        self.assertTrue(any(e["status"] == "accepted_nonterminal" for e in ep["events"]))
        self.assertTrue(all(e["status"] != "rejected" for e in ep["events"]))
        self.assertTrue(replay_episode(FakeLean(), case, ep))

    def test_rejects_disallowed_axioms(self):
        case = self._case()
        ep = run_episode(FakeLean(bad_axiom=True), case, self.model,
                         policy="auto8", checks=8, depth=2)
        self.assertEqual(ep["status"], "certificate_rejected")
        self.assertFalse(ep["live_kernel"])
        self.assertEqual(ep["events"][-1]["kernel"]["axioms"], ["sorryAx"])

    def test_invalid_or_missing_local_hypotheses_are_not_invented(self):
        lean = FakeLean()
        state = lean.init("ignored")
        self.assertEqual(candidate_menu(lean, state), ("apply r0",))
        lean.contexts.clear()
        with self.assertRaisesRegex(ValueError, "missing Lean local context"):
            candidate_menu(lean, state)


class LiveCampaignTests(unittest.TestCase):
    def test_simulated_support_query_campaign_replays_without_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = root / "suite"
            create_suite(suite, seed=57, per_split=2)
            train, source = load_suite(suite, "train")
            queries, _ = load_suite(suite, "calibration")
            torch.manual_seed(17)
            model = PlasticProofTransformer(
                Config(plan="b", width=16, heads=4, max_bytes=32)
            ).eval()
            session = root / "campaign"
            report = run_campaign(
                FakeLean(), model, train, queries, out=session, checks=8, depth=2,
                max_seconds=60.0, certified_backend=False,
                certifier=mock_certify, source={
                    "suite_manifest_sha256": source["suite_manifest_sha256"],
                    "checkpoint_evidence": "synthetic_fixture",
                    "lean": {"actual_lean_process": False},
                },
            )
            self.assertFalse(report["actual_lean_process"])
            self.assertFalse(report["architecture_promotion_eligible"])
            self.assertEqual(report["online_query_updates"], 0)
            self.assertEqual(report["support_roots"], 2)
            self.assertEqual(report["fully_evaluated_query_roots"], 2)
            self.assertTrue(report["arms"]["auto8"]["standalone_verified_closures"] >= 1)
            replay = audit_campaign(FakeLean(), suite, session)
            self.assertTrue(replay["replay_passed"])
            self.assertEqual(replay["episodes"], 2 + 2 * 6)
            with (session / "queries.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(canonical({"tampered": True}) + "\n")
            with self.assertRaisesRegex(ValueError, "artifact hash changed"):
                audit_campaign(FakeLean(), suite, session)


if __name__ == "__main__":
    unittest.main()

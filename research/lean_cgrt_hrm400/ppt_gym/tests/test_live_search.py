"""PPT-G03B test: rollback rescue is independent of model predictions."""
from __future__ import annotations

import copy
import unittest

import torch

from ppt_gym.core import Config, PlasticProofTransformer
from ppt_gym.data import digest
from ppt_gym.live_search import replay_search, search_episode
from ppt_gym.live_suite import LiveCase


class BranchRepl:
    """Simulated proof tree. Only the native Lean workflow establishes proof facts."""
    version = "FAKE Lean 4.19.0"

    def __init__(self):
        self.contexts = {}
        self.calls = 0
        self.reset()

    def reset(self):
        self.calls += 1
        self.contexts = {"ctx": [
            {"name": "P0", "type": {"sort": "Prop"}},
            {"name": "P1", "type": {"sort": "Prop"}},
            {"name": "P2", "type": {"sort": "Prop"}},
            {"name": "h0", "type": {"a": "P0"}},
            {"name": "r0", "type": {"imp": [{"a": "P2"}, {"a": "P1"}]}},
            {"name": "r1", "type": {"imp": [{"a": "P0"}, {"a": "P1"}]}},
        ]}
        return {"ok": True}

    def init(self, statement, pp=True):
        self.calls += 1
        return {"sid": 1, "goals": [{"ctx": "ctx", "target": {"a": "P1"}, "pp": "⊢ P1"}]}

    def run(self, sid, tactic):
        self.calls += 1
        after = {
            (1, "apply r0"): {"sid": 2, "goals": [
                {"ctx": "ctx", "target": {"a": "P2"}, "pp": "⊢ P2"}]},
            (1, "apply r1"): {"sid": 3, "goals": [
                {"ctx": "ctx", "target": {"a": "P0"}, "pp": "⊢ P0"}]},
            (3, "apply h0"): {"sid": 4, "goals": []},
        }.get((sid, tactic))
        return {"ok": True, "state": after} if after else {
            "ok": False, "error": "no matching proof hypothesis"}

    def kernel(self, sid):
        self.calls += 1
        return {"ok": sid == 4, "axioms": [] if sid == 4 else ["sorryAx"],
                "term": "r1 h0" if sid == 4 else ""}


class BacktrackingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.case = LiveCase(
            root=digest("backtrack-example"), lineage="fake-proof-tree",
            split="train", family="fake", seed=1, depth=1,
            tid="backtrack-test",
            statement="∀ (P0 P1 P2 : Prop) (h0 : P0) (r0 : P2 → P1) (r1 : P0 → P1), P1",
        )
        self.model = PlasticProofTransformer(
            Config(plan="b", width=16, heads=4, max_bytes=32)
        ).eval()

    def test_auto_backtracks_from_dead_premise_to_certified_proof(self):
        ep = search_episode(BranchRepl(), self.case, self.model, policy="auto8",
                            depth=2, check_budget=8, rollback_budget=3)
        self.assertEqual(ep["status"], "kernel_closed")
        self.assertEqual(ep["tactic_checks"], 3)
        self.assertEqual(ep["rollbacks"], 1)
        self.assertEqual(ep["model_calls"], 0)
        self.assertEqual(ep["accepted_script"], ["apply r1", "apply h0"])
        self.assertEqual([s["kind"] for s in ep["events"]],
                         ["tactic", "rollback", "tactic", "tactic"])
        audit = replay_search(BranchRepl(), self.case, ep)
        self.assertTrue(audit["replay_passed"])
        self.assertEqual(audit["tactic_checks"], 3)
        self.assertEqual(audit["rollbacks"], 1)

    def test_zero_rollback_control_exposes_dead_branch_at_equal_check_limit(self):
        control = search_episode(BranchRepl(), self.case, self.model, policy="auto8",
                                 depth=2, check_budget=8, rollback_budget=0)
        self.assertFalse(control["live_kernel"])
        self.assertEqual(control["status"], "rollback_limit")
        self.assertEqual(control["tactic_checks"], 1)
        self.assertEqual(control["rollbacks"], 0)
        self.assertTrue(replay_search(BranchRepl(), self.case, control)["replay_passed"])

    def test_recurrent_ranker_has_same_check_budget_and_verified_result(self):
        ep = search_episode(BranchRepl(), self.case, self.model, policy="frozen",
                            depth=2, check_budget=8, rollback_budget=3)
        self.assertEqual(ep["status"], "kernel_closed")
        self.assertTrue(2 <= ep["tactic_checks"] <= 3)
        self.assertLessEqual(ep["tactic_checks"], 8)
        self.assertTrue(replay_search(BranchRepl(), self.case, ep)["replay_passed"])

    def test_detects_tampered_rollback_and_budget(self):
        ep = search_episode(BranchRepl(), self.case, self.model, policy="auto8",
                            depth=2, check_budget=8, rollback_budget=3)
        bad = copy.deepcopy(ep)
        bad["events"][1]["abandoned_tactic"] = "apply r1"
        with self.assertRaisesRegex(ValueError, "rollback proof-state witness differs"):
            replay_search(BranchRepl(), self.case, bad)
        bad = copy.deepcopy(ep)
        bad["tactic_checks"] = 2
        with self.assertRaisesRegex(ValueError, "replayed action cost"):
            replay_search(BranchRepl(), self.case, bad)

    def test_rejects_unbounded_or_invalid_search_policies(self):
        with self.assertRaises(ValueError):
            search_episode(BranchRepl(), self.case, self.model, check_budget=0)
        with self.assertRaises(ValueError):
            search_episode(BranchRepl(), self.case, self.model, policy="unbounded")


if __name__ == "__main__":
    unittest.main()

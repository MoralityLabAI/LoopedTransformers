"""LSA-02I: simulated public proof search, rollback cost and native replay law.

These engineering fixtures do not constitute Lean proof certificates.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import unittest

import torch

from ppt_gym.backtrack_search import run_episode, replay_episode, SCHEMA


CONTEXT = [
    {"name": "P0", "type": {"sort": "Prop"}},
    {"name": "P1", "type": {"sort": "Prop"}},
    {"name": "P2", "type": {"sort": "Prop"}},
    {"name": "P3", "type": {"sort": "Prop"}},
    {"name": "h0", "type": {"a": "P0"}},
    {"name": "r0", "type": {"imp": [{"a": "P3"}, {"a": "P1"}]}},
    {"name": "r1", "type": {"imp": [{"a": "P0"}, {"a": "P1"}]}},
    {"name": "r2", "type": {"imp": [{"a": "P1"}, {"a": "P1"}]}},
]


class BranchLean:
    """Provides accepted but dead/looping alternatives and one complete proof."""
    version = "FAKE_LEAN_NOT_NATIVE"

    def __init__(self, *, cycle=False):
        self.cycle = cycle
        self.contexts = {"ctx": CONTEXT}
        self.states = {}
        self.n = 0

    def reset(self):
        self.states = {}
        self.n = 0
        return {"ok": True}

    def new(self, goal):
        self.n += 1
        row = {
            "sid": self.n,
            "goals": [] if goal is None else [{
                "ctx": "ctx", "target": {"a": goal}, "pp": f"⊢ {goal}",
            }],
        }
        self.states[self.n] = row
        return row

    def init(self, statement, pp=True):
        return self.new("P1")

    def run(self, sid, tactic):
        goal = self.states[sid]["goals"][0]["target"]["a"]
        transitions = {
            ("P1", "apply r0"): "P1" if self.cycle else "P3",
            ("P1", "apply r1"): "P0",
            ("P0", "apply h0"): None,
        }
        if (goal, tactic) not in transitions:
            return {"ok": False, "error": "Lean tactic is invalid in simulated state"}
        return {"ok": True, "state": self.new(transitions[(goal, tactic)])}

    def kernel(self, sid):
        closed = not self.states[sid]["goals"]
        return {"ok": closed, "axioms": [] if closed else ["unproved"]}


@dataclass
class Case:
    root: str = "synthetic_branch_root"
    lineage: str = "synthetic_branch_lineage"
    family: str = "chain-dead"
    split: str = "calibration"
    statement: str = "simulation only"


class Ranker:
    """Makes the tempting but dead r0 more attractive than recoverable r1."""
    def __call__(self, packet, candidates, *, depth, checks_remaining):
        scores = []
        for candidate in candidates:
            scores.append(10.0 if candidate == "apply r0" else
                          4.0 if candidate == "apply r1" else
                          1.0 if candidate == "apply h0" else 0.0)
        return [{"logits": torch.tensor(scores), "features": torch.zeros(len(scores), 1)}]


class BoundedNativeBacktrackTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_legal_dead_end_costed_and_then_recovered_with_native_stack(self):
        case = Case()
        repl = BranchLean()
        greedy = run_episode(
            repl, case, Ranker(), policy="frozen_greedy", effort_budget=16,
        )
        self.assertEqual(greedy["terminal_status"], "no_candidates")
        self.assertEqual(greedy["tactic_checks"], 1)
        self.assertEqual(greedy["backtracks"], 0)
        self.assertFalse(greedy["live_kernel"])
        self.assertEqual(greedy["accepted_script"], ["apply r0"])
        self.assertTrue(replay_episode(repl, case, greedy))

        dfs = run_episode(
            repl, case, Ranker(), policy="frozen_dfs", effort_budget=16,
        )
        self.assertEqual(dfs["schema"], SCHEMA)
        self.assertEqual(dfs["terminal_status"], "kernel_closed")
        self.assertTrue(dfs["live_kernel"])
        self.assertEqual(dfs["accepted_script"], ["apply r1", "apply h0"])
        self.assertEqual(dfs["tactic_checks"], 3)
        self.assertEqual(dfs["backtracks"], 1)
        self.assertEqual(dfs["effort_spent"], 4)
        self.assertEqual([e["kind"] for e in dfs["events"]],
                         ["TACTIC", "RECONSIDER", "TACTIC", "TACTIC"])
        self.assertEqual(dfs["events"][1]["abandoned_prefix"], ["apply r0"])
        self.assertEqual(dfs["events"][1]["script_after"], [])
        self.assertEqual(dfs["events"][1]["reason"], "exhausted_menu")
        self.assertTrue(replay_episode(repl, case, dfs))
        self.assertFalse(dfs["scientific_eligible"])

    def test_a_cycle_also_requires_and_pays_for_reconsideration(self):
        case = Case()
        repl = BranchLean(cycle=True)
        result = run_episode(
            repl, case, Ranker(), policy="frozen_dfs", effort_budget=16,
        )
        self.assertTrue(result["live_kernel"])
        self.assertEqual(result["events"][1]["kind"], "RECONSIDER")
        self.assertEqual(result["events"][1]["reason"], "cycle")
        self.assertEqual(result["backtracks"], 1)
        self.assertTrue(replay_episode(repl, case, result))

    def test_insufficient_budget_cannot_create_a_free_success(self):
        case = Case()
        ep = run_episode(
            BranchLean(), case, Ranker(), policy="frozen_dfs", effort_budget=3,
        )
        self.assertFalse(ep["live_kernel"])
        self.assertEqual(ep["effort_spent"], 3)
        self.assertEqual(ep["accepted_script"], ["apply r1"])
        self.assertTrue(replay_episode(BranchLean(), case, ep))
        with self.assertRaisesRegex(ValueError, "effort"):
            run_episode(BranchLean(), case, Ranker(), policy="frozen_dfs", effort_budget=0)

    def test_independent_transcript_replay_refuses_forged_backtrack(self):
        case = Case()
        ep = run_episode(
            BranchLean(), case, Ranker(), policy="frozen_dfs", effort_budget=16,
        )
        altered = copy.deepcopy(ep)
        altered["events"][1]["script_after"] = ["apply r0"]
        with self.assertRaisesRegex(ValueError, "different script"):
            replay_episode(BranchLean(), case, altered)
        altered = copy.deepcopy(ep)
        altered["events"][0]["tactic"] = "apply r1"
        with self.assertRaisesRegex(ValueError, "menu"):
            replay_episode(BranchLean(), case, altered)
        altered = copy.deepcopy(ep)
        altered["backtracks"] = 0
        with self.assertRaisesRegex(ValueError, "total"):
            replay_episode(BranchLean(), case, altered)

    def test_model_free_auto_and_root_seeded_uniform_are_replayable(self):
        for policy in ("auto_dfs", "uniform_dfs"):
            case = Case()
            ep = run_episode(
                BranchLean(), case, None, policy=policy,
                effort_budget=16, random_seed=17,
            )
            self.assertEqual(ep["model_calls"], 0)
            self.assertTrue(ep["live_kernel"])
            self.assertTrue(replay_episode(BranchLean(), case, ep))
            second = run_episode(
                BranchLean(), case, None, policy=policy,
                effort_budget=16, random_seed=17,
            )
            self.assertEqual(ep["events"], second["events"])


if __name__ == "__main__":
    unittest.main()

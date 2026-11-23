"""LSA-02J synthetic-only regressions for public branch labels and utility search.

FakeLean proves only the engineering wiring. It is never used as evidence for
the real native Lean corpus or the scientific proof-completion gate.
"""
from __future__ import annotations

import copy
import json
import unittest

import torch

from ppt_gym.branch_utility import BranchUtility, action_features, public_horn_costs, public_horn_order
from ppt_gym.branch_utility_source import action_cost, label_frame
from ppt_gym.branch_utility_train import _shuffled_targets
from ppt_gym.backtrack_search import run_episode, replay_episode
from ppt_gym.live_lean import state_input, candidate_menu
from ppt_gym.tests.test_backtrack_search import BranchLean, Case, Ranker


class LearnableViabilityRegressions(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.repl = BranchLean()
        state = self.repl.init("simulated")
        self.packet, _ = state_input(self.repl, state, (), 8)
        self.menu = list(candidate_menu(self.repl, state))

    def test_public_horn_fixedpoint_controls_dead_and_cyclic_rules(self):
        self.assertIn("apply r0", self.menu)
        self.assertIn("apply r1", self.menu)
        self.assertIn("apply r2", self.menu)
        costs = public_horn_costs(self.packet, self.menu)
        mapped = dict(zip(self.menu, costs))
        self.assertIsNone(mapped["apply r0"])
        self.assertEqual(mapped["apply r1"], 2)
        self.assertEqual(mapped["apply r2"], 3)
        self.assertEqual(self.menu[public_horn_order(self.packet, self.menu)], "apply r1")
        features = action_features(self.packet, self.menu)
        self.assertEqual(features.shape, (len(self.menu), 8))
        self.assertTrue(torch.isfinite(features).all())

    def test_lean_graph_training_labels_keep_dead_and_finite_costs_distinct(self):
        oracle = {
            "table": {
                "P1": {"apply r0": ["P3"],
                       "apply r1": ["P0"],
                       "apply r2": ["P1"]},
                "P0": {"apply h0": []},
                "P3": {},
            },
            "size": {"P0": 1, "P1": 2, "P3": None},
        }
        assert action_cost(oracle, json.loads(self.packet)["state"], "apply r0") is None
        costs = [
            action_cost(oracle, json.loads(self.packet)["state"], tac)
            for tac in self.menu
        ]
        self.assertEqual(costs, [None, 2, 3])
        frame = {
            "root": "root_train_only", "lineage": "lineage_train_only",
            "step": 0, "goal": self.packet, "public_state_sha256": "synthetic_hash",
            "candidates": self.menu, "checks_remaining": 8, "labels": [0,1,0],
        }
        tagged = label_frame(frame, oracle)
        self.assertEqual(tagged["minimum_whole_state_tactic_cost"], costs)
        self.assertEqual(tagged["finite_viability"], [0,1,1])
        self.assertEqual(tagged["bounded_viability"], [0,1,1])
        self.assertTrue(tagged["has_mixed_finite_actions"])
        self.assertTrue(tagged["source_only"])
        with self.assertRaisesRegex(ValueError, "contradict"):
            label_frame({**frame,"labels":[1,0,0]}, oracle)

    def test_sham_preserves_labels_counts_and_pairs_with_cost(self):
        row = {
            "root": "training_root", "step": 1,
            "viability": torch.tensor([1.,0.,1.,0.]),
            "cost": torch.tensor([0.125,1.,0.625,1.]),
        }
        row["cost_known"] = row["viability"] > 0.5
        v, c, mask = _shuffled_targets(row, 183)
        self.assertEqual(v.sum().item(), 2.0)
        self.assertEqual(len(v), len(row["viability"]))
        self.assertTrue(torch.equal(mask, v > 0.5))
        self.assertEqual(
            sorted(zip(v.tolist(),c.tolist())),
            sorted(zip(row["viability"].tolist(),row["cost"].tolist())),
        )
        v2,c2,m2 = _shuffled_targets(row, 183)
        self.assertTrue(torch.equal(v,v2))
        self.assertTrue(torch.equal(c,c2))

    def test_probe_readout_has_registered_shape_and_no_backbone_gradient(self):
        torch.manual_seed(33)
        head = BranchUtility(width=32)
        frozen = torch.randn(3,32)
        out = head(frozen, action_features(self.packet,self.menu))
        self.assertEqual(out.shape,(3,2))
        score = head.additive_scores(frozen, self.packet, self.menu)
        self.assertEqual(score.shape,(3,))
        self.assertTrue(torch.isfinite(score).all())

    def test_frozen_model_uses_ranker_without_oracle_or_free_reconsideration(self):
        class Pred:
            def __call__(self, packet, options, *, depth, checks_remaining):
                base = Ranker()(packet, options, depth=depth,
                                checks_remaining=checks_remaining)[-1]
                base["features"] = torch.zeros((len(options),32))
                return [base]
        class Boost:
            def additive_scores(self, features, packet, candidates):
                return torch.tensor([
                    20. if c=="apply r1" else 0. for c in candidates
                ])
        case=Case()
        naive = run_episode(BranchLean(), case, Pred(), policy="frozen_dfs")
        ranked = run_episode(BranchLean(), case, Pred(), policy="utility_dfs",
                             utility=Boost())
        self.assertEqual(naive["terminal_status"], "kernel_closed")
        self.assertEqual(ranked["terminal_status"], "kernel_closed")
        self.assertEqual(ranked["accepted_script"], ["apply r1","apply h0"])
        self.assertEqual(ranked["backtracks"], 0)
        self.assertLess(ranked["effort_spent"], naive["effort_spent"])
        self.assertTrue(replay_episode(BranchLean(),case,ranked))
        public = run_episode(BranchLean(),case,None,policy="public_horn_dfs")
        self.assertTrue(public["live_kernel"])
        self.assertEqual(public["accepted_script"], ["apply r1","apply h0"])
        self.assertEqual(public["model_calls"], 0)
        self.assertTrue(replay_episode(BranchLean(),case,public))
        with self.assertRaisesRegex(ValueError, "registered frozen model"):
            run_episode(BranchLean(),case,Pred(),policy="utility_dfs")
        with self.assertRaisesRegex(ValueError, "registered frozen model"):
            run_episode(BranchLean(),case,Pred(),policy="frozen_dfs",utility=Boost())

    def test_reference_solver_only_uses_public_types_not_oracle_metadata(self):
        packet = json.loads(self.packet)
        packet["private_oracle_label"] = "SHOULD_FAIL"
        with self.assertRaisesRegex(ValueError, "shape"):
            public_horn_costs(json.dumps(packet), self.menu)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

import torch

from ppt_gym.data import canonical
from ppt_gym.hybrid_rl.__main__ import reinforce
from ppt_gym.hybrid_rl.corpus import freeze, load
from ppt_gym.hybrid_rl.mathcan import MathCanReplay
from ppt_gym.hybrid_rl.model import ProofPolicy


class HybridProofPolicyTests(unittest.TestCase):
    def test_tied_bank_intervention_and_trm_parameter_matching(self):
        torch.random.default_generator.manual_seed(14)
        hybrid = ProofPolicy("hybrid", 32)
        trm = ProofPolicy("trm", 32)
        self.assertLess(abs(hybrid.parameter_count() / trm.parameter_count() - 1), 0.07)
        packet = canonical(self.packet())
        actions = ["apply r0", "apply r1"]
        normal, control, sham = [], [], []
        hybrid(packet, actions, 4, trace=normal)
        hybrid(packet, actions, 4, intervention=("zero", 2), trace=control)
        hybrid(packet, actions, 4, intervention=("sham", 2), trace=sham)
        torch.testing.assert_close(normal[0]["state"], control[0]["state"])
        self.assertEqual(float(control[1]["received_bank"].abs().sum()), 0)
        torch.testing.assert_close(normal[1]["received_bank"].norm(), sham[1]["received_bank"].norm())
        self.assertFalse(torch.allclose(normal[1]["state"], sham[1]["state"]))

    def test_actor_critic_updates_from_rollouts_without_gold_actions(self):
        model = ProofPolicy("hybrid", 32)
        optim = torch.optim.AdamW(model.parameters(), lr=0.001)
        before = model.cell.linear1.weight.detach().clone()
        episode = {"verified_closed": True, "within_cpu_budget": True, "tactic_checks": 3, "depth": 4,
                   "events": [{"kind": "tactic", "attempt": 1, "input": self.packet(),
                               "candidate_menu": ["apply r0", "apply r1"], "selected_index": 1}]}
        result = reinforce(model, [episode], optim)
        self.assertEqual(result["decisions"], 1)
        self.assertFalse(torch.equal(before, model.cell.linear1.weight))
        failure = reinforce(model, [{**episode, "verified_closed": False}], optim)
        self.assertLess(failure["mean_return"], 0)

    def test_frozen_sources_are_disjoint_and_include_bpd(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cohort.json"
            freeze(path, 2, 4, 3)
            cohort = load(path)
            self.assertTrue({c.cluster for c in cohort["train"]}.isdisjoint(c.cluster for c in cohort["dev"]))
            for split in ("train", "dev"):
                self.assertEqual({"bpd_lg02", "loopt_horn"}, {c.source for c in cohort[split]})

    def test_mathcan_replay_refuses_equal_goal_from_a_different_prefix(self):
        graph = MathCanReplay.__new__(MathCanReplay)
        root = ("MathCan/Test.lean", "test", ())
        goal = {"goals": ["⊢ True"], "definitions": []}
        graph.states = {root: {"intro": {"after": {**goal, "accepted": True}}}}
        graph.observations = {(*root[:2], ("wrong",)): goal}
        self.assertEqual(graph.step(root, "intro"), ("censored", None))
        next_prefix = (*root[:2], ("intro",))
        graph.observations[next_prefix] = goal
        self.assertEqual(graph.step(root, "intro"), ("accepted_nonterminal", next_prefix))

    @staticmethod
    def packet():
        return {"state": {"goals": [{"target": {"a": "P2"}, "hypotheses": [
            {"name": "r0", "type": {"imp": [{"a": "P0"}, {"a": "P2"}]}},
            {"name": "r1", "type": {"imp": [{"a": "P1"}, {"a": "P2"}]}}]}]},
            "failed_choices": [], "checks_remaining": 8}

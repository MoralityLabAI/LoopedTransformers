import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from configs.probe_ppt_hybrid_bank import EditedPolicy

from ppt_gym.data import canonical
from ppt_gym.hybrid_rl.__main__ import paired_credit, reinforce
from ppt_gym.hybrid_rl.corpus import freeze, load
from ppt_gym.hybrid_rl.graph_model import FACT_CHANNELS, GraphProofPolicy, ResidualFactTransport, visible_graph
from ppt_gym.hybrid_rl.mathcan import MathCanReplay
from ppt_gym.hybrid_rl.model import ProofPolicy


class HybridProofPolicyTests(unittest.TestCase):
    def test_paired_credit_cancels_shared_bias_and_learns_fact_response(self):
        class CounterfactualPolicy(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.tensor(0.7))
                self.response = nn.Parameter(torch.tensor(0.0))

            def forward(self, packet, actions, passes):
                sign = json.loads(packet)["sign"]
                return torch.stack((self.bias + sign * self.response, self.bias.new_zeros(()))), self.bias

        policy = CounterfactualPolicy()
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
        episodes = [
            {"verified_closed": passed, "within_cpu_budget": True, "tactic_checks": 8, "depth": 4,
             "root_proposal_probability": 0.5,
             "events": [{"input": {"sign": sign}, "candidate_menu": ["a", "b"], "selected_index": 0}]}
            for sign, passed in ((1, True), (-1, False))
        ]
        stats = paired_credit(policy, episodes, optimizer)
        self.assertEqual(stats["decisions"], 2)
        self.assertEqual(float(policy.bias.grad), 0.0)
        self.assertGreater(float(policy.response.detach()), 0.0)

    def test_fact_transport_keeps_signal_across_six_rules(self):
        transport = ResidualFactTransport()
        heads = torch.tensor([14, 14, *range(1, 7), *range(8, 14)])
        sources = torch.tensor([[7], [14], *[[n] for n in range(1, 7)], *[[n] for n in range(8, 14)]])
        first, second = torch.zeros(15, 1), torch.zeros(15, 1)
        first[0], second[7] = 1, 1
        torch.testing.assert_close(transport(first, sources, heads, 8), torch.zeros(len(heads)))
        with torch.no_grad():
            transport.source.weight.fill_(1)
            transport.action.weight.fill_(1 / FACT_CHANNELS)
        torch.testing.assert_close(transport(first, sources, heads, 5)[:2], torch.zeros(2))
        torch.testing.assert_close(transport(first, sources, heads, 8)[:2],
                                   transport(second, sources, heads, 8)[:2].flip(0))
        self.assertGreater(float(transport(first, sources, heads, 8)[0].detach()), 5.0)
        transport(first, sources, heads, 8)[0].backward()
        self.assertGreater(float(transport.retention.grad.abs().sum()), 0.0)

    def test_shared_public_graph_preserves_the_fact_omitted_by_lexical_rankers(self):
        actions = ["apply r0", "apply r1"]
        packets = []
        for known in ("P0", "P1"):
            packet = self.packet()
            packet["state"]["goals"][0]["hypotheses"].append({"name": "h0", "type": {"a": known}})
            packets.append(canonical(packet))
        graphs = [visible_graph(packet, actions) for packet in packets]
        self.assertEqual(graphs[0].actions, graphs[1].actions)
        self.assertEqual(graphs[0].features[0][0], 1.0)
        self.assertEqual(graphs[1].features[1][0], 1.0)
        lexical = ProofPolicy("hybrid", 32)
        torch.testing.assert_close(lexical(packets[0], actions, 4)[0], lexical(packets[1], actions, 4)[0])
        for arch, passes in (("hybrid", 4), ("trm", 1)):
            model = GraphProofPolicy(arch, 32)
            a, value = model(packets[0], actions, passes)
            b, _ = model(packets[1], actions, passes)
            self.assertFalse(torch.allclose(a, b))
            (a[0] + value).backward()
            self.assertIsNotNone(model.graph.node.weight.grad)

    def test_bank_probe_removes_only_values_and_matches_control_dose(self):
        class Recall(nn.Module):
            def forward(self, query, keys, values):
                return query, keys, values

        class Policy(nn.Module):
            architecture = "hybrid"

            def __init__(self):
                super().__init__()
                self.recall = Recall()

            def macs(self, candidates, passes):
                return 0

            def forward(self, bank):
                return [self.recall(bank + i, bank - i, bank * (i + 1)) for i in range(3)]

        bank = torch.arange(1.0, 13.0).reshape(1, 3, 4)
        edit = EditedPolicy(Policy())
        clean = edit(bank)
        with edit.edit("ablate-pass-2"):
            cut = edit(bank)
        with edit.edit("control-pass-2"):
            sham = edit(bank)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(clean[0][:2], cut[0][:2])))
        self.assertEqual(float(cut[0][2].abs().sum()), 0)
        torch.testing.assert_close((clean[0][2] - cut[0][2]).norm(), (clean[0][2] - sham[0][2]).norm())
        self.assertTrue(torch.equal(clean[1][2], cut[1][2]))
        self.assertTrue(torch.equal(clean[1][2], sham[1][2]))

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

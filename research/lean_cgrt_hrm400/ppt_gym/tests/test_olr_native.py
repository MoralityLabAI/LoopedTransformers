"""OLR-01B public-data and paired-structural preflight tests.

These tests create synthetic Lean-shaped JSON for protocol checks only.
Native validity and certificates are checked by the separate CI Lean job.
"""
import unittest

from lcgrt import fragment as F
from ppt_gym.olr_native import (
    atom, horn_type, pair_worlds, parse_public, rank, rule_swap,
)
from ppt_gym.obligation_ledger import mean_premise_cue, propagate


def public_packet(theorem: F.Theorem):
    hyps = [{"name": f"h{i}", "type": {"a": f"P{p}"}}
            for i, p in enumerate(theorem.facts)]
    for i, (premises, head) in enumerate(theorem.rules):
        ty = {"a": f"P{head}"}
        for p in reversed(premises):
            ty = {"imp": [{"a": f"P{p}"}, ty]}
        hyps.append({"name": f"r{i}", "type": ty})
    return {"goals": [{"target": {"a": f"P{theorem.goal[0]}"},
                       "hypotheses": hyps}]}


class OLREvidenceInterfaceTests(unittest.TestCase):
    def test_pair_reverses_witness_with_same_cue_and_menu(self):
        pair = pair_worlds(20261123, 2)
        self.assertEqual(pair[0].lineage, pair[1].lineage)
        self.assertNotEqual(F.statement(pair[0].theorem), F.statement(pair[1].theorem))
        self.assertEqual(pair[0].witness, "apply r0")
        self.assertEqual(pair[1].witness, "apply r1")
        for w in pair:
            visible = public_packet(w.theorem)
            parsed = parse_public(visible)
            self.assertEqual(len(parsed.rules), 4)
            self.assertEqual(mean_premise_cue(parsed.graph, parsed.target), 0.5)
            trace = propagate(parsed.graph, fact_cost=1.)
            self.assertEqual(trace.costs[parsed.target], 3.)
            self.assertEqual(rank(visible, ("apply r0", "apply r1"), "ledger",
                                  at_root=True)[0], w.witness)
            self.assertNotEqual(rank(visible, ("apply r0", "apply r1"), "edge_swap",
                                     at_root=True)[0], w.witness)

    def test_mean_only_cannot_separate_two_worlds(self):
        p = pair_worlds(20261124, 3)
        cues = [rank(public_packet(w.theorem), ("apply r0", "apply r1"),
                     "mean_only", at_root=True) for w in p]
        self.assertEqual(cues[0], cues[1])
        self.assertEqual(cues[0][1]["pooled_cue"], 0.5)

    def test_eight_variants_get_structurally_distinct_roots(self):
        hashes = [pair_worlds(20261123 + i, i)[0].lineage for i in range(8)]
        self.assertEqual(len(set(hashes)), 8)

    def test_parser_uses_active_goal_and_preserves_pending(self):
        case = pair_worlds(42, 1)[0]
        shown = public_packet(case.theorem)
        shown["goals"].append({"target": {"a": "P9999"}, "hypotheses": []})
        parsed = parse_public(shown)
        self.assertEqual(parsed.target, case.theorem.goal[0])
        self.assertEqual(len(parsed.rules), 3)

    def test_malformed_and_raw_edges_are_never_promoted(self):
        shown = {"goals": [{"target": {"a": "P2"}, "hypotheses": [
            {"name": "h0", "type": {"a": "P0"}},
            {"name": "r0", "type": {"raw": "decide condition"}},
            {"name": "r1", "type": {"imp": [{"a": "P0"}, {"a": "P2"}]}},
        ]}]}
        parsed = parse_public(shown)
        self.assertEqual(parsed.unparsed, ("r0",))
        self.assertNotIn("r0", parsed.rules)
        self.assertEqual(rank(shown, ("apply r0", "apply r1"),
                              "ledger", at_root=True)[0], "apply r1")

    def test_bad_atoms_and_malformed_implications(self):
        self.assertIsNone(atom({"a": "Nat.add"}))
        self.assertIsNone(horn_type({"imp": ["P0", {"a": "P1"}]}))
        self.assertIsNone(horn_type({"imp": [{"a": "P0"}]}))
        with self.assertRaises(ValueError):
            parse_public({"goals": []})

    def test_forced_root_choice_uses_menu_only(self):
        shown = public_packet(pair_worlds(77, 0)[0].theorem)
        self.assertEqual(rank(shown, ("apply r0", "apply r1"),
                              "force_r1", at_root=True)[0], "apply r1")
        with self.assertRaises(ValueError):
            rank(shown, ("apply r0",), "force_r1", at_root=True)

    def test_rule_swap_preserves_graph_size_and_fact_set(self):
        p = parse_public(public_packet(pair_worlds(7, 2)[0].theorem))
        q = rule_swap(p)
        self.assertEqual(p.graph.nodes, q.graph.nodes)
        self.assertEqual(p.graph.facts, q.graph.facts)
        self.assertEqual(p.rules["r0"].premises, q.rules["r1"].premises)
        self.assertEqual(p.rules["r1"].premises, q.rules["r0"].premises)


if __name__ == "__main__":
    unittest.main()

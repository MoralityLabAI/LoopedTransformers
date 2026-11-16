"""AF-24 pure-source preflight, before empirical baseline scoring."""
from __future__ import annotations
import unittest

from experiments.activation_funnel_af24 import symbolic as ctl
from experiments.activation_funnel_af23 import branching_gym as gym


class SymbolicControls(unittest.TestCase):
    def setUp(self):
        # Public graph with a productive 0 -> 2 -> goal route and dead branch.
        self.graph = ctl.PublicHorn(
            rules=((0, (0,), 1), (1, (1,), 3),
                   (2, (0,), 2), (3, (2,), 4)),
            initial=frozenset((0,)), goal=4,
        )

    def test_public_graph_only(self):
        for arm in (*ctl.CORE, *ctl.RANDOM, "fixed"):
            row = ctl.choose(self.graph, self.graph.initial, arm, 17, "synthetic", 0)
            self.assertIn(row["action"], (0, 2))
            self.assertGreaterEqual(row["rule_examinations"], 0)
            self.assertGreaterEqual(row["wall_ns"], 0)
        for cap in (4, 16, 64):
            a = ctl.choose(self.graph, self.graph.initial, f"bfs_cap{cap}")
            self.assertLessEqual(a["expanded"], cap)

    def test_closure_picks_valid_proof(self):
        start = self.graph.initial
        first = ctl.choose(self.graph, start, "closure_direct")["action"]
        self.assertEqual(first, 2)
        after = self.graph.fire(start, first)
        second = ctl.choose(self.graph, after, "closure_direct")["action"]
        self.assertEqual(second, 3)
        self.assertIn(self.graph.goal, self.graph.fire(after, second))

    def test_bfs16_finds_path(self):
        first = ctl.choose(self.graph, self.graph.initial, "bfs_cap16")
        self.assertEqual(first["action"], 2)
        self.assertLessEqual(first["expanded"], 16)

    def test_goal_impossible_closure_stops(self):
        graph = ctl.PublicHorn(rules=((0, (1,), 2), (1, (2,), 3)),
                               initial=frozenset((0,)), goal=3)
        row = ctl.choose(graph, graph.initial, "closure_direct")
        self.assertIsNone(row["action"])
        self.assertTrue(row["search_exhausted"])

    def test_bfs_exhaustion_stops(self):
        graph = ctl.PublicHorn(rules=((0, (0,), 1),),
                               initial=frozenset((0,)), goal=4)
        row = ctl.choose(graph, graph.initial, "bfs_cap16")
        self.assertIsNone(row["action"])
        self.assertEqual(row["expanded"], 2)

    def test_random_reproducible(self):
        a = ctl.choose(self.graph, self.graph.initial, "ancestor_random_1",
                       seed=103, case_id="x", step=0)
        b = ctl.choose(self.graph, self.graph.initial, "ancestor_random_1",
                       seed=103, case_id="x", step=0)
        self.assertEqual(a["action"], b["action"])

    def test_disallow_oracle_at_inference(self):
        with self.assertRaises(ValueError):
            ctl.choose(self.graph, self.graph.initial, "shortest")
        with self.assertRaises(ValueError):
            ctl.choose(self.graph, self.graph.initial, "not_registered")

    def test_host_invariance_under_action(self):
        c = gym.cases_for("held_depth4")[0]
        graph = ctl.PublicHorn.from_public_features(gym.public_features(c))
        start = graph.initial
        for action, _ in graph.enabled(start):
            after = graph.fire(start, action)
            self.assertGreaterEqual(len(after), len(start))
            self.assertEqual(len(after), len(start) + 1)
        self.assertEqual(len(graph.rules), gym.RULES)

    def test_no_oracle_dependency(self):
        """Intervention never passes a Case or hidden label to the controller."""
        graph = ctl.PublicHorn(
            rules=((0, (0,), 1), (1, (1,), 2)),
            initial=frozenset((0,)), goal=2
        )
        # A non-oracular dynamic-programming controller can derive the path.
        self.assertEqual(ctl.choose(graph, graph.initial,
                                    "closure_direct")["action"], 0)


if __name__ == "__main__":
    unittest.main()

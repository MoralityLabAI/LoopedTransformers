"""AF-23 preregistered static structural and numerical preflight tests."""
import copy
import unittest

from experiments.activation_funnel_af23 import branching_gym as g
from experiments.activation_funnel_af23 import stop_scoring as sc


def synthetic_plan():
    root = [0, 1, 2, 3]
    left = [0, 1, 2, 3, 6]
    right = [0, 1, 2, 3, 13]
    goal = [0, 1, 2, 3, 6, 13, 14]
    return {
        "root_raw": [0.0] * (g.RULES + 1),
        "nodes": [
            {"state": root, "edges": {"0": left, "1": right},
             "value0": 7.0, "stop_cost": 8.0,
             "reach_logit": 4.0, "distance_logit": 1.0},
            {"state": left, "edges": {"2": goal},
             "value0": 5.0, "stop_cost": 1.0,
             "reach_logit": -3.0, "distance_logit": 1.0},
            {"state": right, "edges": {"3": goal},
             "value0": 5.0, "stop_cost": .5,
             "reach_logit": -5.0, "distance_logit": 1.0},
            {"state": goal, "edges": {},
             "value0": 0.0, "stop_cost": 0.0,
             "reach_logit": -50.0, "distance_logit": 7.0},
        ],
    }


class GraphTests(unittest.TestCase):
    def test_split_sizes_and_facts(self):
        self.assertEqual([len(g.cases_for(s)) for s in g.SPLITS],
                         [768, 384, 32, 192, 192])
        for split, truth in [("held_depth4", 80), ("held_depth5", 80)]:
            cs = g.cases_for(split)
            self.assertEqual(sum(c.provable_design for c in cs), truth)
            self.assertEqual(len({c.id for c in cs}), len(cs))

    def test_exact_depth_and_branching(self):
        for split, depth in [("held_depth4", 4), ("held_depth5", 5)]:
            for c in g.cases_for(split)[:40]:
                self.assertTrue(c.provable_design)
                self.assertEqual(g.shortest_steps(c), depth)
                self.assertEqual(g.independent_bfs(c), depth)
                legal = g.enabled(c, g.initial_facts(c))
                optimal = g.optimal_actions(c, g.initial_facts(c))
                self.assertGreaterEqual(len(legal), 5)
                self.assertLessEqual(len(optimal) / len(legal), .5)

    def test_unprovable_control(self):
        for split in ("held_depth4", "held_depth5"):
            for c in g.cases_for(split):
                if c.provable_design:
                    continue
                self.assertIsNone(g.shortest_steps(c))
                self.assertIsNone(g.independent_bfs(c))
                self.assertFalse(g.symbolic_rollout(c, "shortest")["solved"])
                self.assertTrue(g.symbolic_rollout(c, "shortest")["initial_stop"])

    def test_closure_invariance_and_illegal_action(self):
        c = g.cases_for("held_depth4")[0]
        s = g.initial_facts(c)
        before = g.GOAL in g.closure(c, s)
        for a in g.enabled(c, s):
            future, proof = g.sound_fire(c, s, a)
            self.assertIsNotNone(proof)
            self.assertEqual(before, g.GOAL in g.closure(c, future))
        with self.assertRaises(ValueError):
            g.sound_fire(c, s, -1)

    def test_oracle_terminates_with_shortest_distance(self):
        for split in ("held_depth4", "held_depth5"):
            for c in g.cases_for(split)[:80]:
                self.assertEqual(
                    g.symbolic_rollout(c, "shortest")["solved"], c.provable_design
                )

    def test_manifest_uses_registered_structural_gate(self):
        receipt = g.manifest()
        self.assertTrue(receipt["no_cross_split_public_feature_overlap"])
        self.assertEqual(receipt["splits"]["held_depth5"]["designed_positive"], 80)
        self.assertGreaterEqual(
            len(receipt["manifest_sha256"]), 64
        )


class ScorerTests(unittest.TestCase):
    def test_original_vi2_and_ablations(self):
        p = synthetic_plan()
        scores = [-1e4] * (g.STOP + 1)
        scores[0] = -2.0
        scores[1] = -1.5
        scores[g.STOP] = -8.0
        result = sc.compare_original_vi2(p, p["nodes"][0]["state"], scores)
        self.assertEqual(result["chosen"], 1)
        self.assertEqual(result["interior_stop_bind"], 2)
        no_stop = sc.score(p, p["nodes"][0]["state"], "no_interior_stop")
        self.assertEqual(no_stop["chosen"], 0)
        tied = sc.score(p, p["nodes"][0]["state"], "root_tied_reach")
        self.assertEqual(tied["chosen"], 0)
        both = sc.score(p, p["nodes"][0]["state"], "root_tied_no_interior_stop")
        self.assertEqual(both["chosen"], 0)

    def test_goal_boundary_clamp(self):
        p = synthetic_plan()
        tied = sc.score(p, p["nodes"][0]["state"], "root_tied_reach")
        self.assertEqual(tied["scores"]["0"], -2.0)

    def test_tamper_rejection(self):
        p = synthetic_plan()
        p["nodes"][0]["edges"]["0"] = [14, 99]
        with self.assertRaises(ValueError):
            sc.score(p, p["nodes"][0]["state"], "root_tied_reach")
        p = synthetic_plan()
        with self.assertRaises(ValueError):
            sc.compare_original_vi2(
                p, p["nodes"][0]["state"], [-2., -100.] + [-1e4] * 12 + [-8.]
            )

    def test_no_legal_successor_has_finite_root_stop(self):
        root = [0, 1, 2, 3]
        p = {"root_raw": [0] * 15,
             "nodes": [{"state": root, "edges": {}, "value0": 7,
                        "stop_cost": 8, "reach_logit": 0,
                        "distance_logit": 0}]}
        for arm in (*sc.ARMS, *sc.AUDIT_ONLY):
            self.assertEqual(sc.score(p, root, arm)["chosen"], sc.STOP)


if __name__ == "__main__":
    unittest.main()

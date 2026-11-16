"""AF-24B source-only gym contract; never opens a neural checkpoint."""
import unittest

from experiments.activation_funnel_af24b import gym
from experiments.activation_funnel_af24 import symbolic


class NewGrammarContracts(unittest.TestCase):
    def test_registered_split_counts(self):
        for split, depth in zip(gym.SPLITS, gym.DEPTHS):
            cases = gym.cases_for(split)
            self.assertEqual(len(cases), 192)
            self.assertEqual(sum(c.designed_positive for c in cases), 80)
            self.assertEqual({c.depth for c in cases}, {depth})
            self.assertEqual(len({c.id for c in cases}), 192)

    def test_new_goal_topologies_and_root_branching(self):
        for split in gym.SPLITS:
            for family in gym.FAMILIES:
                cases = [c for c in gym.cases_for(split) if c.family == family]
                self.assertEqual(len(cases), 64)
                self.assertEqual(
                    len([c for c in cases if c.designed_positive]),
                    gym.POSITIVE_COUNT[family]
                )
                for c in cases[:min(8, gym.POSITIVE_COUNT[family])]:
                    rs = gym.logical_rules(c)
                    self.assertEqual(sum(h == gym.GOAL for _, h in rs), 2)
                    self.assertLessEqual(len(rs), gym.RULES)
                    legal = gym.enabled(c, gym.initial_facts(c))
                    optimal = gym.optimal_actions(c, gym.initial_facts(c))
                    self.assertGreaterEqual(len(legal), 2)
                    self.assertLessEqual(len(optimal)/len(legal), .75)
                    self.assertEqual(gym.shortest_steps(c), c.depth)

    def test_or_and_shared_counterexamples(self):
        for split in gym.SPLITS:
            for family in gym.FAMILIES:
                c = next(c for c in gym.cases_for(split)
                         if c.family == family and c.designed_positive)
                self.assertEqual(gym.independent_bfs(c), c.depth)
                if family == "or_goal_competition":
                    self.assertEqual(
                        sorted(len(pre) for pre, h in gym.logical_rules(c)
                               if h == gym.GOAL), [1,1])
                else:
                    self.assertEqual(
                        sorted(len(pre) for pre, h in gym.logical_rules(c)
                               if h == gym.GOAL), [1,2])

    def test_provable_horn_closure_preserved_by_any_rule(self):
        for split in gym.SPLITS:
            for c in gym.cases_for(split)[:28]:
                known = gym.initial_facts(c)
                before = gym.GOAL in gym.closure(c, known)
                for a in gym.enabled(c, known):
                    after, proof = gym.sound_fire(c, known, a)
                    self.assertIsNotNone(proof)
                    self.assertEqual(before, gym.GOAL in gym.closure(c, after))
                with self.assertRaises(ValueError):
                    gym.sound_fire(c, known, -1)

    def test_no_supervision_in_symbolic_graph(self):
        c = gym.cases_for("held_depth5")[0]
        f = gym.public_features(c)
        self.assertNotIn("optimal_steps", f)
        self.assertNotIn("goal_truth", f)
        graph = symbolic.PublicHorn.from_public_features(f)
        self.assertEqual(len(graph.rules), len(gym.logical_rules(c)))
        self.assertEqual(
            sorted(graph.initial),
            sorted(gym.physical_maps(c)[0][a] for a in gym.initial_facts(c))
        )

    def test_independent_source_manifest(self):
        receipt = gym.manifest()
        self.assertTrue(receipt["source_only"])
        self.assertFalse(receipt["model_checkpoints_opened"])
        self.assertEqual(receipt["original_af23_goal_rule_count"], 1)
        self.assertEqual(receipt["new_goal_rule_count"], 2)
        self.assertEqual(len(receipt["manifest_sha256"]), 64)
        for split in gym.SPLITS:
            self.assertLessEqual(
                receipt["symbolic_pretest_closure_solved"][split], 60
            )


if __name__ == "__main__":
    unittest.main()

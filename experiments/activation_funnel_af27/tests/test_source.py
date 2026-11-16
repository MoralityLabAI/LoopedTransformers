"""AF-27 source-only graph and inference-policy isolation preflight."""
import unittest
from experiments.activation_funnel_af27 import gym
from experiments.activation_funnel_af24 import symbolic


class NewGrammarContracts(unittest.TestCase):
    def test_case_sizes_root_fact_cardinality(self):
        for split in gym.SPLITS:
            cs=gym.cases_for(split)
            self.assertEqual(len(cs),192)
            self.assertEqual(sum(c.designed_positive for c in cs),80)
            self.assertEqual(len({c.id for c in cs}),192)
            for c in cs:
                self.assertEqual(len(gym.initial_facts(c)),4)

    def test_independent_proof_bfs_and_unprovable_roots(self):
        for split in gym.SPLITS:
            for c in gym.cases_for(split):
                d=gym.shortest_steps(c)
                self.assertEqual(d,gym.independent_bfs(c))
                self.assertEqual(d is not None,c.designed_positive)
                self.assertEqual(gym.GOAL in gym.closure(c),c.designed_positive)
                if c.designed_positive:
                    self.assertEqual(d,c.depth)

    def test_goal_rules_are_new_family_shapes(self):
        for split in gym.SPLITS:
            cases=gym.cases_for(split)
            for c in cases:
                rs=gym.logical_rules(c)
                self.assertEqual(sum(h==gym.GOAL for _,h in rs),2)
                self.assertLessEqual(len(rs),gym.RULES)
                self.assertLessEqual(max(len(pre) for pre,h in rs),2)
                self.assertEqual(len(gym.public_features(c)["rules_physical"]),len(rs))

    def test_public_input_excludes_oracle_label(self):
        for split in gym.SPLITS:
            for family in gym.FAMILIES:
                c=next(c for c in gym.cases_for(split)
                       if c.family==family and c.designed_positive)
                public=gym.public_features(c)
                for forbidden in ("oracle","shortest","goal_truth","optimal_steps"):
                    self.assertNotIn(forbidden,public)
                graph=symbolic.PublicHorn.from_public_features(public)
                self.assertEqual(len(graph.initial),4)
                self.assertEqual(len(graph.rules),len(gym.logical_rules(c)))

    def test_horn_positivity_and_sound_transitions(self):
        for split in gym.SPLITS:
            for c in gym.cases_for(split)[:48]:
                known=gym.initial_facts(c)
                before=gym.GOAL in gym.closure(c,known)
                for action in gym.enabled(c,known):
                    after,evidence=gym.sound_fire(c,known,action)
                    self.assertIsNotNone(evidence)
                    self.assertEqual(len(after),len(known)+1)
                    self.assertEqual(before,gym.GOAL in gym.closure(c,after))
                with self.assertRaises(ValueError):
                    gym.sound_fire(c,known,-1)

    def test_future_count_preflight(self):
        manifest=gym.manifest()
        self.assertTrue(manifest["source_only"])
        self.assertFalse(manifest["original_model_opened"])
        self.assertFalse(manifest["original_parent_public_features_overlap"])
        for split in gym.SPLITS:
            s=manifest["splits"][split]
            self.assertEqual((s["cases"],s["provable"],s["unprovable"]), (192,80,112))
            self.assertEqual(s["root_fact_count"],4)
            for fam in gym.FAMILIES:
                self.assertEqual(s["family_statistics"][fam]["positive_count"],40)
                self.assertGreaterEqual(
                    s["family_statistics"][fam]["root_suboptimal_positive"],20
                )


if __name__=="__main__":
    unittest.main()

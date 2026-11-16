"""AF-25 stage-0 pre-data source-only graph and symbolic-adversary checks."""
import unittest
from experiments.activation_funnel_af25 import gym
from experiments.activation_funnel_af24 import symbolic


class AdversarialHornContracts(unittest.TestCase):
    def test_every_case_has_four_initial_facts(self):
        for split in gym.SPLITS:
            cases=gym.cases_for(split)
            self.assertEqual(len(cases),192)
            self.assertEqual(sum(c.designed_positive for c in cases),80)
            self.assertEqual(len({c.id for c in cases}),192)
            for c in cases:
                self.assertEqual(len(gym.initial_facts(c)),4)

    def test_exact_host_independent_bitset_oracle(self):
        for split in gym.SPLITS:
            cases=gym.cases_for(split)
            for c in cases:
                d=gym.shortest_steps(c)
                self.assertEqual(d,gym.independent_bitset_bfs(c))
                self.assertEqual(d is not None,c.designed_positive)
                if c.designed_positive:
                    self.assertEqual(d,c.depth)
                self.assertEqual(gym.GOAL in gym.closure(c),d is not None)

    def test_true_distinct_branch_choices_and_goal_arity(self):
        for split in gym.SPLITS:
            for c in gym.cases_for(split):
                if not c.designed_positive:
                    continue
                legal=gym.enabled(c,gym.initial_facts(c))
                successors={gym.sound_fire(c,gym.initial_facts(c),a)[0]
                            for a in legal}
                self.assertGreaterEqual(len(successors),4)
                self.assertGreaterEqual(len(legal),4)
                self.assertEqual(
                    sum(h==gym.GOAL for _,h in gym.logical_rules(c)),2)
                self.assertLessEqual(len(gym.logical_rules(c)),gym.RULES)

    def test_unprovable_swap_preserves_four_facts(self):
        for split in gym.SPLITS:
            for c in gym.cases_for(split):
                if c.designed_positive:
                    self.assertEqual(gym.initial_facts(c),frozenset((0,1,2,3)))
                    continue
                facts=gym.initial_facts(c)
                self.assertNotIn(2,facts)
                self.assertTrue({0,1,3}.issubset(facts))
                self.assertEqual(gym.independent_bitset_bfs(c),None)

    def test_shared_cost_and_false_ancestors(self):
        for split in gym.SPLITS:
            for c in gym.cases_for(split):
                if c.family!="decoy_and_shared_overcount" or not c.designed_positive:
                    continue
                rs=gym.logical_rules(c)
                incoming=sorted(len(pre) for pre,h in rs if h==gym.GOAL)
                self.assertEqual(incoming,[2,2])
                root=gym.initial_facts(c)
                relevant=gym.goal_ancestor_atoms(c)
                optimal=set(gym.optimal_actions(c,root))
                misleading=[
                    a for a in gym.enabled(c,root)
                    if gym.physical_rules(c)[a][1] in relevant and a not in optimal
                ]
                self.assertGreaterEqual(len(misleading),2)

    def test_public_features_admit_original_symbolic_controller(self):
        for split in gym.SPLITS:
            c=gym.cases_for(split)[0]
            f=gym.public_features(c)
            self.assertNotIn("goal_truth",f)
            self.assertNotIn("optimal_steps",f)
            public=symbolic.PublicHorn.from_public_features(f)
            self.assertEqual(len(public.rules),len(gym.logical_rules(c)))
            self.assertEqual(len(public.initial),4)
            first=symbolic.choose(public,public.initial,"ancestor_greedy",17,c.id,0)
            self.assertIn(first["action"],dict(public.enabled(public.initial)))

    def test_registered_structural_gate_before_any_model(self):
        manifest=gym.manifest()
        self.assertTrue(manifest["structural_gate_pass"])
        self.assertFalse(manifest["original_model_checkpoint_opened"])
        self.assertTrue(manifest["no_cross_parent_or_internal_public_feature_leakage"])
        self.assertEqual(len(manifest["manifest_sha256"]),64)
        for split in gym.SPLITS:
            row=manifest["splits"][split]
            self.assertEqual((row["cases"],row["provable"],row["unprovable"]),
                             (192,80,112))
            self.assertEqual(row["initial_facts_per_case"],4)
            self.assertEqual(row["oracle_positive_success"],80)
            for arm in gym.CHEAP:
                self.assertGreaterEqual(row["source_only_positive_success"][arm],20)
                self.assertLessEqual(row["source_only_positive_success"][arm],60)
                self.assertLessEqual(row["hard_family_positive_success"][arm],24)


if __name__=="__main__":
    unittest.main()

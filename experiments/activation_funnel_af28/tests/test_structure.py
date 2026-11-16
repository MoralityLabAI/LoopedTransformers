"""AF-28 pre-training, pre-evaluation source-only structural requirements."""
import unittest
from experiments.activation_funnel_af28 import gym
from experiments.activation_funnel_af24 import symbolic


class GrammarRegistration(unittest.TestCase):
    def test_train_dev_held_layout_disjoint(self):
        layouts=[]
        for split in gym.SPLITS:
            cs=gym.cases_for(split)
            self.assertEqual(len(cs),192)
            self.assertEqual(sum(c.designed_positive for c in cs),80)
            layouts.extend(c.layout for c in cs)
            self.assertEqual(len({c.id for c in cs}),192)
        self.assertEqual(len(set(layouts)),16)

    def test_four_known_facts_and_exact_truth(self):
        for split in gym.SPLITS:
            for c in gym.cases_for(split):
                self.assertEqual(len(gym.initial_facts(c)),4)
                d=gym.shortest_steps(c)
                self.assertEqual(d,gym.independent_bfs(c))
                self.assertEqual(gym.GOAL in gym.closure(c),d is not None)
                self.assertEqual(d is not None,c.designed_positive)
                if c.designed_positive:
                    self.assertEqual(d,c.depth)
                else:
                    self.assertNotIn(2,gym.initial_facts(c))

    def test_alternate_goal_guards(self):
        for split in gym.SPLITS:
            for fam in gym.TEMPLATES[split]:
                c=next(c for c in gym.cases_for(split) if c.family==fam)
                rules=gym.logical_rules(c)
                self.assertLessEqual(len(rules),14)
                goals=[p for p,h in rules if h==gym.GOAL]
                self.assertEqual(len(goals),2)
                self.assertIn(2,goals[-1])

    def test_source_hidden_supervision_absent(self):
        for split in gym.SPLITS:
            for fam in gym.TEMPLATES[split]:
                c=next(c for c in gym.cases_for(split) if c.family==fam)
                p=gym.public_features(c)
                for x in ("goal_truth","optimal_steps","case_id","family"):
                    self.assertNotIn(x,p)
                graph=symbolic.PublicHorn.from_public_features(p)
                self.assertEqual(graph.goal,gym.physical_maps(c)[0][gym.GOAL])
                self.assertEqual(len(graph.initial),4)
                self.assertEqual(len(graph.rules),len(gym.logical_rules(c)))

    def test_training_state_schedule_source_only(self):
        for split in ("train","dev"):
            for c in gym.cases_for(split):
                states=gym.training_states(c)
                self.assertTrue(1<=len(states)<=3)
                self.assertEqual(states[0],gym.initial_facts(c))
                for a,b in zip(states,states[1:]):
                    self.assertGreater(len(b),len(a))
                    self.assertTrue(a.issubset(b))
                for s in states:
                    self.assertNotIn(gym.GOAL,s)

    def test_independent_manifest_and_topology(self):
        m=gym.manifest()
        self.assertTrue(m["source_only"])
        self.assertFalse(m["model_checkpoint_opened"])
        self.assertTrue(m["grammar_disjoint_logical_incidence"])
        self.assertTrue(m["public_overlap_with_AF25_AF27"] is False)
        self.assertEqual(len(m["manifest_sha256"]),64)
        for split in gym.SPLITS:
            self.assertEqual(m["split_receipts"][split]["positive"],80)
            self.assertEqual(m["split_receipts"][split]["negative"],112)
            self.assertEqual(m["split_receipts"][split]["four_initial_facts"],True)


if __name__=="__main__":
    unittest.main()

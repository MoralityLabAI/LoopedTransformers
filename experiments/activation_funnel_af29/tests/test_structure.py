"""AF-29 proof-pair structural tests independent of original model weights."""
import unittest
from experiments.activation_funnel_af29 import gym
from experiments.activation_funnel_af24 import symbolic


class PairedReachability(unittest.TestCase):
    def test_every_split_balanced_matched(self):
        for split in gym.SPLITS:
            cs=gym.cases_for(split)
            self.assertEqual(len(cs),192)
            self.assertEqual(sum(c.designed_positive for c in cs),96)
            self.assertEqual(len({c.id for c in cs}),192)
            self.assertEqual(len({c.pair_id for c in cs}),96)
            self.assertTrue(all(gym.initial_facts(c)==frozenset((0,1,2,3))
                                for c in cs))

    def test_every_pair_matches_local_structure_and_global_degree(self):
        for split in gym.SPLITS:
            for family in gym.FAMILIES[split]:
                positives=[c for c in gym.cases_for(split)
                           if c.family==family and c.reachable]
                for pos in positives:
                    positive,negative=gym.pair_mates(pos)
                    self.assertEqual(positive,pos)
                    record=gym.pair_structural_record(positive,negative)
                    self.assertEqual(len(record["root_enabled_physical"]),
                                     len(gym.enabled(negative,gym.SOURCE_ROOTS)))
                    self.assertEqual(record["positive_shortest"],pos.depth)
                    self.assertIsNone(record["negative_shortest"])
                    self.assertEqual(gym.radius2_anchor_signature(positive),
                                     gym.radius2_anchor_signature(negative))
                    self.assertEqual(gym.node_degrees(positive),
                                     gym.node_degrees(negative))
                    self.assertEqual(gym.physical_maps(positive),
                                     gym.physical_maps(negative))

    def test_independent_host_and_bitset_bfs(self):
        for split in gym.SPLITS:
            for c in gym.cases_for(split):
                d=gym.shortest_steps(c)
                self.assertEqual(d,gym.independent_bfs(c))
                self.assertEqual(d is not None,c.reachable)
                self.assertEqual(gym.GOAL in gym.closure(c),c.reachable)
                if c.reachable:
                    self.assertEqual(d,c.depth)
                self.assertGreaterEqual(len(gym.enabled(c,gym.SOURCE_ROOTS)),2)

    def test_true_goal_reachability_invariant_along_all_legal_steps(self):
        for split in gym.SPLITS:
            for c in gym.cases_for(split)[:40]:
                s=gym.initial_facts(c)
                truth=gym.GOAL in gym.closure(c,s)
                for a in gym.enabled(c,s):
                    after,proof=gym.sound_fire(c,s,a)
                    self.assertIsNotNone(proof)
                    self.assertEqual(len(after),len(s)+1)
                    self.assertEqual(gym.GOAL in gym.closure(c,after),truth)
                with self.assertRaises(ValueError):
                    gym.sound_fire(c,s,-1)

    def test_only_two_rules_change_and_root_actions_agree(self):
        for split in gym.SPLITS:
            for family in gym.FAMILIES[split]:
                c=next(c for c in gym.cases_for(split)
                       if c.family==family and c.reachable)
                pos,neg=gym.pair_mates(c)
                posrules=gym.physical_rules(pos)
                negrules=gym.physical_rules(neg)
                changed=[i for i in posrules if posrules[i]!=negrules[i]]
                self.assertEqual(len(changed),2)
                self.assertEqual(gym.enabled(pos,gym.SOURCE_ROOTS),
                                 gym.enabled(neg,gym.SOURCE_ROOTS))
                self.assertEqual(sum(h==gym.GOAL for p,h in posrules.values()),2)
                self.assertEqual(gym.public_features(pos)["goal_slot"],
                                 gym.public_features(neg)["goal_slot"])

    def test_public_features_do_not_expose_labels(self):
        c=gym.cases_for("held_depth5")[0]
        pos,neg=gym.pair_mates(c)
        for original in (pos,neg):
            features=gym.public_features(original)
            for prohibited in ("designed_positive","label","case_id",
                               "pair_id","optimal_steps","goal_truth"):
                self.assertNotIn(prohibited,features)
            graph=symbolic.PublicHorn.from_public_features(features)
            self.assertEqual(len(graph.initial),4)
            self.assertEqual(len(graph.rules),len(gym.logical_rules(original)))

    def test_source_only_manifest_complete_and_historical_disjoint(self):
        m=gym.manifest()
        self.assertTrue(m["source_only"])
        self.assertFalse(m["original_model_checkpoint_opened"])
        self.assertTrue(m["all_pairs_radius2_local_equal"])
        self.assertTrue(m["all_pairs_exact_per_node_degrees_equal"])
        self.assertTrue(m["all_pairs_goal_truth_disagrees"])
        self.assertTrue(m["cross_split_and_historical_graph_disjoint"])
        self.assertEqual(len(m["manifest_sha256"]),64)
        for split in gym.SPLITS:
            r=m["splits"][split]
            self.assertEqual((r["cases"],r["positive"],r["negative"],r["pairs"]),
                             (192,96,96,96))


if __name__=="__main__":
    unittest.main()

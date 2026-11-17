"""AF-32 source-only adversarial structural controls and independent BFS."""
import json
import unittest
from experiments.activation_funnel_af32 import gym


class SourceStructure(unittest.TestCase):
    def test_balanced_splits_and_swap_shift(self):
        for split in gym.SPLITS:
            roster = gym.cases_for(split)
            self.assertEqual(len(roster), 192)
            self.assertEqual(sum(c.reachable for c in roster), 96)
            self.assertEqual(len(set(c.id for c in roster)), 192)
            self.assertEqual(len(set(c.pair_id for c in roster)), 96)
            count = {x: 0 for x in gym.POSITIONS[split]}
            for c in roster:
                if c.reachable:
                    count[c.swap_position] += 1
            expected = ({1: 48, 2: 48} if split in ("train", "dev")
                        else {1: 32, 2: 32, 3: 32})
            self.assertEqual(count, expected)

    def test_every_pair_exact_local_match_but_global_truth_disagrees(self):
        for split in gym.SPLITS:
            for c in gym.cases_for(split):
                if not c.reachable:
                    continue
                p, n = gym.mates(c)
                rec = gym.pair_certificate(p, n)
                self.assertEqual(rec["swap_position"], c.swap_position)
                self.assertEqual(rec["positive_shortest"], c.depth)
                self.assertEqual(len(rec["changed_slots"]), 2)
                self.assertEqual(len(rec["two_goal_visible_distraction_rule_slots"]),2)
                self.assertGreaterEqual(rec["root_legal_count"],3)
                roots=gym.enabled(p,gym.ROOTS)
                self.assertTrue(set(rec["two_goal_visible_distraction_rule_slots"])<=set(roots))
                for head in (7,12):
                    self.assertIn(((head,10),13),gym.logical_rules(p))
                self.assertEqual(gym.radius2_signature(p), gym.radius2_signature(n))
                self.assertEqual(gym.degrees(p), gym.degrees(n))
                self.assertEqual(gym.enabled(p, gym.ROOTS),
                                 gym.enabled(n, gym.ROOTS))
                self.assertEqual(gym.physical_maps(p), gym.physical_maps(n))
                self.assertEqual(p.cycle_length,n.cycle_length)
                self.assertIn(p.cycle_length,(2,3))
                for source in (p,n):
                    logical=gym.logical_rules(source)
                    heads={h for _,h in logical}
                    self.assertTrue(set(range(4,15))<=heads)
                    physical=gym.physical_maps(source)[0]
                    reachable,_=gym.public_fixed_point(gym.public_features(source))
                    self.assertTrue(all(physical[a] not in reachable for a in (8,10,13)))
                    self.assertIn(((8,),10),logical)
                    if p.cycle_length==2:
                        self.assertIn(((10,),8),logical)
                    else:
                        self.assertIn(((7,10),13),logical)
                        self.assertIn(((13,),8),logical)


    def test_public_physical_bitset_bfs_and_firing_replay(self):
        for split in gym.SPLITS:
            for c in gym.cases_for(split):
                public = gym.public_features(c)
                self.assertEqual(gym.public_bitset_bfs(public),
                                 gym.shortest_steps(c))
                fixed, receipt = gym.public_fixed_point(public)
                self.assertEqual(public["goal_slot"] in fixed, c.reachable)
                self.assertGreater(receipt["rule_examinations"], 0)
                mapping = dict((slot, (tuple(prem), head))
                               for slot, prem, head in public["rules_physical"])
                s = gym.ROOTS
                physical = gym.to_physical(c, s)
                public_legal = sorted(slot for slot, (prem, head) in mapping.items()
                                      if head not in physical and
                                      all(p in physical for p in prem))
                self.assertEqual(public_legal, list(gym.enabled(c, s)))
                for slot in gym.enabled(c, s):
                    ns, proof = gym.sound_fire(c, s, slot)
                    self.assertEqual(gym.to_physical(c, ns),
                                     physical | {mapping[slot][1]})
                    self.assertEqual(proof["slot"], slot)
                for forbidden in (-1, 999, gym.STOP + 1):
                    with self.assertRaises(ValueError):
                        gym.sound_fire(c, s, forbidden)

    def test_no_generator_labels_in_public_policy_view(self):
        for c in gym.cases_for("held_depth6"):
            pub = gym.public_features(c)
            self.assertEqual(set(pub), {
                "atoms", "rules", "initial_physical", "goal_slot",
                "subgoal_slots", "rules_physical"
            })
            serialized = json.dumps(pub)
            for token in ("case_id", "pair_index", "reachable",
                          "swap_position", "optimal_steps", "split"):
                self.assertNotIn(token, serialized)

    def test_manifest_matches_unique_source_protocol(self):
        result = gym.manifest()
        self.assertEqual(result["total_pairs"], 384)
        self.assertEqual(result["total_cases"], 768)
        self.assertTrue(result["all_public_graphs_unique_and_historically_disjoint"])
        self.assertTrue(result["all_pairs_identical_radius2_degrees_and_root_masks"])
        self.assertTrue(result["all_unreachable_cycle_nodes_have_producers"])
        self.assertTrue(result["all_pairs_two_independent_goal_visible_root_distractors"])
        self.assertTrue(result["all_matched_cycle_SCCs_are_inaccessible"])
        self.assertEqual(result["splits"]["held_depth6"]["cycle_lengths"],
                         {"2":48,"3":48})
        self.assertFalse(result["opened_original_checkpoint"])
        self.assertEqual(len(result["manifest_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()

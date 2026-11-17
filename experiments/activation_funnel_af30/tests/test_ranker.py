"""AF-30 train-only action scoring and full host-legal episode smoke."""
import unittest
from experiments.activation_funnel_af30 import gym, features, ranker, evaluate


class RankerSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train=ranker.training_rows()
        cls.model=ranker.fit(17,cls.train)

    def test_train_roster_and_oracle_labeling(self):
        self.assertEqual(len(self.train),528)
        self.assertEqual(self.model["train_rows"],528)
        self.assertEqual(self.model["train_pair_graphs"],96)
        self.assertEqual(self.model["updates"],500)
        self.assertEqual(len(self.model["weights"]),8)
        self.assertEqual(len(self.model["loss_receipt"]),6)
        self.assertTrue(all(set(r["target_slots"])<=set(r["feature_vectors"])
                            for r in self.train))
        with self.assertRaises(ValueError):
            ranker.training_rows("held_depth5")

    def test_predict_uses_public_features_and_legal_slots_only(self):
        for split in ("train","dev","held_depth5","held_depth6"):
            c=gym.cases_for(split)[0]
            public=gym.public_features(c)
            state=frozenset(public["initial_physical"])
            vec, work=features.extract(public,state)
            self.assertEqual(len(next(iter(vec.values()))),8)
            self.assertEqual(
                sorted(vec),list(sorted(slot for slot,prem,head in public["rules_physical"]
                    if head not in state and all(p in state for p in prem))))
            self.assertGreater(work["feature_rule_examinations"],0)
            self.assertIn(ranker.choose(self.model,vec),vec)
            self.assertIn(ranker.choose(self.model,vec,untrained=True),vec)

    def test_same_seed_determinism(self):
        self.assertEqual(self.model,ranker.fit(17,self.train))
        self.assertNotEqual(self.model["initial_weights"],self.model["weights"])
        self.assertNotEqual(ranker.init_weights(17),ranker.init_weights(29))

    def test_closed_loop_multiple_modes_and_local_graphs(self):
        cases=gym.cases_for("held_depth5")[:4]+gym.cases_for("held_depth6")[:4]
        for case in cases:
            for arm in evaluate.ARMS:
                ep=evaluate.episode(case,arm,self.model,17)
                self.assertEqual(ep["designed_reachable"],case.reachable)
                self.assertFalse(ep["false_accepted_proof"])
                self.assertLessEqual(len(ep["history"]),ep["budget_oracle"])
                self.assertGreaterEqual(ep["total_rule_examinations"],0)
                if arm in evaluate.CERT_ARMS and not case.reachable:
                    self.assertTrue(ep["initial_stop"])

    def test_registered_gate_requires_actual_absolute_proofs_and_gain(self):
        summary={
            split:{arm:{
                "positive_proofs":0,
                "negative_correct_initial_stop":96,
                "false_accepted_proofs":0
            } for arm in evaluate.ARMS} for split in evaluate.SPLITS
        }
        self.assertFalse(evaluate.gate(summary)["both_depths_pass"])
        for split in evaluate.SPLITS:
            summary[split]["RANK_CERT"]["positive_proofs"]=60
        self.assertTrue(evaluate.gate(summary)["both_depths_pass"])
        for split in evaluate.SPLITS:
            summary[split]["PATH_CERT"]["positive_proofs"]=51
        self.assertFalse(evaluate.gate(summary)["both_depths_pass"])


if __name__ == "__main__":
    unittest.main()

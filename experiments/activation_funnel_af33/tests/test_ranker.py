"""AF-33 train-only action scoring and full host-legal episode smoke."""
import unittest
from experiments.activation_funnel_af33 import gym, features, ranker, evaluate, stop, audit


class RankerSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train=ranker.training_rows()
        cls.model=ranker.fit(17,cls.train)
        cls.stop_model={
            "BAL_STOP":stop.fit(17,cls.model,variant="BAL"),
            "J12_STOP":stop.fit(17,cls.model,variant="J12"),
        }

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
            for budget in evaluate.BUDGET_MODES:
                for arm in evaluate.ARMS:
                    ep=evaluate.episode(case,arm,self.model,17,self.stop_model,budget)
                    audit.audit_one(case,ep,self.model,17,self.stop_model)
                    self.assertEqual(ep["designed_reachable"],case.reachable)
                    self.assertEqual(ep["budget_mode"],budget)
                    self.assertFalse(ep["false_accepted_proof"])
                    self.assertLessEqual(len(ep["history"]),ep["budget_oracle"])
                    self.assertGreaterEqual(ep["total_rule_examinations"],0)
                    if arm in evaluate.CERT_ARMS and not case.reachable:
                        self.assertTrue(ep["initial_stop"])
                    if budget=="strict" and case.reachable and ep["solved"]:
                        self.assertFalse(ep["root_first_off_optimal"])

    def test_learned_stop_training_is_train_dev_only(self):
        self.assertEqual(set(self.stop_model),set(evaluate.LEARNED_STOP_ARMS))
        for arm,variant,ntrain,ndev in (
            ("BAL_STOP","BAL",576,96),
            ("J12_STOP","J12",384,64)
        ):
            m=self.stop_model[arm]
            self.assertEqual(m["train_rows"],ntrain)
            self.assertEqual(m["updates"],600)
            self.assertEqual(len(m["weights_bias_first"]),11)
            self.assertEqual(m["dev_positive_count"],ndev)
            self.assertIn(m["dev_threshold"],stop.THRESHOLDS)
            self.assertEqual(m,stop.fit(17,self.model,variant=variant))
        for variant in stop.VARIANTS:
            with self.assertRaises(ValueError):
                stop.make_rows("held_depth5",self.model,variant=variant)
        BAL=stop.make_rows("train",self.model,variant="BAL")
        J12=stop.make_rows("train",self.model,variant="J12")
        self.assertEqual({j:sum(r["source_swap_position"]==j for r in BAL) for j in (1,2,3)},
                         {1:192,2:192,3:192})
        self.assertEqual({j:sum(r["source_swap_position"]==j for r in J12) for j in (1,2,3)},
                         {1:192,2:192,3:0})

    def test_producer_supported_cycles_are_heldout_without_labels(self):
        for split in ("held_depth5","held_depth6"):
            bycycle={2:[],3:[]}
            for c in gym.cases_for(split):
                if c.reachable:
                    bycycle[c.cycle_length].append(c)
                public=gym.public_features(c)
                root=frozenset(public["initial_physical"])
                rootstop, prob, work=stop.choose(self.stop_model["BAL_STOP"],public,root,self.model)
                self.assertIsInstance(rootstop,bool)
                self.assertGreaterEqual(prob,0)
                self.assertLessEqual(prob,1)
                self.assertGreater(work["feature_rule_examinations"],0)
            self.assertEqual((len(bycycle[2]),len(bycycle[3])),(48,48))

    def test_registered_gate_requires_actual_absolute_proofs_and_gain(self):
        summary={
            key:{arm:{
                "positive_proofs":0,
                "negative_correct_initial_stop":96,
                "positive_premature_initial_stop":0,
                "false_accepted_proofs":0
            } for arm in evaluate.ARMS}
            for mode,split in evaluate.COHORTS
            for key in [evaluate.cohort_key(mode,split)]
        }
        self.assertFalse(evaluate.gate(summary)["both_depths_pass"])
        self.assertFalse(evaluate.balanced_stop_gate(summary)["both_depths_pass"])
        self.assertFalse(evaluate.absolute_bal_stop_gate(summary)["both_depths_pass"])
        for split in evaluate.SPLITS:
            summary[evaluate.cohort_key("strict",split)]["RANK_CERT"]["positive_proofs"]=60
        self.assertTrue(evaluate.gate(summary)["both_depths_pass"])
        for split in evaluate.SPLITS:
            summary[evaluate.cohort_key("strict",split)]["PATH_CERT"]["positive_proofs"]=51
        self.assertFalse(evaluate.gate(summary)["both_depths_pass"])
        for split in evaluate.SPLITS:
            x=summary[evaluate.cohort_key("strict",split)]
            x["BAL_STOP"]["positive_proofs"]=80
            x["BAL_STOP"]["negative_correct_initial_stop"]=70
            x["J12_STOP"]["negative_correct_initial_stop"]=40
        self.assertTrue(evaluate.balanced_stop_gate(summary)["both_depths_pass"])
        self.assertTrue(evaluate.absolute_bal_stop_gate(summary)["both_depths_pass"])
        for split in evaluate.SPLITS:
            summary[evaluate.cohort_key("strict",split)]["J12_STOP"]["negative_correct_initial_stop"]=65
        self.assertFalse(evaluate.balanced_stop_gate(summary)["both_depths_pass"])
        self.assertTrue(evaluate.absolute_bal_stop_gate(summary)["both_depths_pass"])


if __name__ == "__main__":
    unittest.main()

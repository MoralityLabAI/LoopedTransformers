"""FC-06 train-only leak guards, exact rule-budget and recurrent joint gradients."""
from __future__ import annotations
from copy import deepcopy
import os
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import fc06_joint as model


class JointFC06Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=os.environ.get("FC06_TRAIN_ROOT")
        if not root:
            raise RuntimeError("FC06_TRAIN_ROOT must point to immutable test-free source")
        cls.train_api,cls.train,cls.dev,cls.digest,_=model.load_train_source(Path(root))
        torch.set_num_threads(2)

    def test_frozen_registration_and_training_isolation(self):
        self.assertEqual(model.PREREG_SHA,"81f53bcf606e45d8b3fea9acba7907641c339cab")
        self.assertEqual(model.TRAIN_DATA_SHA,self.digest)
        self.assertEqual((len(self.train),len(self.dev)),(768,192))
        self.assertFalse({t["family_id"] for t in self.train} &
                         {t["family_id"] for t in self.dev})
        self.assertEqual(model.BUDGET,5)
        self.assertEqual(model.ACTOR_STEPS,300)
        self.assertEqual(model.ACTOR_BATCH,48)

    def test_hidden_labels_do_not_enter_model_features(self):
        t=next(x for x in self.train if x["label"]=="established")
        other=deepcopy(t)
        other["label"]="refuted"
        other["reachable"]=[]
        other["shortest_trace"]=[]
        other["min_steps"]=999
        other["split"]="test_deep"
        other["family_id"]="foreign"
        for known in (set(t["facts"]),set(t["facts"])|{t["goal"]}):
            b1,legal1=model._source_input(t,known)
            b2,legal2=model._source_input(other,known)
            self.assertTrue(torch.equal(legal1,legal2))
            for key in b1:
                self.assertTrue(torch.equal(b1[key],b2[key]),key)

    def test_actor_joint_gradients_and_rule_shapes(self):
        task=next(t for t in self.train if t["label"]=="established")
        b,legal=model._source_input(task,set(task["facts"]))
        a=model.JointActor()
        scores=a(b)
        self.assertEqual(tuple(scores.shape),(1,model.fs.MAX_R))
        self.assertEqual(sum(p.numel() for p in a.parameters()),16226)
        self.assertEqual(sum(p.numel() for p in a.trunk.parameters()),11521)
        self.assertEqual(sum(p.numel() for p in a.head.parameters()),4705)
        y=task["shortest_trace"][0]
        self.assertTrue(legal[y])
        loss=torch.nn.functional.cross_entropy(
            scores.masked_fill(~legal[None],-1e4),
            torch.tensor([y]))
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in a.trunk.cell.parameters()))
        self.assertTrue(any(p.grad is not None for p in a.head.parameters()))

    def test_train_only_teacher_is_minimal_under_budget(self):
        from research.refutation_bench.fc05 import proofs
        for t in self.train:
            if t["label"]!="established":
                continue
            self.assertLessEqual(t["min_steps"],5)
            self.assertTrue(proofs.verify_trace(
                t,t["shortest_trace"],budget=5)["verified"])
            self.assertEqual(len(t["reachable"])-len(t["facts"]),10)

    def test_inference_actor_returns_only_legal_bounded_rules(self):
        task=next(t for t in self.train if t["label"]=="established")
        actor=model.JointActor().eval()
        for method in model.METHODS:
            x=model.propose(actor,task,method)
            self.assertLessEqual(len(x["trace"]),5)
            known=set(task["facts"])
            for ruleid in x["trace"]:
                self.assertIn(ruleid,[j for j,rule in enumerate(task["rules"])
                    if rule["head"] not in known and
                    set(rule["premises"]).issubset(known)])
                known.add(task["rules"][ruleid]["head"])
            self.assertEqual(x["syntactic_goal_reached"],task["goal"] in known)

    def test_random_baseline_repeatable(self):
        task=next(t for t in self.dev if t["label"]=="established")
        actor=model.JointActor().eval()
        a=model.propose(actor,task,"random_legal")
        b=model.propose(actor,task,"random_legal")
        self.assertEqual(a["trace"],b["trace"])
        self.assertEqual(a["actor_recurrent_cells"],0)

    def test_paired_discordances_match_known_example(self):
        tasks=[{"case_id":"A","label":"established"},
               {"case_id":"B","label":"established"}]
        a=[{"case_id":"A","outcome":"established_certified"},
           {"case_id":"B","outcome":"abstain"}]
        b=[{"case_id":"A","outcome":"abstain"},
           {"case_id":"B","outcome":"established_certified"}]
        self.assertEqual(model._paired(tasks,a,b),
                         {"learned_only":1,"baseline_only":1,"net_learned":0})
        with self.assertRaises(AssertionError):
            model._paired(tasks,a,[{"case_id":"Z","outcome":"abstain"},b[1]])

    def test_invalid_method_and_negative_proof_gate(self):
        from research.refutation_bench.fc05 import proofs
        t=next(x for x in self.train if x["label"]=="refuted")
        with self.assertRaises(ValueError):
            model.propose(model.JointActor(),t,"oracle_repair")
        self.assertFalse(proofs.verify_trace(t,[],budget=5)["verified"])


if __name__=="__main__":
    unittest.main()

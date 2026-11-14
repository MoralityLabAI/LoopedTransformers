"""FC-07 prereg source isolation and exact backward-slice/learner pairing."""
from __future__ import annotations
from copy import deepcopy
import os
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import fc07_ancestry as fc07
import fc06_joint as fc06


class FC07LearnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=os.environ.get("FC07_TRAIN_ROOT")
        if not root:
            raise RuntimeError("FC07_TRAIN_ROOT required: must be pinned train-only source")
        cls.root=Path(root)
        cls.train,cls.dev,cls.digest=fc07.load_train_only(cls.root)
        torch.set_num_threads(2)

    def test_source_frozen_and_family_isolated(self):
        self.assertEqual(fc07.PREREG_SHA,"d3216371ae480b7e1b42cb913a33847ed5ad47c2")
        self.assertEqual((len(self.train),len(self.dev)),(768,192))
        self.assertEqual(fc07.FC07_TRAIN_DATA_SHA,self.digest)
        self.assertFalse({x["family_id"] for x in self.train} &
                         {x["family_id"] for x in self.dev})
        self.assertFalse((self.root/"research/refutation_bench/fc07/test_data.py").exists())

    def test_graph_input_independent_of_gold_and_teacher(self):
        t=next(x for x in self.train if x["label"]=="established")
        altered=deepcopy(t)
        altered["label"]="refuted"
        altered["shortest_trace"]=[]
        altered["min_steps"]=999
        altered["countermodel"]={"poison":True}
        altered["reachable"]=[]
        altered["family_id"]="test"
        altered["split"]="test_deep"
        for known in (set(t["facts"]),set(t["facts"])|{t["goal"]}):
            a,mask1=fc06._source_input(t,known)
            b,mask2=fc06._source_input(altered,known)
            self.assertTrue(torch.equal(mask1,mask2))
            for k in a:
                self.assertTrue(torch.equal(a[k],b[k]),k)

    def test_train_only_teacher_shortest_trace_is_valid_and_budgeted(self):
        from research.refutation_bench.fc05 import proofs
        for t in self.train:
            if t["label"]!="established":
                continue
            trace=t["shortest_trace"]
            self.assertEqual(len(trace),t["depth"])
            self.assertLessEqual(len(trace),5)
            self.assertTrue(proofs.verify_trace(t,trace,budget=5)["verified"])

    def test_joint_actor_gradients_and_frozen_architecture(self):
        actor=fc06.JointActor()
        self.assertEqual(sum(x.numel() for x in actor.parameters()),16226)
        t=next(x for x in self.train if x["label"]=="established")
        b,m=fc06._source_input(t,set(t["facts"]))
        logits=actor(b)
        target=t["shortest_trace"][0]
        self.assertTrue(m[target])
        loss=torch.nn.functional.cross_entropy(
            logits.masked_fill(~m[None],-1e4),
            torch.tensor([target]))
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in actor.trunk.cell.parameters()))
        self.assertTrue(any(p.grad is not None for p in actor.head.parameters()))

    def test_backward_slice_has_a_real_failure_on_training_cases(self):
        from research.refutation_bench.fc05 import proofs
        from research.refutation_bench.fc06.posthoc_backward_slice import priority_trace
        positive=[x for x in self.train if x["label"]=="established"]
        certified=sum(proofs.verify_trace(x,priority_trace(x,budget=5),budget=5)["verified"]
                      for x in positive)
        self.assertGreater(certified,0)
        self.assertLess(certified,len(positive))

    def test_backward_baseline_proposals_have_original_rule_indices(self):
        from research.refutation_bench.fc05 import proofs
        from research.refutation_bench.fc06.posthoc_backward_slice import priority_trace
        example=next(x for x in self.dev if x["label"]=="established")
        p={
            "outcome":"established_provisional",
            "reason":"test_fixture",
            "passes":6,
            "model_forward_ms":0.,
            "checker_ms":0.,
            "checker_calls":0,
            "certificate":None,
        }
        response=fc07._ancestry_output(example,p,proofs,priority_trace)
        known=set(example["facts"])
        for action in response["proposal_trace"]:
            self.assertGreaterEqual(action,0)
            self.assertLess(action,len(example["rules"]))
            rule=example["rules"][action]
            self.assertTrue(set(rule["premises"])<=known)
            self.assertNotIn(rule["head"],known)
            known.add(rule["head"])
        self.assertEqual(len(response["proposal_trace"]),response["rule_actions"])
        self.assertEqual(response["actor_recurrent_cells"],0)
        self.assertGreaterEqual(response["backward_slice_ms"],0.)

    def test_pair_count_rejects_wrong_id_and_measures_discordance(self):
        cases=[{"case_id":"A","label":"established"},
               {"case_id":"B","label":"established"}]
        neural=[{"case_id":"A","outcome":"established_certified"},
                {"case_id":"B","outcome":"abstain"}]
        baseline=[{"case_id":"A","outcome":"abstain"},
                  {"case_id":"B","outcome":"established_certified"}]
        self.assertEqual(fc07._pair(cases,neural,baseline),
                         {"learned_only":1,"baseline_only":1,"net_learned":0})
        with self.assertRaises(AssertionError):
            fc07._pair(cases,neural,[{"case_id":"C","outcome":"abstain"},baseline[1]])


if __name__=="__main__":
    unittest.main()

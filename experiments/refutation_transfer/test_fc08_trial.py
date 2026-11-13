"""FC-08: source isolation, model feature masking and bounded search checks."""
from __future__ import annotations
from copy import deepcopy
import os
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import fc08_trial as fc08
import fc08_symbolic as symbolic
import fc06_joint as fc06


class FC08ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source=os.environ.get("FC08_TRAIN_ROOT")
        if not source:
            raise RuntimeError("FC08_TRAIN_ROOT must resolve to pinned source before test module exists")
        cls.root=Path(source)
        cls.training,cls.dev,cls.hash=fc08.load_source_train(cls.root)
        torch.set_num_threads(2)

    def test_train_only_pin_and_test_generator_absent(self):
        self.assertEqual(fc08.PREREG_SHA,"b6c8ffbef6606e89737bbfc32d73b8732b765e8d")
        self.assertEqual(fc08.TRAIN_DATA_SHA,self.hash)
        self.assertEqual((len(self.training),len(self.dev)),(768,192))
        self.assertFalse((self.root/"research/refutation_bench/fc08/test_data.py").exists())
        self.assertFalse((self.root/"research/refutation_bench/fc08/controls.py").exists())
        self.assertFalse(set(x["family_id"] for x in self.training)&
                         set(x["family_id"] for x in self.dev))

    def test_no_goal_only_unary_shortcuts(self):
        from research.refutation_bench.fc07.posthoc_unary import unary_shortest_trace
        for t in self.training:
            self.assertEqual(len(t["atoms"]),12)
            self.assertTrue(all(len(r["premises"])>=2 for r in t["rules"]
                                if r["head"]==t["goal"]))
            self.assertFalse(unary_shortest_trace(t,budget=5))
            if t["label"]=="established":
                self.assertLessEqual(t["min_steps"],5)
                self.assertEqual(t["min_steps"],t["depth"])

    def test_labels_and_source_proof_not_in_model_forward(self):
        t=next(x for x in self.training if x["label"]=="established")
        edited=deepcopy(t)
        edited["shortest_trace"]=[-1]
        edited["label"]="refuted"
        edited["min_steps"]=0
        edited["reachable"]=[]
        edited["countermodel"]={"poison":True}
        edited["family_id"]="test-family"
        edited["split"]="test"
        for known in (set(t["facts"]),set(t["facts"])|{t["goal"]}):
            a,mask1=fc06._source_input(t,known)
            b,mask2=fc06._source_input(edited,known)
            for k in a:
                self.assertTrue(torch.equal(a[k],b[k]),k)
            self.assertTrue(torch.equal(mask1,mask2))

    def test_shared_lr01_actor_is_trainable_and_rule_masked(self):
        actor=fc06.JointActor()
        self.assertEqual(sum(p.numel() for p in actor.parameters()),16226)
        t=next(x for x in self.training if x["label"]=="established")
        features,mask=fc06._source_input(t,set(t["facts"]))
        logits=actor(features)
        self.assertEqual(tuple(logits.shape),(1,fc06.fs.MAX_R))
        label=t["shortest_trace"][0]
        self.assertTrue(mask[label])
        loss=torch.nn.functional.cross_entropy(
            logits.masked_fill(~mask[None],-1e4),torch.tensor([label]))
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in actor.trunk.cell.parameters()))
        self.assertTrue(any(p.grad is not None for p in actor.head.parameters()))

    def test_symbolic_24_state_budget_is_real(self):
        for t in self.training[:20]:
            proposal=symbolic.bounded_best_first(t,max_popped=24)
            self.assertLessEqual(proposal["states_popped"],24)
            self.assertLessEqual(len(proposal["trace"]),5)
            self.assertGreaterEqual(proposal["symbolic_search_ms"],0)

    def test_bounded_search_fails_without_legal_goal_trace(self):
        t=next(x for x in self.training if x["label"]=="refuted")
        p=symbolic.bounded_best_first(t)
        self.assertFalse(p["found_goal"])
        from research.refutation_bench.fc05 import proofs
        self.assertFalse(proofs.verify_trace(t,p["trace"],budget=5)["verified"])

    def test_equal_case_pairing_fails_closed(self):
        cases=[{"case_id":"a","label":"established"},
               {"case_id":"b","label":"established"}]
        model=[{"case_id":"a","outcome":"established_certified"},
               {"case_id":"b","outcome":"abstain"}]
        base=[{"case_id":"a","outcome":"abstain"},
              {"case_id":"b","outcome":"established_certified"}]
        self.assertEqual(fc08._paired(cases,model,base),
              {"learned_only":1,"baseline_only":1,"both_certified":0,"net_learned":0})
        with self.assertRaises(AssertionError):
            fc08._paired(cases,model,[{"case_id":"z","outcome":"abstain"},base[1]])


if __name__=="__main__":
    unittest.main()

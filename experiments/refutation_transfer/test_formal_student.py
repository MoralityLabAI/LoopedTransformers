"""FC-02 smoke tests for actual LR-01 latent closure learner."""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import formal_student as fs


class FormalStudentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=os.environ.get("RB_BENCH_ROOT")
        if not root:
            raise RuntimeError("RB_BENCH_ROOT must point to exact benchmark commit")
        cls.train,cls.dev,cls.test,cls.digest,cls.verifier=fs.load_frozen_curriculum(Path(root))
        torch.set_num_threads(2)

    def test_integrity_counts_and_families(self):
        self.assertEqual((len(self.train),len(self.dev),len(self.test)),(1152,192,192))
        self.assertEqual(self.digest,fs.GENERATOR_SHA)
        self.assertFalse({t["family_id"] for t in self.train} &
                         {t["family_id"] for t in self.dev+self.test})
        self.assertFalse({t["family_id"] for t in self.dev} &
                         {t["family_id"] for t in self.test})

    def test_input_features_ignore_gold_and_provenance(self):
        t=deepcopy(self.train[0])
        altered=deepcopy(t)
        altered["label"]="refuted" if t["label"]=="established" else "established"
        altered["reachable"]=[]
        altered["countermodel"]={"bogus":True}
        altered["split"]="test"
        altered["family_id"]="new"
        altered["case_id"]="modified"
        for k in ("a","r","prem","head","mask","goal_idx"):
            self.assertTrue(torch.equal(fs.encode([t])[k],fs.encode([altered])[k]),k)

    def test_source_names_are_structural(self):
        orig=deepcopy(self.train[0])
        renamed=deepcopy(orig)
        old=orig["atoms"]
        mapping={name:"Z"+str(len(old)-1-i) for i,name in enumerate(old)}
        renamed["atoms"]=sorted(mapping.values())
        renamed["facts"]=[mapping[x] for x in orig["facts"]]
        renamed["goal"]=mapping[orig["goal"]]
        renamed["rules"]=[{"premises":[mapping[x] for x in r["premises"]],
                           "head":mapping[r["head"]]} for r in orig["rules"]]
        renamed["reachable"]=[mapping[x] for x in orig["reachable"]]
        torch.manual_seed(17)
        net=fs.LoopReach(4).eval()
        b1=fs.encode([orig]);b2=fs.encode([renamed])
        with torch.no_grad():
            p1=net(b1)[-1].sigmoid()[0,b1["goal_idx"][0]]
            p2=net(b2)[-1].sigmoid()[0,b2["goal_idx"][0]]
        self.assertTrue(torch.allclose(p1,p2,atol=3e-5,rtol=3e-5))

    def test_forward_and_backward(self):
        data=fs.encode(self.train[:8])
        torch.manual_seed(19)
        for loops in (1,4):
            model=fs.LoopReach(loops)
            logits=model(data)
            self.assertEqual(len(logits),loops)
            self.assertEqual(tuple(logits[-1].shape),(8,fs.MAX_A))
            loss=fs.objective(logits,data)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(any(x.grad is not None for x in model.parameters()))

    def test_reproducible_training_without_dev(self):
        a,meta_a=fs.train_model(self.train[:16],loops=1,seed=17,steps=3,batch_size=8)
        b,meta_b=fs.train_model(self.train[:16],loops=1,seed=17,steps=3,batch_size=8)
        self.assertEqual(meta_a["losses"],meta_b["losses"])
        for name,tensor in a.state_dict().items():
            self.assertTrue(torch.equal(tensor,b.state_dict()[name]),name)

    def test_verifier_rejects_unrepaired_model_counterexample(self):
        task=next(t for t in self.dev if t["label"]=="refuted")
        invalid={atom:True for atom in task["atoms"]}
        f=task["formal"]
        self.assertFalse(type(self).verifier(f["premises"],f["conclusion"],invalid)["verified"])
        expected=task["countermodel"]["assignment"]
        self.assertTrue(type(self).verifier(f["premises"],f["conclusion"],expected)["verified"])


if __name__=="__main__":
    unittest.main()

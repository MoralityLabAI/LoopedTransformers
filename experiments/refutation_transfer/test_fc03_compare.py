"""FC-03 prereg controls: matched cell steps, train/test isolation, blind forward."""
from __future__ import annotations
from copy import deepcopy
import os
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import fc03_compare as fc


class FC03CompareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=os.environ.get("FC02_ROOT")
        if not root:
            raise RuntimeError("FC02_ROOT must point to pinned train-only source repo")
        cls.train,cls.dev,cls.digest=fc.load_source_training(Path(root))
        torch.set_num_threads(2)

    def test_parameter_parity_and_equal_pass_count(self):
        models={name:fc.make_model(name) for name in fc.TRAIN_KINDS}
        counts={name:sum(p.numel() for p in model.parameters())
                for name,model in models.items()}
        self.assertEqual(counts,{
            "tied4_width24":11521,
            "untied4_width24":45073,
            "untied4_width12":11593,
        })
        self.assertLess(abs(counts["untied4_width12"]-counts["tied4_width24"])/11521,.01)
        self.assertEqual(len(models["untied4_width24"].blocks),4)
        self.assertEqual(models["tied4_width24"].loops,4)

    def test_input_projection_excludes_labels(self):
        t=deepcopy(self.train[0])
        changed=deepcopy(t)
        changed["label"]="refuted" if t["label"]=="established" else "established"
        changed["reachable"]=[]
        changed["countermodel"]={"fake":True}
        changed["family_id"]="unknown"
        changed["split"]="test_deep"
        changed["case_id"]="unseen"
        b1=fc.fs.encode([{**{key:t[key] for key in
            ("case_id","atoms","facts","rules","goal")},
            "label":"refuted","reachable":[]}])
        b2=fc.fs.encode([{**{key:changed[key] for key in
            ("case_id","atoms","facts","rules","goal")},
            "label":"refuted","reachable":[]}])
        for key in ("a","r","prem","head"):
            self.assertTrue(torch.equal(b1[key],b2[key]),key)

    def test_same_seed_minibatches_and_reproducibility(self):
        data=self.train[:16]
        old_steps=fc.STEPS
        try:
            fc.STEPS=3
            for kind in ("tied4_width24","untied4_width12"):
                net1,r1=fc._train_one(data,kind,17)
                net2,r2=fc._train_one(data,kind,17)
                self.assertEqual(r1["losses"],r2["losses"])
                for name,weight in net1.state_dict().items():
                    self.assertTrue(torch.equal(weight,net2.state_dict()[name]),name)
        finally:
            fc.STEPS=old_steps

    def test_architecture_forward_finite_and_noparam_change_after_extra(self):
        cases=self.train[:5]
        for name in fc.TRAIN_KINDS:
            net=fc.make_model(name)
            x=fc.fs.encode(cases)
            y=net(x)
            self.assertEqual(len(y),4)
            self.assertTrue(torch.isfinite(y[-1]).all())
        net=fc.make_model("tied4_width24")
        state={key:v.clone() for key,v in net.state_dict().items()}
        _,ms=fc.predict_blind(net,cases,passes=8)
        self.assertGreater(ms,0)
        self.assertEqual(net.loops,4)
        for key,v in net.state_dict().items():
            self.assertTrue(torch.equal(v,state[key]))

    def test_pair_scoring_examples(self):
        tasks=[{"case_id":"a","label":"established"},
               {"case_id":"b","label":"refuted"},
               {"case_id":"c","label":"established"}]
        t=[{"case_id":x["case_id"],"predicted":p} for x,p in
           zip(tasks,["established","refuted","refuted"])]
        u=[{"case_id":x["case_id"],"predicted":p} for x,p in
           zip(tasks,["refuted","refuted","established"])]
        self.assertEqual(fc.paired(tasks,t,u),
            {"tied_right_untied_wrong":1,
             "untied_right_tied_wrong":1,
             "net_tied_advantage":0})


if __name__=="__main__":
    unittest.main()

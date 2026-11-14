"""FC-05 positive inference policy tests with FC-02-only train checkout."""
from __future__ import annotations
from copy import deepcopy
import os
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import fc05_dual as dual
import fc03_compare as fc03


class DualCertificateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source=os.environ.get("FC02_ROOT")
        if not source:
            raise RuntimeError("FC02_ROOT must point to the pinned old training checkout")
        cls.train,cls.dev,cls.digest,cls.verifier=dual.fc04.load_train_only(Path(source))
        torch.set_num_threads(2)

    def test_prior_data_and_protocol_isolation(self):
        self.assertEqual(dual.PREREG_SHA,"e6ab5716e3316441adad131b087a50d1ca0f2734")
        self.assertEqual(self.digest,fc03.FC02_DATA_SHA)
        self.assertEqual((len(self.train),len(self.dev)),(1152,192))
        self.assertEqual(dual.RULE_BUDGET,10)
        self.assertEqual(dual.TRAIN_HEAD_STEPS,250)
        self.assertFalse({c["family_id"] for c in self.train} &
                         {c["family_id"] for c in self.dev})

    def test_training_oracle_trace_is_a_valid_legal_derivation(self):
        for task in self.train:
            if task["label"]!="established":
                continue
            trace=dual.shortest_training_trace(task)
            self.assertLessEqual(len(trace),10)
            known=set(task["facts"])
            for action in trace:
                self.assertIn(action,dual.legal_rules(task,known))
                known.add(task["rules"][action]["head"])
            self.assertIn(task["goal"],known)

    def test_policy_ignores_answer_fields(self):
        task=next(x for x in self.dev if x["label"]=="established")
        other=deepcopy(task)
        other["label"]="refuted"
        other["reachable"]=[]
        other["countermodel"]={"bad":True}
        other["family_id"]="unseen"
        model=fc03.make_model("tied4_width24").eval()
        with torch.no_grad():
            a,m1=dual._features(model,task,set(task["facts"]))
            b,m2=dual._features(model,other,set(task["facts"]))
        self.assertTrue(torch.equal(a,b))
        self.assertTrue(torch.equal(m1,m2))

    def test_model_action_is_always_currently_legal(self):
        t=next(x for x in self.dev if x["label"]=="established")
        model=fc03.make_model("tied4_width24").eval()
        head=dual.RuleHead().eval()
        for method in dual.RULE_METHODS:
            r=dual.propose_trace(model,head,t,method)
            self.assertLessEqual(len(r["trace"]),10)
            known=set(t["facts"])
            for action in r["trace"]:
                self.assertIn(action,dual.legal_rules(t,known))
                known.add(t["rules"][action]["head"])
            self.assertEqual(r["goal_reached_syntactically"],t["goal"] in known)

    def test_random_policy_reproducibility(self):
        t=next(x for x in self.train if x["label"]=="established")
        model=fc03.make_model("tied4_width24")
        head=dual.RuleHead()
        a=dual.propose_trace(model,head,t,"random_legal")
        b=dual.propose_trace(model,head,t,"random_legal")
        self.assertEqual(a["trace"],b["trace"])
        self.assertEqual(a["head_calls"],0)

    def test_proof_head_shape_and_gradient(self):
        x=torch.randn(4,dual.fs.MAX_R,4*dual.fs.WIDTH)
        head=dual.RuleHead()
        out=head(x)
        self.assertEqual(out.shape,(4,dual.fs.MAX_R))
        self.assertEqual(sum(p.numel() for p in head.parameters()),4705)
        out.square().mean().backward()
        self.assertTrue(all(p.grad is not None for p in head.parameters()))

    def test_wrong_nonnumeric_method_is_rejected(self):
        t=self.train[0]
        model=fc03.make_model("tied4_width24")
        with self.assertRaises(ValueError):
            dual.propose_trace(model,dual.RuleHead(),t,"oracle_repair")


if __name__=="__main__":
    unittest.main()

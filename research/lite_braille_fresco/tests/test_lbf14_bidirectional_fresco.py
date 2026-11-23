"""Layer scheduler, exact counterfactual and gate treatment checks."""
import os
from copy import deepcopy
from pathlib import Path
import sys
import unittest
try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name!="torch":raise
    raise unittest.SkipTest("optional CPU Torch is absent")
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import lbf14_bidirectional_fresco as b


class GateTreatment(unittest.TestCase):
    def test_initialization_differs_only_at_two_biases(self):
        a,initial=b.new_model(107,"standard")
        z,other=b.new_model(107,"opened")
        self.assertEqual(initial,other)
        different=[]
        for name,t in a.state_dict().items():
            u=z.state_dict()[name]
            if not torch.equal(t,u):
                different.append(name)
                self.assertTrue(torch.all(t==-6))
                self.assertTrue(torch.all(u==-2))
        self.assertEqual(set(different),{"gate_a.gate.bias","gate_b.gate.bias"})


@unittest.skipUnless(os.environ.get("LBF14_UPSTREAM"),"requires pinned rewrite source")
class LayerScheduler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rg=b.base.load_upstream(Path(os.environ["LBF14_UPSTREAM"]))
        cases,_=cls.rg.generate(20261219,{"train":8})
        cls.cases=cases["train"]

    def test_pinned_equivalence_at_multiple_caps(self):
        for c in self.cases:
            for budget in (1,4,16,128):
                ref=self.rg.search(c,"bidir",budget=budget,depth=16)
                actual=b.run(self.rg,c,"bidir",budget=budget)
                self.assertEqual(ref,{k:actual[k] for k in ref})

    def test_forced_smaller_reproduces_continuation_without_mutation(self):
        for c in self.cases:
            s=b.State.start(c)
            while s.live(128):
                old=deepcopy(s)
                side=b.smaller(s)
                cost,used,path=b.forced(self.rg,c,s,side)
                self.assertEqual(s,old)
                direct=b.continue_smaller(self.rg,c,deepcopy(s))
                self.assertEqual(used,direct.work-s.work)
                self.assertEqual(path,direct.path)
                self.assertEqual(cost,used if path is not None else 129-s.work)
                b.advance(self.rg,c,s,side)

    def test_public_feature_symmetry_and_accounting(self):
        c=self.cases[0];s=b.State.start(c)
        p,menu,x,raw,work=b.features(self.rg,c,s)
        self.assertEqual(len(menu),2)
        self.assertEqual(x[0],x[1])
        self.assertEqual(raw[0],raw[1])
        self.assertEqual(work,3)
        self.assertIn("|W:128",p)
        self.assertEqual(len(x[0]),14)
        self.assertNotIn("witness",p)


if __name__=="__main__":unittest.main()

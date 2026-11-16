"""RAM-16 training-codepath CPU micro-smoke, entirely synthetic source goals."""
from __future__ import annotations
from pathlib import Path
import sys
import unittest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import train

class Objective(unittest.TestCase):
    def test_architectures_are_parameter_matched(self):
        a=train.make_model("loop64",17)
        b=train.make_model("flat71",17)
        x=sum(p.numel() for p in a.parameters())
        y=sum(p.numel() for p in b.parameters())
        self.assertLess(abs(x-y)/x,.015)
        self.assertEqual(a.config.loops,4)
        self.assertEqual(b.config.loops,0)
        self.assertEqual(len(train.FAMILIES23),23)
    def test_original_first_twenty_family_ids_stable(self):
        self.assertEqual(tuple(train.FAMILIES23[:20]),tuple(train.old.FAMILIES))
        self.assertEqual(train.FAMILIES23[20:],("nat_induction","simp_prior","rewrite_prior"))
    def test_paired_objective_init_stable(self):
        a=train.make_model("loop64",43)
        b=train.make_model("loop64",43)
        self.assertEqual(train.state_fingerprint(a.state_dict()),
                         train.state_fingerprint(b.state_dict()))
        a=train.make_model("flat71",29)
        b=train.make_model("flat71",29)
        self.assertEqual(train.state_fingerprint(a.state_dict()),
                         train.state_fingerprint(b.state_dict()))
    def test_registered_exact_10_percent_macro_sample_count(self):
        dummy=dict(before={"goals":["n : Nat\n⊢ n = n"],"definitions":[]},
                   script="omega",family="omega",delta=1,label=3)
        macro=dict(dummy,family="nat_induction",
                   script="induction n <;> simp_all")
        import random
        rng=random.Random(17)
        n=sum(train.draw_batch([dummy],[macro],17,i,"task_plus_structural",rng)[1]
              for i in range(400))
        self.assertEqual(n,2560)
        self.assertEqual(train.draw_batch([dummy],[macro],17,1,
                            "task_only",rng)[1],0)
    def test_cpu_original_lean_goal_encoding_and_two_step_gradients(self):
        original=dict(before={"goals":["n : Nat\n⊢ n = n"],"definitions":[]},
            script="rfl",family="rfl",label=3,delta=1)
        macro=dict(before={"goals":["n : Nat\n⊢ n = n"],"definitions":[]},
            script="induction n <;> simp_all",family="nat_induction",label=3,delta=1)
        m=train.make_model("loop64",17)
        import torch
        torch.set_num_threads(2)
        optim=torch.optim.AdamW(m.parameters(),lr=.002)
        for _ in range(2):
            optim.zero_grad()
            loss=train.single_loss(m,[original,macro])
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optim.step()
    def test_task_holdback_split_by_original_theorem_root(self):
        oldtrain=[]
        for n in range(100):
            for a in range(3):
                oldtrain.append(dict(source_file="MathCan/TOENonAbelian.lean",
                   theorem="generated_train_"+str(n),development=False))
        task,dev,receipt=train.task_split(oldtrain)
        self.assertTrue(task)
        self.assertTrue(dev)
        roots1={(x["source_file"],x["theorem"]) for x in task}
        roots2={(x["source_file"],x["theorem"]) for x in dev}
        self.assertFalse(roots1&roots2)
        self.assertEqual(len(task)+len(dev),300)
        self.assertTrue(receipt["root_disjoint"])

if __name__=="__main__":unittest.main()

"""LOOPSCOPE-04 CPU source-independent tests: exact shapes and causal arithmetic."""
from __future__ import annotations

import math
import unittest

import torch
from torch.nn import functional as F

from mechinterp.loopscope04.gradient_flow import (
    SCHEMA,ReasoningBlock,TinyRecursiveModel,frozen_inputs,
    fixed_data,train_one,per_token_ce,head_rowspace,policy_decision,
    reward,finite_class,profile,intervention,GOALS,cos_flat,run,head_only_bayes,
)


class GradientLoopGymTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_exact_source_model_size_and_tied_update(self):
        m=TinyRecursiveModel(48,4)
        self.assertEqual(sum(p.numel() for p in m.parameters()),44001)
        x=torch.ones((2,81),dtype=torch.long)
        s=m.reason_states(x)
        self.assertEqual(len(s),5)
        self.assertTrue(all(t.shape==(2,81,48) for t in s))
        self.assertIs(m.block,m.block)
        with torch.no_grad():
            self.assertTrue(torch.equal(m.head(s[-1]),m(x)))

    def test_cyclic_copy_data_and_split_source_hash(self):
        data,manifest=frozen_inputs(smoke=True)
        self.assertEqual(set(manifest),{"train","val","test"})
        self.assertEqual(len({manifest[k]["x_sha256"] for k in manifest}),3)
        for x,y in data:
            self.assertTrue(torch.equal(y,x.roll(1,1)))
            self.assertTrue(bool(((1<=x)&(x<=9)).all()))
            self.assertEqual(x.shape[-1],81)

    def test_local_ce_gradient_matches_autograd_independently(self):
        m=TinyRecursiveModel(48,4)
        h=torch.randn(3,81,48,requires_grad=True)
        y=torch.randint(1,10,(3,81))
        logits=m.head(h)
        token=per_token_ce(logits,y)
        derivative=torch.autograd.grad(token.sum(),h)[0]
        with torch.no_grad():
            analytical=(torch.softmax(logits,-1)-F.one_hot(y,10).float())@m.head.weight
        self.assertTrue(torch.allclose(derivative,analytical,atol=1e-5,rtol=1e-5))
        self.assertAlmostEqual(cos_flat(analytical,-analytical),-1.,places=10)

    def test_rowspace_projection_kills_head_null_component(self):
        m=TinyRecursiveModel(48,4)
        basis,rank=head_rowspace(m)
        self.assertLessEqual(rank,10)
        self.assertGreater(rank,0)
        vector=torch.randn(48,dtype=torch.float64)
        d=vector-basis@(basis.T@vector)
        output=m.head.weight.double()@d
        self.assertLess(float(output.norm()),1e-9)
        self.assertLess(float((basis.T@d).norm()),1e-9)

    def test_exact_bayes_actions_and_original_abstain_rewards(self):
        p=[.7]+[.3/8]*8
        d=[1.]+[-1.]*8
        decision,rec=policy_decision(d,p)
        self.assertEqual(decision,1)
        self.assertAlmostEqual(rec["p_plus"],.7)
        self.assertAlmostEqual(rec["p_minus"],.3)
        self.assertAlmostEqual(reward(1,1),.9)
        self.assertAlmostEqual(reward(0,0),1.)
        self.assertAlmostEqual(reward(1,0),-1.1)
        self.assertEqual(finite_class(.15),0)
        self.assertEqual(finite_class(-.2),-1)
        abstain,_=policy_decision([0.]*9,p)
        self.assertEqual(abstain,0)

    def test_vector_bayes_exact_parity_with_scalar_nine_digit_policy(self):
        torch.manual_seed(1327)
        m=TinyRecursiveModel()
        h=torch.randn(2,81,48)
        delta=torch.randn(2,81,48)*.25
        targets=torch.randint(1,10,(2,81))
        vec=head_only_bayes(m,h,delta,targets,None)
        with torch.no_grad():
            clean=m.head(h)
            plus=m.head(h+delta)
            minus=m.head(h-delta)
            source=[]
            for i in range(2):
                r=[]
                for j in range(81):
                    logits=clean[i,j].tolist()
                    q=torch.softmax(clean[i,j,1:10].double(),dim=-1).tolist()
                    effect=[]
                    for digit in range(1,10):
                        rival=max((x for x in range(10) if x!=digit),
                                  key=lambda x:logits[x])
                        effect.append(float((plus[i,j,digit]-plus[i,j,rival])-
                                            (minus[i,j,digit]-minus[i,j,rival])))
                    a,_=policy_decision(effect,q)
                    b,_=policy_decision(effect,[1./9.]*9)
                    truth=finite_class(effect[int(targets[i,j])-1])
                    r.append((reward(a,truth),reward(b,truth),reward(0,truth),
                              int(a!=0),int(a==truth and truth!=0)))
                source.append([sum(item[c] for item in r)/81 for c in range(5)])
        keys=("bayes_utility","uniform_utility","always_abstain_utility",
              "action_rate","correct_nonzero")
        for i in range(2):
            for k,name in enumerate(keys):
                self.assertAlmostEqual(vec[i][name],source[i][k],places=10)

    def test_optimizer_vs_recurrence_gradient_ledger(self):
        x,y=fixed_data(48,31041)
        model,snapshots,hist=train_one(103,"final_only",x,y,updates=5,check_steps=(0,5))
        self.assertEqual(set(snapshots),{0,5})
        self.assertEqual(len(hist),1)
        measured=hist["5"]
        self.assertGreater(measured["optimizer_gradient_l2"],0)
        self.assertGreater(measured["optimizer_update_l2"],0)
        self.assertGreater(measured["optimizer_update_cos_negative_BPTT_grad"],-1.001)
        self.assertLessEqual(measured["optimizer_update_cos_negative_BPTT_grad"],1.)
        self.assertEqual(model.n_steps,4)

    def test_head_only_and_native_remaining_loop_measured_separately(self):
        model=TinyRecursiveModel()
        x,y=fixed_data(2,31043)
        p=profile(model,x,y)
        self.assertEqual(set(p["per_loop"]),{"0","1","2","3"})
        self.assertEqual(p["n_independent_sequences"],2)
        self.assertEqual(len(p["per_loop"]["0"]["per_sequence"]),2)
        self.assertTrue(math.isfinite(p["per_loop"]["2"]["mean"]["cosine"]))
        a=intervention(model,x,y,103,"final_only",checkpoint=0,max_sequences=1)
        self.assertEqual(set(a),{"1","2"})
        self.assertEqual(a["1"]["n_goals"],8)
        self.assertTrue(all(set(c["arms"])=={"clean","plus","minus","random"}
                            for c in a["1"]["cases"][0]["goals"]))

    def test_end_to_end_smoke_preserves_source_and_all_controls(self):
        out=run(smoke=True)
        self.assertEqual(out["schema"],SCHEMA)
        self.assertEqual(out["registered_study_status"],"PILOT_ONLY")
        self.assertEqual(set(out["results"]["103"]),{"final_only","deep_020"})
        for arm in ("final_only","deep_020"):
            item=out["results"]["103"][arm]
            self.assertEqual(set(item["checkpoints"]),{"0","5","10"})
            self.assertEqual(item["native_remaining_loop_intervention"]["1"]["n_puzzles"],8)
            self.assertEqual(item["registered_diagnostics"]["task_admitted"],
                             item["registered_diagnostics"]["test_final_acc"]>.20)


if __name__=="__main__":
    unittest.main()

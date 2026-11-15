"""AF-11 source, Boolean oracle, causal cut, gradients and numerical guards."""
import unittest
import tempfile
import json
from pathlib import Path
import os
import torch
from experiments.activation_funnel_af05 import gym
from experiments.activation_funnel_af05.model import GraphLooper
from experiments.activation_funnel_af11 import data,core,experiment as ex,verify

class AF11Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.train=data.Dataset('train')

    def test_previous_encoding_exact_and_new_permutations_nontrivial(self):
        for perm in (0,1,4,7):
            case=gym.Case('test',3,('and','or'),'and',(1,0,1,1),perm)
            a=data.encode(case);b=gym.encode_case(case)
            for key in a.__dataclass_fields__:
                self.assertTrue(torch.equal(getattr(a,key),getattr(b,key)),key)
        self.assertNotEqual(data.physical_permutations(8),gym.permutation_maps(7))
        with self.assertRaises(ValueError):data.physical_permutations(14)

    def test_frozen_fresh_manifest_complete_and_disjoint(self):
        m=data.fresh_manifest()
        self.assertEqual({x:m['fresh_splits'][x]['cases'] for x in data.FRESH},
                         dict(fresh_depth3=128,fresh_comp=16,fresh_depth4=256,fresh_depth5=256))
        self.assertEqual(m,json.loads(ex.FRESH_MANIFEST.read_text()))
        self.assertTrue(ex.verify_fresh_manifest())

    def test_exact_horn_closure_matches_direct_boolean(self):
        for split in data.FRESH:
            ds=data.Dataset(split)
            for case in ds.cases:
                self.assertEqual(verify.exact_boolean(case),gym.oracle(case)[0])
                self.assertEqual(int(case.depth),data.DEPTH[split])

    def test_donor_roster_unchanged_other_branch_and_graph(self):
        for split in data.FRESH:
            ds=data.Dataset(split)
            self.assertEqual(len(ds.donors),2*len(ds.cases))
            for i,c in enumerate(ds.cases):
                for j in (0,1):
                    k,hybrid=ds.donors[i,j]
                    d=ds.cases[k]
                    self.assertEqual(c.skeleton,d.skeleton)
                    o=1-j
                    self.assertEqual(c.facts[2*o:2*o+2],d.facts[2*o:2*o+2])
                    self.assertEqual(gym.oracle_interchange(c,d,j),hybrid)

    def test_exact_influence_every_gate_case_and_donor(self):
        for split in data.FRESH:
            ds=data.Dataset(split)
            for i,c in enumerate(ds.cases):
                for j in (0,1):
                    _,hybrid=ds.donors[i,j]
                    y=ds.batch.targets[i:i+1]
                    z=torch.tensor([hybrid],dtype=torch.long)
                    delta,influ=core.signed_goal_difference(y,z,torch.tensor([j]),
                                                             torch.tensor([int(c.gate=='or')]))
                    other=gym.oracle(c)[0][1-j]
                    want=other if c.gate=='and' else 1-other
                    self.assertEqual(int(influ[0]),want)
                    self.assertEqual(int(delta[0]),hybrid[2]-int(y[0,2]))

    def test_influence_rejects_unmatched_boolean_oracle(self):
        r=torch.tensor([[1,0,0]]);d=torch.tensor([[0,0,1]])
        with self.assertRaisesRegex(ValueError,'contradicts'):
            core.signed_goal_difference(r,d,torch.tensor([0]),torch.tensor([0]))

    def test_causal_cut_carries_actual_q_codes_and_other_patch_control(self):
        model=GraphLooper(2)
        cases=self.train.cases
        rx=self.train.get([0,1]);donor,y=self.train.pair([0,1],[0,1])
        out=core.replay(model,rx,donor,torch.tensor([0,1]),4,cut=True)
        self.assertEqual(tuple(out.clean.q_codes.shape),(2,3,2,2))
        self.assertTrue(torch.equal(out.patched.q_codes[:,0],out.patched.q_codes[:,-1]))
        self.assertTrue(torch.equal(out.control.q_codes[:,0],out.control.q_codes[:,-1]))
        protected=out.patched.q_codes.gather(2,
                        torch.tensor([1,0]).view(2,1,1,1).expand(2,3,1,2))
        expected=out.clean.q_codes.gather(2,
                        torch.tensor([1,0]).view(2,1,1,1).expand(2,3,1,2))
        self.assertTrue(torch.equal(protected,expected))

    def test_native_gradient_through_goal_influence(self):
        torch.manual_seed(23)
        m=GraphLooper(2)
        cases=self.train.cases
        chosen=[0,1]
        rx=self.train.get(chosen);dx,y=self.train.pair(chosen,[0,1])
        p=core.replay(m,rx,dx,torch.tensor([0,1]),4,cut=True)
        gates=torch.tensor([int(cases[i].gate=='or') for i in chosen])
        loss,margin,_=core.influence_loss(p.clean.goal,p.patched.goal,rx.targets,y,
                                          torch.tensor([0,1]),gates)
        (loss+margin).backward()
        self.assertGreater(sum(float(w.grad.square().sum()) for w in m.cell.parameters()
             if w.grad is not None),0.)
        self.assertTrue(all(torch.isfinite(w.grad).all() for w in m.cell.parameters()
             if w.grad is not None))

    def test_fixed_fitting_config_rejects_mismatched_budget(self):
        with self.assertRaises(ValueError):
            ex.train('/nonexistent',Path('/tmp/af11-should-not-exist'),17,adapt=3,steps=4,smoke=False)

    def test_boolean_logit_and_probability_independent_arithmetic(self):
        clean=torch.tensor([[3.,-2.],[0.,2.]])
        patch=torch.tensor([[-2.,3.],[2.,0.]])
        delta=core.soft_signed_delta(clean,patch)
        self.assertGreater(float(delta[0]),.9)
        self.assertLess(float(delta[1]),-.5)
        self.assertTrue(verify.equal_logits([1.,2.],[1.0000001,2.]))
        self.assertFalse(verify.equal_logits([1.,2.],[1.,3.]))

    def test_source_zip_pinned_and_validated(self):
        if os.getenv('AF11_TEST_SOURCE_ZIP'):
            seed=int(os.environ.get('AF11_TEST_SEED','17'))
            _,m,sha,*_=ex.check_source(os.environ['AF11_TEST_SOURCE_ZIP'],seed)
            self.assertEqual(sha,ex.SOURCE[str(seed)])
            self.assertTrue(all(torch.isfinite(w).all() for w in m.state_dict().values()))

    @unittest.skipUnless(os.getenv('AF11_RUN_SMOKE'),'Engineering smoke selected explicitly')
    def test_end_to_end_training_and_independent_replay(self):
        source=os.environ['AF11_TEST_SOURCE_ZIP']
        with tempfile.TemporaryDirectory() as d:
            td=Path(d)
            receipt=ex.train(source,td/'train',17,adapt=2,steps=2,smoke=True)
            self.assertEqual(len(receipt['arms']),4)
            score=ex.evaluate(td/'train',td/'eval',17,smoke=True)
            audit=verify.audit(td/'train',td/'eval',17,td/'eval'/'audit.json',smoke=True)
            self.assertTrue(audit['verified'])
            self.assertEqual(score['counterfactual_rows'],224*2*4)
            # Mutating saved goal logits and recomputing file digests must fail.
            records=(td/'eval'/'counterfactuals.jsonl').read_text().splitlines()
            row=json.loads(records[0]);row['patch_goal_logits'][0]+=2
            records[0]=json.dumps(row)
            ex.jsonl_write(td/'eval'/'counterfactuals.jsonl',[json.loads(x) for x in records])
            desc=json.loads((td/'eval'/'evaluation.json').read_text())
            desc['counterfactual_sha256']=ex.sha_file(td/'eval'/'counterfactuals.jsonl')
            ex.json_write(td/'eval'/'evaluation.json',desc)
            with self.assertRaises(ValueError):
                verify.audit(td/'train',td/'eval',17,td/'eval'/'audit2.json',smoke=True)

if __name__=='__main__':unittest.main()

"""AF-13 source, asynchronous oracle, differentiable losses and replay guards."""
from __future__ import annotations
import json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from experiments.activation_funnel_af05 import gym
from experiments.activation_funnel_af05.model import GraphLooper
from experiments.activation_funnel_af10.core import latch_suffix
from experiments.activation_funnel_af11.core import replay
from experiments.activation_funnel_af13 import core,data,experiment as ex,verify

class AF13Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_new_prospective_roster_and_manifest(self):
        self.assertEqual(tuple(len(data.cases_for(x)) for x in data.FRESH),(128,16,256,256))
        self.assertEqual(ex.verify_manifest(),ex.sha(ex.MANIFEST))
        self.assertEqual(data.LAYOUTS['fresh_depth4'],(20,21))

    def test_independent_bounds_match_oracle_and_physical_order(self):
        for split in ('train','dev')+data.FRESH:
            for c in data.Dataset(split).cases[::31]:
                a,b=core.closure_bounds(c)
                ia,ib=verify.independent_bounds(c)
                self.assertEqual({k for k,v in enumerate(a.tolist()) if v},ia)
                self.assertEqual({k for k,v in enumerate(b.tolist()) if v},ib)
                pos,_=data.physical_permutations(c.permutation)
                target=gym.oracle(c)[0]
                self.assertEqual(tuple(int(b[pos[q]]) for q in gym.SUBGOALS),target[:2])

    def test_delayed_legal_clock_does_not_need_wave_label(self):
        c=data.Dataset('train').cases[0]
        f,l=core.closure_bounds(c)
        # Repeating initial facts for arbitrary legal stutters, then jumping
        # to the final closure: allowed by endpoint/fact/outside envelope.
        p=torch.stack([f.float(),f.float(),l.float(),l.float()])[None]
        logits=torch.stack([1-p,p],-1)*8-4
        z=core.admissible_status(logits,[c])
        self.assertEqual(z['unsupported_true_count'],0)
        self.assertEqual(z['missing_fact_count'],0)
        self.assertEqual(z['monotonicity_violation_count'],0)
        self.assertEqual(z['end_atom_accuracy'],1.0)
        self.assertEqual(z['fully_admissible_fraction'],1.0)

    def test_forbidden_false_claims_and_erasure_detected(self):
        c=next(x for x in data.Dataset('train').cases if x.facts[0])
        f,l=core.closure_bounds(c)
        p=l.float()[None,None,:].repeat(1,4,1)
        missed=int(torch.where(f)[0][0]);p[:,1,missed]=0
        false_node=int(torch.where(~l)[0][0]);p[:,2,false_node]=1
        logits=torch.stack([1-p,p],-1)*8-4
        z=core.admissible_status(logits,[c])
        self.assertGreater(z['missing_fact_count'],0)
        self.assertGreater(z['unsupported_true_count'],0)
        self.assertGreater(z['monotonicity_violation_count'],0)

    def test_soft_support_gradient(self):
        c=data.Dataset('train').cases[0]
        p=torch.full((1,4,gym.ATOMS),0.25,requires_grad=True)
        loss=core.rule_support_loss(p,[c]);loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(p.grad.abs().sum()),0)

    def test_losses_different_but_all_backpropagate(self):
        m=GraphLooper(2);ds=data.Dataset('train');cases=list(ds.cases[:2]);rx=ds.get([0,1])
        tr=core.trace_upstream(m,rx,6)
        self.assertEqual(tuple(tr.logits.shape),(2,6,15,2))
        self.assertEqual(tuple(core.trace_upstream(m,rx,12).logits.shape),(2,12,15,2))
        term,_=core.asynchronous_loss(tr,cases,'terminal')
        relay,_=core.asynchronous_loss(tr,cases,'relay_final')
        async_loss,terms=core.asynchronous_loss(tr,cases,'async_sound')
        self.assertEqual(float(term.detach()),0)
        self.assertGreater(float(relay.detach()),0)
        self.assertGreater(float(async_loss.detach()),0)
        self.assertGreaterEqual(float(terms['support_penalty']),0)
        async_loss.backward()
        self.assertGreater(sum(float(p.grad.square().sum()) for p in m.cell.parameters() if p.grad is not None),0)

    def test_permutation_equivariance_on_fresh_layouts(self):
        c=data.cases_for('fresh_depth4')[0]
        alt=gym.Case('fresh_depth4',c.depth,c.kinds,c.gate,c.facts,21)
        m=GraphLooper(2)
        with torch.no_grad():
            a=core.trace_upstream(m,data.stack([c]),8).logits[0]
            b=core.trace_upstream(m,data.stack([alt]),8).logits[0]
        amap,_=data.physical_permutations(c.permutation)
        bmap,_=data.physical_permutations(alt.permutation)
        self.assertTrue(torch.allclose(a[:,list(amap)],b[:,list(bmap)],atol=3e-5,rtol=0))

    def test_structural_cut_and_untouched_subgoal(self):
        ds=data.Dataset('train');m=GraphLooper(2)
        rx=ds.get([0,1]);donor,_=ds.pair([0,1],[0,1]);selected=torch.tensor([0,1])
        paired=replay(m,rx,donor,selected,4,cut=True)
        for i in (0,1):
            other=1-int(selected[i]);z=paired.patched.q_codes[i,:,other]
            self.assertTrue(torch.equal(z,z[0].expand_as(z)))

    def test_eligibility_requires_both_split_and_joint_q(self):
        from experiments.activation_funnel_af13.aggregate import qualified
        m={a:{s:{'goal_accuracy':1.,'joint_q_accuracy':1.} for s in ('train','dev')} for a in ex.ARMS}
        self.assertTrue(qualified(m,'async_sound'))
        m['async_sound']['dev']['goal_accuracy']=.89
        self.assertFalse(qualified(m,'async_sound'))

    def test_smoke_and_independent_rehash_tamper_detection(self):
        parent=os.environ.get('AF13_TEST_SOURCE_ZIP','/mnt/data/af06-s17-w2-training.zip')
        if not Path(parent).exists():self.skipTest('Exact qualified AF06 artifact ZIP unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);tr=p/'training';ev=p/'eval'
            ex.train(parent,tr,17,steps=2,smoke=True)
            ex.evaluate(tr,ev,17,smoke=True)
            audit=verify.audit(tr,ev,17,p/'audit.json',smoke=True)
            self.assertTrue(audit['passed'])
            self.assertEqual(audit['clean_rows'],3*224)
            f=ev/'clean.jsonl';rows=[json.loads(x) for x in f.read_text().splitlines()]
            rows[0]['async_stage_logits'][0][0][1]+=8.
            f.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in rows))
            meta=json.loads((ev/'evaluation.json').read_text());meta['clean_sha256']=ex.sha(f)
            ex.write_json(ev/'evaluation.json',meta)
            with self.assertRaisesRegex(ValueError,'scoring mismatch|source-bound clean forward'):
                verify.audit(tr,ev,17,p/'tampered.json',smoke=True)

if __name__=='__main__':unittest.main()

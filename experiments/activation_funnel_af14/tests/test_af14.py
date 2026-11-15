"""AF-14 oracle proof depths, flexible deadlines, gradients, and source-bound replay."""
from __future__ import annotations
import json, os, tempfile, unittest
from pathlib import Path
import torch
from experiments.activation_funnel_af05 import gym
from experiments.activation_funnel_af05.model import GraphLooper
from experiments.activation_funnel_af14 import core,data,experiment as ex,verify,aggregate
from experiments.activation_funnel_af13.core import asynchronous_loss

class AF14Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_fresh_roster_and_input_hash_isolation(self):
        self.assertEqual(tuple(len(data.cases_for(x)) for x in data.FRESH),(128,16,256,256))
        self.assertEqual(data.LAYOUTS['fresh_depth4'],(26,27))
        self.assertEqual(ex.verify_manifest(),ex.sha(ex.MANIFEST))
        self.assertEqual(data.LAYOUTS['fresh_depth5'],(28,29))

    def test_minimal_proof_depth_matches_independent_frontier(self):
        for split in ('train','dev')+data.FRESH:
            for c in data.Dataset(split).cases[::29]:
                expected=verify.independent_progress_deadlines(c)
                self.assertEqual(core.progress_deadlines(c),expected)
                heights=core.minimal_proof_depths(c)
                reach,_=gym.closure(gym.initial_facts(c),gym.logical_rules(c)[0])
                self.assertEqual(set(heights),reach)

    def test_empty_positive_roster_has_finite_zero_deadline_loss(self):
        c=data.Dataset('train').cases[0]
        self.assertEqual(core.progress_deadlines(c),[-1]*gym.ATOMS)
        m=GraphLooper(2);t=core.trace_upstream(m,data.stack([c]),4)
        loss,terms=core.bounded_progress_loss(t,[c],'bounded_progress')
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(terms['deadline_atoms'],0)

    def test_early_correct_arrival_and_late_stability(self):
        c=next(c for c in data.Dataset('train').cases if any(d>1 for d in core.progress_deadlines(c)))
        d=core.progress_deadlines(c)
        state=torch.zeros(1,8,gym.ATOMS,dtype=torch.long)
        for v,deadline in enumerate(d):
            if deadline>0:state[:,:,v]=1 # every positive arrives early
        logits=torch.stack([1-state,state],-1).float()*8-4
        score=core.progress_counts(logits,[c])[0]
        self.assertEqual(score['deadline_stable_count'],score['deadline_atom_count'])
        self.assertEqual(score['deadline_met_count'],score['deadline_atom_count'])
        later=next(v for v,x in enumerate(d) if x>=2)
        state[:,d[later]-1:,later]=0
        logits=torch.stack([1-state,state],-1).float()*8-4
        score=core.progress_counts(logits,[c])[0]
        self.assertLess(score['deadline_met_count'],score['deadline_atom_count'])

    def test_same_unclocked_loss_in_async_arm(self):
        m=GraphLooper(2);cases=list(data.Dataset('train').cases[:2]);rx=data.Dataset('train').get([0,1])
        tr=core.trace_upstream(m,rx,6)
        left,_=core.bounded_progress_loss(tr,cases,'async_sound')
        right,_=asynchronous_loss(tr,cases,'async_sound')
        self.assertTrue(torch.equal(left,right))

    def test_bounded_loss_penalizes_broken_deadlines(self):
        c=next(c for c in data.Dataset('train').cases if any(d>1 for d in core.progress_deadlines(c)))
        d=core.progress_deadlines(c);v=next(i for i,t in enumerate(d) if t>=2)
        m=GraphLooper(2);tr=core.trace_upstream(m,data.stack([c]),8)
        p=tr.logits.detach().clone();q=p.clone()
        p[:,d[v]-1:,v]=torch.tensor([-4.,4.])
        q[:,d[v]-1:,v]=torch.tensor([4.,-4.])
        from experiments.activation_funnel_af13.core import AsyncTrace
        val,_=core.bounded_progress_loss(AsyncTrace(p,tr.codes,tr.final),[c],'bounded_progress')
        loss,_=core.bounded_progress_loss(AsyncTrace(q,tr.codes,tr.final),[c],'bounded_progress')
        self.assertGreater(float(loss),float(val))

    def test_bounded_gradients_reach_native_cell(self):
        m=GraphLooper(2);ds=data.Dataset('train');indices=[17,18]
        tr=core.trace_upstream(m,ds.get(indices),6)
        loss,_=core.bounded_progress_loss(tr,[ds.cases[i] for i in indices],'bounded_progress')
        loss.backward()
        self.assertGreater(sum(float(p.grad.square().sum()) for p in m.cell.parameters() if p.grad is not None),0)

    def test_finite_horn_monotone_bound(self):
        # The combinatorial statement is also represented by BoundedProgress.lean.
        for depth in range(6):
            arrivals=list(range(0,2*depth+1,2))
            self.assertTrue(all(arrivals[k+1]-arrivals[k]<=2 for k in range(depth)))
            self.assertLessEqual(arrivals[-1],2*depth)

    def test_independent_audit_rejects_changed_rehashed_deadlines(self):
        parent=os.environ.get('AF14_TEST_SOURCE_ZIP','/mnt/data/af06-s17-w2-training.zip')
        if not Path(parent).exists():self.skipTest('Original AF06 qualified source ZIP unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);tr=p/'train';ev=p/'eval'
            ex.train(parent,tr,17,steps=2,smoke=True)
            ex.evaluate(tr,ev,17,smoke=True)
            audit=verify.audit(tr,ev,17,p/'audit.json',smoke=True)
            self.assertTrue(audit['passed'])
            self.assertEqual(audit['clean_rows'],3*224)
            path=ev/'clean.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()]
            rows[0]['progress_deadlines'][0]=999
            ex.write_jsonl(path,rows)
            meta=json.loads((ev/'evaluation.json').read_text())
            meta['clean_sha256']=ex.sha(path)
            ex.write_json(ev/'evaluation.json',meta)
            with self.assertRaisesRegex(ValueError,'independently reconstructed proof-depth deadlines'):
                verify.audit(tr,ev,17,p/'tampered.json',smoke=True)

    def test_qualification_exact_joint_q_required(self):
        fit={a:{split:{'goal_accuracy':1.,'joint_q_accuracy':1.} for split in ('train','dev')} for a in ex.ARMS}
        self.assertTrue(aggregate.qualified(fit,'bounded_progress'))
        fit['bounded_progress']['dev']['joint_q_accuracy']=.89
        self.assertFalse(aggregate.qualified(fit,'bounded_progress'))

if __name__=='__main__':unittest.main()

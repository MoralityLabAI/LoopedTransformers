"""Causal-cut invariants with the real source-hash-verified TypedLoop."""
import unittest
import os
import tempfile
from pathlib import Path
import json
import torch
from experiments.activation_funnel_af05 import gym
from experiments.activation_funnel_af05.model import GraphLooper
from experiments.activation_funnel_af05.experiment import Dataset
from experiments.activation_funnel_af10.core import (
    latch_suffix, paired, exact_prefix_factorization, cross_subgoal_jacobian)

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.ds=Dataset('train')
        cls.model=GraphLooper(2)
        cls.model.eval()
        cls.b=cls.ds.get([0,1])

    def test_actual_code_and_logit_are_bitwise_latched_across_five_steps(self):
        model=self.model;b=self.b;pre=model.prefix(b,4)
        trace=latch_suffix(model,b,pre,5)
        expected=pre.az.gather(1,b.read_slots[:,:2].unsqueeze(-1).expand(-1,-1,2))
        for t in range(5):
            self.assertTrue(torch.equal(trace.q_codes[:,t],expected))
            self.assertTrue(torch.equal(trace.q_steps[:,t],trace.initial_q))

    def test_protected_subgoal_is_invariant_under_addressed_and_negative_patch(self):
        ds=self.ds;rx=ds.get([0,1]);donor,truth=ds.pair([0,1],[0,1])
        x=paired(self.model,rx,donor,torch.tensor([0,1]),4,5,cut=True)
        rows=exact_prefix_factorization(x,rx.targets,truth)
        self.assertEqual(float(rows['protected_code_max_difference_by_step'].detach().abs().max()),0.)
        for i in range(2):
            j=1-i
            self.assertTrue(torch.equal(x.clean.q_steps[i,:,j],x.patched.q_steps[i,:,j]))

    def test_selected_donor_and_recipient_q_codes_remain_distinct_when_changed(self):
        ds=self.ds;rx=ds.get([0,1]);donor,_=ds.pair([0,1],[0,1])
        x=paired(self.model,rx,donor,torch.tensor([0,1]),4,3)
        for i in range(2):
            j=i;slot=int(rx.read_slots[i,j])
            self.assertTrue(torch.equal(x.patched.q_codes[i,0,j],x.prefix_donor.az[i,slot]))

    def test_latch_keeps_goal_path_differentiable(self):
        ds=self.ds;rx=ds.get([0,1]);donor,_=ds.pair([0,1],[0,1])
        model=GraphLooper(2)
        result=paired(model,rx,donor,torch.tensor([0,1]),4,3,cut=True)
        result.patched.goal.square().mean().backward()
        grad=sum(float(p.grad.square().sum()) for p in model.cell.parameters() if p.grad is not None)
        self.assertGreater(grad,0)

    def test_source_has_measurable_cross_subgoal_jacobian(self):
        torch.manual_seed(12)
        jac=cross_subgoal_jacobian(GraphLooper(2),self.ds.get([0]),4,0)
        self.assertEqual(jac.shape,(2,2))
        self.assertTrue(torch.isfinite(jac).all())
        self.assertGreater(float(jac.norm()),1e-10)

    def test_invalid_loop_and_duplicate_slots_rejected(self):
        pre=self.model.prefix(self.b,4)
        with self.assertRaises(ValueError):latch_suffix(self.model,self.b,pre,4)
        b=self.ds.get([0,1]);b.read_slots[:,1]=b.read_slots[:,0]
        with self.assertRaises(ValueError):latch_suffix(self.model,b,pre,3)


    def test_retained_affine_replay_rejects_rehashed_semantic_tamper(self):
        source=os.environ.get('AF10_TEST_SOURCE_ZIP')
        if not source:self.skipTest('AF-09 pinned source zip needed for integration test')
        from experiments.activation_funnel_af10.diagnostic import run
        from experiments.activation_funnel_af10.verify import check_rows
        from experiments.activation_funnel_af05.experiment import sha_file
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/'scored'
            report=run(Path(source),17,folder,splits=('dev',),chunk=64)
            verified=check_rows(Path(source),17,folder,replay=True)
            self.assertTrue(verified['verified'])
            rows=(folder/'cut_replay.jsonl').read_text().splitlines()
            first=json.loads(rows[0])
            first['cut_patch_q_logits'][1-first['selected']][0]+=0.25
            rows[0]=json.dumps(first,sort_keys=True)
            (folder/'cut_replay.jsonl').write_text('\n'.join(rows)+'\n')
            report['row_sha256']=sha_file(folder/'cut_replay.jsonl')
            (folder/'summary.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
            (folder/'summary.sha256').write_text(sha_file(folder/'summary.json')+'\n')
            with self.assertRaisesRegex(ValueError,'affine cut readout mismatch'):
                check_rows(Path(source),17,folder,replay=False)

if __name__=='__main__':unittest.main()

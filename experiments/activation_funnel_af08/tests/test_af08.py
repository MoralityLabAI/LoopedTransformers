"""AF-08 source integrity, noncollapse, intervened transport and independent replay."""
from __future__ import annotations
import json
import os
from pathlib import Path
import tempfile
import unittest

import torch

from experiments.activation_funnel_af05 import gym
from experiments.activation_funnel_af05.model import GraphLooper
from experiments.activation_funnel_af05.experiment import Dataset
from experiments.activation_funnel_af07.core import paired_traces
from experiments.activation_funnel_af08 import core
from experiments.activation_funnel_af08 import experiment as ex
from experiments.activation_funnel_af08 import verify


class AF08Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_parent_corpus_is_exactly_frozen(self):
        self.assertEqual(ex.validate_manifest(), ex.sha_file(ex.PINNED_MANIFEST))
        self.assertEqual(len(gym.cases_for('train')),672)
        self.assertEqual(len(gym.cases_for('dev')),224)
        self.assertEqual(sum(len(gym.cases_for(x)) for x in ex.EVAL_LOOPS if x.startswith('held_')),496)

    def test_qualified_gate_requires_both_train_dev_and_joint_q(self):
        perfect={'train':{'goal_accuracy':1.,'subgoal_exact':1.},
                 'dev':{'goal_accuracy':1.,'subgoal_exact':1.}}
        self.assertTrue(core.qualified(perfect))
        perfect['dev']['subgoal_exact']=.89
        self.assertFalse(core.qualified(perfect))

    def test_noncollapse_margin_rejects_stable_degenerate_code(self):
        truth=torch.tensor([[0,0],[1,1],[0,1],[1,0]])
        collapsed=torch.ones(4,3,2,2,requires_grad=True)
        a=core.class_separation(collapsed,truth,margin=.45)
        self.assertGreater(float(a.detach()),.15)
        a.backward()
        self.assertIsNotNone(collapsed.grad)
        separated=collapsed.detach().clone()
        separated[:, :, 0, 0]=truth[:,0,None]*2.
        separated[:, :, 1, 0]=truth[:,1,None]*2.
        self.assertEqual(float(core.class_separation(separated,truth,margin=.45)),0.)
        self.assertEqual(float(core.class_separation(torch.ones(2,3,2,2),torch.zeros(2,2,dtype=torch.long))),0.)

    def test_changed_truth_patch_noncollapse_penalty(self):
        g=torch.zeros(2,2); q=torch.zeros(2,2,2)
        clean=type('Trace',(),{'goal':g,'q_codes':torch.zeros(2,3,2,2)})()
        patch=type('Trace',(),{'goal':g,'q_codes':torch.zeros(2,3,2,2)})()
        t=core.Triple(clean,patch,patch,torch.tensor([True,False]))
        select=torch.tensor([0,0]);rx=torch.tensor([[0,1,0],[0,1,0]])
        dy=torch.tensor([[1,1,1],[1,1,1]])
        self.assertGreater(float(core.selected_code_change(t,select,rx,dy)),0.)
        t.addressed[:]=False
        self.assertEqual(float(core.selected_code_change(t,select,rx,dy)),0.)

    def test_paired_triple_matches_unmodified_parent_patch(self):
        torch.manual_seed(101)
        model=GraphLooper(2)
        ds=Dataset('train')
        receiver=ds.get([0,1]);donor,hybrid=ds.pair([0,1],[0,1])
        selected=torch.tensor([0,1]);active=torch.tensor([True,False])
        t=core.triple_traces(model,receiver,donor,selected,active,4)
        original=model.patched(receiver,donor,selected,condition='addressed',pre_loops=4)
        irrelevant=model.patched(receiver,donor,selected,condition='other',pre_loops=4)
        self.assertTrue(torch.allclose(t.patched.q[0],original.subgoals[0],atol=2e-5,rtol=0))
        self.assertTrue(torch.allclose(t.patched.q[1],irrelevant.subgoals[1],atol=2e-5,rtol=0))
        self.assertTrue(torch.allclose(t.control.goal,irrelevant.goal,atol=2e-5,rtol=0))

    def test_loss_backpropagates_into_real_typedloop_and_code(self):
        torch.manual_seed(127)
        model=GraphLooper(2)
        ds=Dataset('train');rx=ds.get([0,1,2,3]);donor,hybrid=ds.pair([0,1,2,3],[0,1,0,1])
        anchor=[x.detach().clone() for x in model.parameters()]
        loss,parts=ex.loss_for(model,anchor,rx,donor,torch.tensor([0,1,0,1]),
                     hybrid,torch.ones(4,dtype=torch.bool),4,'noncollapse')
        loss.backward()
        cell=sum(float(p.grad.abs().sum()) for p in model.cell.parameters() if p.grad is not None)
        codec=sum(float(p.grad.abs().sum()) for p in model.codec.parameters() if p.grad is not None)
        self.assertGreater(cell,0.)
        self.assertGreater(codec,0.)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn('separation',parts)

    def test_equal_code_centroids_and_zero_delta_have_finite_gradients(self):
        # Original sqrt(sum(square)) produced NaN backward at a coincident code.
        # vector_norm defines a zero gradient at zero distance and preserves the
        # preregistered Euclidean distance loss value away from that singularity.
        truth=torch.tensor([[0,0],[1,1]],dtype=torch.long)
        c=torch.zeros(2,3,2,2,requires_grad=True)
        separated=core.class_separation(c,truth)
        separated.backward()
        self.assertTrue(torch.isfinite(c.grad).all())
        self.assertGreater(float(separated.detach()),0.)
        g=torch.zeros(1,2);a=torch.zeros(1,3,2,2,requires_grad=True)
        b=torch.zeros(1,3,2,2,requires_grad=True)
        x=type('Trace',(),{'goal':g,'q_codes':a})()
        y=type('Trace',(),{'goal':g,'q_codes':b})()
        t=core.Triple(x,y,y,torch.ones(1,dtype=torch.bool))
        loss=core.selected_code_change(t,torch.tensor([0]),
                         torch.tensor([[0,1,0]]),torch.tensor([[1,1,1]]))
        loss.backward()
        self.assertTrue(torch.isfinite(a.grad).all())
        self.assertTrue(torch.isfinite(b.grad).all())
        self.assertGreater(float(loss.detach()),0.)

    def test_source_artifact_hash_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'corrupt.zip';path.write_bytes(b'not original')
            with self.assertRaisesRegex(ValueError,'checksum'):
                ex.source_from_zip(path,17)

    def test_source_identical_weight_hashes_and_qualified_warmup(self):
        path=os.environ.get('AF08_TEST_SOURCE_ZIP')
        if not path:self.skipTest('set AF08_TEST_SOURCE_ZIP to checked AF-06 s17 width2 artifact')
        receipt,model,_,_,_=ex.source_from_zip(path,17)
        self.assertTrue(core.qualified(receipt['arms']['warmup']['fit_metrics']))
        self.assertEqual(len(model.cell.state_dict()),len(GraphLooper(2).cell.state_dict()))

    def test_end_to_end_smoke_and_rehashed_mutation(self):
        path=os.environ.get('AF08_TEST_SOURCE_ZIP')
        if not path:self.skipTest('requires original AF-06 seed17 parent ZIP')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            train=root/'train';scored=root/'evaluation'
            rec=ex.train_one(path,train,17,steps=2,smoke=True)
            self.assertEqual(rec['training_rows'],3*896)
            self.assertEqual({k:rec['arms'][k]['steps'] for k in ex.ARMS},dict.fromkeys(ex.ARMS,2))
            entry=ex.evaluate_one(train,scored,17,smoke=True)
            self.assertEqual(entry['clean_rows'],3*224)
            self.assertEqual(entry['patch_rows'],6*224)
            report=verify.verify_one(train,scored,17,smoke=True,replay_cases=1)
            self.assertTrue(report['verified'])
            rows=verify.read_jsonl(scored/'stage_predictions.jsonl')
            rows[0]['patched_q_steps'][0][0][0]+=0.4
            ex.jsonl_write(scored/'stage_predictions.jsonl',rows)
            evaluation=json.loads((scored/'evaluation.json').read_text())
            evaluation['files']['stage_predictions.jsonl']=ex.sha_file(scored/'stage_predictions.jsonl')
            ex.json_write(scored/'evaluation.json',evaluation)
            (scored/'evaluation.sha256').write_text(ex.sha_file(scored/'evaluation.json')+'\n')
            with self.assertRaisesRegex(ValueError,'stage|replay|semantic|readout'):
                verify.verify_one(train,scored,17,smoke=True,replay_cases=1)


if __name__=='__main__':unittest.main()

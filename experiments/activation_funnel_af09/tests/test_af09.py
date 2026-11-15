"""AF-09 exact semantic margin, source binding, training gradient and replay."""
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
from experiments.activation_funnel_af09 import core
from experiments.activation_funnel_af09 import experiment as ex
from experiments.activation_funnel_af09 import verify


class AF09Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_original_data_and_parent_models_are_immutable(self):
        self.assertEqual(ex.validate_manifest(),ex.sha_file(ex.PINNED_MANIFEST))
        self.assertEqual({s:len(gym.cases_for(s)) for s in gym.SPLITS},
                         {'train':672,'dev':224,'held_composition':16,'held_layout':224,
                          'held_depth4':128,'held_depth5':128})
        self.assertEqual(ex.ARMS,('status','locality','semantic_margin'))
        self.assertEqual(ex.STEPS,160)

    def test_mathematical_affine_certificate_is_exact_and_sharp(self):
        torch.manual_seed(4009)
        model=GraphLooper(2)
        ds=Dataset('train')
        receiver=ds.get([0,1,2,3]);donor,hybrid=ds.pair([0,1,2,3],[0,1,0,1])
        selected=torch.tensor([0,1,0,1]);addressed=torch.ones(4,dtype=torch.bool)
        clean,changed=paired_traces(model,receiver,donor,selected,addressed,4)
        x=core.certificate_tensors(model,clean,changed,selected,receiver.targets)
        k=x['sensitivity'][:,None]*x['displacement']
        self.assertTrue(torch.all(x['patched_margin']>=x['lower_bound']-2e-5))
        self.assertTrue(torch.all(torch.abs(x['patched_margin']-x['clean_margin'])<=k+2e-5))
        saved=core.certificate_rows(model,clean,changed,selected,receiver.targets)
        self.assertEqual(len(saved),4)
        for r in saved:
            self.assertEqual(len(r['clean_code']),3)
            self.assertEqual(len(r['certificate_positive']),3)
            self.assertEqual(sum(c and not y for c,y in zip(r['certificate_positive'],r['patched_correct'])),0)

    def test_constant_codes_do_not_earn_semantic_certificate(self):
        torch.manual_seed(101)
        model=GraphLooper(2)
        with torch.no_grad():
            model.subgoal_head.weight.zero_()
            model.subgoal_head.bias.zero_()
        ds=Dataset('train');rx=ds.get([0,1]);donor,_=ds.pair([0,1],[0,1])
        selected=torch.tensor([0,1]);yes=torch.ones(2,dtype=torch.bool)
        clean,patch=paired_traces(model,rx,donor,selected,yes,4)
        c=core.certificate_tensors(model,clean,patch,selected,rx.targets)
        self.assertTrue(torch.equal(c['lower_bound'],torch.zeros_like(c['lower_bound'])))
        self.assertFalse(any(any(r['certificate_positive']) for r in core.certificate_rows(model,clean,patch,selected,rx.targets)))
        a,b,_=core.semantic_margin_loss(model,clean,patch,selected,rx.targets,yes)
        self.assertGreater(float(a),.8)
        self.assertGreater(float(b),.8)

    def test_zero_drift_has_finite_gradient(self):
        torch.manual_seed(94)
        model=GraphLooper(2)
        ds=Dataset('train');batch=ds.get([0,1]);donor,_=ds.pair([0,1],[0,1])
        selected=torch.tensor([0,1]);yes=torch.ones(2,dtype=torch.bool)
        clean,patch=paired_traces(model,batch,donor,selected,yes,4)
        c=core.certificate_tensors(model,clean,clean,selected,batch.targets)
        self.assertEqual(float(c['displacement'].max()),0.)
        a,b,_=core.semantic_margin_loss(model,clean,clean,selected,batch.targets,yes)
        (a+b).backward()
        self.assertTrue(all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None))

    def test_source_uses_validated_parent_sha(self):
        with tempfile.TemporaryDirectory() as temp:
            bad=Path(temp)/'bad.zip';bad.write_bytes(b'invalid')
            with self.assertRaisesRegex(ValueError,'checksum'):
                ex.source_from_zip(bad,17)
        parent=os.environ.get('AF09_TEST_SOURCE_ZIP')
        if parent:
            receipt,model,h,_,_=ex.source_from_zip(parent,17)
            self.assertTrue(core.qualified(receipt['arms']['warmup']['fit_metrics']))
            self.assertEqual(h,ex.SOURCE_ARTIFACTS[17]['sha256'])

    def test_new_margin_gradient_reaches_original_native_cell(self):
        torch.manual_seed(107)
        model=GraphLooper(2)
        ds=Dataset('train');rx=ds.get([0,1,2,3]);donor,hybrid=ds.pair([0,1,2,3],[0,1,0,1])
        anchor=[x.detach().clone() for x in model.parameters()]
        loss,parts=ex.loss_for(model,anchor,rx,donor,torch.tensor([0,1,0,1]),
                      hybrid,torch.ones(4,dtype=torch.bool),4,'semantic_margin')
        self.assertGreater(parts['certified_margin_loss'],0.)
        loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.cell.parameters() if p.grad is not None),0.)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.codec.atom.decoder.parameters() if p.grad is not None),0.)
        self.assertTrue(torch.isfinite(loss))

    def test_addressed_patch_matches_frozen_parent_semantics(self):
        torch.manual_seed(112)
        model=GraphLooper(2);ds=Dataset('train')
        rx=ds.get([0,1]);donor,hybrid=ds.pair([0,1],[0,1])
        pick=torch.tensor([0,1]);yes=torch.ones(2,dtype=torch.bool)
        trio=core.triple_traces(model,rx,donor,pick,yes,4)
        original=model.patched(rx,donor,pick,condition='addressed',pre_loops=4)
        self.assertTrue(torch.allclose(trio.patched.q,original.subgoals,atol=2e-5,rtol=0))
        self.assertTrue(torch.allclose(trio.patched.goal,original.goal,atol=2e-5,rtol=0))

    def test_ramp_zero_mask_gets_no_extra_loss(self):
        torch.manual_seed(223)
        model=GraphLooper(2);ds=Dataset('train');rx=ds.get([0,1]);donor,_=ds.pair([0,1],[0,1])
        selected=torch.tensor([0,1]);off=torch.zeros(2,dtype=torch.bool)
        t=core.triple_traces(model,rx,donor,selected,off,4)
        a,b,_=core.semantic_margin_loss(model,t.clean,t.patched,selected,rx.targets,off)
        self.assertEqual(float(a),0.);self.assertEqual(float(b),0.)

    def test_end_to_end_smoke_and_rehash_tampering(self):
        parent=os.environ.get('AF09_TEST_SOURCE_ZIP')
        if not parent:self.skipTest('requires pinned AF-06 seed17 ZIP')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);train=root/'train';scored=root/'scored'
            rec=ex.train_one(parent,train,17,steps=2,smoke=True)
            self.assertEqual(rec['training_rows'],3*896)
            self.assertEqual(set(rec['arms']),set(ex.ARMS))
            result=ex.evaluate_one(train,scored,17,smoke=True)
            self.assertEqual(result['clean_rows'],3*224)
            self.assertEqual(result['patch_rows'],6*224)
            self.assertEqual(result['margin_rows'],6*224)
            audit=verify.verify_one(train,scored,17,smoke=True,replay_cases=1)
            self.assertTrue(audit['verified'])
            rows=verify.read_jsonl(scored/'margin_predictions.jsonl')
            rows[0]['certified_lower_bound'][0]+=0.6
            ex.jsonl_write(scored/'margin_predictions.jsonl',rows)
            metadata=json.loads((scored/'evaluation.json').read_text())
            metadata['files']['margin_predictions.jsonl']=ex.sha_file(scored/'margin_predictions.jsonl')
            ex.json_write(scored/'evaluation.json',metadata)
            (scored/'evaluation.sha256').write_text(ex.sha_file(scored/'evaluation.json')+'\n')
            with self.assertRaisesRegex(ValueError,'margin|certificate|affine|signed'):
                verify.verify_one(train,scored,17,smoke=True,replay_cases=1)

if __name__=='__main__':unittest.main()

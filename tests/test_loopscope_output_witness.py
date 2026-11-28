"""Controls for exact-current-output intervention assay; random weights only."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from research_gym.scripts.loopscope_toy import LoopedChase, make_batch
from research_gym.scripts.loopscope_semantic_echo import native_trace, rollout, load_model
from research_gym.scripts.loopscope_output_witness import (
    audit_pair, check_close, edge_patch, evaluate, json_sha, main,
    probabilities, resolve_checkpoints, select_pairs, summarize, tv,
)


class OutputWitnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        cls.model = LoopedChase(n_nodes=4,d=16,layers=1,heads=2).double().eval()
        cls.tokens, cls.oracle = make_batch(4,3,51,12)
        with torch.inference_mode():
            cls.embedding, cls.states = native_trace(cls.model,cls.tokens,12)

    def test_native_trace_matches_original_forward(self):
        with torch.inference_mode():
            expected = self.model(self.tokens,12)
            torch.testing.assert_close(self.states[1:,:,-1],expected,rtol=1e-12,atol=1e-12)

    def test_edge_patch_keeps_all_current_logits_exact(self):
        base,donor = self.states[4,0:1],self.states[1,0:1]
        for kind in ('edge_donor','edge_random','edge_layernorm_gauge','sham'):
            with torch.inference_mode():
                x = edge_patch(base,donor,4,.5,kind,99)
                self.assertTrue(torch.equal(x[:,4:],base[:,4:]))
                self.assertTrue(torch.equal(self.model.readout(x[:,-1]),self.model.readout(base[:,-1])))

    def test_random_and_gauge_norms_match_each_edge_token(self):
        base,donor=self.states[4,0:1],self.states[1,0:1]
        desired=(.5*(donor-base)[:,:4]).norm(dim=-1)
        for kind in ('edge_donor','edge_random','edge_layernorm_gauge'):
            x=edge_patch(base,donor,4,.5,kind,9)
            torch.testing.assert_close((x-base)[:,:4].norm(dim=-1),desired)

    def test_random_control_is_repeatable_and_seed_sensitive(self):
        b,d=self.states[4,0:1],self.states[1,0:1]
        a=edge_patch(b,d,4,1,'edge_random',9)
        self.assertTrue(torch.equal(a,edge_patch(b,d,4,1,'edge_random',9)))
        self.assertFalse(torch.equal(a,edge_patch(b,d,4,1,'edge_random',10)))

    def test_zero_dose_identity(self):
        b,d=self.states[4,0:1],self.states[1,0:1]
        for k in ('edge_donor','edge_random','edge_layernorm_gauge'):
            self.assertTrue(torch.equal(b,edge_patch(b,d,4,0,k,5)))

    def test_layernorm_gauge_vanishes_on_continuation(self):
        b,d=self.states[4,0:1],self.states[1,0:1]
        x=edge_patch(b,d,4,1,'edge_layernorm_gauge',12)
        with torch.inference_mode():
            ref=rollout(self.model,b,self.embedding[0:1],4)
            altered=rollout(self.model,x,self.embedding[0:1],4)
        torch.testing.assert_close(altered,ref,atol=1e-12,rtol=1e-12)

    def test_full_pair_audit_and_replay(self):
        row=audit_pair(self.model,self.states,self.embedding,
                       dict(item=0,t=1,u=4,stratum='training_depth'),self.oracle,
                       horizon=3,random_controls=2,seed=70,rtol=1e-10,atol=1e-12,effect_tv=.01)
        self.assertEqual(len(row['interventions']),13)
        self.assertTrue(all(x['current_logits_bitwise_equal'] for x in row['interventions']))
        sham=row['interventions'][0]
        self.assertEqual(sham['future_tv'],[0.,0.,0.])
        self.assertEqual(row['oracle_future'],self.oracle[0,4:7].tolist())

    @staticmethod
    def artificial_endpoints():
        p=torch.tensor([[[.9,.1]]]*8,dtype=torch.float64)
        oracle=torch.zeros((1,8),dtype=torch.long)
        return p,oracle

    def test_no_correct_endpoint_pairs_gives_empty(self):
        p,o=self.artificial_endpoints()
        o[:]=1
        selected,counts=select_pairs(p,o,horizon=2,training_loops=8,tau=.01,per_item=2,seed=1)
        self.assertEqual(selected,[])
        self.assertEqual(counts[0]['both_correct_pairs'],0)

    def test_distribution_gate_excludes_same_argmax(self):
        p=torch.tensor([[[.9,.1]],[[.6,.4]],[[.8,.2]],[[.7,.3]]],dtype=torch.float64)
        o=torch.zeros((1,4),dtype=torch.long)
        selected,counts=select_pairs(p,o,horizon=1,training_loops=4,tau=.01,per_item=8,seed=1)
        self.assertEqual(selected,[])
        self.assertGreater(counts[0]['both_correct_pairs'],0)

    def test_selection_reproducible_capped_and_stratified(self):
        p,o=self.artificial_endpoints()
        kw=dict(horizon=2,training_loops=5,tau=.01,per_item=2,seed=17)
        a,_=select_pairs(p,o,**kw)
        b,_=select_pairs(p,o,**kw)
        self.assertEqual(a,b)
        self.assertEqual(len(a),4)
        self.assertEqual({x['stratum'] for x in a},{'training_depth','depth_extrapolation'})
        for x in a:
            self.assertEqual(x['stratum']=='training_depth',x['u']+2<=5)

    def test_selection_does_not_use_final_continuation_outputs(self):
        p,o=self.artificial_endpoints()
        kw=dict(horizon=2,training_loops=8,tau=.01,per_item=3,seed=1)
        a,_=select_pairs(p,o,**kw)
        changed=p.clone();changed[-2:]=torch.tensor([.1,.9])
        b,_=select_pairs(changed,o,**kw)
        self.assertEqual(a,b)

    def test_future_only_channel_positive_and_negative_controls(self):
        class Mix(nn.Module):
            def forward(self,h):
                h=h.clone();h[:,-1]+=h[:,0];return h
        class Control(nn.Module):
            def __init__(self,coupled):
                super().__init__();self.inject=nn.LayerNorm(2).double()
                self.core=nn.ModuleList([Mix()] if coupled else [])
            def readout(self,h): return h
        b=torch.tensor([[[1.,0.],[0.,1.]]],dtype=torch.float64)
        d=torch.tensor([[[0.,1.],[0.,1.]]],dtype=torch.float64)
        x=edge_patch(b,d,1,1,'edge_donor',1)
        for coupled in (False,True):
            m=Control(coupled)
            with torch.inference_mode():
                self.assertTrue(torch.equal(m.readout(b[:,-1]),m.readout(x[:,-1])))
                ref=rollout(m,b,torch.zeros_like(b),1)
                alt=rollout(m,x,torch.zeros_like(b),1)
                value=float(tv(probabilities(ref[0,:,-1]),probabilities(alt[0,:,-1])))
            self.assertGreater(value,.1) if coupled else self.assertEqual(value,0)

    def test_checkpoint_absence_never_falls_back_to_random(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):resolve_checkpoints(directories=[directory])
            with self.assertRaises(ValueError):resolve_checkpoints(checkpoints=[directory+'/missing.pt'])
            with self.assertRaises(ValueError):resolve_checkpoints()
            with self.assertRaises(ValueError):resolve_checkpoints(checkpoints=['x'],smoke=True)

    def test_checkpoint_loader_rejects_wrong_task(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'bad.final.pt'
            torch.save({'cfg':{'fixed_depth':False},'model':{}},p)
            with self.assertRaises(ValueError):load_model(p,smoke=False,seed=1,device='cpu',dtype='float64')

    def test_native_checkpoint_round_trip(self):
        cfg=dict(n_nodes=4,d=16,layers=1,heads=2,fixed_depth=True,loops=8)
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'fixture.final.pt'
            torch.save({'cfg':cfg,'model':self.model.state_dict()},p)
            restored,_=load_model(p,smoke=False,seed=9,device='cpu',dtype='float64')
            with torch.inference_mode():
                torch.testing.assert_close(restored(self.tokens,3),self.model(self.tokens,3),atol=2e-6,rtol=2e-5)
            self.assertEqual(resolve_checkpoints(directories=[directory]),[p.resolve()])

    def test_smoke_is_labelled_and_cannot_emit_trained_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            args=argparse.Namespace(smoke=True,seed=7,device='cpu',dtype='float64',items=4,
                max_loops=10,horizon=2,endpoint_tv=.01,pairs_per_item=1,random_controls=1,effect_tv=.01)
            r=evaluate(None,args)
            self.assertEqual(r['status'],'UNTRAINED_SMOKE')
            self.assertIsNone(r['summary']);self.assertIsNone(r['checkpoint_sha256'])
            self.assertFalse(r['engineering_control']['endpoint_gate'])
            json.dumps(r,allow_nan=False)

    def test_empty_summary_is_undefined_not_zero(self):
        r=summarize([],7)
        self.assertIsNone(r['training_depth']['mean_future_tv'])
        self.assertEqual(r['training_depth']['items'],0)

    def test_fail_loud_on_invalid_patch_or_nonfinite(self):
        b,d=self.states[4,0:1],self.states[1,0:1]
        with self.assertRaises(ValueError):edge_patch(b,d,5,1,'edge_donor',1)
        with self.assertRaises(ValueError):edge_patch(b,d,4,1,'bogus',1)
        with self.assertRaises(ValueError):probabilities(torch.tensor([float('nan'),0.]))
        with self.assertRaises(RuntimeError):check_close(b,d,rtol=0,atol=0,name='tampered')

    def test_manifest_and_existing_output_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            argv=['--smoke','--out',directory,'--items','2','--max-loops','6',
                  '--horizon','2','--random-controls','1','--dtype','float64']
            self.assertEqual(main(argv),0)
            manifest=json.loads((Path(directory)/'manifest.json').read_text())
            self.assertTrue(manifest['complete'])
            self.assertEqual(manifest['runs'][0]['status'],'UNTRAINED_SMOKE')
            with self.assertRaises(ValueError):main(argv)


if __name__=='__main__':unittest.main()

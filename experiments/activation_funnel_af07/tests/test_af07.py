"""AF-07 independent mechanism and tamper tests on the frozen Horn corpus."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
import torch

from experiments.activation_funnel_af05 import gym
from experiments.activation_funnel_af05.model import GraphLooper
from experiments.activation_funnel_af05.experiment import Dataset
from experiments.activation_funnel_af06.core import ramp_mask
from experiments.activation_funnel_af07 import aggregate, experiment as ex, verify
from experiments.activation_funnel_af07.core import (paired_traces,patch_prefix,
    trace_suffix,locality_components,trace_metrics,step_status_loss,protect_index)


class AF07Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.data=Dataset('train')
        cls.receiver=cls.data.get([0,1])
        cls.donor,cls.composed=cls.data.pair([0,1],[0,1])
        cls.select=torch.tensor([0,1],dtype=torch.long)
        cls.addressed=torch.tensor([True,False])

    def test_protocol_and_parent_corpus_are_unchanged(self):
        self.assertEqual(ex.validate_manifest(),ex.sha_file(ex.PINNED_MANIFEST))
        protocol=json.loads(Path(ex.__file__).with_name('PROTOCOL.json').read_text())
        self.assertEqual(protocol['experiment'],'AF-07')
        self.assertEqual(protocol['training']['continuation_steps_each'],160)
        self.assertEqual(protocol['evaluation']['primary_width'],2)
        self.assertEqual(ex.ARMS,('status','ramp','locality'))
        self.assertEqual(set(ex.source_receipt()),set(ex.source_receipt()))

    def test_native_trace_matches_original_suffix_readout(self):
        torch.manual_seed(91);model=GraphLooper(2)
        pre=model.prefix(self.receiver,4)
        forward=model(self.receiver,pre_loops=4)
        trace=trace_suffix(model,self.receiver,pre,3)
        self.assertTrue(torch.allclose(trace.goal,forward.goal,atol=2e-5,rtol=0))
        self.assertTrue(torch.allclose(trace.q,forward.subgoals,atol=2e-5,rtol=0))
        self.assertEqual(tuple(trace.q_steps.shape),(2,3,2,2))
        self.assertEqual(tuple(trace.q_codes.shape),(2,3,2,2))
        self.assertTrue(torch.equal(trace.q,trace.q_steps[:,-1]))

    def test_patched_trace_exactly_matches_original_interchange(self):
        model=GraphLooper(2)
        for boolval,condition in ((True,'addressed'),(False,'other')):
            flags=torch.tensor([boolval,boolval])
            _,p=paired_traces(model,self.receiver,self.donor,self.select,flags,4)
            reference=model.patched(self.receiver,self.donor,self.select,condition=condition,pre_loops=4)
            self.assertTrue(torch.allclose(p.goal,reference.goal,atol=2e-5,rtol=0))
            self.assertTrue(torch.allclose(p.q,reference.subgoals,atol=2e-5,rtol=0))
            self.assertTrue(torch.allclose(p.prefix.az,reference.pre.az,atol=2e-5,rtol=0))

    def test_only_selected_token_changes_at_boundary(self):
        m=GraphLooper(1)
        r=m.prefix(self.receiver,4);d=m.prefix(self.donor,4)
        p=patch_prefix(m,self.receiver,self.donor,self.select,self.addressed,r,d)
        for j in (0,1):
            address=self.select[j] if self.addressed[j] else 1-self.select[j]
            slot=int(self.receiver.read_slots[j,address])
            mask=torch.ones(gym.ATOMS,dtype=torch.bool);mask[slot]=False
            self.assertTrue(torch.equal(p.az[j,mask],r.az[j,mask]))
            self.assertTrue(torch.equal(p.az[j,slot],d.az[j,slot]))
            self.assertTrue(torch.equal(p.rz[j],r.rz[j]))

    def test_source_donor_guard_is_host_owned(self):
        model=GraphLooper(2);r=model.prefix(self.receiver,4);d=model.prefix(self.donor,4)
        bad=copy.deepcopy(self.donor)
        bad.rule_features[0,0,0]=123
        with self.assertRaisesRegex(ValueError,'structural'):
            patch_prefix(model,self.receiver,bad,self.select,self.addressed,r,d)

    def test_locality_loss_gradient_reaches_native_loop_and_codec(self):
        torch.manual_seed(12)
        m=GraphLooper(2)
        loss,out=ex.model_loss(m,self.receiver,self.donor,self.select,self.composed,
                               self.addressed,4,'locality')
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        g=sum(float(p.grad.square().sum()) for p in m.cell.parameters() if p.grad is not None)**.5
        h=sum(float(p.grad.square().sum()) for p in m.codec.parameters() if p.grad is not None)**.5
        self.assertGreater(g,0);self.assertGreater(h,0)

    def test_locality_regularizer_sees_only_unaffected_q(self):
        m=GraphLooper(2)
        clean,patched=paired_traces(m,self.receiver,self.donor,self.select,self.addressed,4)
        k,code=locality_components(clean,patched,self.select,self.addressed)
        self.assertTrue(torch.isfinite(k));self.assertTrue(torch.isfinite(code))
        self.assertTrue(torch.equal(protect_index(self.select,self.addressed),torch.tensor([1,1])))
        altered=copy.copy(patched)
        altered.q_steps=patched.q_steps.clone()
        altered.q_codes=patched.q_codes.clone()
        altered.q_steps[:,:,0]+=20
        altered.q_codes[:,:,0]+=20
        newk,newcode=locality_components(clean,altered,self.select,self.addressed)
        self.assertTrue(torch.allclose(newk.detach(),k.detach(),atol=1e-7,rtol=0))
        self.assertTrue(torch.allclose(newcode.detach(),code.detach(),atol=1e-7,rtol=0))
        altered.q_steps[:,:,1]+=torch.tensor([5.,-5.])
        altered.q_codes[:,:,1]+=5
        newk,newcode=locality_components(clean,altered,self.select,self.addressed)
        self.assertGreater(float(newk.detach()),float(k.detach())+1e-3)
        self.assertGreater(float(newcode.detach()),float(code.detach())+1e-2)

    def test_extra_recurrence_reuses_exact_oracle(self):
        m=GraphLooper(2)
        c,p=paired_traces(m,self.receiver,self.donor,self.select,
                          torch.ones(2,dtype=torch.bool),4,post_loops=5)
        self.assertEqual(p.q_steps.shape[1],5)
        meta=trace_metrics(c,p,self.receiver.targets,self.composed,self.select,
                           torch.ones(2,dtype=torch.bool))
        self.assertEqual(len(meta['protected_q_correct_by_step']),2)
        self.assertEqual(len(meta['protected_q_correct_by_step'][0]),5)
        self.assertEqual(len(meta['patched_q_correct_by_step'][0]),5)
        with self.assertRaises(ValueError):
            trace_suffix(m,self.receiver,m.prefix(self.receiver,4),8)

    def test_sampler_and_ramp_exposure_are_reproducible(self):
        depths={d:[i for i,c in enumerate(self.data.cases) if c.depth==d] for d in (2,3)}
        a=ex.sample_batches(self.data,depths,0,17,2,'continuation')
        b=ex.sample_batches(self.data,depths,0,17,2,'continuation')
        for step in range(5):
            first,p1=next(a);second,p2=next(b)
            self.assertEqual(first,second);self.assertTrue(torch.equal(p1,p2))
            self.assertTrue(torch.equal(ramp_mask(32,step,160,p1),ramp_mask(32,step,160,p2)))
        count=sum(int(ramp_mask(32,t,160,next(a)[1]).sum()) for t in range(160))
        # 0..32 counts are symmetric over 160 steps, total 2560 exposures.
        self.assertEqual(count,2560)

    def test_primary_science_gate_rejects_unqualified_and_locality_failures(self):
        fit={'train':{'goal_accuracy':1.,'subgoal_exact':1.},
             'dev':{'goal_accuracy':1.,'subgoal_exact':1.}}
        prot={'goal_accuracy':.8,'false_goal_attempts':0}
        patch={'untouched_subgoal_accuracy':.5,'informative_goal_accuracy':.75,
               'other_patch_goal_retention':.6}
        trace={'3':{'protected_q_step_accuracy':[.5,.5,.5],
                   'mean_protected_code_l2_by_step':[.1,.1,.1]},
               '5':{'protected_q_step_accuracy':[.5]*5}}
        run={'training':{a:{'fit_metrics':copy.deepcopy(fit)} for a in ('warmup',)+ex.ARMS},
             'scores':{a:{s:copy.deepcopy(prot) for s in aggregate.PRIMARY} for a in ex.ARMS},
             'patches':{a:{s:copy.deepcopy(patch) for s in aggregate.PRIMARY} for a in ex.ARMS},
             'trajectories':{a:{s:copy.deepcopy(trace) for s in aggregate.PRIMARY} for a in ex.ARMS}}
        self.assertEqual(aggregate.judge(run)['verdict'],'fail')
        run['training']['warmup']['fit_metrics']['dev']['subgoal_exact']=.7
        self.assertEqual(aggregate.judge(run)['verdict'],'inconclusive')

    def test_smoke_training_evaluation_audit_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td)
            fit=ex.train_one(td/'train',17,1,warm_steps=2,cont_steps=2,smoke=True)
            self.assertEqual(set(fit['arms']),{'warmup','status','ramp','locality'})
            self.assertEqual(fit['arms']['ramp']['addressed_rows'],32)
            self.assertEqual(fit['arms']['locality']['addressed_rows'],32)
            ev=ex.evaluate_one(td/'train',td/'eval',17,1,smoke=True)
            self.assertEqual(ev['clean_rows'],3*224)
            self.assertEqual(ev['patch_rows'],3*2*224)
            self.assertEqual(ev['trajectory_rows'],3*2*224)
            result=verify.verify_one(td/'eval',td/'train',17,1,smoke=True,replay_samples=2)
            self.assertTrue(result['passed'])
            path=td/'eval'/'trajectory_predictions.jsonl'
            lines=path.read_text().splitlines()
            row=json.loads(lines[0]);row['protected_q_correct_by_step'][0]=not row['protected_q_correct_by_step'][0]
            lines[0]=json.dumps(row)
            path.write_text('\n'.join(lines)+'\n')
            evalfile=td/'eval'/'evaluation.json'
            payload=json.loads(evalfile.read_text())
            payload['files'][path.name]=ex.sha_file(path)
            ex.json_write(evalfile,payload)
            (td/'eval'/'evaluation.sha256').write_text(ex.sha_file(evalfile)+'\n')
            with self.assertRaisesRegex(ValueError,'untouched Q semantic mismatch'):
                verify.verify_one(td/'eval',td/'train',17,1,smoke=True,replay_samples=1)

if __name__=='__main__': unittest.main()

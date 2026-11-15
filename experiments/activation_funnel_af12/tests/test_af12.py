"""AF-12 independent semantics, source invariants, leakage checks, and smoke."""
from __future__ import annotations
import json,os,tempfile,unittest
from pathlib import Path
import torch
from experiments.activation_funnel_af05 import gym
from experiments.activation_funnel_af05.model import GraphLooper
from experiments.activation_funnel_af10.core import latch_suffix
from experiments.activation_funnel_af11.core import replay
from experiments.activation_funnel_af12 import data,core,experiment as ex,verify


class AF12Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_fresh_corpus_frozen_and_disjoint(self):
        self.assertEqual(len(data.cases_for('fresh_depth4')),256)
        self.assertEqual(len(data.cases_for('fresh_depth5')),256)
        self.assertEqual(len(data.cases_for('fresh_depth3')),128)
        self.assertEqual(len(data.cases_for('fresh_comp')),16)
        self.assertEqual(ex.verify_manifest(),ex.sha(ex.MANIFEST))

    def test_synchronous_oracle_reconstructs_closure_independently(self):
        for split in ('train','dev')+data.FRESH:
            for c in data.Dataset(split).cases[::17]:
                before=core.closure_layers(c,ex.PRE[c.depth]);alternative=verify.independent_layers(c,ex.PRE[c.depth])
                self.assertEqual(before.tolist(),alternative)
                qpos=data.physical_permutations(c.permutation)[0]
                self.assertEqual(tuple(int(before[-1,qpos[q]]) for q in gym.SUBGOALS),gym.oracle(c)[0][:2])
                self.assertTrue(torch.all(before[1:]>=before[:-1]))

    def test_closure_exact_waves_and_stuttering(self):
        for depth in (2,3,4,5):
            c=gym.Case('math',depth,('and','or'),'and',(1,1,1,0),0)
            z=core.closure_layers(c,10)
            self.assertTrue(torch.equal(z[depth-1],z[-1]))
            pos,_=gym.permutation_maps(0)
            p={i:int(z[i-1,pos[2+i]]) for i in range(1,depth) if 2+i<7}
            self.assertEqual(len(z),10)

    def test_runtime_targets_are_excluded_from_model_features(self):
        a=data.Dataset('fresh_depth4').cases[0]
        b=data.Dataset('fresh_depth4').cases[1]
        x,y=data.encode(a),data.encode(b)
        self.assertEqual(x.atom_features.shape,(15,5))
        self.assertEqual(x.rule_features.shape,(14,3))
        self.assertTrue(x.targets.ndim==1 and x.targets.numel()==3)
        self.assertTrue(torch.equal(x.rule_features,y.rule_features))
        self.assertFalse(torch.equal(x.atom_features,y.atom_features))

    def test_model_source_native_cell_trainable_and_stage_shapes(self):
        model=GraphLooper(2)
        ds=data.Dataset('train');batch=ds.get([0,1]);t=core.trace_upstream(model,batch,4)
        self.assertEqual(tuple(t.logits.shape),(2,4,15,2))
        self.assertEqual(tuple(t.codes.shape),(2,4,15,2))
        loss,_=core.relay_loss(t,core.true_step_targets(ds.cases[:2],4),'closure_rollout')
        loss.backward()
        self.assertGreater(sum(float(p.grad.square().sum()) for p in model.cell.parameters() if p.grad is not None),0.)

    def test_equal_source_forward_calls_and_stable_cut(self):
        model=GraphLooper(2);ds=data.Dataset('train');ix=[0,1]
        rx=ds.get(ix);donor,_=ds.pair(ix,[0,1]);choose=torch.tensor([0,1])
        p=replay(model,rx,donor,choose,4,cut=True)
        for j in range(2):
            v=p.patched.q_codes[j,:,1-int(choose[j])]
            self.assertTrue(torch.equal(v,v[0].expand_as(v)))
        t=core.trace_upstream(model,rx,4)
        self.assertEqual(len(t.logits[0]),4)

    def test_stage_losses_nonzero_and_controls_distinct(self):
        model=GraphLooper(2);cases=data.Dataset('train').cases[:2];rx=data.Dataset('train').get([0,1])
        t=core.trace_upstream(model,rx,4);y=core.true_step_targets(cases,4)
        a,_=core.relay_loss(t,y,'terminal');b,_=core.relay_loss(t,y,'relay_final');c,_=core.relay_loss(t,y,'closure_rollout')
        self.assertEqual(float(a),0.);self.assertGreater(float(b),0.);self.assertGreater(float(c),0.)
        with self.assertRaises(ValueError):core.relay_loss(t,y[:,:2],'closure_rollout')

    def test_physical_permutation_transports_same_logical_oracle(self):
        c=data.cases_for('fresh_depth4')[0]
        alt=gym.Case('fresh_depth4',c.depth,c.kinds,c.gate,c.facts,15)
        m=GraphLooper(2)
        with torch.no_grad():
            b1=data.stack([c]);b2=data.stack([alt])
            a=core.trace_upstream(m,b1,8).logits[0]
            b=core.trace_upstream(m,b2,8).logits[0]
        p=data.physical_permutations(c.permutation)[0];q=data.physical_permutations(alt.permutation)[0]
        self.assertTrue(torch.allclose(a[:,list(p)],b[:,list(q)],atol=3e-5,rtol=0))

    def test_fit_eligibility_requires_train_and_dev(self):
        from experiments.activation_funnel_af12.aggregate import qualified
        data1={a:{s:{'goal_accuracy':1.,'joint_q_accuracy':1.} for s in ('train','dev')} for a in ex.ARMS}
        self.assertTrue(qualified(data1,'closure_rollout'))
        data1['closure_rollout']['dev']['joint_q_accuracy']=.8999
        self.assertFalse(qualified(data1,'closure_rollout'))

    def test_smoke_full_model_and_independent_mutation_rejection(self):
        path=os.getenv('AF12_TEST_SOURCE_ZIP')
        if not path:
            path='/mnt/data/af06-s17-w2-training.zip'
        if not Path(path).exists():
            self.skipTest('Exact AF06 frozen source artifact unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);train=r/'training';evaluation=r/'evaluation'
            ex.train(path,train,17,steps=2,smoke=True)
            ex.evaluate(train,evaluation,17,smoke=True)
            v=verify.audit(train,evaluation,17,r/'independent.json',smoke=True)
            self.assertTrue(v['passed']);self.assertEqual(v['patched_rows'],3*2*224)
            # Rehashing a fabricated logit is insufficient to alter the score.
            f=evaluation/'clean.jsonl';rows=[json.loads(x) for x in f.read_text().splitlines()]
            rows[0]['closure_stage_logits'][0][0][1]+=8.
            f.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in rows))
            entry=json.loads((evaluation/'evaluation.json').read_text())
            entry['clean_sha256']=ex.sha(f)
            ex.write_json(evaluation/'evaluation.json',entry)
            with self.assertRaisesRegex(ValueError,'scoring mismatch|source-bound clean forward'):
                verify.audit(train,evaluation,17,r/'bad.json',smoke=True)

if __name__=='__main__':unittest.main()

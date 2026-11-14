"""AF-06 checks for phased identity, exact patch semantics, and independent replay."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from experiments.activation_funnel_af05 import gym
from experiments.activation_funnel_af05.model import GraphLooper, parameter_count
from experiments.activation_funnel_af06 import experiment as ex
from experiments.activation_funnel_af06.core import mixed_patch, mixed_targets, ramp_mask
from experiments.activation_funnel_af06 import aggregate, verify


class AF06Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_frozen_source_manifest_and_roster(self):
        self.assertEqual(ex.validate_manifest(),ex.sha_file(ex.PINNED_MANIFEST))
        self.assertEqual(len(gym.cases_for('train')),672)
        self.assertEqual(len(gym.cases_for('dev')),224)
        self.assertEqual({d:len([x for x in gym.cases_for('train') if x.depth==d]) for d in (2,3)}, {2:336,3:336})
        self.assertEqual(ex.WARM_STEPS,480)
        self.assertEqual(ex.CONT_STEPS,160)
        self.assertEqual(ex.ARMS,('status','ramp','abrupt'))

    def test_monotone_patch_curriculum_has_known_extrema(self):
        n=32;perm=torch.randperm(n,generator=torch.Generator().manual_seed(1))
        counts=[int(ramp_mask(n,t,160,perm).sum()) for t in range(160)]
        self.assertEqual(counts[0],0)
        self.assertEqual(counts[-1],32)
        self.assertTrue(all(x<=y for x,y in zip(counts,counts[1:])))
        self.assertTrue(all(0<=x<=32 for x in counts))

    def test_schedule_rejects_bad_permutation(self):
        with self.assertRaises(ValueError):ramp_mask(4,0,160,torch.tensor([0,0,2,3]))
        with self.assertRaises(ValueError):ramp_mask(4,0,1,torch.arange(4))

    def test_mixed_patch_matches_parent_for_uniform_conditions(self):
        torch.manual_seed(23);model=GraphLooper(2).eval()
        ds=ex.Dataset('train');indices=[0,1,2,3];choices=torch.tensor([0,1,0,1]);
        receiver=ds.get(indices);donor,_=ds.pair(indices,choices.tolist())
        for branch,condition in ((True,'addressed'),(False,'other')):
            with torch.no_grad():
                expected=model.patched(receiver,donor,choices,condition=condition,pre_loops=4)
                actual=mixed_patch(model,receiver,donor,choices,
                                   torch.full((4,),branch,dtype=torch.bool),4)
            self.assertTrue(torch.equal(actual.subgoals,expected.subgoals))
            self.assertTrue(torch.equal(actual.goal,expected.goal))
            self.assertTrue(torch.equal(actual.pre.az,expected.pre.az))

    def test_row_mixed_patch_and_oracle_targets(self):
        torch.manual_seed(17);model=GraphLooper(1).eval()
        data=ex.Dataset('train');cases=[0,1,2,3];sel=torch.tensor([0,1,0,1])
        receiver=data.get(cases);donor,hybrid=data.pair(cases,sel.tolist())
        mask=torch.tensor([True,False,False,True])
        actual=mixed_patch(model,receiver,donor,sel,mask,4)
        targets=mixed_targets(receiver.targets,hybrid,mask)
        for j in range(4):
            m='addressed' if bool(mask[j]) else 'other'
            with torch.no_grad():
                reference=model.patched(data.get([cases[j]]),data.get([data.donors[cases[j],int(sel[j])][0]]),
                                        sel[j:j+1],condition=m,pre_loops=4)
            self.assertTrue(torch.allclose(actual.goal[j],reference.goal[0],atol=2e-5,rtol=0))
            self.assertEqual(targets[j].tolist(),(hybrid if bool(mask[j]) else receiver.targets)[j].tolist())

    def test_bad_structural_donor_is_rejected(self):
        model=GraphLooper(1);ds=ex.Dataset('train');r=ds.get([0]);d=ds.get([0]);
        d.read_slots=d.read_slots.flip(-1)
        with self.assertRaisesRegex(ValueError,'structurally invalid'):
            mixed_patch(model,r,d,torch.tensor([0]),torch.tensor([True]),4)

    def test_native_cell_and_codec_receive_patch_gradients(self):
        model=GraphLooper(2);ds=ex.Dataset('train');indices=[0,1];sel=torch.tensor([0,1]);
        r=ds.get(indices);d,targets=ds.pair(indices,sel.tolist())
        out=mixed_patch(model,r,d,sel,torch.tensor([True,False]),4)
        loss=ex.logits_ce(out.goal,mixed_targets(r.targets,targets,torch.tensor([True,False]))[:,2])
        loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.cell.parameters() if p.grad is not None),0)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.codec.parameters() if p.grad is not None),0)
        self.assertEqual(parameter_count(model)['native_cell_parameters'],5024)

    def test_horn_intervention_ground_truth_is_independent(self):
        for case in gym.cases_for('held_depth5')[:20]:
            self.assertEqual(gym.oracle(case)[0][2],int(all(gym.oracle(case)[0][:2]))
                             if case.gate=='and' else int(any(gym.oracle(case)[0][:2])))
        self.assertEqual(len(gym.donors_for('held_depth4')),256)

    def test_one_phase_ramp_shared_budget_and_reproducible_batch(self):
        ds=ex.Dataset('train');by={d:[i for i,c in enumerate(ds.cases) if c.depth==d] for d in (2,3)}
        a=next(ex.sample_batches(ds,by,0,17,2,'continuation'))
        b=next(ex.sample_batches(ds,by,0,17,2,'continuation'))
        self.assertEqual(a[0],b[0]);self.assertTrue(torch.equal(a[1],b[1]))
        self.assertEqual([len(a[0][d]) for d in (2,3)],[16,16])

    def test_registered_threshold_includes_warmup_eligibility(self):
        good={'train':{'goal_accuracy':.9,'subgoal_exact':.9},
              'dev':{'goal_accuracy':.9,'subgoal_exact':.9}}
        bad={'train':good['train'],'dev':{'goal_accuracy':.79,'subgoal_exact':.95}}
        self.assertTrue(aggregate.eligible_metrics(good))
        self.assertFalse(aggregate.eligible_metrics(bad))

    def test_engineering_smoke_and_mutation_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            train=Path(tmp)/'train';score=Path(tmp)/'score'
            receipt=ex.train_one(train,17,1,warm_steps=1,cont_steps=2,smoke=True)
            self.assertEqual(receipt['train_dev_predictions'],4*(672+224))
            self.assertEqual(receipt['arms']['status']['addressed_rows'],0)
            self.assertEqual(receipt['arms']['ramp']['addressed_rows'],32)
            self.assertEqual(receipt['arms']['abrupt']['addressed_rows'],64)
            res=ex.evaluate_one(train,score,17,1,smoke=True)
            self.assertEqual(res['clean_rows'],3*224)
            self.assertEqual(res['patch_rows'],6*224)
            audited=verify.verify_one(score,train,17,1,smoke=True,replay_samples=1)
            self.assertTrue(audited['passed'])
            (score/'patch_predictions.jsonl').write_text('tampered\n')
            with self.assertRaisesRegex(ValueError,'corrupt evaluation'):
                verify.verify_one(score,train,17,1,smoke=True,replay_samples=1)


if __name__=='__main__':unittest.main()

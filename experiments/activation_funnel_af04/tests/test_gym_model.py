"""AF-04 exact Horn semantics, two-stage causal masks and gradient regression."""
import json
import tempfile
import unittest
from pathlib import Path

import torch

from experiments.activation_funnel_af04 import gym
from experiments.activation_funnel_af04 import experiment as ex
from experiments.activation_funnel_af04.model import GraphLooper, parameter_count


class AF04Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_frozen_split_counts_and_exclusive_inputs(self):
        m=ex.validate_manifest()
        self.assertEqual(len(m),64)
        self.assertEqual([len(gym.cases_for(s)) for s in gym.SPLITS],
                         [336,112,16,112,128])
        self.assertEqual([len(gym.donors_for(s)) for s in gym.SPLITS],
                         [672,224,32,224,256])
        self.assertEqual(len({c.id for s in gym.SPLITS for c in gym.cases_for(s)}),704)

    def test_oracle_closure_and_explicit_boolean_check(self):
        for split in gym.SPLITS:
            for case in gym.cases_for(split):
                result,history,goal_history=gym.oracle(case)
                self.assertEqual(result[2],
                                 int(all(result[:2])) if case.gate=='and' else int(any(result[:2])))
                self.assertEqual(all(v in (0,1) for v in result),True)
                self.assertLessEqual(len(history)+len(goal_history),7)
                self.assertNotIn(gym.GOAL,[x[1] for x in history])

    def test_encoder_never_exposes_derived_truth(self):
        c=gym.Case('train',2,('and','or'),'and',(0,0,0,0),0)
        d=gym.Case('train',2,('and','or'),'and',(1,1,1,1),0)
        a,b=gym.encode_case(c),gym.encode_case(d)
        self.assertFalse(torch.equal(a.targets,b.targets))
        self.assertTrue(torch.equal(a.rule_features,b.rule_features))
        for key in ('prem_up','head_up','prem_down','head_down','read_slots'):
            self.assertTrue(torch.equal(getattr(a,key),getattr(b,key)))
        diff=a.atom_features-b.atom_features
        self.assertEqual(int((diff!=0).sum()),4)
        self.assertEqual(float(diff[:,1:].abs().sum()),0.)
        self.assertEqual(int(a.atom_features[:,1].sum()),1)
        self.assertEqual(int(a.atom_features[:,2].sum()),1)
        self.assertEqual(int(a.atom_features[:,3].sum()),1)

    def test_donor_pairs_change_exactly_one_inferred_subgoal(self):
        for split in gym.SPLITS:
            cases=gym.cases_for(split)
            for (i,j),(di,truth) in gym.donors_for(split).items():
                a,b=cases[i],cases[di]
                ra=gym.oracle(a)[0]; db=gym.oracle(b)[0]
                self.assertEqual(a.skeleton,b.skeleton)
                self.assertEqual(ra[1-j],db[1-j])
                self.assertNotEqual(ra[j],db[j])
                self.assertEqual(truth[j],db[j])
                self.assertEqual(truth[1-j],ra[1-j])
                self.assertEqual(truth[2],int(all(truth[:2])) if a.gate=='and' else int(any(truth[:2])))

    def test_graph_slots_permutation_is_transport_not_shortcut(self):
        torch.manual_seed(13)
        m=GraphLooper(2).eval()
        source=gym.Case('train',2,('and','or'),'and',(1,1,0,1),0)
        x=gym.stack_cases([source])
        expected=m(x)
        for perm in (1,2,3,4,5,6):
            c=gym.Case('train',2,source.kinds,source.gate,source.facts,perm)
            actual=m(gym.stack_cases([c]))
            self.assertTrue(torch.allclose(actual.goal,expected.goal,atol=3e-5,rtol=0))
            self.assertTrue(torch.allclose(actual.subgoals,expected.subgoals,atol=3e-5,rtol=0))

    def test_patch_whole_and_self_are_exact(self):
        torch.manual_seed(21)
        m=GraphLooper(1).eval()
        cs=gym.cases_for('train')
        donor=gym.donors_for('train')
        ix=[0,1,2,3]
        orig=gym.stack_cases([cs[i] for i in ix])
        dn=gym.stack_cases([cs[donor[i,0][0]] for i in ix])
        selection=torch.zeros(len(ix),dtype=torch.long)
        whole=m.patched(orig,dn,selection,condition='whole')
        direct=m(dn)
        self.assertTrue(torch.equal(whole.goal,direct.goal))
        self.assertTrue(torch.equal(whole.subgoals,direct.subgoals))
        selfpatch=m.patched(orig,dn,selection,condition='self')
        clean=m(orig)
        self.assertTrue(torch.equal(selfpatch.goal,clean.goal))
        self.assertTrue(torch.equal(selfpatch.subgoals,clean.subgoals))

    def test_address_patch_only_selected_latent_theorem_node(self):
        torch.manual_seed(17)
        model=GraphLooper(2)
        cs=gym.cases_for('train')
        roster=gym.donors_for('train')
        ix=list(range(8))
        selects=torch.tensor([i%2 for i in ix])
        r=gym.stack_cases([cs[i] for i in ix])
        d=gym.stack_cases([cs[roster[i,i%2][0]] for i in ix])
        rp,dp=model.prefix(r),model.prefix(d)
        patched=model.patched(r,d,selects)
        for k in range(len(ix)):
            addr=r.read_slots[k,selects[k]]
            self.assertTrue(torch.equal(patched.pre.az[k,addr],dp.az[k,addr]))
            unchanged=[i for i in range(gym.ATOMS) if i!=addr]
            self.assertTrue(torch.equal(patched.pre.az[k,unchanged],rp.az[k,unchanged]))
            self.assertTrue(torch.equal(patched.pre.rz[k],rp.rz[k]))

    def test_causal_oracle_gradients_reach_original_typed_cell(self):
        torch.manual_seed(33)
        model=GraphLooper(2)
        ds=ex.Dataset('train')
        selected=[0,1,2,3]
        r=ds.get(selected)
        d,label=ds.pair(selected,[0,1,0,1])
        o=model.patched(r,d,torch.tensor([0,1,0,1]))
        loss=ex.logits_ce(o.goal,label[:,2]) + ex.logits_ce(o.subgoals,label[:,:2])
        loss.backward()
        native=sum(float(p.grad.abs().sum()) for p in model.cell.parameters() if p.grad is not None)
        codec=sum(float(p.grad.abs().sum()) for p in model.codec.parameters() if p.grad is not None)
        self.assertGreater(native,0)
        self.assertGreater(codec,0)
        self.assertLess(parameter_count(model)['parameters'],6000)

    def test_receipt_tamper_and_negative_control(self):
        m=GraphLooper(1)
        data=ex.Dataset('dev')
        s,rows=ex.score_counterfactual(m,data,chunk=64)
        self.assertEqual(s['pairs'],224)
        self.assertEqual(s['informative_goal_pairs'],104)
        self.assertEqual(s['whole_direct_max_logit_error'],0.)
        self.assertEqual(len(rows),224)
        self.assertIn('patched_subgoal_code',rows[0])
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'new'
            ex.train_one(out,seed=17,width=1,steps=1,smoke=True)
            rec=json.loads((out/'train_receipt.json').read_text())
            self.assertEqual(rec['train_dev_predictions'],3*(336+112))
            (out/'status.npz').write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError,'changed artifact'):
                ex.load_frozen(out,17,1,smoke=True)

    def test_lean_closed_proof_terms_are_graph_derived(self):
        src=gym.lean_certificate('held_composition')
        self.assertEqual(src.count('theorem af04_'),3*16)
        self.assertIn('by decide',src)
        self.assertIn('&&',src)
        self.assertIn('||',src)
        self.assertNotIn('sorry',src)

if __name__=='__main__':unittest.main()

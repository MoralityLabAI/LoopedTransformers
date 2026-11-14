"""AF-05 frozen corpus, deeper rule closure, causal abstraction and training gates."""
from pathlib import Path
import json
import tempfile
import unittest

import torch

from experiments.activation_funnel_af05 import gym
from experiments.activation_funnel_af05 import experiment as ex
from experiments.activation_funnel_af05 import aggregate
from experiments.activation_funnel_af05.model import GraphLooper, parameter_count


class AF05Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_registered_corpus_and_source_receipt(self):
        expected = [672,224,16,224,128,128]
        self.assertEqual([len(gym.cases_for(s)) for s in gym.SPLITS],expected)
        self.assertEqual(sum(expected),1392)
        self.assertEqual(len({gym.digest(gym.encode_case(c).atom_features.tolist() +
                              gym.encode_case(c).rule_features.tolist()+
                              [gym.encode_case(c).prem_up.tolist(),gym.encode_case(c).head_up.tolist()])
                              for split in gym.SPLITS for c in gym.cases_for(split)}),1392)
        self.assertEqual(ex.validate_manifest(),ex.sha_file(ex.PINNED_MANIFEST))
        self.assertEqual(len(gym.donors_for('train')),1344)
        self.assertEqual(ex.EVAL_LOOPS['held_depth5'],10)
        self.assertEqual(aggregate.PRIMARY,('held_depth4','held_depth5'))

    def test_all_depths_match_independent_boolean_semantics(self):
        for split in gym.SPLITS:
            for case in gym.cases_for(split):
                q,up,down=gym.oracle(case)
                self.assertEqual(q[2],int(all(q[:2])) if case.gate=='and' else int(any(q[:2])))
                self.assertTrue(all(x in (0,1) for x in q))
                self.assertNotIn(gym.GOAL, [h for _,h in up])
                self.assertTrue(len(up)<=gym.ATOMS)

    def test_depth_five_uses_four_relay_edges_and_new_nodes(self):
        case=gym.cases_for('held_depth5')[0]
        up,down=gym.logical_rules(case)
        self.assertEqual(case.depth,5)
        self.assertTrue(any(h==5 for _,h in up))
        self.assertTrue(any(h==12 for _,h in up))
        self.assertTrue(any(h==gym.SUBGOALS[0] for _,h in up))
        self.assertEqual(gym.initial_facts(gym.Case('train',2,('and','and'),'and',(1,1,1,1),0)), {0,1,7,8})

    def test_only_initial_facts_exposed(self):
        a=gym.Case('train',2,('and','or'),'and',(0,0,0,0),0)
        b=gym.Case('train',2,('and','or'),'and',(1,1,1,1),0)
        u,v=gym.encode_case(a),gym.encode_case(b)
        self.assertFalse(torch.equal(u.targets,v.targets))
        self.assertTrue(torch.equal(u.rule_features,v.rule_features))
        self.assertEqual(int((u.atom_features-v.atom_features).abs().sum()),4)
        self.assertEqual(float((u.atom_features[:,1:]-v.atom_features[:,1:]).abs().sum()),0.)

    def test_donor_oracle_retains_untouched_subgoal_all_families(self):
        for split in gym.SPLITS:
            cases=gym.cases_for(split)
            for (i,selected),(j,target) in gym.donors_for(split).items():
                recipient,donor=cases[i],cases[j]
                self.assertEqual(recipient.skeleton,donor.skeleton)
                q=gym.oracle(recipient)[0]; dq=gym.oracle(donor)[0]
                self.assertEqual(q[1-selected],dq[1-selected])
                self.assertNotEqual(q[selected],dq[selected])
                self.assertEqual(target[selected],dq[selected])
                self.assertEqual(target[1-selected],q[1-selected])

    def test_physical_graph_permutations_transport_logits(self):
        torch.manual_seed(13)
        m=GraphLooper(2).eval()
        orig=gym.Case('train',3,('and','or'),'and',(1,1,0,1),0)
        with torch.no_grad():
            expected=m(gym.stack_cases([orig]),pre_loops=6)
            for perm in range(1,8):
                test=gym.Case('train',3,orig.kinds,orig.gate,orig.facts,perm)
                actual=m(gym.stack_cases([test]),pre_loops=6)
                self.assertTrue(torch.allclose(actual.goal,expected.goal,atol=3e-5,rtol=0))
                self.assertTrue(torch.allclose(actual.subgoals,expected.subgoals,atol=3e-5,rtol=0))

    def test_pre_depth_variants_patch_whole_exactly(self):
        torch.manual_seed(29)
        m=GraphLooper(2)
        ds=ex.Dataset('held_depth5')
        selected=[0,1,2,3]; labels=[0,1,0,1]
        r=ds.get(selected);don,_=ds.pair(selected,labels)
        for loops in (4,6,8,10):
            with torch.no_grad():
                whole=m.patched(r,don,torch.tensor(labels),condition='whole',pre_loops=loops)
                full=m(don,pre_loops=loops)
                same=m.patched(r,don,torch.tensor(labels),condition='self',pre_loops=loops)
                rec=m(r,pre_loops=loops)
            self.assertTrue(torch.equal(whole.goal,full.goal))
            self.assertTrue(torch.equal(whole.subgoals,full.subgoals))
            self.assertTrue(torch.equal(same.goal,rec.goal))
            self.assertTrue(torch.equal(same.subgoals,rec.subgoals))

    def test_selected_code_patch_is_local_and_native_gradients_exist(self):
        torch.manual_seed(7)
        m=GraphLooper(2)
        ds=ex.Dataset('train'); idx=[0,1,2,3]; sels=[0,1,0,1]
        r=ds.get(idx);don,target=ds.pair(idx,sels)
        one=m.prefix(r,loops=4);two=m.prefix(don,loops=4)
        patched=m.patched(r,don,torch.tensor(sels),pre_loops=4)
        for k,sel in enumerate(sels):
            pos=int(r.read_slots[k,sel]);rest=[i for i in range(gym.ATOMS) if i!=pos]
            self.assertTrue(torch.equal(patched.pre.az[k,pos],two.az[k,pos]))
            self.assertTrue(torch.equal(patched.pre.az[k,rest],one.az[k,rest]))
            self.assertTrue(torch.equal(patched.pre.rz[k],one.rz[k]))
        loss=ex.logits_ce(patched.goal,target[:,2])+ex.logits_ce(patched.subgoals,target[:,:2])
        loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in m.cell.parameters() if p.grad is not None),0.)
        self.assertEqual(parameter_count(m)['native_cell_parameters'],5024)
        self.assertEqual(parameter_count(m)['carried_state_floats'],58)

    def test_held_family_patches_have_nontrivial_goal_flips(self):
        for family in ('held_composition','held_layout','held_depth4','held_depth5'):
            data=ex.Dataset(family)
            count=0
            for (i,_),(j,hybrid) in data.donors.items():
                count += int(hybrid[2] != gym.oracle(data.cases[i])[0][2])
            self.assertGreater(count,0)

    def test_lean_certification_text_uses_all_deeper_rules(self):
        cert=gym.lean_certificate('held_depth5')
        self.assertEqual(cert.count('theorem af05_'),128*3)
        self.assertIn('by decide',cert)
        self.assertIn('&&',cert)
        self.assertIn('||',cert)
        self.assertNotIn('sorry',cert)

    def test_smoke_training_and_independent_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            train=Path(tmp)/'training';evaldir=Path(tmp)/'evaluation'
            receipt=ex.train_one(train,17,2,steps=1,smoke=True)
            self.assertEqual(receipt['train_dev_predictions'],3*(672+224))
            self.assertEqual(set(receipt['arms']),set(ex.ARMS))
            result=ex.evaluate_one(train,evaldir,17,2,smoke=True)
            self.assertEqual(result['clean_rows'],3*224)
            self.assertEqual(result['patch_rows'],6*224)
            from experiments.activation_funnel_af05 import verify
            audit=verify.verify_one(evaldir,train,17,2,smoke=True,replay_samples=1)
            self.assertTrue(audit['passed'])
            (evaldir/'patch_predictions.jsonl').write_text('corrupted')
            with self.assertRaisesRegex(ValueError,'corrupt evaluation'):
                verify.verify_one(evaldir,train,17,2,smoke=True,replay_samples=1)


if __name__=='__main__':unittest.main()

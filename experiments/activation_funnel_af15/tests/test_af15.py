"""AF-15 host authority and typed-recurrence regression tests."""
from __future__ import annotations
import copy,json,tempfile,unittest
from pathlib import Path
import torch
from experiments.activation_funnel_af05 import gym as base
from experiments.activation_funnel_af15 import gym,model,experiment,verify

class AF15Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.train=gym.cases_for('train')

    def test_fresh_manifest_distinct_and_complete(self):
        r=gym.manifest()
        self.assertEqual(r['splits']['train']['count'],672)
        self.assertEqual([r['splits'][n]['count'] for n in gym.FRESH],[128,16,256,256])
        self.assertEqual(r['manifest_sha256'],gym.base.digest({k:v for k,v in r.items() if k!='manifest_sha256'}))

    def test_sound_fire_blocks_invalid_and_never_erases_known(self):
        c=self.train[0];s=frozenset(base.initial_facts(c))
        with self.assertRaisesRegex(ValueError,'inadmissible'):
            gym.sound_fire(c,s,999)
        for _ in range(20):
            choices=gym.enabled(c,s)
            if not choices:break
            nxt,proof=gym.sound_fire(c,s,choices[0])
            self.assertTrue(s<=nxt)
            self.assertIn(proof['head'],gym.final_closure(c))
            s=nxt
        self.assertTrue(s<=gym.final_closure(c))

    def test_exact_minimal_goal_step_and_legal_reference(self):
        for c in self.train[::41]:
            states=gym.reference_episode(c)
            self.assertGreaterEqual(len(states),1)
            for state,optimal in states:
                self.assertTrue(set(optimal)<=(set(gym.enabled(c,state))|{gym.STOP}))
            initial=frozenset(base.initial_facts(c));steps=gym.optimal_remaining(c,initial)
            self.assertEqual(steps<10**6,bool(base.oracle(c)[0][2]))

    def test_reference_no_label_leakage_in_model_observations(self):
        c=next(c for c in self.train if base.oracle(c)[0][2])
        x,bits,gate=model.observations([c],[frozenset(base.initial_facts(c))])
        self.assertEqual(tuple(x.atom_features.shape),(1,base.ATOMS,5))
        self.assertEqual(tuple(bits.shape),(1,base.ATOMS))
        self.assertTrue(bool(gate[0,gym.STOP]))
        self.assertFalse(torch.any(x.atom_features[:,:,0]!=bits))

    def test_original_native_typedloop_is_trainable(self):
        torch.manual_seed(5);m=model.TypedHornPolicy()
        cases=list(self.train[:16]);states=[frozenset(base.initial_facts(c)) for c in cases]
        x,bits,gate,opt=model.state_inputs(cases,states)
        out=m(x,bits);loss,_=model.objective(out,gate,opt,'frontier');loss.backward()
        self.assertEqual(sum(t.numel() for t in m.parameters()),5234)
        self.assertGreater(sum(float(t.grad.square().sum()) for t in m.cell.parameters() if t.grad is not None),0)

    def test_raw_logits_permutation_transport(self):
        c=base.Case('fresh_depth4',4,('or','and'),'or',(1,0,1,1),32)
        d=base.Case('fresh_depth4',4,c.kinds,c.gate,c.facts,33)
        torch.manual_seed(8);m=model.TypedHornPolicy().eval()
        state=frozenset(base.initial_facts(c))
        x,bits,_=model.observations([c],[state]);x2,b2,_=model.observations([d],[state])
        with torch.no_grad():a=m(x,bits)[0];b=m(x2,b2)[0]
        _,r0=gym.physical_maps(32);_,r1=gym.physical_maps(33)
        for k in range(len(gym.full_rules(c))):
            self.assertAlmostEqual(float(a[r0[k]]),float(b[r1[k]]),places=5)
        self.assertAlmostEqual(float(a[-1]),float(b[-1]),places=5)

    def test_frontier_loss_uses_only_legal_actions(self):
        out=torch.tensor([[9.,-1.,-2.]],requires_grad=True)
        valid=torch.tensor([[False,True,True]])
        targets=torch.tensor([[False,True,False]])
        val,_=model.objective(out,valid,targets,'frontier');val.backward()
        self.assertEqual(float(out.grad[0,0]),0.)

    def test_independent_host_edge_decoding_matches_source(self):
        for c in list(self.train[:10])+list(gym.cases_for('fresh_depth4')[:10]):
            edges=verify.physical_edges(c)
            slots,_=gym.physical_maps(c.permutation)
            for i,(prem,head) in gym.physical_rules(c).items():
                self.assertEqual(edges[i],(tuple(sorted(slots[p] for p in prem)),slots[head]))

    def test_smoke_and_rehashed_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);tr=root/'train';ev=root/'eval'
            experiment.train(17,tr,steps=2,smoke=True)
            experiment.evaluate(17,tr,ev,smoke=True)
            audit=verify.audit(17,tr,ev,root/'audit.json',replays=1,smoke=True)
            self.assertTrue(audit['passed'])
            rows=[json.loads(x) for x in (ev/'episodes.jsonl').read_text().splitlines()]
            target=next(x for x in rows if x['actions'])
            target['actions'][0]['enabled']=[999]
            experiment.write_jsonl(ev/'episodes.jsonl',rows)
            meta=json.loads((ev/'evaluation.json').read_text())
            meta['episodes_sha256']=experiment.sha(ev/'episodes.jsonl')
            experiment.write(ev/'evaluation.json',meta)
            with self.assertRaisesRegex(ValueError,'eligibility'):
                verify.audit(17,tr,ev,root/'bad.json',replays=1,smoke=True)

if __name__=='__main__':unittest.main()

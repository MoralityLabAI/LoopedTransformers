"""AF-18 preregistered causal cone, Bellman target and evidence contracts."""
from __future__ import annotations
import json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from experiments.activation_funnel_af05 import gym as base
from experiments.activation_funnel_af17 import gym as parent
from experiments.activation_funnel_af18 import gym,model,experiment as ex,verify

class AF18Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_disjoint_manifest_and_frozen_train_dev(self):
        self.assertEqual({s:len(gym.cases_for(s)) for s in gym.SPLITS},
            {'train':928,'dev':352,'held_layout':32,'held_depth4':192,'held_depth5':192})
        self.assertEqual(gym.cases_for('train'),parent.cases_for('train'))
        self.assertEqual(gym.cases_for('dev'),parent.cases_for('dev'))
        self.assertTrue(gym.manifest()['host_rule_gate'])
        self.assertEqual(ex.check_manifest(),ex.sha(ex.HERE/'fresh_manifest.json'))

    def test_exact_shortest_and_independent_physical_bfs(self):
        for sp in gym.SPLITS:
            cases=gym.cases_for(sp)
            for c in cases[::max(1,len(cases)//22)]:
                _,_,goal,d,closure=verify.independent_reference(c)
                dist=gym.optimal_remaining(c,gym.initial_facts(c))
                self.assertEqual(d,None if dist>=gym.INFINITY else dist)
                self.assertEqual(goal in closure,dist<gym.INFINITY)

    def test_cost_cap16_restores_strict_long_proof_gap(self):
        self.assertEqual(gym.CAP,16)
        examples=[c for c in gym.cases_for('held_depth5') if gym.optimal_remaining(c,gym.initial_facts(c))>=10
                  and gym.optimal_remaining(c,gym.initial_facts(c))<gym.INFINITY]
        self.assertTrue(any(gym.optimal_remaining(c,gym.initial_facts(c))==11 for c in examples))
        for c in examples:
            initial=gym.initial_facts(c);d=gym.optimal_remaining(c,initial)
            ten=gym.action_costs(c,initial,10);sixteen=gym.action_costs(c,initial,16)
            for a in gym.optimal_actions(c,initial):
                self.assertEqual(sixteen[a],d)
                self.assertLess(sixteen[a],16)
                self.assertEqual(ten[a],10)
            self.assertEqual(sixteen[gym.STOP],16)
            _,by_bfs=verify.independent_costs(c,initial,16)
            self.assertEqual({i:x for i,x in enumerate(sixteen) if x==x},by_bfs)

    def test_causal_cone_structural_paired_truth(self):
        self.assertEqual([gym.graph_distance(a,0) for a,b in gym.probe_pairs()],[8,10,12])
        torch.manual_seed(19)
        for l in model.LOOPS:
            m=model.SweepPolicy(l).eval()
            for a,b in gym.probe_pairs():
                self.assertFalse(base.GOAL in gym.closure(a))
                self.assertTrue(base.GOAL in gym.closure(b))
                x,k,_=gym.observations([a,b],[gym.initial_facts(a),gym.initial_facts(b)])
                with torch.no_grad():scores,dist,reach=m.full(x,k)
                if gym.graph_distance(a,0)>2*l:
                    self.assertTrue(torch.allclose(scores[:,-1][0],scores[:,-1][1],atol=1e-6,rtol=0))
                    self.assertTrue(torch.allclose(dist[0],dist[1],atol=1e-6,rtol=0))
                    self.assertTrue(torch.allclose(reach[0],reach[1],atol=1e-6,rtol=0))

    def test_shared_initialization_and_variable_native_call_gradients(self):
        torch.manual_seed(19);a=model.SweepPolicy(2)
        torch.manual_seed(19);b=model.SweepPolicy(8)
        self.assertEqual(set(a.state_dict()),set(b.state_dict()))
        self.assertTrue(all(torch.equal(a.state_dict()[x],b.state_dict()[x]) for x in a.state_dict()))
        cases=gym.cases_for('train')[:5];states=[gym.initial_facts(x) for x in cases]
        x,k,gate,opt,cost,d,reach,stop=ex.supervision(cases,states,16)
        raw,dhat,rhat=b.full(x,k)
        loss,_=ex.objective(raw,dhat,rhat,gate,opt,cost,d,reach,stop)
        loss.backward()
        self.assertGreater(sum(float(v.grad.square().sum()) for v in b.cell.parameters() if v.grad is not None),0.)

    def test_host_gate_soundness_and_readout_legal_mask(self):
        c=gym.cases_for('held_depth5')[0];known=gym.initial_facts(c)
        with self.assertRaisesRegex(ValueError,'inadmissible'):
            gym.sound_fire(c,known,999)
        for _ in range(base.ATOMS):
            acts=gym.enabled(c,known)
            if not acts:break
            known,proof=gym.sound_fire(c,known,acts[0])
            self.assertIn(proof['head'],gym.closure(c))

    def test_layout_transport_for_new_physical_permutations(self):
        from dataclasses import replace
        c=gym.cases_for('held_depth4')[0];d=replace(c,permutation=59)
        m=model.SweepPolicy(6).eval();state=gym.initial_facts(c)
        x,known,_=gym.observations([c],[state]);y,known_y,_=gym.observations([d],[state])
        with torch.no_grad():a=m(x,known)[0];b=m(y,known_y)[0]
        _,r0=gym.physical_maps(c.permutation);_,r1=gym.physical_maps(d.permutation)
        for i in range(len(gym.logical_rules(c))):
            self.assertAlmostEqual(float(a[r0[i]]),float(b[r1[i]]),places=4)
        self.assertAlmostEqual(float(a[-1]),float(b[-1]),places=4)

    def test_smoke_training_scoring_and_mutation_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);t=d/'train';e=d/'evaluation'
            tr=ex.train(17,t,smoke=True)
            self.assertEqual(set(tr['weights']),set(ex.CONFIGS))
            report=ex.evaluate(17,t,e,smoke=True)
            self.assertEqual(report['probes'],15)
            audit=verify.audit(17,t,e,d/'audit.json',smoke=True)
            self.assertTrue(audit['passed']);self.assertGreater(audit['independent_bfs_cost_checks'],0)
            raw=[json.loads(line) for line in (e/'episodes.jsonl').read_text().splitlines()]
            target=next(r for r in raw if r['actions']);target['actions'][0]['enabled']=[999]
            ex.jsonl(e/'episodes.jsonl',raw)
            p=e/'evaluation.json';meta=json.loads(p.read_text());meta['episodes_sha256']=ex.sha(e/'episodes.jsonl');ex.write(p,meta)
            with self.assertRaisesRegex(ValueError,'enabled-action'):
                verify.audit(17,t,e,d/'mutation.json',smoke=True)

if __name__=='__main__':unittest.main()

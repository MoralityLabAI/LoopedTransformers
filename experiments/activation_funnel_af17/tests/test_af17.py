"""AF-17 frozen source contracts, Bellman semantics and independent replay."""
import json,tempfile,unittest
from pathlib import Path
import torch
from experiments.activation_funnel_af05 import gym as base
from experiments.activation_funnel_af16 import gym as af16
from experiments.activation_funnel_af17 import gym,model,experiment,verify

class AF17Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.ds={k:gym.cases_for(k) for k in gym.SPLITS}

    def test_manifest_roster_and_source_hashes(self):
        self.assertEqual({k:len(v) for k,v in self.ds.items()},
         {'train':928,'dev':352,'held_layout':32,'held_depth4':192,'held_depth5':192})
        self.assertEqual(experiment.check_manifest(),experiment.sha(experiment.HERE/'fresh_manifest.json'))
        self.assertEqual([c.id for c in self.ds['train']],[c.id for c in af16.cases_for('train')])

    def test_heldout_public_disjointness_and_novel_structure(self):
        self.assertTrue(gym.manifest()['all_heldout_feature_hashes_disjoint_from_AF16'])
        held=self.ds['held_depth5']
        self.assertEqual(set(c.variant for c in held),set(gym.HELD_VARIANTS))
        self.assertTrue(any(base.SUBGOALS[0] in prem for c in held for prem,_ in gym.logical_rules(c)))
        self.assertTrue(any(head==base.GOAL and 8 in prem for c in held for prem,head in gym.logical_rules(c)))

    def test_independent_bfs_and_shortest_dp_agree(self):
        for sp in gym.SPLITS:
            for c in self.ds[sp][::max(1,len(self.ds[sp])//21)]:
                _,_,goal,d,closure=verify.independent_reference(c)
                dp=gym.optimal_remaining(c,gym.initial_facts(c))
                self.assertEqual(d,dp if dp<gym.INFINITY else None)
                self.assertEqual(d is not None,goal in closure)

    def test_host_soundness_and_stop_legal(self):
        c=self.ds['held_depth5'][0];state=frozenset(gym.initial_facts(c))
        self.assertEqual(gym.sound_fire(c,state,gym.STOP),(state,None))
        with self.assertRaisesRegex(ValueError,'inadmissible'):
            gym.sound_fire(c,state,999)
        for _ in range(base.ATOMS):
            actions=gym.enabled(c,state)
            if not actions:break
            state,proof=gym.sound_fire(c,state,actions[0])
            self.assertTrue(set(state)<=gym.closure(c))
            self.assertTrue(proof)

    def test_bellman_action_costs_with_independent_physical_bfs(self):
        for c in self.ds['held_depth4'][::15]+self.ds['train'][::37]:
            state=frozenset(gym.initial_facts(c))
            d,independent=verify.independent_costs(c,state)
            exact=gym.action_costs(c,state)
            for i,value in independent.items():self.assertEqual(value,exact[i])
            if d is not None and d>0:
                optimal=gym.optimal_actions(c,state)
                self.assertTrue(all(exact[i]==float(d) for i in optimal))
                self.assertEqual(exact[gym.STOP],float(gym.CAP))

    def test_stop_calibration_targets_unreachable_and_reachable(self):
        provable=next(c for c in self.ds['train'] if gym.optimal_remaining(c,gym.initial_facts(c))<gym.INFINITY)
        impossible=next(c for c in self.ds['train'] if gym.optimal_remaining(c,gym.initial_facts(c))>=gym.INFINITY)
        x,known,gate,opt,costs,d,reach,stop=experiment.supervision([provable,impossible],
          [frozenset(gym.initial_facts(provable)),frozenset(gym.initial_facts(impossible))])
        self.assertEqual(stop.tolist(),[0.,1.])
        self.assertEqual(reach.tolist(),[1.,0.])
        self.assertEqual(costs[0,gym.STOP],0.)
        self.assertEqual(costs[1,gym.STOP],gym.CAP/3.)
        self.assertTrue(bool(gate[:,gym.STOP].all()))

    def test_identical_model_architecture_and_native_gradients(self):
        m=model.BellmanTypedPolicy()
        self.assertEqual(sum(p.numel() for p in m.parameters()),5268)
        cc=[r[0] for r in experiment.training_states()[0][:12]]
        state=[r[1] for r in experiment.training_states()[0][:12]]
        x,bits,gate,opt,costs,d,reach,stop=experiment.supervision(cc,state)
        raw,value,p=m.full(x,bits)
        self.assertEqual(raw.shape,(12,15));self.assertEqual(value.shape,(12,));self.assertEqual(p.shape,(12,))
        for arm in ('imitation','stop','bellman'):
            m.zero_grad(set_to_none=True)
            loss,_=experiment.objective(raw,value,p,gate,opt,costs,d,reach,stop,arm)
            loss.backward(retain_graph=True)
            self.assertGreater(sum(float(q.grad.square().sum()) for q in m.cell.parameters() if q.grad is not None),0.)

    def test_physical_layout_transport(self):
        from dataclasses import replace
        c=self.ds['held_depth4'][0];other=replace(c,permutation=54)
        m=model.BellmanTypedPolicy().eval();state=frozenset(gym.initial_facts(c))
        x,known,_=gym.observations([c],[state]);y,other_known,_=gym.observations([other],[state])
        with torch.no_grad():a=m(x,known)[0];b=m(y,other_known)[0]
        _,perm=gym.physical_maps(c.permutation);_,other_perm=gym.physical_maps(other.permutation)
        for i in range(len(gym.logical_rules(c))):self.assertAlmostEqual(float(a[perm[i]]),float(b[other_perm[i]]),places=4)
        self.assertAlmostEqual(float(a[-1]),float(b[-1]),places=4)

    def test_engineering_smoke_and_independent_mutation_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);tr=d/'training';ev=d/'evaluation'
            experiment.train(17,tr,smoke=True)
            experiment.evaluate(17,tr,ev,smoke=True)
            a=verify.audit(17,tr,ev,d/'audit.json',smoke=True)
            self.assertTrue(a['passed']);self.assertGreater(a['independent_cost_label_checks'],0)
            lines=[json.loads(r) for r in (ev/'episodes.jsonl').read_text().splitlines()]
            row=next(x for x in lines if x['actions']);row['actions'][0]['enabled']=[999]
            experiment.jsonl(ev/'episodes.jsonl',lines)
            report=json.loads((ev/'evaluation.json').read_text())
            report['episode_sha256']=experiment.sha(ev/'episodes.jsonl')
            experiment.write(ev/'evaluation.json',report)
            with self.assertRaisesRegex(ValueError,'enabled-action'):
                verify.audit(17,tr,ev,d/'tampered.json',smoke=True)

if __name__=='__main__':unittest.main()

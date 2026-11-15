"""AF-16 independent rule transfer, proof gates, data and smoke audit tests."""
import json,tempfile,unittest
from dataclasses import replace
from pathlib import Path
import torch
from experiments.activation_funnel_af05 import gym as base
from experiments.activation_funnel_af15.model import TypedHornPolicy,objective
from experiments.activation_funnel_af16 import gym,experiment,verify

class AF16Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.data={s:gym.cases_for(s) for s in gym.SPLITS}

    def test_fresh_family_counts(self):
        self.assertEqual({s:len(v) for s,v in self.data.items()},
             {'train':928,'dev':352,'novel_layout':64,'novel_depth4':192,'novel_depth5':192})
        self.assertEqual(experiment.check_manifest(),experiment.sha(experiment.MANIFEST) if hasattr(experiment,'MANIFEST') else experiment.sha(experiment.HERE/'fresh_manifest.json'))

    def test_new_composition_is_structurally_novel(self):
        for split in ('novel_depth4','novel_depth5'):
            self.assertTrue(all(isinstance(c,gym.Case) and c.variant in gym.HELD_VARIANTS for c in self.data[split]))
            self.assertTrue(any(base.SUBGOALS[0] in prem for c in self.data[split] for prem,_ in gym.logical_rules(c)))
            self.assertTrue(any(8 in prem and head==base.GOAL for c in self.data[split] for prem,head in gym.logical_rules(c)))

    def test_independent_shortest_and_closure_agree(self):
        for split in gym.SPLITS:
            for c in self.data[split][::max(1,len(self.data[split])//15)]:
                rules,initial,goal,d,closure=verify.independent_reference(c)
                self.assertEqual((d is not None),base.GOAL in gym.complete_closure(c))
                self.assertEqual(d is None,gym.optimal_remaining(c,gym.initial_facts(c))>=10**6)
                if d is not None:self.assertEqual(d,gym.optimal_remaining(c,gym.initial_facts(c)))

    def test_host_gate_rejects_illegal_action_and_enforces_soundness(self):
        for c in self.data['novel_depth5'][::18]:
            known=frozenset(gym.initial_facts(c))
            with self.assertRaisesRegex(ValueError,'inadmissible'):
                gym.sound_fire(c,known,999)
            for _ in range(base.ATOMS+1):
                enabled=gym.enabled(c,known)
                if not enabled:break
                nxt,edge=gym.sound_fire(c,known,enabled[0])
                self.assertTrue(known<=nxt<=gym.complete_closure(c));known=nxt

    def test_heldout_disjointness(self):
        m=gym.manifest();self.assertEqual(m['manifest_sha256'],base.digest({k:v for k,v in m.items() if k!='manifest_sha256'}))
        prev=set()
        for split in gym.SPLITS:
            rows={gym.case_hash(c) for c in self.data[split]}
            self.assertFalse(prev&rows);prev|=rows

    def test_rule_layout_transport_logits(self):
        c=self.data['novel_depth4'][0]
        d=replace(c,permutation=45)
        torch.manual_seed(6);m=TypedHornPolicy().eval()
        q=frozenset(gym.initial_facts(c))
        a,aa,_=gym.observations([c],[q]);b,bb,_=gym.observations([d],[q])
        with torch.no_grad():x=m(a,aa)[0];y=m(b,bb)[0]
        _,p=gym.physical_maps(44);_,r=gym.physical_maps(45)
        for j in range(len(gym.logical_rules(c))):
            self.assertAlmostEqual(float(x[p[j]]),float(y[r[j]]),places=4)
        self.assertAlmostEqual(float(x[-1]),float(y[-1]),places=4)

    def test_rule_priority_gradient_through_original_cell(self):
        torch.manual_seed(16)
        model=TypedHornPolicy()
        selected=[(c,gym.reference_episode(c)[0][0]) for c in self.data['train'][:12]]
        x,bits,gate,target=gym.observations([c for c,_ in selected],[s for _,s in selected],with_labels=True)
        loss,_=objective(model(x,bits),gate,target,'frontier');loss.backward()
        self.assertEqual(sum(p.numel() for p in model.parameters()),5234)
        self.assertGreater(sum(float(t.grad.square().sum()) for t in model.cell.parameters() if t.grad is not None),0)

    def test_shortest_host_action_has_valid_proof_trace(self):
        batch=experiment.rollout(None,list(self.data['novel_depth4'][:12]),baseline='shortest')
        for c,row in zip(self.data['novel_depth4'][:12],batch):
            self.assertTrue(verify.verify_row(c,row))
            self.assertEqual(row['solved'],bool(row['goal_truth']))

    def test_engineering_smoke_replay_and_tampered_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory);tr=d/'training';ev=d/'evaluation'
            experiment.train(17,tr,smoke=True)
            experiment.evaluate(17,tr,ev,smoke=True)
            a=verify.audit(17,tr,ev,d/'audit.json',smoke=True)
            self.assertTrue(a['passed'])
            rows=[json.loads(x) for x in (ev/'episodes.jsonl').read_text().splitlines()]
            for row in rows:
                if row['actions']:
                    row['actions'][0]['enabled']=[999]
                    break
            experiment.jsonl(ev/'episodes.jsonl',rows)
            summary=json.loads((ev/'evaluation.json').read_text())
            summary['episode_sha256']=experiment.sha(ev/'episodes.jsonl')
            experiment.write(ev/'evaluation.json',summary)
            with self.assertRaisesRegex(ValueError,'eligibility'):
                verify.audit(17,tr,ev,d/'bad.json',smoke=True)

if __name__=='__main__':unittest.main()

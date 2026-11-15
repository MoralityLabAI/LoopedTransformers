"""AF-19 preregistration controls for graph ancestry, proof validity and STOP sensitivity."""
from __future__ import annotations
from dataclasses import replace,fields
import json,tempfile,unittest
from pathlib import Path
import torch
from experiments.activation_funnel_af05 import gym as base
from experiments.activation_funnel_af19 import gym,model,experiment as ex,verify

class AF19Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_frozen_disjoint_manifest(self):
        self.assertEqual({s:len(gym.cases_for(s)) for s in gym.SPLITS},
            {'train':928,'dev':352,'held_layout':32,'held_depth4':192,'held_depth5':192})
        self.assertEqual(gym.manifest()['heldout_new_layouts'],[62,63,64,65,66])
        self.assertEqual(ex.check_manifest(),ex.sha(ex.HERE/'fresh_manifest.json'))

    def test_independent_rule_ancestry_all_splits_and_layouts(self):
        for split in gym.SPLITS:
            cases=gym.cases_for(split)
            for c in cases[::max(1,len(cases)//30)]:
                x,known,_=gym.observations([c],[gym.initial_facts(c)])
                local,anc=model.relevance_masks(x)
                src_l,src_a=verify.independent_rule_masks(c)
                self.assertEqual(set(torch.where(local[0])[0].tolist()),src_l)
                self.assertEqual(set(torch.where(anc[0])[0].tolist()),src_a)
                self.assertTrue(src_l <= src_a)

    def test_readout_parameter_and_initializer_match(self):
        torch.manual_seed(73);local=model.ReadoutPolicy('local',2)
        torch.manual_seed(73);ancestor=model.ReadoutPolicy('ancestor',2)
        self.assertEqual(sum(p.numel() for p in local.parameters()),6036)
        self.assertEqual(set(local.state_dict()),set(ancestor.state_dict()))
        self.assertTrue(all(torch.equal(v,ancestor.state_dict()[k]) for k,v in local.state_dict().items()))
        self.assertEqual(set(ex.CONFIGS),{'local_l2','ancestor_l2','local_l4','ancestor_l4'})
        self.assertEqual(sum(p.numel() for p in model.ReadoutPolicy('ancestor',4).parameters()),6036)

    def test_padded_rule_cannot_affect_stop_readout(self):
        c=gym.cases_for('held_depth5')[0];x,k,_=gym.observations([c],[gym.initial_facts(c)])
        missing=next(i for i in range(base.RULES) if i not in gym.physical_rules(c))
        altered=base.GraphBatch(*(getattr(x,f.name).clone() for f in fields(base.GraphBatch)))
        altered.rule_features[:,missing,:]=torch.tensor([4.5,2.5,-9.])
        for mode in model.MODES:
            m=model.ReadoutPolicy(mode,4).eval()
            with torch.no_grad():raw,d,reach=m.full(x,k);rr,dd,rh=m.full(altered,k)
            self.assertTrue(torch.equal(raw[:,-1],rr[:,-1]))
            self.assertTrue(torch.equal(d,dd))
            self.assertTrue(torch.equal(reach,rh))

    def test_physically_disconnected_irrelevant_fact_is_stable(self):
        c=gym.cases_for('held_depth4')[0]
        used={n for p,h in gym.logical_rules(c) for n in (*p,h)}
        irrelevant=next(n for n in range(base.ATOMS) if n not in used)
        x,k,_=gym.observations([c],[gym.initial_facts(c)])
        altered=base.GraphBatch(*(getattr(x,f.name).clone() for f in fields(base.GraphBatch)))
        apos,_=gym.physical_maps(c.permutation);slot=apos[irrelevant]
        altered.atom_features[:,slot,0]=1.
        k2=k.clone();k2[:,slot]=1.
        for mode in model.MODES:
            m=model.ReadoutPolicy(mode,4).eval()
            with torch.no_grad():a=m.full(x,k);b=m.full(altered,k2)
            for i in range(3):
                if i==0:self.assertTrue(torch.equal(a[i][:,-1],b[i][:,-1]))
                else:self.assertTrue(torch.equal(a[i],b[i]))

    def test_distant_fact_reaches_ancestor_pool_before_goal_cone(self):
        torch.manual_seed(71)
        local=model.ReadoutPolicy('local',2).eval()
        ancestor=model.ReadoutPolicy('ancestor',2).eval()
        ancestor.load_state_dict(local.state_dict())
        for a,b in gym.probe_pairs():
            self.assertGreater(gym.graph_distance(a,0),4)
            x,k,_=gym.observations([a,b],[gym.initial_facts(a),gym.initial_facts(b)])
            with torch.no_grad():l=local.full(x,k);g=ancestor.full(x,k)
            self.assertLess(float((l[0][0,-1]-l[0][1,-1]).abs()),1e-6)
            self.assertGreater(float((g[0][0,-1]-g[0][1,-1]).abs()),1e-5)
            self.assertNotEqual(base.GOAL in gym.closure(a),base.GOAL in gym.closure(b))

    def test_graph_layout_transport_preserves_logits(self):
        c=gym.cases_for('held_depth4')[0];d=replace(c,permutation=64)
        for mode in model.MODES:
            m=model.ReadoutPolicy(mode,4).eval()
            x,k,_=gym.observations([c],[gym.initial_facts(c)])
            y,kk,_=gym.observations([d],[gym.initial_facts(d)])
            with torch.no_grad():a=m.full(x,k);b=m.full(y,kk)
            _,r0=gym.physical_maps(c.permutation);_,r1=gym.physical_maps(d.permutation)
            for i in range(len(gym.logical_rules(c))):
                self.assertAlmostEqual(float(a[0][0,r0[i]]),float(b[0][0,r1[i]]),places=4)
            for j in range(3):
                aa=a[j][0,-1] if j==0 else a[j][0]
                bb=b[j][0,-1] if j==0 else b[j][0]
                self.assertAlmostEqual(float(aa),float(bb),places=4)

    def test_host_rule_gate_and_independent_bfs_costs(self):
        for c in gym.cases_for('held_depth5')[::30]:
            state=gym.initial_facts(c)
            _,costs=verify.independent_costs(c,state,16)
            oracle=gym.action_costs(c,state,16)
            self.assertEqual({i:x for i,x in enumerate(oracle) if x==x},costs)
            with self.assertRaisesRegex(ValueError,'inadmissible'):
                gym.sound_fire(c,state,base.RULES+1)

    def test_ancestry_path_gradients_reach_native_cell(self):
        c=gym.cases_for('train')[:4];s=[gym.initial_facts(x) for x in c]
        x,bits,gate,optimal,q,dist,reach,stop=ex.supervision(c,s,16)
        m=model.ReadoutPolicy('ancestor',2)
        out,dh,rh=m.full(x,bits)
        loss,_=ex.objective(out,dh,rh,gate,optimal,q,dist,reach,stop)
        loss.backward()
        self.assertGreater(sum(float(p.grad.square().sum()) for p in m.cell.parameters() if p.grad is not None),0)
        self.assertGreater(sum(float(p.grad.square().sum()) for p in m.summary_value.parameters() if p.grad is not None),0)

    def test_smoke_fit_eval_replay_and_tamper_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);training=p/'training';evaluation=p/'evaluation'
            rec=ex.train(17,training,smoke=True)
            self.assertEqual(set(rec['weights']),set(ex.CONFIGS))
            scored=ex.evaluate(17,training,evaluation,smoke=True)
            self.assertEqual(scored['probes'],12)
            audit=verify.audit(17,training,evaluation,p/'audit.json',smoke=True)
            self.assertTrue(audit['passed'])
            self.assertEqual(audit['cone_probe_replays'],12)
            rows=[json.loads(x) for x in (evaluation/'episodes.jsonl').read_text().splitlines()]
            target=next(x for x in rows if x['actions'])
            target['actions'][0]['enabled']=[999]
            ex.jsonl(evaluation/'episodes.jsonl',rows)
            scored['episodes_sha256']=ex.sha(evaluation/'episodes.jsonl')
            ex.write(evaluation/'evaluation.json',scored)
            with self.assertRaisesRegex(ValueError,'enabled-action'):
                verify.audit(17,training,evaluation,p/'tampered.json',smoke=True)

if __name__=='__main__':unittest.main()

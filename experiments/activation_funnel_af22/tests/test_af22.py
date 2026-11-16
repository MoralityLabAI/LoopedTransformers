"""AF-22 frozen proof-state source and neural Bellman planning contracts."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from experiments.activation_funnel_af05 import gym as base
from experiments.activation_funnel_af19.model import ReadoutPolicy
from experiments.activation_funnel_af21 import gym as parent
from experiments.activation_funnel_af22 import gym,planner,experiment as ex,verify

class AF22Test(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(2)

 def test_corpus_fixed_and_disjoint(self):
  self.assertEqual([len(gym.cases_for(s)) for s in gym.SPLITS],[928,352,32,192,192])
  self.assertIs(gym.cases_for('train'),parent.cases_for('train'))
  self.assertIs(gym.cases_for('dev'),parent.cases_for('dev'))
  self.assertEqual(ex.check_manifest(),ex.sha(ex.HERE/'fresh_manifest.json'))

 def test_fresh_independent_public_bfs(self):
  for sp in ('held_layout','held_depth4','held_depth5'):
   for c in gym.cases_for(sp)[::13]:
    rules,initial,goal=verify.public_graph(c)
    d=verify.shortest_bfs(rules,initial,goal)
    target=gym.optimal_remaining(c,gym.initial_facts(c))
    self.assertEqual(d,None if target>=gym.INFINITY else target)
    self.assertEqual(len(rules),len(gym.logical_rules(c)))

 def test_host_bars_forged_edge(self):
  c=gym.cases_for('held_depth5')[0]
  with self.assertRaisesRegex(ValueError,'host rejected'):
   gym.sound_fire(c,gym.initial_facts(c),gym.STOP+3)

 def test_bellman_backup_prefers_exact_shortest_edge(self):
  examples=[c for c in gym.cases_for('train') if 0<gym.optimal_remaining(c,gym.initial_facts(c))<gym.INFINITY and len(gym.enabled(c,gym.initial_facts(c)))>1]
  self.assertTrue(examples)
  c=examples[0];s=frozenset(gym.initial_facts(c))
  nodes,edges=planner.state_graph(c,s)
  def truecost(x):
   d=gym.optimal_remaining(c,x)
   return float(16 if d>=gym.INFINITY else d)
  v0={x:truecost(x) for x in nodes}
  stop={x:(0. if gym.optimal_remaining(c,x)==0 or gym.optimal_remaining(c,x)>=gym.INFINITY else 16.) for x in nodes}
  score,selected,_=planner.score_from_values(c,s,edges,v0,stop,[0.]*(gym.STOP+1))
  self.assertIn(selected['vi1'],gym.optimal_actions(c,s))
  self.assertIn(selected['vi2'],gym.optimal_actions(c,s))

 def test_cone_neighborhood_uses_only_legal_rule_fire(self):
  for c in gym.cases_for('held_depth5')[::26]:
   nodes,edges=planner.state_graph(c,gym.initial_facts(c))
   self.assertLessEqual(len(nodes),1+base.RULES+base.RULES**2)
   for s in nodes:
    for a,t in edges[s].items():
     self.assertIn(a,gym.enabled(c,s));self.assertEqual(t,gym.sound_fire(c,s,a)[0])

 def test_neural_value_terminal_boundary_and_finite(self):
  d=torch.tensor([0.,-2.,2.],requires_grad=True)
  r=torch.tensor([3.,-4.,0.],requires_grad=True)
  v,c=planner.neural_values(d,r,torch.tensor([True,False,False]))
  self.assertEqual(float(v[0].detach()),0.);self.assertEqual(float(c[0].detach()),0.)
  self.assertTrue(bool(torch.isfinite(v).all()))
  (v.sum()+c.sum()).backward()
  self.assertGreater(float(r.grad[1].abs()),0.)

 def test_shared_model_capacity_and_direct_audit(self):
  torch.manual_seed(3);model=ReadoutPolicy('ancestor',4)
  self.assertEqual(sum(p.numel() for p in model.parameters()),6036)
  c=next(c for c in gym.cases_for('train') if gym.enabled(c,gym.initial_facts(c)))
  s=frozenset(gym.initial_facts(c))
  out=planner.decisions(model,[c],[s])[0]
  p=planner.check_trace_math(c,s,out['trace'],out['scores'])
  self.assertEqual(p,out['selected'])
  self.assertEqual(set(out['scores']),set(planner.MODES))

 def test_planner_inference_never_calls_proof_oracle(self):
  c=next(c for c in gym.cases_for('held_depth4') if gym.enabled(c,gym.initial_facts(c)))
  m=ReadoutPolicy('ancestor',4)
  with patch.object(gym,'optimal_remaining',side_effect=AssertionError('oracle leakage')),\
       patch.object(gym,'optimal_actions',side_effect=AssertionError('action oracle leakage')):
   p=planner.decisions(m,[c],[frozenset(gym.initial_facts(c))])[0]
  self.assertEqual(set(p['selected']),set(planner.MODES))

 def test_bellman_boundary_tamper_rejected(self):
  c=next(c for c in gym.cases_for('train') if gym.enabled(c,gym.initial_facts(c)))
  m=ReadoutPolicy('ancestor',4);s=frozenset(gym.initial_facts(c))
  o=planner.decisions(m,[c],[s])[0]
  o['trace']['nodes'][0]['value0']+=1.
  with self.assertRaisesRegex(ValueError,'boundary arithmetic'):
   planner.check_trace_math(c,s,o['trace'],o['scores'])

 def test_full_smoke_and_forged_proof_rejected(self):
  with tempfile.TemporaryDirectory() as temp:
   p=Path(temp);tr=ex.train(17,p/'train',smoke=True)
   self.assertEqual(tr['parameters'],6036)
   ev=ex.evaluate(17,p/'train',p/'eval',smoke=True)
   audit=verify.audit(17,p/'train',p/'eval',p/'audit.json',smoke=True,replays=1)
   self.assertTrue(audit['verified']);self.assertGreater(audit['checkpoint_forward_replays'],0)
   rows=[json.loads(x) for x in (p/'eval'/'episodes.jsonl').read_text().splitlines()]
   target=next(r for r in rows if r['mode']=='vi2' and r['actions'])
   target['actions'][0]['action']=gym.STOP+1
   ex.jsonl(p/'eval'/'episodes.jsonl',rows)
   ev['episodes_sha256']=ex.sha(p/'eval'/'episodes.jsonl');ex.write(p/'eval'/'evaluation.json',ev)
   with self.assertRaises(ValueError):
    verify.audit(17,p/'train',p/'eval',p/'tampered.json',smoke=True,replays=1)

if __name__=='__main__':unittest.main()

"""AF-20 frozen split, matched objectives, independent Horn and replay checks."""
import unittest,tempfile,json
from pathlib import Path
from dataclasses import fields,replace
import torch
from experiments.activation_funnel_af05 import gym as base
from experiments.activation_funnel_af19.model import ReadoutPolicy
from experiments.activation_funnel_af20 import gym,experiment as ex,verify

class AF20Contracts(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(2)

 def test_manifest_counts_and_disjoint_new_layouts(self):
  self.assertEqual({s:len(gym.cases_for(s)) for s in gym.SPLITS},
   {'train':928,'dev':352,'held_layout':32,'held_depth4':192,'held_depth5':192})
  self.assertEqual(gym.manifest()['heldout_new_layouts'],list(range(67,72)))
  self.assertEqual(ex.check_manifest(),ex.sha(ex.HERE/'fresh_manifest.json'))

 def test_exact_pair_rosters_and_connected_semantics(self):
  counts={'train':(304,484),'dev':(118,176),'held_layout':(16,7),
   'held_depth4':(92,66),'held_depth5':(92,66)}
  for split,(flip,noop) in counts.items():
   ps=gym.pairs_for(split);self.assertEqual((sum(p.kind=='flip' for p in ps),
    sum(p.kind=='connected_invariant' for p in ps)),(flip,noop))
   cases=gym.cases_for(split)
   for p in ps[::max(1,len(ps)//15)]:
    x=verify.independent_pair(cases[p.left],cases[p.right])
    self.assertEqual(x['changed_atom'],p.changed_atom)
    self.assertEqual(x['incidence_distance'],p.graph_distance)
    self.assertGreater(p.graph_distance,0)
    if p.kind=='connected_invariant':
     self.assertEqual(x['minimum'][0],x['minimum'][1])
     self.assertEqual(x['truth'],[1,1])

 def test_public_physical_layout_and_incidence_bfs(self):
  for split in ('held_layout','held_depth4','held_depth5'):
   for c in gym.cases_for(split)[::25]:
    r,initial,goal,dist,closure=verify.independent_reference(c)
    d=gym.optimal_remaining(c,gym.initial_facts(c))
    self.assertEqual(dist,None if d>=gym.INFINITY else d)
    self.assertEqual(int(goal in closure),int(base.GOAL in gym.closure(c)))
    self.assertEqual(len(r),len(gym.logical_rules(c)))

 def test_equal_model_capacity_and_tied_initializer(self):
  torch.manual_seed(7);a=ReadoutPolicy('ancestor',2)
  torch.manual_seed(7);b=ReadoutPolicy('ancestor',4)
  self.assertEqual(sum(p.numel() for p in a.parameters()),6036)
  self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))

 def test_paired_and_bce_share_same_inputs_and_native_cell_gradients(self):
  torch.manual_seed(11);a=ReadoutPolicy('ancestor',2)
  torch.manual_seed(11);b=ReadoutPolicy('ancestor',2)
  cs=gym.cases_for('train')
  p=[next(p for p in gym.pairs_for('train') if p.kind==k)
     for k in ('flip','connected_invariant')]
  out_a,_=ex.pair_losses(a,cs,p,'bce');out_b,_=ex.pair_losses(b,cs,p,'pair')
  self.assertGreater(float(out_b.detach()),float(out_a.detach()))
  out_b.backward()
  self.assertGreater(sum(float(x.grad.square().sum()) for x in b.cell.parameters()
    if x.grad is not None),0.)
  self.assertGreater(sum(float(x.grad.square().sum()) for x in b.summary_value.parameters()
    if x.grad is not None),0.)

 def test_connected_distractor_is_conditional_not_a_global_structural_invariant(self):
  cs=gym.cases_for('held_depth4');p=next(p for p in gym.pairs_for('held_depth4')
     if p.kind=='connected_invariant')
  a,b=cs[p.left],cs[p.right]
  self.assertEqual(gym.logical_rules(a),gym.logical_rules(b))
  self.assertNotEqual(gym.initial_facts(a),gym.initial_facts(b))
  self.assertEqual(gym.optimal_remaining(a,gym.initial_facts(a)),
                   gym.optimal_remaining(b,gym.initial_facts(b)))
  self.assertGreater(gym.incidence_distance(a,p.changed_atom),0)

 def test_paired_forward_and_labels_have_nontrivial_rule_choices(self):
  cases=gym.cases_for('train')
  p=[next(p for p in gym.pairs_for('train') if p.kind==k)
    for k in ('flip','connected_invariant')]
  selected=[cases[z] for pair in p for z in (pair.left,pair.right)]
  x,bits,gate=gym.observations(selected,[gym.initial_facts(c) for c in selected])
  self.assertTrue(bool(gate[:,:-1].any(-1).all()))
  raw,dist,reach=ReadoutPolicy('ancestor',2).full(x,bits)
  self.assertEqual(tuple(ex.stop_gap(raw,gate).shape),(4,))

 def test_host_rejects_illegal_rule_and_cannot_forge_goal(self):
  for c in gym.cases_for('held_depth5')[::28]:
   with self.assertRaisesRegex(ValueError,'inadmissible'):
    gym.sound_fire(c,gym.initial_facts(c),base.RULES+1)

 def test_matched_pair_sampler_is_deterministic(self):
  a=ex.sample_pairs('train',torch.Generator().manual_seed(13))
  b=ex.sample_pairs('train',torch.Generator().manual_seed(13))
  self.assertEqual(a,b)
  self.assertEqual({x.kind for x in a},set(ex.PAIRS))

 def test_smoke_end_to_end_and_rehashed_tamper(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t)
   rec=ex.train(17,root/'training',smoke=True)
   self.assertEqual(set(rec['weights']),set(ex.CONFIGS))
   result=ex.evaluate(17,root/'training',root/'evaluation',smoke=True)
   audit=verify.audit(17,root/'training',root/'evaluation',root/'audit.json',smoke=True,replays=1)
   self.assertTrue(audit['verified'])
   self.assertGreater(audit['counterfactual_rows'],0)
   rows=[json.loads(s) for s in (root/'evaluation'/'paired_initial.jsonl').read_text().splitlines()]
   rows[0]['truth']=[1-row for row in rows[0]['truth']]
   ex.jsonl(root/'evaluation'/'paired_initial.jsonl',rows)
   result['paired_sha256']=ex.sha(root/'evaluation'/'paired_initial.jsonl')
   ex.write(root/'evaluation'/'evaluation.json',result)
   with self.assertRaisesRegex(ValueError,'counterfactual .* differs'):
    verify.audit(17,root/'training',root/'evaluation',root/'tampered.json',smoke=True,replays=1)

if __name__=='__main__':unittest.main()

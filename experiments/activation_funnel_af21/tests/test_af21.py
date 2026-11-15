"""AF-21 frozen mathematical, source, engineering and tamper contracts."""
import unittest,tempfile,json,math
from pathlib import Path
from dataclasses import replace
import torch
from experiments.activation_funnel_af05 import gym as base
from experiments.activation_funnel_af20 import gym as parent
from experiments.activation_funnel_af19.model import ReadoutPolicy
from experiments.activation_funnel_af21 import gym,experiment as ex,verify

class AF21Test(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(2)

 def test_new_logic_and_feature_split(self):
  self.assertEqual({s:len(gym.cases_for(s)) for s in gym.SPLITS},
    {'train':928,'dev':352,'held_layout':32,'held_depth4':192,'held_depth5':192})
  self.assertIs(gym.cases_for('train'),parent.cases_for('train'))
  self.assertIs(gym.cases_for('dev'),parent.cases_for('dev'))
  self.assertEqual(set(gym.HELD_VARIANTS),{'bridged_shortcut','crossed_detour','double_dependency'})
  self.assertEqual(ex.check_manifest(),ex.sha(ex.HERE/'fresh_manifest.json'))

 def test_host_oracle_independent_each_new_composition(self):
  for split in ('held_layout','held_depth4','held_depth5'):
   cs=gym.cases_for(split)
   for c in cs[::17]:
    rules,init,goal,d,closure=verify.independent_reference(c)
    teacher=gym.optimal_remaining(c,gym.initial_facts(c))
    self.assertEqual(d,None if teacher>=gym.INFINITY else teacher)
    self.assertEqual(int(goal in closure),int(base.GOAL in gym.closure(c)))
    self.assertEqual(len(rules),len(gym.logical_rules(c)))
    self.assertLessEqual(len(rules),base.RULES)
    if d is not None and d>0:
     good=gym.optimal_actions(c,gym.initial_facts(c))
     self.assertTrue(good)
     for r in good:
      s,_=gym.sound_fire(c,gym.initial_facts(c),r)
      self.assertEqual(gym.optimal_remaining(c,s),d-1)

 def test_paired_source_train_equal_parent(self):
  self.assertEqual(gym.pairs_for('train'),parent.pairs_for('train'))
  self.assertEqual(gym.pair_receipt('train')['digest'],parent.pair_receipt('train')['digest'])
  for s in ('held_depth4','held_depth5'):
   cases=gym.cases_for(s)
   self.assertEqual(len(gym.pairs_for(s)),152)
   for p in gym.pairs_for(s)[::13]:
    oracle=verify.independent_pair(cases[p.left],cases[p.right])
    self.assertEqual(oracle['truth'],[int(p.reachable_left),int(p.reachable_right)])

 def test_greedy_margin_uses_true_max_instead_of_logsumexp(self):
  raw=torch.zeros(2,base.RULES+1)
  gate=torch.zeros_like(raw,dtype=torch.bool)
  gate[:,-1]=True;gate[:,:2]=True
  raw[:,0]=.0;raw[:,1]=.0;raw[0,-1]=.2;raw[1,-1]=-.2
  greedy,mask=ex.greedy_gap(raw,gate)
  mass=ex.stop_gap(raw,gate)
  self.assertTrue(bool(mask.all()))
  self.assertAlmostEqual(float(greedy[0]),.2,places=6)
  self.assertLess(float(mass[0]),0.)
  self.assertAlmostEqual(float(greedy[1]),-.2,places=6)
  self.assertGreater(float((greedy-mass).min()),.69-1e-5)

 def test_stop_and_rule_signs_and_gradients(self):
  c=next(c for c in gym.cases_for('train') if
    0<gym.optimal_remaining(c,gym.initial_facts(c))<gym.INFINITY and
    len(gym.enabled(c,gym.initial_facts(c)))>1)
  x,known,gate,opt=gym.observations([c],[gym.initial_facts(c)],with_labels=True)
  torch.manual_seed(7);m=ReadoutPolicy('ancestor',4)
  raw,_,_=m.full(x,known)
  s,r,diag=ex.decision_losses(raw,gate,opt,[gym.initial_facts(c)],[c])
  self.assertTrue(bool(torch.isfinite(s+r)))
  self.assertGreaterEqual(diag['continue_examples'],1)
  (s+r).backward()
  self.assertGreater(sum(float(p.grad.square().sum()) for p in m.cell.parameters()
    if p.grad is not None),0.)

 def test_zero_legal_rule_is_stop_without_nan(self):
  raw=torch.zeros(2,base.RULES+1,requires_grad=True)
  gate=torch.zeros_like(raw,dtype=torch.bool);gate[:,-1]=True
  gap,valid=ex.greedy_gap(raw,gate)
  self.assertFalse(bool(valid.any()))
  self.assertTrue(bool(torch.isfinite(gap).all()))

 def test_same_model_capacity_and_initialization(self):
  torch.manual_seed(5);a=ReadoutPolicy('ancestor',4)
  torch.manual_seed(5);b=ReadoutPolicy('ancestor',4)
  self.assertEqual(sum(p.numel() for p in a.parameters()),6036)
  self.assertTrue(all(torch.equal(x,b.state_dict()[k]) for k,x in a.state_dict().items()))
  self.assertEqual(set(ex.CONFIGS),{'bce_l4','greedy_l4','greedy_rule_l4','greedy_rule_extra_l4'})

 def test_host_never_accepts_unsupported_rule(self):
  for c in gym.cases_for('held_depth5')[::32]:
   with self.assertRaisesRegex(ValueError,'host rejected'):
    gym.sound_fire(c,gym.initial_facts(c),base.RULES+1)

 def test_pair_sampler_determinism(self):
  self.assertEqual(ex.sample_pairs('train',torch.Generator().manual_seed(117)),
                   ex.sample_pairs('train',torch.Generator().manual_seed(117)))

 def test_smoke_train_score_replay_and_rehashed_tamper(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);tr=ex.train(17,p/'training',smoke=True)
   self.assertEqual(set(tr['weights']),set(ex.CONFIGS))
   ev=ex.evaluate(17,p/'training',p/'evaluation',smoke=True)
   chk=verify.audit(17,p/'training',p/'evaluation',p/'audit.json',smoke=True,replays=1)
   self.assertTrue(chk['verified'])
   self.assertGreater(chk['forward_replays'],0)
   records=[json.loads(x) for x in (p/'evaluation'/'paired_initial.jsonl').read_text().splitlines()]
   records[0]['truth']=[1-int(v) for v in records[0]['truth']]
   ex.jsonl(p/'evaluation'/'paired_initial.jsonl',records)
   ev['paired_sha256']=ex.sha(p/'evaluation'/'paired_initial.jsonl')
   ex.write(p/'evaluation'/'evaluation.json',ev)
   with self.assertRaisesRegex(ValueError,'counterfactual .* differs'):
    verify.audit(17,p/'training',p/'evaluation',p/'tampered.json',smoke=True,replays=1)

if __name__=='__main__':unittest.main()

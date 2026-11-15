"""Independent public-incidence Horn BFS and AF-21 proof/pair outcome replay.

This oracle reads physical premise/head tensors. The tested model's Horn-case
logical-rule generator, optimal_remaining, and pair labels are never the source
of independent ground truth.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
from collections import deque
from functools import lru_cache
import torch
from experiments.activation_funnel_af05 import gym as base
from . import experiment as ex,gym


def public_graph(c):
 x=gym.encode(c);p=x.prem_up+x.prem_down;h=x.head_up+x.head_down
 rules={}
 for j in range(base.RULES):
  if float(x.rule_features[j,2])<.5:continue
  prem=tuple(torch.where(p[j]>.5)[0].tolist());head=tuple(torch.where(h[j]>.5)[0].tolist())
  if not prem or len(prem)>2 or len(head)!=1:raise ValueError('invalid encoded public rule')
  rules[j]=(prem,head[0])
 initial=frozenset(torch.where(x.atom_features[:,0]>.5)[0].tolist())
 return rules,initial,int(x.read_slots[2])


def bfs(rules,known,goal):
 known=frozenset(known)
 if goal in known:return 0
 queue=deque([(known,0)]);seen={known}
 while queue:
  s,d=queue.popleft()
  for premise,head in rules.values():
   if head in s or not set(premise)<=s:continue
   nxt=s|{head}
   if goal in nxt:return d+1
   if nxt not in seen:seen.add(nxt);queue.append((nxt,d+1))
 return None

@lru_cache(maxsize=4096)
def independent_reference(c):
 rules,initial,goal=public_graph(c)
 state=set(initial)
 for _ in range(base.ATOMS+1):
  nxt=state|{h for p,h in rules.values() if set(p)<=state}
  if state==nxt:break
  state=nxt
 else:raise ValueError('independent Horn closure cannot converge')
 d=bfs(rules,initial,goal)
 if (d is None)!=(goal not in state):raise ValueError('BFS/closure disagree')
 return rules,initial,goal,d,frozenset(state)


def independent_connected_distance(c,atom):
 rules,_,goal,_,_=independent_reference(c)
 amap,_=gym.physical_maps(c.permutation);start=amap[atom]
 adj={('a',i):set() for i in range(base.ATOMS)}
 for j,(p,h) in rules.items():
  rule=('r',j);adj[rule]=set()
  for i in (*p,h):adj[('a',i)].add(rule);adj[rule].add(('a',i))
 todo=deque([(('a',start),0)]);seen={('a',start)}
 while todo:
  node,n=todo.popleft()
  if node==('a',goal):return n
  for k in adj[node]:
   if k not in seen:seen.add(k);todo.append((k,n+1))
 return None


def independent_pair(c0,c1):
 r0,init0,goal0,d0,_=independent_reference(c0)
 r1,init1,goal1,d1,_=independent_reference(c1)
 if r0!=r1 or goal0!=goal1:raise ValueError('counterfactual pair has different public Horn program')
 a0,_=gym.physical_maps(c0.permutation);a1,_=gym.physical_maps(c1.permutation)
 if a0!=a1:raise ValueError('pair changed public graph layout')
 delta=list(init0^init1)
 if len(delta)!=1:raise ValueError('counterfactual must toggle exactly one initial fact')
 atom=a0.index(delta[0]);distance=independent_connected_distance(c0,atom)
 if distance is None:raise ValueError('distractor graph-disconnected')
 legal0=[i for i,(p,h) in r0.items() if h not in init0 and set(p)<=init0]
 legal1=[i for i,(p,h) in r1.items() if h not in init1 and set(p)<=init1]
 if not legal0 or not legal1:raise ValueError('pair has trivially forced STOP')
 if (d0 is None)!=(d1 is None):kind='flip'
 elif d0 is not None and d0==d1:kind='connected_invariant'
 else:raise ValueError('pair does not meet exact counterfactual label contract')
 return {'kind':kind,'changed_atom':atom,'incidence_distance':distance,
    'truth':[int(d0 is not None),int(d1 is not None)],'minimum':[d0,d1]}


def verify_proof_row(c,row):
 if (row['split'],row['case_id'])!=(c.split,c.id):raise ValueError('episode case identity wrong')
 rules,init,goal,d,closure=independent_reference(c)
 amap,_=gym.physical_maps(c.permutation);inv={a:b for b,a in enumerate(amap)}
 if row['initial']!=sorted(inv[i] for i in init):raise ValueError('episode initial facts changed')
 if row['goal_truth']!=int(goal in closure):raise ValueError('goal truth changed')
 budget=2 if d is None else d+1
 if row['budget']!=budget or row['optimal_steps']!=d:raise ValueError('episode proof budget changed')
 if len(row['actions'])>budget:raise ValueError('episode exceeds proof budget')
 state=set(init);stopped=False
 for t,step in enumerate(row['actions']):
  if step['before']!=sorted(inv[k] for k in state):raise ValueError('prior Horn state differs')
  legal=sorted(i for i,(p,h) in rules.items() if h not in state and set(p)<=state)
  if step['enabled']!=legal:raise ValueError('enabled Horn rule roster changed')
  act=step['action'];proof=step['proof']
  if act==gym.STOP:
   if proof is not None:raise ValueError('STOP cannot carry a proof')
   stopped=True
  else:
   if act not in legal:raise ValueError('host accepted unsupported rule')
   p,h=rules[act]
   if not isinstance(proof,dict) or proof.get('rule_slot')!=act or proof.get('head')!=inv[h] or sorted(proof.get('premises',[]))!=sorted(inv[i] for i in p):
    raise ValueError('host proof trace not bound to original rule incidence')
   state.add(h)
  if step['after']!=sorted(inv[k] for k in state):raise ValueError('after-state tampered')
  raw=step['raw_logits']
  if raw is not None:
   if len(raw)!=gym.STOP+1 or any(not math.isfinite(v) for v in raw):raise ValueError('malformed raw policy')
   if max(legal+[gym.STOP],key=lambda j:raw[j])!=act:raise ValueError('policy argmax does not match action')
  if stopped or goal in state:
   if t!=len(row['actions'])-1:raise ValueError('episode kept running after terminal state')
 solved=goal in state
 if row['solved']!=solved or row['stopped']!=(stopped or solved) or row['used_actions']!=len(row['actions']):
  raise ValueError('episode terminal metadata mismatch')
 if row['final']!=sorted(inv[k] for k in state):raise ValueError('final Horn state changed')
 if row['early_stop']!=bool(d is not None and not solved and stopped):raise ValueError('early stop flag changed')
 if row['host_accepted_false_proof'] or solved and goal not in closure:raise ValueError('unsound Horn proof')


def verify_pair_row(cases,row):
 a=next((c for c in cases if c.id==row['left_id']),None)
 b=next((c for c in cases if c.id==row['right_id']),None)
 if a is None or b is None:raise ValueError('unknown pair identity')
 data=independent_pair(a,b)
 for k,v in data.items():
  if row[k]!=v:raise ValueError('counterfactual '+k+' differs from independent Horn oracle')
 if row['split']!=a.split:raise ValueError('pair split inconsistent')
 for key in ('reach_logits','stop_gaps','stop_mass_gaps','stop_raw'):
  if len(row[key])!=2 or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in row[key]):
   raise ValueError('pair logits invalid')


def calc_proof_score(rows):
 good=[r for r in rows if r['goal_truth']]
 return {'cases':len(rows),'provable':len(good),'solved':sum(bool(r['solved']) for r in good),
         'proof_success':sum(bool(r['solved']) for r in good)/len(good) if good else 1.,
         'early_stop_positive':sum(bool(r['early_stop']) for r in good)/len(good) if good else 0.,
         'false_accepted_proofs':sum(bool(r['host_accepted_false_proof']) for r in rows)}


def calc_pair_score(rows):
 flips=[r for r in rows if r['kind']=='flip'];noops=[r for r in rows if r['kind']=='connected_invariant']
 if not flips or not noops:raise ValueError('missing required counterfactual class')
 sigmoid=lambda x:1/(1+math.exp(-max(-60,min(60,x))))
 correct=reachcorrect=0
 for row in flips:
  t0,t1=row['truth'];r0,r1=row['reach_logits'];s0,s1=row['stop_gaps']
  sign=1 if t0 else -1
  reachcorrect+=int((r0>0)==bool(t0) and (r1>0)==bool(t1))
  correct+=int(sign*(r0-r1)>0 and sign*(s0-s1)<0)
 dr=[abs(sigmoid(r['reach_logits'][0])-sigmoid(r['reach_logits'][1])) for r in noops]
 ds=[abs(sigmoid(r['stop_gaps'][0])-sigmoid(r['stop_gaps'][1])) for r in noops]
 return {'flip_pairs':len(flips),'connected_invariant_pairs':len(noops),
  'flip_reach_accuracy':reachcorrect/len(flips),'flip_joint_order_accuracy':correct/len(flips),
  'connected_mean_abs_reach_delta':sum(dr)/len(dr),
  'connected_mean_abs_stop_delta':sum(ds)/len(ds),
  'connected_both_within_0p1':sum(a<=.1 and b<=.1 for a,b in zip(dr,ds))/len(noops)}


def audit(seed,training,evaluation,out,smoke=False,replays=2):
 training=Path(training);evaluation=Path(evaluation)
 tr,models=ex.load_models(training,seed,smoke)
 result=json.loads((evaluation/'evaluation.json').read_text())
 if (result['experiment'],result['seed'],result['smoke'])!=('AF-21',seed,smoke):raise ValueError('AF21 score identity changed')
 if result['source_hashes']!=ex.source_hashes() or result['manifest_sha256']!=ex.check_manifest() or \
    result['training_sha256']!=ex.sha(training/'train_receipt.json') or \
    result['episode_sha256']!=ex.sha(evaluation/'episodes.jsonl') or \
    result['paired_sha256']!=ex.sha(evaluation/'paired_initial.jsonl'):
  raise ValueError('AF21 source or scored file checksum changed')
 episode=[json.loads(x) for x in (evaluation/'episodes.jsonl').read_text().splitlines()]
 pair=[json.loads(x) for x in (evaluation/'paired_initial.jsonl').read_text().splitlines()]
 epidx={};pairidx={};replayed=0;replayed_pairs=0
 for row in episode:
  key=(row['arm'],row['split'],row['case_id'])
  if key in epidx:raise ValueError('duplicate episode row')
  epidx[key]=row
 for row in pair:
  key=(row['arm'],row['split'],row['left_id'],row['right_id'])
  if key in pairidx:raise ValueError('duplicate paired row')
  pairidx[key]=row
 families=('dev',) if smoke else ('held_layout','held_depth4','held_depth5')
 expected_episode=sum(len(gym.cases_for(sp)) for sp in families)*len(result['scores'])
 expected_pair=sum(len(gym.pairs_for(sp)) for sp in families)*len(ex.CONFIGS)
 if len(episode)!=expected_episode or len(pair)!=expected_pair or \
    result['episode_rows']!=expected_episode or result['paired_rows']!=expected_pair:
  raise ValueError('AF21 scored row roster incomplete')
 for arm in result['scores']:
  for split in families:
   cs=gym.cases_for(split);obs=[]
   for c in cs:
    key=(arm,split,c.id)
    if key not in epidx:raise ValueError('missing episode')
    row=epidx[key];verify_proof_row(c,row);obs.append(row)
   if result['scores'][arm][split]!=calc_proof_score(obs):raise ValueError('independent proof score differs')
   if arm in models:
    for c in list(cs[:replays])+list(cs[-replays:]):
     for step in epidx[(arm,split,c.id)]['actions']:
      x,k,_=gym.observations([c],[frozenset(step['before'])])
      with torch.no_grad():pred=models[arm](x,k)[0]
      if not torch.allclose(pred,torch.tensor(step['raw_logits']),atol=1e-5,rtol=0):
       raise ValueError('checkpoint forward proof-logit replay mismatch')
      replayed+=1
 for arm in ex.CONFIGS:
  for split in families:
   cs=gym.cases_for(split);obs=[]
   for p in gym.pairs_for(split):
    key=(arm,split,cs[p.left].id,cs[p.right].id)
    if key not in pairidx:raise ValueError('missing registered counterfactual')
    row=pairidx[key];verify_pair_row(cs,row);obs.append(row)
   observed=calc_pair_score(obs);expected=result['pairs'][arm][split]
   for k in observed:
    if isinstance(observed[k],float):
     if abs(observed[k]-expected[k])>1e-12:raise ValueError('independent paired statistic mismatch')
    elif observed[k]!=expected[k]:raise ValueError('independent paired count mismatch')
   for row in obs[:replays]+obs[-replays:]:
    left=next(c for c in cs if c.id==row['left_id'])
    right=next(c for c in cs if c.id==row['right_id'])
    x,k,gate=gym.observations([left,right],[gym.initial_facts(left),gym.initial_facts(right)])
    with torch.no_grad():raw,dist,reach=models[arm].full(x,k)
    gaps,_=ex.greedy_gap(raw,gate)
    mass=ex.stop_gap(raw,gate)
    for label,actual in [('reach_logits',reach.tolist()),('stop_gaps',gaps.tolist()),('stop_mass_gaps',mass.tolist()),('stop_raw',raw[:,-1].tolist())]:
     if max(abs(x-y) for x,y in zip(row[label],actual))>1e-5:
      raise ValueError('saved counterfactual logits differ from actual checkpoint')
    replayed_pairs+=1
 # Recompute exact training pairing target roster from physical incidence even on source data.
 checked_training=0
 train_cases=gym.cases_for('train')
 for p in gym.pairs_for('train'):
  data=independent_pair(train_cases[p.left],train_cases[p.right])
  if (data['kind'],data['changed_atom'],data['incidence_distance'],data['truth'],data['minimum'])!=\
     (p.kind,p.changed_atom,p.graph_distance,[int(p.reachable_left),int(p.reachable_right)],
      [p.minimum_left,p.minimum_right]):raise ValueError('training pair oracle mismatch')
  checked_training+=1
 rec={'experiment':'AF-21','seed':seed,'verified':True,'host_proof_rows':len(episode),
   'counterfactual_rows':len(pair),'forward_replays':replayed,'pair_forward_replays':replayed_pairs,
   'independent_train_pair_oracle_checks':checked_training,'false_accepted_proofs':0,
   'evaluation_sha256':ex.sha(evaluation/'evaluation.json'),
   'training_sha256':ex.sha(training/'train_receipt.json'),'source_hashes':ex.source_hashes()}
 ex.write(out,rec);return rec


def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True)
 p.add_argument('--training',type=Path,required=True);p.add_argument('--evaluation',type=Path,required=True)
 p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true')
 a=p.parse_args();print(json.dumps(audit(a.seed,a.training,a.evaluation,a.out,a.smoke)))
if __name__=='__main__':main()

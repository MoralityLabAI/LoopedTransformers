"""AF-21: directly train greedy STOP and optimal-rule margins.

All training labels come from Horn closure and exact shortest-goal search. The
neural rule proposals are host-gated; training and scoring use disjoint layouts.
"""
from __future__ import annotations
import argparse,hashlib,json,time,math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiments.activation_funnel_af05 import gym as base
from experiments.activation_funnel_af19 import experiment as parent
from experiments.activation_funnel_af19.model import ReadoutPolicy
from . import gym

HERE=Path(__file__).parent
SEEDS=(17,29,43,61,73,89,103,131)
CONFIGS={'bce_l4':('bce',4,240),'greedy_l4':('greedy',4,240),
         'greedy_rule_l4':('greedy_rule',4,240),
         'greedy_rule_extra_l4':('greedy_rule',4,360)}
STEPS=240;BATCH=32;PAIR_COUNT=8
PAIRS=('flip','connected_invariant')
GAMMA=.50
STOP_WEIGHT=.50
PAIR_STOP_WEIGHT=.25
RULE_WEIGHT=.40
BASELINES=('fixed','uniform','shortest')
SOURCE_FILES=['gym.py','experiment.py','verify.py','aggregate.py','PROTOCOL.json',
 'fresh_manifest.json','GreedyStop.lean','README.md','tests/test_af21.py',
 '../../.github/workflows/activation-funnel-af21.yml',
 '../activation_funnel_af20/gym.py','../activation_funnel_af20/experiment.py',
 '../activation_funnel_af20/verify.py','../activation_funnel_af19/gym.py','../activation_funnel_af19/model.py',
 '../activation_funnel_af19/experiment.py','../activation_funnel_af19/verify.py',
 '../activation_funnel_af18/gym.py','../activation_funnel_af17/gym.py',
 '../activation_funnel_af17/model.py','../activation_funnel_af17/experiment.py',
 '../activation_funnel_af16/gym.py','../activation_funnel_af15/model.py',
 '../activation_funnel_af02/model.py','../lean_reasoner/train.py',
 '../lean_reasoner/proof_data.py']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def source_hashes():return {n:sha(HERE/n) for n in SOURCE_FILES}
def write(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n')
def jsonl(path,rows):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w') as f:
  for r in rows:f.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')
def check_manifest():
 frozen=json.loads((HERE/'fresh_manifest.json').read_text())
 if gym.manifest()!=frozen:raise ValueError('AF21 frozen corpus/pair roster changed')
 return sha(HERE/'fresh_manifest.json')

def primary_loss(model,cases,states):
 x,bits,gate,optimal=gym.observations(cases,states,with_labels=True)
 raw,dist,reach=model.full(x,bits)
 n=len(cases)
 q=torch.zeros_like(raw);dd=[];rr=[];ss=[]
 for i,(case,s) in enumerate(zip(cases,states)):
  d=gym.optimal_remaining(case,s)
  rr.append(float(d<gym.INFINITY))
  ss.append(float(d==0 or d>=gym.INFINITY))
  dd.append(float(16 if d>=gym.INFINITY else min(d,16)))
  costs=gym.action_costs(case,s,cap=16)
  for j in torch.where(gate[i])[0].tolist():q[i,j]=(16-costs[j])/3.
 loss,parts=parent.objective(raw,dist,reach,gate,optimal,q,
     torch.tensor(dd),torch.tensor(rr),torch.tensor(ss))
 return loss,parts


def stop_gap(logits,gate):
 """STOP logit versus logsumexp of legal rule logits; call only on nonempty rules."""
 if not bool(gate[:,:-1].any(-1).all()):raise ValueError('STOP contrast requires enabled rule on every input')
 return logits[:,-1]-torch.logsumexp(logits[:,:-1].masked_fill(~gate[:,:-1],-1e4),-1)


def pair_losses(model, cases, pairs, objective):
 """Same pair forwards for every arm, only direct-decision losses differ."""
 if objective not in ('bce','greedy','greedy_rule'):
  raise ValueError('unregistered AF21 training objective')
 cc=[cases[p.left] for p in pairs]+[cases[p.right] for p in pairs]
 states=[gym.initial_facts(c) for c in cc]
 x,bits,gate,optimal=gym.observations(cc,states,with_labels=True)
 logits,dist,reach=model.full(x,bits)
 n=len(pairs)
 labels=torch.tensor([float(p.reachable_left) for p in pairs]+
                     [float(p.reachable_right) for p in pairs])
 # Preserve AF20's identical pair classification data and supervision.
 mass_gap=stop_gap(logits,gate)
 reach_loss=F.binary_cross_entropy_with_logits(reach,labels)
 stop_loss=F.binary_cross_entropy_with_logits(mass_gap,1-labels)
 return .6*(reach_loss+stop_loss),{'pair_reach':float(reach_loss.detach()),
   'pair_mass_stop':float(stop_loss.detach())},(logits,gate,optimal,cc,states)


def greedy_gap(raw,gate):
 """STOP minus the actual best enabled rule, with no STOP in rule maximum."""
 if raw.shape!=gate.shape or gate.dtype!=torch.bool or not bool(gate[:,-1].all()):
  raise ValueError('greedy margin must use host legal-rule mask')
 rule=raw[:,:-1].masked_fill(~gate[:,:-1],-1e4)
 best=rule.amax(-1)
 valid=gate[:,:-1].any(-1)
 return torch.where(valid,raw[:,-1]-best,torch.zeros_like(best)),valid


def decision_losses(raw,gate,optimal,states,cases):
 """Actual greedy STOP decision plus optimal-rule selection, separately."""
 gap,has_rule=greedy_gap(raw,gate)
 reach=torch.tensor([0<gym.optimal_remaining(c,s)<gym.INFINITY
                    for c,s in zip(cases,states)],dtype=torch.bool)
 must_continue=has_rule & reach
 must_stop=has_rule & ~reach
 stop_parts=[]
 if bool(must_continue.any()):stop_parts.append(F.softplus(GAMMA+gap[must_continue]))
 if bool(must_stop.any()):stop_parts.append(F.softplus(GAMMA-gap[must_stop]))
 stop_loss=torch.cat(stop_parts).mean() if stop_parts else raw.sum()*0.
 legal=gate[:,:-1];opt=optimal[:,:-1]
 ranked=must_continue & opt.any(-1) & (legal & ~opt).any(-1)
 if bool(ranked.any()):
  best_good=raw[:,:-1].masked_fill(~opt,float('-inf')).amax(-1)
  best_bad=raw[:,:-1].masked_fill(~(legal&~opt),float('-inf')).amax(-1)
  rule_loss=F.softplus(GAMMA+best_bad[ranked]-best_good[ranked]).mean()
 else:rule_loss=raw.sum()*0.
 return stop_loss,rule_loss,{
  'greedy_margin_mean':float(gap[has_rule].detach().mean()) if bool(has_rule.any()) else 0.,
  'stop_loss':float(stop_loss.detach()),'rule_loss':float(rule_loss.detach()),
  'continue_examples':int(must_continue.sum()),'stop_examples':int(must_stop.sum()),
  'ranked_examples':int(ranked.sum())}


def sample_pairs(split,generator,smoke=False):
 rows=gym.pairs_for(split)
 grouped={kind:[p for p in rows if p.kind==kind] for kind in PAIRS}
 count=2 if smoke else PAIR_COUNT
 result=[]
 for kind in PAIRS:
  ix=torch.randint(len(grouped[kind]),(count,),generator=generator)
  result.extend(grouped[kind][i] for i in ix.tolist())
 return result


def rollout(model,cases,baseline=None,seed=17):
 if model is not None:model.eval()
 states=[frozenset(gym.initial_facts(c)) for c in cases]
 initial=list(states);done=[False]*len(cases);history=[[] for _ in cases]
 mins=[gym.optimal_remaining(c,s) for c,s in zip(cases,states)]
 budgets=[d+1 if d<gym.INFINITY else 2 for d in mins]
 gen=torch.Generator().manual_seed(202000+seed)
 for t in range(max(budgets)):
  active=[i for i in range(len(cases)) if not done[i] and t<budgets[i]]
  if not active:break
  for off in range(0,len(active),32):
   idx=active[off:off+32];cc=[cases[i] for i in idx];ss=[states[i] for i in idx]
   x,bits,gate=gym.observations(cc,ss)
   if model is not None:
    with torch.no_grad():raw=model(x,bits)
    action=raw.masked_fill(~gate,-1e4).argmax(-1).tolist();logits=raw.tolist()
   else:
    action=[];logits=[None]*len(idx)
    for c,s in zip(cc,ss):
     legal=gym.enabled(c,s)
     if baseline=='fixed':a=legal[0] if legal else gym.STOP
     elif baseline=='uniform':a=legal[int(torch.randint(len(legal),(1,),generator=gen))] if legal else gym.STOP
     elif baseline=='shortest':a=gym.optimal_actions(c,s)[0]
     else:raise ValueError('unknown host baseline')
     action.append(a)
   for j,i in enumerate(idx):
    before=states[i];a=action[j];after,proof=gym.sound_fire(cases[i],before,a)
    history[i].append({'before':sorted(before),'enabled':list(gym.enabled(cases[i],before)),
      'action':a,'proof':proof,'after':sorted(after),'raw_logits':logits[j]})
    states[i]=after
    if a==gym.STOP or base.GOAL in after:done[i]=True
 rows=[]
 for i,c in enumerate(cases):
  provable=base.GOAL in gym.closure(c);solved=base.GOAL in states[i]
  rows.append({'case_id':c.id,'split':c.split,'initial':sorted(initial[i]),
    'final':sorted(states[i]),'goal_truth':int(provable),
    'optimal_steps':None if mins[i]>=gym.INFINITY else mins[i],
    'budget':budgets[i],'actions':history[i],'solved':solved,
    'early_stop':bool(provable and not solved and any(x['action']==gym.STOP for x in history[i])),
    'stopped':bool(done[i]),'used_actions':len(history[i]),'host_accepted_false_proof':False})
 return rows


def summarize(rows):return parent.summarize(rows)


def train(seed,out,smoke=False):
 out=Path(out)
 if out.exists() and any(out.iterdir()):raise FileExistsError('AF21 training output must be fresh')
 out.mkdir(parents=True,exist_ok=True)
 if seed not in SEEDS:raise ValueError('unregistered seed')
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 check_manifest();torch.manual_seed(212100+seed)
 initial=ReadoutPolicy('ancestor',loops=4)
 shared={k:v.detach().clone() for k,v in initial.state_dict().items()}
 legacy,composed=parent.training_states()
 traincases=gym.cases_for('train')
 weights={};fit={};timing={}
 for name,(obj,loops,budget) in CONFIGS.items():
  model=ReadoutPolicy('ancestor',loops);model.load_state_dict(shared)
  opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
  rng=torch.Generator().manual_seed(212117+seed)
  steps=2 if smoke else budget
  began=time.perf_counter();first_grad=None;losses=[]
  for t in range(steps):
   ai=torch.randint(len(legacy),(BATCH//2,),generator=rng).tolist()
   bi=torch.randint(len(composed),(BATCH//2,),generator=rng).tolist()
   batch=[legacy[i] for i in ai]+[composed[i] for i in bi]
   cases=[v[0] for v in batch];states=[v[1] for v in batch]
   chosen=sample_pairs('train',rng,smoke=smoke)
   model.train();opt.zero_grad(set_to_none=True)
   # Same primary input, native graph cell calls and AF20 base loss for all arms.
   primary,parts=primary_loss(model,cases,states)
   paired,pair_parts,(paired_raw,paired_gate,paired_opt,paired_cases,paired_states)=pair_losses(
    model,traincases,chosen,obj)
   loss=primary+.4*paired
   dparts={'stop_loss':0.,'rule_loss':0.}
   x,k,legal,opt_targets=gym.observations(cases,states,with_labels=True)
   raw,dist,reach=model.full(x,k)
   stop,rank,dparts=decision_losses(raw,legal,opt_targets,states,cases)
   pair_stop,pair_rank,_=decision_losses(paired_raw,paired_gate,paired_opt,
                                        paired_states,paired_cases)
   if obj!='bce':
    loss=loss+STOP_WEIGHT*stop+PAIR_STOP_WEIGHT*pair_stop
    if obj=='greedy_rule':loss=loss+RULE_WEIGHT*(rank+.5*pair_rank)
   if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite AF21 training loss')
   loss.backward()
   if first_grad is None:
    first_grad=sum(float(p.grad.square().sum()) for p in model.cell.parameters()
                   if p.grad is not None)**.5
    if first_grad<=0:raise RuntimeError('native cell received no gradient')
   torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
   opt.step()
   if t%60==0 or t==steps-1:
    losses.append({'step':t,'loss':float(loss.detach()),**parts,**pair_parts,
                  **dparts,'pair_stop_direct':float(pair_stop.detach()),
                  'pair_rule_direct':float(pair_rank.detach())})
  if not all(bool(torch.isfinite(v).all()) for v in model.state_dict().values()):
   raise FloatingPointError('nonfinite AF21 model')
  torch.save({'experiment':'AF-21','seed':seed,'config':name,'objective':obj,
              'loops':loops,'budget':budget,'state_dict':model.state_dict()},out/f'{name}.pt')
  np.savez(out/f'{name}.npz',**{k:v.detach().numpy() for k,v in model.state_dict().items()})
  weights[name]={ext:sha(out/f'{name}.{ext}') for ext in ('pt','npz')}
  fit[name]={sp:summarize(rollout(model,gym.cases_for(sp),seed=seed)) for sp in ('train','dev')}
  timing[name]={'seconds':time.perf_counter()-began,'updates':steps,
    'native_calls':steps*(BATCH+2*PAIR_COUNT*2+BATCH)*loops if not smoke else steps*(BATCH+8+BATCH)*loops,
    'first_grad_l2':first_grad,'losses':losses}
  print(json.dumps({'seed':seed,'arm':name,'dev':fit[name]['dev']['proof_success'],
    'seconds':round(timing[name]['seconds'],2)}),flush=True)
 receipt={'experiment':'AF-21','seed':seed,'smoke':smoke,'torch':torch.__version__,
  'weights':weights,'fit':fit,'timing':timing,'configs':CONFIGS,
  'source_hashes':source_hashes(),'manifest_sha256':check_manifest(),
  'trained_parameters':sum(p.numel() for p in initial.parameters()),
  'same_initial_parameters':True,'same_training_inputs_within_first_240':True,
  'pair_train_digest':gym.pair_receipt('train')['digest'],
  'fit_split_access':['train','dev_eligibility_only']}
 write(out/'train_receipt.json',receipt)
 (out/'train_receipt.sha256').write_text(sha(out/'train_receipt.json')+'\n')
 return receipt


def load_models(training,seed,smoke=False):
 p=Path(training);source=p/'train_receipt.json'
 if sha(source)!=(p/'train_receipt.sha256').read_text().strip():raise ValueError('training receipt changed')
 rec=json.loads(source.read_text())
 if (rec['experiment'],rec['seed'],rec['smoke'])!=('AF-21',seed,smoke):raise ValueError('AF21 checkpoint identity changed')
 if rec['source_hashes']!=source_hashes() or rec['manifest_sha256']!=check_manifest():
  raise ValueError('AF21 source or manifest hash changed')
 models={}
 for name,(obj,loops,budget) in CONFIGS.items():
  for ext in ('pt','npz'):
   if sha(p/f'{name}.{ext}')!=rec['weights'][name][ext]:raise ValueError('checkpoint bytes changed')
  pkg=torch.load(p/f'{name}.pt',weights_only=True,map_location='cpu')
  if (pkg['experiment'],pkg['seed'],pkg['config'],pkg['objective'],pkg['loops'],pkg['budget'])!=('AF-21',seed,name,obj,loops,budget):
   raise ValueError('model checkpoint metadata changed')
  with np.load(p/f'{name}.npz',allow_pickle=False) as arr:
   if set(arr.files)!=set(pkg['state_dict']) or any(not np.array_equal(arr[k],v.numpy())
       for k,v in pkg['state_dict'].items()):raise ValueError('Torch/NumPy weights disagree')
  model=ReadoutPolicy('ancestor',loops);model.load_state_dict(pkg['state_dict']);model.eval()
  models[name]=model
 return rec,models


def pair_predictions(model,cases,pairs,arm,split):
 rows=[];model.eval()
 for off in range(0,len(pairs),32):
  subset=pairs[off:off+32]
  cc=[cases[p.left] for p in subset]+[cases[p.right] for p in subset]
  ss=[gym.initial_facts(c) for c in cc];x,bits,gate=gym.observations(cc,ss)
  with torch.no_grad():raw,dist,reach=model.full(x,bits)
  gaps,_=greedy_gap(raw,gate)
  mass=stop_gap(raw,gate)
  n=len(subset)
  for i,p in enumerate(subset):
   rows.append({'arm':arm,'split':split,'left_id':cases[p.left].id,'right_id':cases[p.right].id,
     'kind':p.kind,'changed_atom':p.changed_atom,'incidence_distance':p.graph_distance,
     'truth':[int(p.reachable_left),int(p.reachable_right)],
     'minimum':[p.minimum_left,p.minimum_right],
     'reach_logits':[float(reach[i]),float(reach[i+n])],
     'stop_gaps':[float(gaps[i]),float(gaps[i+n])],
     'stop_mass_gaps':[float(mass[i]),float(mass[i+n])],
     'stop_raw':[float(raw[i,-1]),float(raw[i+n,-1])],
     'pair_index':off+i})
 return rows


def pair_summary(rows):
 flips=[r for r in rows if r['kind']=='flip'];invars=[r for r in rows if r['kind']=='connected_invariant']
 if not flips or not invars:raise ValueError('missing pair class for scoring')
 sig=lambda x:1/(1+math.exp(-max(-60,min(60,x))))
 flip_correct=0;flip_reach=0
 for row in flips:
  ra,rb=row['reach_logits'];sa,sb=row['stop_gaps']
  t0,t1=row['truth']
  flip_reach+=int((ra>0)==bool(t0) and (rb>0)==bool(t1))
  flip_correct+=int(((ra-rb)*(1 if t0 else -1)>0) and
                    ((sa-sb)*(1 if t0 else -1)<0))
 deltas_reach=[abs(sig(r['reach_logits'][0])-sig(r['reach_logits'][1])) for r in invars]
 deltas_stop=[abs(sig(r['stop_gaps'][0])-sig(r['stop_gaps'][1])) for r in invars]
 return {'flip_pairs':len(flips),'connected_invariant_pairs':len(invars),
   'flip_reach_accuracy':flip_reach/len(flips),'flip_joint_order_accuracy':flip_correct/len(flips),
   'connected_mean_abs_reach_delta':sum(deltas_reach)/len(deltas_reach),
   'connected_mean_abs_stop_delta':sum(deltas_stop)/len(deltas_stop),
   'connected_both_within_0p1':sum(a<=.1 and b<=.1 for a,b in zip(deltas_reach,deltas_stop))/len(invars)}


def evaluate(seed,training,out,smoke=False):
 out=Path(out)
 if out.exists() and any(out.iterdir()):raise FileExistsError('AF21 scored output must be fresh')
 out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
 receipt,models=load_models(training,seed,smoke=smoke)
 families=('dev',) if smoke else ('held_layout','held_depth4','held_depth5')
 rows=[];probes=[];scores={};paired={}
 for name,model in models.items():
  scores[name]={};paired[name]={}
  for split in families:
   cases=gym.cases_for(split)
   outrows=rollout(model,cases,seed=seed)
   rows.extend({'arm':name,**r} for r in outrows)
   scores[name][split]=summarize(outrows)
   ps=pair_predictions(model,cases,gym.pairs_for(split),name,split)
   probes.extend(ps)
   paired[name][split]=pair_summary(ps)
 for mode in BASELINES:
  scores[mode]={}
  for split in families:
   cc=gym.cases_for(split);rr=rollout(None,cc,baseline=mode,seed=seed)
   rows.extend({'arm':mode,**r} for r in rr)
   scores[mode][split]=summarize(rr)
 jsonl(out/'episodes.jsonl',rows);jsonl(out/'paired_initial.jsonl',probes)
 summary={'experiment':'AF-21','seed':seed,'smoke':smoke,'scores':scores,'pairs':paired,
   'episode_rows':len(rows),'paired_rows':len(probes),
   'episode_sha256':sha(out/'episodes.jsonl'),'paired_sha256':sha(out/'paired_initial.jsonl'),
   'training_sha256':sha(Path(training)/'train_receipt.json'),
   'source_hashes':source_hashes(),'manifest_sha256':check_manifest()}
 write(out/'evaluation.json',summary)
 return summary


def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['manifest','train','evaluate'])
 p.add_argument('--seed',type=int,default=17);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--training',type=Path);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.command=='manifest':write(a.out,gym.manifest())
 elif a.command=='train':train(a.seed,a.out,a.smoke)
 else:
  if not a.training:p.error('evaluate requires --training')
  evaluate(a.seed,a.training,a.out,a.smoke)
if __name__=='__main__':main()

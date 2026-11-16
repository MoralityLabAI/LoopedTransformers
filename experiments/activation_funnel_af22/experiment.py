"""AF-22: one shared neural value checkpoint, three frozen inference algorithms.

Direct, one-step and two-step inference always use the SAME trained weights.
The only policy change is the min-plus backup over exact host-verified
successor states. No shortest-proof oracle call occurs inside the planner.
"""
from __future__ import annotations
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiments.activation_funnel_af05 import gym as base
from experiments.activation_funnel_af19.model import ReadoutPolicy
from experiments.activation_funnel_af21 import experiment as parent
from . import gym,planner

HERE=Path(__file__).parent
SEEDS=gym.SEEDS
STEPS=240
BATCH=32
MODES=planner.MODES
BASELINES=('fixed','shortest')
SOURCE_FILES=[
 'PROTOCOL.json','fresh_manifest.json','README.md','BellmanIteration.lean',
 'gym.py','planner.py','experiment.py','verify.py','aggregate.py','tests/test_af22.py',
 '../../.github/workflows/activation-funnel-af22.yml',
 '../activation_funnel_af21/gym.py','../activation_funnel_af21/experiment.py',
 '../activation_funnel_af19/model.py','../activation_funnel_af19/experiment.py',
 '../activation_funnel_af17/model.py','../activation_funnel_af17/gym.py',
 '../activation_funnel_af15/model.py','../activation_funnel_af05/gym.py',
 '../activation_funnel_af02/model.py','../lean_reasoner/train.py','../lean_reasoner/proof_data.py']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def hashes():return {n:sha(HERE/n) for n in SOURCE_FILES}
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n')
def jsonl(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w') as f:
  for r in rows:f.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')
def check_manifest():
 if json.loads((HERE/'fresh_manifest.json').read_text())!=gym.manifest():
  raise ValueError('AF22 frozen manifest changed')
 return sha(HERE/'fresh_manifest.json')

def train(seed,out,smoke=False):
 if seed not in SEEDS:raise ValueError('unregistered seed')
 out=Path(out)
 if out.exists() and any(out.iterdir()):raise FileExistsError('training folder must be fresh')
 out.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 check_manifest();torch.manual_seed(220000+seed)
 model=ReadoutPolicy('ancestor',4)
 opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
 generator=torch.Generator().manual_seed(220079+seed)
 legacy,composed=parent.parent.training_states() if hasattr(parent,'parent') else ([],[])
 if not legacy or not composed:raise RuntimeError('registered AF21 train-only roster absent')
 steps=2 if smoke else STEPS
 began=time.perf_counter();log=[];first_grad=None
 for t in range(steps):
  ia=torch.randint(len(legacy),(BATCH//2,),generator=generator).tolist()
  ib=torch.randint(len(composed),(BATCH//2,),generator=generator).tolist()
  batch=[legacy[i] for i in ia]+[composed[i] for i in ib]
  cc=[r[0] for r in batch];ss=[r[1] for r in batch]
  model.train();opt.zero_grad(set_to_none=True)
  main,parts=parent.primary_loss(model,cc,ss)
  # Exactly one host-gated random legal successor per chosen training state.
  succ_cases=[];succ_states=[];succ_dist=[];succ_reach=[];root_indices=[];opt_successor=[]
  for i,(c,s) in enumerate(batch):
   legal=gym.enabled(c,s)
   if not legal:continue
   selected=legal[int(torch.randint(len(legal),(1,),generator=generator))]
   nxt,_=gym.sound_fire(c,s,selected)
   distance=gym.optimal_remaining(c,nxt)
   succ_cases.append(c);succ_states.append(nxt)
   succ_dist.append(float(16 if distance>=gym.INFINITY else min(distance,16)))
   succ_reach.append(float(distance<gym.INFINITY));root_indices.append(i)
   opt_successor.append(selected in gym.optimal_actions(c,s))
  if not succ_cases:raise RuntimeError('AF22 training sampled no host legal successor')
  sx,sbits,_=gym.observations(succ_cases,succ_states)
  _,sd,sr=model.full(sx,sbits)
  target_d=torch.tensor(succ_dist);target_r=torch.tensor(succ_reach)
  succ_loss=F.smooth_l1_loss(sd,target_d)+.5*F.binary_cross_entropy_with_logits(sr,target_r)
  # Residual is enforced only for exact-optimal source transitions, so a
  # sampled detour cannot falsely be labelled a Bellman minimizer.
  opt_indices=[j for j,good in enumerate(opt_successor) if good]
  residual=main*0.
  if opt_indices:
   xx,bb,_=gym.observations([cc[root_indices[j]] for j in opt_indices],
                                  [ss[root_indices[j]] for j in opt_indices])
   _,rd,_=model.full(xx,bb)
   residual=F.smooth_l1_loss(rd,1.+sd[opt_indices])
  loss=main+.3*succ_loss+.1*residual
  if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite AF22 training loss')
  loss.backward()
  if first_grad is None:
   first_grad=sum(float(p.grad.square().sum()) for p in model.cell.parameters()
       if p.grad is not None)**.5
   if first_grad<=0:raise RuntimeError('native cell received zero gradient')
  torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
  opt.step()
  if t%60==0 or t==steps-1:log.append({'update':t,'total':float(loss.detach()),
    'primary':float(main.detach()),'successor_cost':float(succ_loss.detach()),
    'optimal_residual':float(residual.detach()),'successors':len(succ_cases)})
 pkg={'experiment':'AF-22','seed':seed,'smoke':smoke,'steps':steps,
      'state_dict':model.state_dict(),'parameters':sum(p.numel() for p in model.parameters())}
 torch.save(pkg,out/'value_policy.pt')
 np.savez(out/'value_policy.npz',**{k:v.detach().numpy() for k,v in model.state_dict().items()})
 fit={}
 for sp in ('train','dev'):
  cs=gym.cases_for(sp)
  rows=rollout(model,cs,seed=seed,modes=('direct',),include_baselines=False)
  fit[sp]=summarize(rows['direct'])
  fit[sp]['value_mae'],fit[sp]['reachable_accuracy']=value_fit(model,cs)
 receipt={'experiment':'AF-22','seed':seed,'smoke':smoke,'torch':torch.__version__,
  'updates':steps,'parameters':pkg['parameters'],'native_cell_grad_initial':first_grad,
  'time_seconds':time.perf_counter()-began,'losses':log,'fit':fit,
  'source_hashes':hashes(),'manifest_sha256':check_manifest(),
  'weights':{s:sha(out/f'value_policy.{s}') for s in ('pt','npz')},
  'inference_comparisons':'single checkpoint, direct/vi1/vi2 without refitting',
  'training_splits':['train','dev_eligibility_only']}
 write(out/'train_receipt.json',receipt)
 (out/'train_receipt.sha256').write_text(sha(out/'train_receipt.json')+'\n')
 print(json.dumps({'seed':seed,'steps':steps,'fit':fit,'sec':round(receipt['time_seconds'],1)}),flush=True)
 return receipt


def load_model(path,seed,smoke=False):
 path=Path(path)
 if sha(path/'train_receipt.json')!=(path/'train_receipt.sha256').read_text().strip():raise ValueError('train receipt hash mismatch')
 r=json.loads((path/'train_receipt.json').read_text())
 if (r['experiment'],r['seed'],r['smoke'])!=('AF-22',seed,smoke):raise ValueError('training source identity changed')
 if r['source_hashes']!=hashes() or r['manifest_sha256']!=check_manifest():raise ValueError('source or manifest differs')
 for ext in ('pt','npz'):
  if sha(path/f'value_policy.{ext}')!=r['weights'][ext]:raise ValueError('weight hash mismatch')
 pkg=torch.load(path/'value_policy.pt',weights_only=True,map_location='cpu')
 if (pkg['experiment'],pkg['seed'],pkg['smoke'],pkg['steps'])!=('AF-22',seed,smoke,r['updates']):
  raise ValueError('checkpoint metadata mismatch')
 with np.load(path/'value_policy.npz',allow_pickle=False) as arr:
  if set(arr.files)!=set(pkg['state_dict']):raise ValueError('named weights missing')
  for k,v in pkg['state_dict'].items():
   if not np.array_equal(arr[k],v.detach().numpy()):raise ValueError('weight mismatch')
 model=ReadoutPolicy('ancestor',4);model.load_state_dict(pkg['state_dict']);model.eval()
 if sum(p.numel() for p in model.parameters())!=6036:raise ValueError('AF22 parameter count changed')
 return r,model


def value_fit(model,cases):
 predictions=[];observed=[];good=[];correct=[]
 model.eval()
 with torch.no_grad():
  for off in range(0,len(cases),64):
   cs=cases[off:off+64];states=[frozenset(gym.initial_facts(c)) for c in cs]
   x,bits,_=gym.observations(cs,states)
   _,d,r=model.full(x,bits)
   for i,(c,s) in enumerate(zip(cs,states)):
    truth=gym.optimal_remaining(c,s)
    if truth<gym.INFINITY:predictions.append(float(d[i]));observed.append(truth)
    good.append(truth<gym.INFINITY);correct.append(bool(float(r[i])>0))
 return (sum(abs(a-b) for a,b in zip(predictions,observed))/len(observed) if observed else 0.,
         sum(a==b for a,b in zip(good,correct))/len(good))


def rollout(model,cases,seed=17,modes=MODES,include_baselines=True):
 modes=tuple(modes)
 if any(m not in MODES for m in modes):raise ValueError('unknown planner mode')
 names=modes+(BASELINES if include_baselines else ())
 states={name:[frozenset(gym.initial_facts(c)) for c in cases] for name in names}
 histories={name:[[] for _ in cases] for name in names}
 done={name:[False]*len(cases) for name in names}
 mins=[gym.optimal_remaining(c,frozenset(gym.initial_facts(c))) for c in cases]
 budgets=[d+1 if d<gym.INFINITY else 2 for d in mins]
 gen=torch.Generator().manual_seed(220000+seed)
 for t in range(max(budgets)):
  jobs=[]
  for mode in modes:
   for i,c in enumerate(cases):
    if not done[mode][i] and t<budgets[i]:jobs.append((mode,i,c,states[mode][i]))
  planned=[]
  for off in range(0,len(jobs),32):
   batch=jobs[off:off+32]
   planned.extend(planner.decisions(model,[a[2] for a in batch],[a[3] for a in batch]))
  for (mode,i,c,state),plan in zip(jobs,planned):
   action=plan['selected'][mode];nxt,proof=gym.sound_fire(c,state,action)
   histories[mode][i].append({'before':sorted(state),'enabled':list(gym.enabled(c,state)),
      'action':action,'proof':proof,'after':sorted(nxt),
      'decision_scores':plan['scores'][mode], 'plan_trace':plan['trace']})
   states[mode][i]=nxt
   if action==gym.STOP or base.GOAL in nxt:done[mode][i]=True
  for mode in BASELINES:
   if mode not in names:continue
   for i,c in enumerate(cases):
    if done[mode][i] or t>=budgets[i]:continue
    s=states[mode][i];legal=gym.enabled(c,s)
    if mode=='fixed':a=legal[0] if legal else gym.STOP
    else:a=gym.optimal_actions(c,s)[0]
    nxt,proof=gym.sound_fire(c,s,a)
    histories[mode][i].append({'before':sorted(s),'enabled':list(legal),'action':a,
      'proof':proof,'after':sorted(nxt),'decision_scores':None,'plan_trace':None})
    states[mode][i]=nxt
    if a==gym.STOP or base.GOAL in nxt:done[mode][i]=True
 rows={}
 for mode in names:
  rows[mode]=[]
  for i,c in enumerate(cases):
   initial=sorted(gym.initial_facts(c));out=states[mode][i]
   truth=base.GOAL in gym.closure(c);solved=base.GOAL in out
   h=histories[mode][i]
   rows[mode].append({'mode':mode,'case_id':c.id,'split':c.split,'initial':initial,
     'final':sorted(out),'goal_truth':int(truth),
     'optimal_steps':None if mins[i]>=gym.INFINITY else mins[i],
     'budget':budgets[i],'actions':h,'solved':solved,
     'early_stop':bool(truth and not solved and any(a['action']==gym.STOP for a in h)),
     'stopped':bool(done[mode][i]),'used_actions':len(h),'host_accepted_false_proof':False})
 return rows


def summarize(rows):
 positive=[r for r in rows if r['goal_truth']]
 return {'cases':len(rows),'provable':len(positive),
  'solved':sum(bool(r['solved']) for r in positive),
  'proof_success':sum(bool(r['solved']) for r in positive)/len(positive) if positive else 1.,
  'early_stop_positive':sum(bool(r['early_stop']) for r in positive)/len(positive) if positive else 0.,
  'false_accepted_proofs':sum(bool(r['host_accepted_false_proof']) for r in rows)}


def evaluate(seed,training,out,smoke=False):
 out=Path(out)
 if out.exists() and any(out.iterdir()):raise FileExistsError('evaluation folder must be fresh')
 out.mkdir(parents=True,exist_ok=True)
 _,model=load_model(training,seed,smoke)
 families=('dev',) if smoke else ('held_layout','held_depth4','held_depth5')
 all_rows=[];scores={}
 for split in families:
  cases=gym.cases_for(split)
  results=rollout(model,cases,seed=seed)
  scores[split]={k:summarize(v) for k,v in results.items()}
  for mode,rows in results.items():all_rows.extend(rows)
 jsonl(out/'episodes.jsonl',all_rows)
 receipt={'experiment':'AF-22','seed':seed,'smoke':smoke,'scores':scores,
  'episode_count':len(all_rows),'episodes_sha256':sha(out/'episodes.jsonl'),
  'training_sha256':sha(Path(training)/'train_receipt.json'),
  'source_hashes':hashes(),'manifest_sha256':check_manifest(),
  'modes':'three decision rules applied to the identical frozen trained weights'}
 write(out/'evaluation.json',receipt)
 return receipt


def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['manifest','train','evaluate'])
 p.add_argument('--seed',type=int,default=17);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--training',type=Path);p.add_argument('--smoke',action='store_true');args=p.parse_args()
 if args.command=='manifest':write(args.out,gym.manifest())
 elif args.command=='train':train(args.seed,args.out,args.smoke)
 else:
  if args.training is None:p.error('--training is required for evaluate')
  evaluate(args.seed,args.training,args.out,args.smoke)
if __name__=='__main__':main()

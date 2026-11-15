"""AF-19 controlled recurrent-depth and Bellman-cap study, host-sound Horn search.

Every configuration starts from the same parameter tensors per seed and sees
identical AF-17 train/dev examples and sampled source states. The only varied
training dimensions are recurrent call count and Bellman target cap.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiments.activation_funnel_af05 import gym as base
from experiments.activation_funnel_af17 import experiment as parent
from . import gym
from .model import ReadoutPolicy

HERE=Path(__file__).parent
SEEDS=(17,29,43,61)
CONFIGS={'local_l2':('local',2),'ancestor_l2':('ancestor',2),
         'local_l4':('local',4),'ancestor_l4':('ancestor',4)}
BASELINES=('fixed','uniform','shortest')
STEPS=240
BATCH=32
SOURCE_FILES=['gym.py','model.py','experiment.py','verify.py','aggregate.py',
    'PROTOCOL.json','fresh_manifest.json','AncestorReadout.lean','README.md','tests/test_af19.py',
    '../../.github/workflows/activation-funnel-af19.yml',
    '../activation_funnel_af18/gym.py',
    '../activation_funnel_af17/gym.py','../activation_funnel_af17/model.py',
    '../activation_funnel_af17/experiment.py','../activation_funnel_af16/gym.py',
    '../activation_funnel_af15/model.py','../activation_funnel_af02/model.py',
    '../lean_reasoner/train.py','../lean_reasoner/proof_data.py']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def hashes():return {p:sha(HERE/p) for p in SOURCE_FILES}
def write(path,record):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(record,sort_keys=True,indent=2,allow_nan=False)+'\n')
def jsonl(path,rows):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w') as f:
        for row in rows:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')

def check_manifest():
    actual=gym.manifest()
    if json.loads((HERE/'fresh_manifest.json').read_text())!=actual:
        raise ValueError('AF19 registered corpus differs from committed manifest')
    return sha(HERE/'fresh_manifest.json')

def training_states():
    # AF-17's exact train-only states; reuse without any label or heldout change.
    return parent.training_states()

def supervision(cases,states,cap):
    batch,bits,gate,optimal=gym.observations(cases,states,with_labels=True)
    q=torch.zeros(gate.shape)
    distances=[];reachable=[];should_stop=[]
    for i,(c,state) in enumerate(zip(cases,states)):
        d=gym.optimal_remaining(c,state)
        reachable.append(float(d<gym.INFINITY))
        should_stop.append(float(d==0 or d>=gym.INFINITY))
        distances.append(float(cap if d>=gym.INFINITY else min(d,cap)))
        costs=gym.action_costs(c,state,cap=cap)
        for j in torch.where(gate[i])[0].tolist():q[i,j]=(cap-costs[j])/3.
    return (batch,bits,gate,optimal,q,torch.tensor(distances),
            torch.tensor(reachable),torch.tensor(should_stop))

def objective(raw,dist,reach,gate,optimal,target_q,target_dist,target_reach,target_stop):
    scores=raw.masked_fill(~gate,-1e4)
    ce=(-torch.logsumexp(scores.masked_fill(~optimal,-1e4),-1)
        +torch.logsumexp(scores,-1)).mean()
    valid=gate[:,:-1].any(-1)
    gap=raw[:,-1]-torch.logsumexp(scores[:,:-1],-1)
    stop_loss=(F.binary_cross_entropy_with_logits(gap[valid],target_stop[valid])
               if bool(valid.any()) else ce*0.)
    reach_loss=F.binary_cross_entropy_with_logits(reach,target_reach)
    q_loss=F.smooth_l1_loss(raw[gate],target_q[gate])
    dist_loss=F.smooth_l1_loss(dist,target_dist/3.)
    total=ce+.65*(stop_loss+.25*reach_loss)+.55*(q_loss+.5*dist_loss)
    return total,{'ce':float(ce.detach()),'stop':float(stop_loss.detach()),
                  'reach':float(reach_loss.detach()),'q':float(q_loss.detach()),
                  'distance':float(dist_loss.detach())}

def rollout(model,cases,baseline=None,seed=17):
    """Host-executed proof traces; network never changes proved atoms directly."""
    if model is not None:model.eval()
    states=[frozenset(gym.initial_facts(c)) for c in cases]
    orig=list(states);done=[False]*len(cases);history=[[] for _ in cases]
    min_steps=[gym.optimal_remaining(c,s) for c,s in zip(cases,states)]
    budgets=[d+1 if d<gym.INFINITY else 2 for d in min_steps]
    rng=torch.Generator().manual_seed(181800+seed)
    for t in range(max(budgets)):
        active=[i for i in range(len(cases)) if not done[i] and t<budgets[i]]
        if not active:break
        for off in range(0,len(active),32):
            idx=active[off:off+32];cc=[cases[i] for i in idx];ss=[states[i] for i in idx]
            batch,bits,gate=gym.observations(cc,ss)
            if model is not None:
                with torch.no_grad():
                    raw=model(batch,bits)
                    choice=raw.masked_fill(~gate,-1e4).argmax(-1).tolist()
                    logits=raw.tolist()
            else:
                choice=[];logits=[None]*len(idx)
                for c,s in zip(cc,ss):
                    enabled=gym.enabled(c,s)
                    if baseline=='fixed':action=enabled[0] if enabled else gym.STOP
                    elif baseline=='uniform':
                        action=enabled[int(torch.randint(len(enabled),(1,),generator=rng))] if enabled else gym.STOP
                    elif baseline=='shortest':action=gym.optimal_actions(c,s)[0]
                    else:raise ValueError('invalid host baseline')
                    choice.append(action)
            for j,i in enumerate(idx):
                act=choice[j];before=states[i]
                after,proof=gym.sound_fire(cases[i],before,act)
                history[i].append({'before':sorted(before),'enabled':list(gym.enabled(cases[i],before)),
                    'action':act,'proof':proof,'after':sorted(after),'raw_logits':logits[j]})
                states[i]=after
                if act==gym.STOP or base.GOAL in after:done[i]=True
    rows=[]
    for i,c in enumerate(cases):
        solvable=base.GOAL in gym.closure(c);solved=base.GOAL in states[i]
        if solved and not solvable:raise AssertionError('false host acceptance')
        early=bool(solvable and not solved and any(x['action']==gym.STOP for x in history[i]))
        rows.append({'case_id':c.id,'split':c.split,'initial':sorted(orig[i]),'final':sorted(states[i]),
            'goal_truth':int(solvable),'optimal_steps':None if min_steps[i]>=gym.INFINITY else min_steps[i],
            'budget':budgets[i],'actions':history[i],'solved':solved,'early_stop':early,
            'stopped':bool(done[i]),'used_actions':len(history[i]),'host_accepted_false_proof':False})
    return rows

def summarize(rows):
    positives=[r for r in rows if r['goal_truth']]
    solved=sum(int(r['solved']) for r in positives)
    return {'cases':len(rows),'provable':len(positives),'solved':solved,
        'proof_success':solved/len(positives) if positives else 1.,
        'early_stop_positive':sum(int(r['early_stop']) for r in positives)/len(positives) if positives else 0.,
        'false_accepted_proofs':sum(int(r['host_accepted_false_proof']) for r in rows)}

def train(seed,out,smoke=False):
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('AF19 training directory must be fresh')
    out.mkdir(parents=True,exist_ok=True)
    if seed not in SEEDS:raise ValueError('unregistered seed')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    check_manifest();torch.manual_seed(81800+seed)
    initial=ReadoutPolicy('local',loops=2)
    state={k:v.detach().clone() for k,v in initial.state_dict().items()}
    legacy,composed=training_states();weights={};fit={};timings={}
    for name,(mode,loops) in CONFIGS.items():
        cap=16
        model=ReadoutPolicy(mode,loops=loops);model.load_state_dict(state)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
        rng=torch.Generator().manual_seed(818017+seed)
        count=2 if smoke else STEPS
        start=time.perf_counter();first_grad=None;losses=[]
        for t in range(count):
            a=torch.randint(len(legacy),(BATCH//2,),generator=rng).tolist()
            b=torch.randint(len(composed),(BATCH//2,),generator=rng).tolist()
            samples=[legacy[i] for i in a]+[composed[i] for i in b]
            cc=[s[0] for s in samples];ss=[s[1] for s in samples]
            x,bits,gate,optimal,q,dist,reach,stop=supervision(cc,ss,cap)
            model.train();optimizer.zero_grad(set_to_none=True)
            raw,dh,rh=model.full(x,bits)
            loss,parts=objective(raw,dh,rh,gate,optimal,q,dist,reach,stop)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite loss')
            loss.backward()
            if first_grad is None:
                first_grad=sum(float(p.grad.square().sum()) for p in model.cell.parameters()
                    if p.grad is not None)**.5
                if first_grad<=0:raise RuntimeError('native cell gradient missing')
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            if t%60==0 or t==count-1:losses.append({'step':t,'loss':float(loss.detach()),**parts})
        model.eval()
        if not all(bool(torch.isfinite(v).all()) for v in model.state_dict().values()):
            raise FloatingPointError('nonfinite checkpoint')
        torch.save({'experiment':'AF-19','seed':seed,'config':name,'loops':loops,'cap':cap,'mode':mode,
            'state_dict':model.state_dict()},out/f'{name}.pt')
        np.savez(out/f'{name}.npz',**{k:v.detach().numpy() for k,v in model.state_dict().items()})
        weights[name]={ext:sha(out/f'{name}.{ext}') for ext in ('pt','npz')}
        fit[name]={sp:summarize(rollout(model,gym.cases_for(sp),seed=seed)) for sp in ('train','dev')}
        timings[name]={'seconds':time.perf_counter()-start,'steps':count,'gradient_l2':first_grad,
            'source_cell_calls':count*BATCH*loops,'loss_history':losses}
        print(json.dumps({'seed':seed,'config':name,'dev_success':fit[name]['dev']['proof_success'],
             'seconds':round(timings[name]['seconds'],2)}),flush=True)
    states=legacy+composed
    labels=[{'case_id':c.id,'state':sorted(s),'distance':None if gym.optimal_remaining(c,s)>=gym.INFINITY
             else gym.optimal_remaining(c,s),
             'cost_cap16':{str(i):v for i,v in enumerate(gym.action_costs(c,s,16)) if v==v}}
            for c,s in states]
    jsonl(out/'train_cost_receipt.jsonl',labels)
    rec={'experiment':'AF-19','seed':seed,'smoke':smoke,'torch':torch.__version__,'steps':timings,
         'configs':CONFIGS,'fit':fit,'weights':weights,'source_hashes':hashes(),
         'manifest_sha256':check_manifest(),'cost_receipt_sha256':sha(out/'train_cost_receipt.jsonl'),
         'train_state_count':{'legacy':len(legacy),'composed':len(composed)},
         'model_parameter_count':sum(p.numel() for p in initial.parameters()),
         'readout_comparison':'parameter-matched full rule-slot attention with local-vs-ancestor public mask'}
    write(out/'train_receipt.json',rec)
    (out/'train_receipt.sha256').write_text(sha(out/'train_receipt.json')+'\n')
    return rec

def load_models(training,seed,smoke=False):
    p=Path(training);r=p/'train_receipt.json'
    if sha(r)!=(p/'train_receipt.sha256').read_text().strip():raise ValueError('receipt hash changed')
    rec=json.loads(r.read_text())
    if (rec['experiment'],rec['seed'],rec['smoke'])!=('AF-19',seed,smoke):
        raise ValueError('checkpoint identity mismatch')
    if rec['source_hashes']!=hashes() or rec['manifest_sha256']!=check_manifest():
        raise ValueError('source or manifest mismatch')
    if rec['cost_receipt_sha256']!=sha(p/'train_cost_receipt.jsonl'):
        raise ValueError('cost receipt changed')
    out={}
    for name,(mode,loops) in CONFIGS.items():
        cap=16
        for ext in ('pt','npz'):
            if sha(p/f'{name}.{ext}')!=rec['weights'][name][ext]:
                raise ValueError('model file changed')
        pkg=torch.load(p/f'{name}.pt',weights_only=True,map_location='cpu')
        if (pkg['experiment'],pkg['seed'],pkg['config'],pkg['loops'],pkg['cap'],pkg['mode'])!=('AF-19',seed,name,loops,cap,mode):
            raise ValueError('model identity changed')
        model=ReadoutPolicy(mode,loops);model.load_state_dict(pkg['state_dict']);model.eval()
        with np.load(p/f'{name}.npz',allow_pickle=False) as z:
            if set(z.files)!=set(pkg['state_dict']) or any(not np.array_equal(z[k],v.numpy())
                for k,v in pkg['state_dict'].items()):raise ValueError('Torch/NumPy model mismatch')
        out[name]=model
    return rec,out

def diagnostic(model,seed,config):
    """Grounded relevant-fact pairs and causally irrelevant padded/isolated inputs."""
    from .model import relevance_masks
    rows=[]
    for a,b in gym.probe_pairs():
        x,k,_=gym.observations([a,b],[gym.initial_facts(a),gym.initial_facts(b)])
        with torch.no_grad():logits,dist,reach=model.full(x,k)
        distance=gym.graph_distance(a,0)
        diffs={'stop':float((logits[0,-1]-logits[1,-1]).abs()),
               'reach':float((reach[0]-reach[1]).abs()),
               'distance':float((dist[0]-dist[1]).abs())}
        if model.mode=='local' and distance>2*model.loops and any(v>1e-6 for v in diffs.values()):
            raise AssertionError('goal-local readout violates its structural cone')
        # Perturb the metadata of one physically absent rule. Its zero incidence
        # prevents any message passing, and both ancestor/local masks exclude it.
        rules=gym.physical_rules(a)
        absent=next((j for j in range(gym.base.RULES) if j not in rules),None)
        if absent is None:raise ValueError('probe needs padded rule slot')
        altered=gym.base.GraphBatch(*(getattr(x,f.name).clone() for f in __import__('dataclasses').fields(gym.base.GraphBatch)))
        altered.rule_features[:,absent,:]=torch.tensor([1.7,-2.3,0.0])
        with torch.no_grad():raw_alt,dist_alt,reach_alt=model.full(altered,k)
        irrelevant=max(float((raw_alt[:,-1]-logits[:,-1]).abs().max()),
            float((dist_alt-dist).abs().max()),float((reach_alt-reach).abs().max()))
        if irrelevant>1e-5:raise AssertionError('padded-rule perturbation changed STOP readout')
        local_mask,ancestor_mask=relevance_masks(x)
        rows.append({'seed':seed,'config':config,'depth':a.depth,'graph_distance':distance,
            'loop_calls':model.loops,'readout_mode':model.mode,
            'outside_local_cone':distance>2*model.loops,
            'truth_a':int(gym.base.GOAL in gym.closure(a)),
            'truth_b':int(gym.base.GOAL in gym.closure(b)),
            'local_rule_count':int(local_mask[0].sum()),
            'ancestor_rule_count':int(ancestor_mask[0].sum()),
            'padded_rule_slot':absent,'irrelevant_perturbation_max':irrelevant,
            **{key+'_difference':value for key,value in diffs.items()}})
    return rows


def evaluate(seed,training,out,smoke=False):
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('AF19 evaluation must be fresh')
    out.mkdir(parents=True,exist_ok=True)
    rec,models=load_models(training,seed,smoke)
    rows=[];scores={};probes=[]
    families=('dev',) if smoke else ('held_layout','held_depth4','held_depth5')
    for name,m in {**models,**{x:None for x in BASELINES}}.items():
        scores[name]={}
        for sp in families:
            scored=rollout(m,gym.cases_for(sp),baseline=name if m is None else None,seed=seed)
            rows.extend({'seed':seed,'config':name,**r} for r in scored)
            scores[name][sp]=summarize(scored)
        if m is not None:probes.extend(diagnostic(m,seed,name))
    jsonl(out/'episodes.jsonl',rows);jsonl(out/'causal_cone.jsonl',probes)
    result={'experiment':'AF-19','seed':seed,'smoke':smoke,'source_hashes':hashes(),
        'manifest_sha256':check_manifest(),'train_sha256':sha(Path(training)/'train_receipt.json'),
        'episodes_sha256':sha(out/'episodes.jsonl'),'cone_sha256':sha(out/'causal_cone.jsonl'),
        'rows':len(rows),'probes':len(probes),'scores':scores,'fit':rec['fit']}
    write(out/'evaluation.json',result)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('manifest','train','evaluate'))
    p.add_argument('--seed',type=int);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--training',type=Path);p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.command=='manifest':write(a.out,gym.manifest())
    elif a.command=='train':train(a.seed,a.out,a.smoke)
    else:evaluate(a.seed,a.training,a.out,a.smoke)
if __name__=='__main__':main()

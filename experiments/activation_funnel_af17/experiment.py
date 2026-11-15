"""AF-17 fixed matched-control training and source-bound test scoring.

Exact Bellman targets are host labels: Q(S,a)=1+d(S_a) for legal rules and
STOP incurs the registered terminal penalty when the goal is reachable.
The model receives only current proved atoms, public graph, and legal mask.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiments.activation_funnel_af05 import gym as base
from . import gym
from .model import BellmanTypedPolicy

HERE=Path(__file__).parent
ROOT=HERE.parents[1]
SEEDS=(17,29,43,61)
ARMS=('imitation','stop','bellman','imitation_extra')
STEPS={'imitation':240,'stop':240,'bellman':240,'imitation_extra':360}
BATCH=32
SOURCE_FILES=['gym.py','model.py','experiment.py','verify.py','aggregate.py','PROTOCOL.json',
 'fresh_manifest.json','BellmanGap.lean','tests/test_af17.py','../../.github/workflows/activation-funnel-af17.yml','../activation_funnel_af16/gym.py',
 '../activation_funnel_af16/experiment.py','../activation_funnel_af15/model.py',
 '../activation_funnel_af05/gym.py','../activation_funnel_af02/model.py',
 '../lean_reasoner/train.py']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def hashes():return {name:sha(HERE/name) for name in SOURCE_FILES}
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n')
def jsonl(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w') as f:
        for r in rows:f.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')

def check_manifest():
    p=HERE/'fresh_manifest.json'
    if json.loads(p.read_text())!=gym.manifest():raise ValueError('AF17 frozen corpus changed')
    return sha(p)


def training_states():
    """Train-only shortest paths plus deterministic legal detours.

    This broadens reachability/STOP examples, without using any dev/test labels.
    """
    legacy=[];composed=[]
    for c in gym.cases_for('train'):
        records={}
        for state,_ in gym.reference_episode(c):records[tuple(sorted(state))]=state
        init=frozenset(gym.initial_facts(c))
        for act in gym.enabled(c,init):
            next_state,_=gym.sound_fire(c,init,act)
            records[tuple(sorted(next_state))]=next_state
            for second in gym.enabled(c,next_state)[:2]:
                after,_=gym.sound_fire(c,next_state,second)
                records[tuple(sorted(after))]=after
        group=composed if isinstance(c,gym.af16.Case) else legacy
        group.extend((c,state) for state in records.values())
    return legacy,composed


def supervision(cases,states):
    batch,known,legal,optimal=gym.observations(cases,states,with_labels=True)
    labels=legal.new_zeros(legal.shape,dtype=torch.float32)
    distance=[];reachable=[];stop_correct=[]
    for i,(c,state) in enumerate(zip(cases,states)):
        d=gym.optimal_remaining(c,state)
        reachable.append(int(d<gym.INFINITY))
        stop_correct.append(int(d==0 or d>=gym.INFINITY))
        distance.append(float(gym.CAP if d>=gym.INFINITY else min(d,gym.CAP)))
        targets=gym.action_costs(c,state)
        for a in torch.where(legal[i])[0].tolist():
            labels[i,a]=(gym.CAP-targets[a])/3.
    return batch,known,legal,optimal,labels,torch.tensor(distance),torch.tensor(reachable,dtype=torch.float32),torch.tensor(stop_correct,dtype=torch.float32)


def objective(raw,d_hat,reach,legal,optimal,cost_scores,distances,can_reach,should_stop,arm):
    if arm not in ARMS:raise ValueError('unknown AF17 arm')
    action=raw.masked_fill(~legal,-1e4)
    ce=(-torch.logsumexp(action.masked_fill(~optimal,-1e4),-1)
        +torch.logsumexp(action,-1)).mean()
    if arm in ('imitation','imitation_extra'):
        return ce,{'imitation':float(ce.detach()),'stop':0.,'bellman':0.}
    # STOP must win on unprovable states and lose on reachable-unsolved states.
    others=action[:,:-1]
    best=torch.logsumexp(others,-1)
    available=legal[:,:-1].any(-1)
    # For no enabled rules, STOP is an unconditional terminal action.
    stop_gap=raw[:,-1]-best
    binary=F.binary_cross_entropy_with_logits(stop_gap[available],should_stop[available]) if bool(available.any()) else ce*0.
    reach_loss=F.binary_cross_entropy_with_logits(reach,can_reach)
    stop_loss=binary+.25*reach_loss
    if arm=='stop':return ce+.65*stop_loss,{'imitation':float(ce.detach()),'stop':float(stop_loss.detach()),'bellman':0.}
    # Bellman Q costs are exact finite targets on all actual enabled rule slots.
    # Inactive slots are excluded entirely; STOP is a real trained alternative.
    cost_loss=F.smooth_l1_loss(raw[legal],cost_scores[legal])
    value_loss=F.smooth_l1_loss(d_hat,distances/3.)
    bellman=cost_loss+.5*value_loss
    loss=ce+.65*stop_loss+.55*bellman
    return loss,{'imitation':float(ce.detach()),'stop':float(stop_loss.detach()),
                 'bellman':float(bellman.detach()),'q_supervised':float(cost_loss.detach()),
                 'state_distance_supervised':float(value_loss.detach())}


def rollout(model,cases,baseline=None,seed=17):
    if model is not None:model.eval()
    state=[frozenset(gym.initial_facts(c)) for c in cases]
    original=list(state);done=[False]*len(cases);hist=[[] for _ in cases]
    shortest=[gym.optimal_remaining(c,s) for c,s in zip(cases,state)]
    budgets=[d+1 if d<gym.INFINITY else 2 for d in shortest]
    rng=torch.Generator().manual_seed(171700+seed)
    for step in range(max(budgets)):
        active=[i for i in range(len(cases)) if not done[i] and step<budgets[i]]
        if not active:break
        for start in range(0,len(active),32):
            idx=active[start:start+32];cc=[cases[i] for i in idx];old=[state[i] for i in idx]
            batch,known,gate=gym.observations(cc,old)
            if model is not None:
                with torch.no_grad():
                    raw=model(batch,known)
                    pred=raw.masked_fill(~gate,-1e4).argmax(-1).tolist()
                    raw_list=raw.tolist()
            else:
                pred=[];raw_list=[None]*len(idx)
                for c,s in zip(cc,old):
                    choices=list(gym.enabled(c,s))
                    if baseline=='uniform':
                        pred.append(choices[int(torch.randint(len(choices),(1,),generator=rng))] if choices else gym.STOP)
                    elif baseline=='fixed':pred.append(choices[0] if choices else gym.STOP)
                    elif baseline=='shortest':pred.append(gym.optimal_actions(c,s)[0])
                    else:raise ValueError('unknown symbolic baseline')
            for j,i in enumerate(idx):
                act=pred[j];after,proof=gym.sound_fire(cases[i],state[i],act)
                hist[i].append({'before':sorted(state[i]),'enabled':list(gym.enabled(cases[i],state[i])),
                    'action':act,'proof':proof,'after':sorted(after),'raw_logits':raw_list[j]})
                state[i]=after
                if act==gym.STOP or base.GOAL in after:done[i]=True
    rows=[]
    for i,c in enumerate(cases):
        provable=base.GOAL in gym.closure(c);solved=base.GOAL in state[i]
        if solved and not provable:raise ValueError('false host proof')
        early_stop=bool(provable and not solved and any(x['action']==gym.STOP for x in hist[i]))
        rows.append({'case_id':c.id,'split':c.split,'initial':sorted(original[i]),
             'final':sorted(state[i]),'goal_truth':int(provable),
             'optimal_steps':shortest[i] if shortest[i]<gym.INFINITY else None,
             'budget':budgets[i],'actions':hist[i],'solved':bool(solved),
             'stopped':bool(done[i]),'early_stop':early_stop,
             'used_actions':len(hist[i]),'host_accepted_false_proof':False})
    return rows

def summarize(rows):
    positives=[r for r in rows if r['goal_truth']];negatives=[r for r in rows if not r['goal_truth']]
    if not positives or not negatives:raise ValueError('expect provable and unreachable cases')
    return {'cases':len(rows),'provable':len(positives),
      'solved':sum(r['solved'] for r in positives),
      'proof_success':sum(r['solved'] for r in positives)/len(positives),
      'early_stop_positive':sum(r['early_stop'] for r in positives)/len(positives),
      'false_accepted':sum(r['host_accepted_false_proof'] for r in rows),
      'negative_safe':sum(not r['host_accepted_false_proof'] for r in negatives)/len(negatives),
      'actions_per_positive':sum(r['used_actions'] for r in positives)/len(positives)}


def train(seed,out,smoke=False):
    if seed not in SEEDS:raise ValueError('seed not registered')
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('AF17 training output must be fresh')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    check_manifest();torch.manual_seed(81700+seed)
    initial_model=BellmanTypedPolicy()
    initial={k:v.detach().clone() for k,v in initial_model.state_dict().items()}
    legacy,novel=training_states();fit={};weights={};runs={}
    for arm in ARMS:
        model=BellmanTypedPolicy();model.load_state_dict(initial)
        opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
        rng=torch.Generator().manual_seed(817017+seed)
        steps=2 if smoke else STEPS[arm]
        start=time.perf_counter();history=[];first_grad=None
        for t in range(steps):
            ai=torch.randint(len(legacy),(BATCH//2,),generator=rng).tolist()
            bi=torch.randint(len(novel),(BATCH//2,),generator=rng).tolist()
            samples=[legacy[i] for i in ai]+[novel[i] for i in bi]
            cc=[s[0] for s in samples];states=[s[1] for s in samples]
            x,known,gate,optimal,q,value,reach,stop=supervision(cc,states)
            model.train();opt.zero_grad(set_to_none=True)
            logits,dhat,rhat=model.full(x,known)
            loss,parts=objective(logits,dhat,rhat,gate,optimal,q,value,reach,stop,arm)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('AF17 loss nonfinite')
            loss.backward()
            if first_grad is None:
                first_grad=sum(float(p.grad.square().sum()) for p in model.cell.parameters() if p.grad is not None)**.5
                if first_grad<=0:raise RuntimeError('AF17 loss gradient did not reach native TypedLoop')
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            if t%60==0 or t==steps-1:history.append({'step':t,'loss':float(loss.detach()),**parts})
        model.eval()
        if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):raise FloatingPointError('nonfinite model')
        torch.save({'experiment':'AF-17','seed':seed,'arm':arm,'state_dict':model.state_dict()},out/f'{arm}.pt')
        np.savez(out/f'{arm}.npz',**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})
        weights[arm]={ext:sha(out/f'{arm}.{ext}') for ext in ('pt','npz')}
        fit[arm]={s:summarize(rollout(model,gym.cases_for(s),seed=seed)) for s in ('train','dev')}
        runs[arm]={'steps':steps,'wall_seconds':time.perf_counter()-start,
                   'loss_history':history,'initial_cell_gradient_l2':first_grad,
                   'native_source_calls_train':steps*BATCH*2}
        print(json.dumps({'seed':seed,'arm':arm,'dev':fit[arm]['dev']['proof_success'],
                          'dev_early_stop':fit[arm]['dev']['early_stop_positive'],
                          'seconds':runs[arm]['wall_seconds']}),flush=True)
    state_rows=[{'case_id':c.id,'state':sorted(s),'d':None if gym.optimal_remaining(c,s)>=gym.INFINITY else gym.optimal_remaining(c,s),
                 'action_costs':{str(k):v for k,v in enumerate(gym.action_costs(c,s)) if v==v}}
                for c,s in legacy+novel]
    jsonl(out/'train_label_receipt.jsonl',state_rows)
    receipt={'experiment':'AF-17','seed':seed,'smoke':smoke,'fit':fit,'training':runs,
       'source_hashes':hashes(),'manifest_sha256':check_manifest(),
       'label_sha256':sha(out/'train_label_receipt.jsonl'),
       'training_states':{'legacy':len(legacy),'novel':len(novel)},
       'weights':weights,'torch':torch.__version__,'threads':2,
       'model_parameters':sum(p.numel() for p in initial_model.parameters())}
    write(out/'train_receipt.json',receipt)
    (out/'train_receipt.sha256').write_text(sha(out/'train_receipt.json')+'\n')
    return receipt


def load_models(training,seed,smoke=False):
    path=Path(training);receipt=path/'train_receipt.json'
    if sha(receipt)!=(path/'train_receipt.sha256').read_text().strip():raise ValueError('train receipt checksum mismatch')
    rec=json.loads(receipt.read_text())
    if (rec['experiment'],rec['seed'],rec['smoke'])!=('AF-17',seed,smoke):raise ValueError('AF17 identity mismatch')
    if rec['source_hashes']!=hashes() or rec['manifest_sha256']!=check_manifest():raise ValueError('source mismatch')
    if rec['label_sha256']!=sha(path/'train_label_receipt.jsonl'):raise ValueError('label receipt hash mismatch')
    result={}
    for arm in ARMS:
        for ext in ('pt','npz'):
            if rec['weights'][arm][ext]!=sha(path/f'{arm}.{ext}'):raise ValueError('model weight digest mismatch')
        pkg=torch.load(path/f'{arm}.pt',map_location='cpu',weights_only=True)
        if (pkg['experiment'],pkg['seed'],pkg['arm'])!=('AF-17',seed,arm):raise ValueError('AF17 checkpoint metadata mismatch')
        model=BellmanTypedPolicy();model.load_state_dict(pkg['state_dict']);model.eval()
        with np.load(path/f'{arm}.npz',allow_pickle=False) as z:
            if set(z.files)!=set(pkg['state_dict']) or any(not np.array_equal(z[k],v.numpy()) for k,v in pkg['state_dict'].items()):
                raise ValueError('numpy weights differ from Torch')
        result[arm]=model
    return rec,result


def evaluate(seed,training,out,smoke=False):
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('AF17 evaluation output must be fresh')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    receipt,models=load_models(training,seed,smoke)
    rows=[];scores={}
    families=('dev',) if smoke else ('held_layout','held_depth4','held_depth5')
    for arm,model in {**models,'fixed':None,'uniform':None,'shortest':None}.items():
        scores[arm]={}
        for split in families:
            batch=rollout(model,gym.cases_for(split),baseline=arm if model is None else None,seed=seed)
            rows.extend({'seed':seed,'arm':arm,**r} for r in batch)
            scores[arm][split]=summarize(batch)
    jsonl(out/'episodes.jsonl',rows)
    result={'experiment':'AF-17','seed':seed,'smoke':smoke,'source_hashes':hashes(),
      'manifest_sha256':check_manifest(),'training_sha256':sha(Path(training)/'train_receipt.json'),
      'episode_sha256':sha(out/'episodes.jsonl'),'rows':len(rows),'scores':scores,'fit':receipt['fit']}
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

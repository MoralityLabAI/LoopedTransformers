"""AF-16: multi-initialization Horn rule-priority transfer to new graph compositions."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from experiments.activation_funnel_af05 import gym as base
from experiments.activation_funnel_af15.model import TypedHornPolicy, objective
from . import gym

HERE=Path(__file__).parent
ROOT=HERE.parents[1]
SEEDS=(17,29,43,61)
ARMS=('imitation','frontier','imitation_extra')
STEPS={'imitation':240,'frontier':240,'imitation_extra':320}
BATCH=32
SOURCE_FILES=['gym.py','experiment.py','verify.py','aggregate.py','PROTOCOL.json',
              'fresh_manifest.json','RulePriority.lean','../activation_funnel_af15/model.py',
              '../activation_funnel_af15/gym.py','../activation_funnel_af05/gym.py',
              '../activation_funnel_af02/model.py','../lean_reasoner/train.py']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def hashes():return {s:sha(HERE/s) for s in SOURCE_FILES}
def write(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
def jsonl(path,rows):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w') as f:
        for row in rows:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')

def check_manifest():
    p=HERE/'fresh_manifest.json'
    if json.loads(p.read_text())!=gym.manifest():raise ValueError('AF16 corpus registration mismatch')
    return sha(p)


def training_states():
    legacy=[];novel=[]
    for c in gym.cases_for('train'):
        store=novel if isinstance(c,gym.Case) else legacy
        store.extend((c,state,opt) for state,opt in gym.reference_episode(c))
    return legacy,novel


def rollout(model,cases,baseline=None,seed=17):
    if model is not None:model.eval()
    initial=[frozenset(gym.initial_facts(c)) for c in cases]
    states=list(initial);done=[False]*len(cases);hist=[[] for _ in cases]
    shortest=[gym.optimal_remaining(c,s) for c,s in zip(cases,initial)]
    budgets=[d+1 if d<10**6 else 2 for d in shortest]
    generator=torch.Generator().manual_seed(16900+seed)
    for t in range(max(budgets)):
        active=[i for i in range(len(cases)) if not done[i] and t<budgets[i]]
        if not active:break
        for begin in range(0,len(active),32):
            ix=active[begin:begin+32];cc=[cases[i] for i in ix];known=[states[i] for i in ix]
            x,bits,gate=gym.observations(cc,known)
            if model is not None:
                with torch.no_grad():
                    raw=model(x,bits);score=raw.masked_fill(~gate,-1e4)
                actions=score.argmax(-1).tolist();logs=raw.tolist()
            else:
                actions=[];logs=[None]*len(ix)
                for c,s in zip(cc,known):
                    legal=list(gym.enabled(c,s))
                    if baseline=='uniform':
                        if legal:actions.append(legal[int(torch.randint(len(legal),(1,),generator=generator).item())])
                        else:actions.append(gym.STOP)
                    elif baseline=='fixed':actions.append(legal[0] if legal else gym.STOP)
                    elif baseline=='shortest':actions.append(gym.optimal_actions(c,s)[0])
                    else:raise ValueError('unknown fixed baseline')
            for j,i in enumerate(ix):
                old=states[i];action=actions[j]
                new,proof=gym.sound_fire(cases[i],old,action)
                hist[i].append({'before':sorted(old),'enabled':list(gym.enabled(cases[i],old)),
                                'action':action,'proof':proof,'after':sorted(new),'raw_logits':logs[j]})
                states[i]=new
                if action==gym.STOP or base.GOAL in new:done[i]=True
    rows=[]
    for i,c in enumerate(cases):
        provable=base.GOAL in gym.complete_closure(c)
        solved=base.GOAL in states[i]
        if solved and not provable:raise ValueError('unsound host rule gate')
        rows.append({'case_id':c.id,'split':c.split,'initial':sorted(initial[i]),
         'final':sorted(states[i]),'goal_truth':int(provable),'optimal_steps':shortest[i] if shortest[i]<10**6 else None,
         'budget':budgets[i],'actions':hist[i],'solved':bool(solved),'stopped':bool(done[i]),
         'used_actions':len(hist[i]),'host_accepted_false_proof':False})
    return rows

def summarize(rows):
    p=[r for r in rows if r['goal_truth']];n=[r for r in rows if not r['goal_truth']]
    if not p or not n:raise ValueError('AF16 expects positive and negative cases')
    return {'cases':len(rows),'provable':len(p),'solved':sum(r['solved'] for r in p),
            'proof_success':sum(r['solved'] for r in p)/len(p),
            'false_accepted':sum(r['host_accepted_false_proof'] for r in rows),
            'negative_safe':sum(not r['host_accepted_false_proof'] for r in n)/len(n),
            'actions_per_positive':sum(r['used_actions'] for r in p)/len(p)}

def train(seed,out,smoke=False):
    if seed not in SEEDS:raise ValueError('unregistered seed')
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('fresh training folder required')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    check_manifest();torch.manual_seed(81600+seed)
    prototype=TypedHornPolicy();initial={k:v.detach().clone() for k,v in prototype.state_dict().items()}
    legacy,novel=training_states();fit={};weights={};logs={}
    for arm in ARMS:
        m=TypedHornPolicy();m.load_state_dict(initial)
        opt=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.01)
        g=torch.Generator().manual_seed(816016+seed)
        steps=2 if smoke else STEPS[arm]
        history=[];first_grad=None;start=time.perf_counter()
        for step in range(steps):
            a=torch.randint(len(legacy),(BATCH//2,),generator=g).tolist()
            b=torch.randint(len(novel),(BATCH//2,),generator=g).tolist()
            samples=[legacy[i] for i in a]+[novel[i] for i in b]
            cases=[p[0] for p in samples];states=[p[1] for p in samples]
            x,bits,gate,pos=gym.observations(cases,states,with_labels=True)
            m.train();opt.zero_grad(set_to_none=True)
            raw=m(x,bits)
            loss,parts=objective(raw,gate,pos,'frontier' if arm=='frontier' else 'imitation')
            if not bool(torch.isfinite(loss)):raise FloatingPointError('AF16 training loss nonfinite')
            loss.backward()
            if first_grad is None:
                first_grad=sum(float(p.grad.square().sum()) for p in m.cell.parameters() if p.grad is not None)**.5
                if first_grad<=0:raise RuntimeError('AF16 gradients did not reach TypedLoop')
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0,error_if_nonfinite=True)
            opt.step()
            if step%60==0 or step==steps-1:history.append({'step':step,'loss':float(loss.detach()),**parts})
        m.eval()
        if any(not bool(torch.isfinite(p).all()) for p in m.parameters()):raise FloatingPointError('nonfinite trained state')
        torch.save({'experiment':'AF-16','seed':seed,'arm':arm,'state_dict':m.state_dict()},out/f'{arm}.pt')
        np.savez(out/f'{arm}.npz',**{k:v.detach().numpy() for k,v in m.state_dict().items()})
        weights[arm]={ext:sha(out/f'{arm}.{ext}') for ext in ('pt','npz')}
        fit[arm]={s:summarize(rollout(m,gym.cases_for(s),seed=seed)) for s in ('train','dev')}
        logs[arm]={'steps':steps,'wall_seconds':time.perf_counter()-start,'losses':history,
                   'cell_initial_gradient_l2':first_grad,'native_source_calls_train':steps*BATCH*2}
        print(json.dumps({'training':seed,'arm':arm,'dev_success':fit[arm]['dev']['proof_success'],
                          'seconds':logs[arm]['wall_seconds']}),flush=True)
    record={'experiment':'AF-16','seed':seed,'smoke':smoke,'fit':fit,'training':logs,
            'source_hashes':hashes(),'manifest_sha256':check_manifest(),
            'training_state_counts':{'legacy':len(legacy),'novel':len(novel)},
            'weights':weights,'torch':torch.__version__,'threads':2,
            'model_parameters':sum(p.numel() for p in prototype.parameters())}
    write(out/'train_receipt.json',record)
    (out/'train_receipt.sha256').write_text(sha(out/'train_receipt.json')+'\n')
    return record


def load_models(path,seed,smoke=False):
    path=Path(path);rec=path/'train_receipt.json'
    if sha(rec)!=(path/'train_receipt.sha256').read_text().strip():raise ValueError('train receipt changed')
    record=json.loads(rec.read_text())
    if (record['experiment'],record['seed'],record['smoke'])!=('AF-16',seed,smoke):raise ValueError('AF16 run identity changed')
    if record['manifest_sha256']!=check_manifest() or record['source_hashes']!=hashes():raise ValueError('AF16 source changed')
    result={}
    for arm in ARMS:
        for ext in ('pt','npz'):
            if sha(path/f'{arm}.{ext}')!=record['weights'][arm][ext]:raise ValueError('AF16 model hash changed')
        pkg=torch.load(path/f'{arm}.pt',map_location='cpu',weights_only=True)
        if (pkg['experiment'],pkg['seed'],pkg['arm'])!=('AF-16',seed,arm):raise ValueError('checkpoint identity mismatch')
        m=TypedHornPolicy();m.load_state_dict(pkg['state_dict']);m.eval()
        with np.load(path/f'{arm}.npz',allow_pickle=False) as z:
            if set(z.files)!=set(pkg['state_dict']) or any(not np.array_equal(z[k],t.numpy()) for k,t in pkg['state_dict'].items()):
                raise ValueError('AF16 named-array export differs')
        result[arm]=m
    return record,result


def evaluate(seed,training,out,smoke=False):
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('fresh evaluation folder required')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);record,models=load_models(training,seed,smoke)
    rows=[];scores={}
    splits=('dev',) if smoke else ('novel_layout','novel_depth4','novel_depth5')
    for arm,m in {**models,'fixed':None,'uniform':None,'shortest':None}.items():
        scores[arm]={}
        for split in splits:
            batch=rollout(m,gym.cases_for(split),baseline=arm if m is None else None,seed=seed)
            rows.extend({'seed':seed,'arm':arm,**r} for r in batch)
            scores[arm][split]=summarize(batch)
    jsonl(out/'episodes.jsonl',rows)
    s={'experiment':'AF-16','seed':seed,'smoke':smoke,'source_hashes':hashes(),
        'manifest_sha256':check_manifest(),'training_sha256':sha(Path(training)/'train_receipt.json'),
        'episode_sha256':sha(out/'episodes.jsonl'),'rows':len(rows),'scores':scores,'fit':record['fit']}
    write(out/'evaluation.json',s)
    return s


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('manifest','train','evaluate'))
    p.add_argument('--seed',type=int);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--training',type=Path);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.command=='manifest':write(a.out,gym.manifest())
    elif a.command=='train':train(a.seed,a.out,a.smoke)
    else:evaluate(a.seed,a.training,a.out,a.smoke)
if __name__=='__main__':main()

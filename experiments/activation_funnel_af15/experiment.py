"""AF-15 frozen-source training and host-checked rule-choice evaluation."""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from experiments.activation_funnel_af05 import gym as base
from . import gym,model as policy

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).parent
SEEDS=(17,29)
ARMS=('imitation','frontier')
STEPS=240
BATCH=32
PROTOCOL=HERE/'PROTOCOL.json'
MANIFEST=HERE/'fresh_manifest.json'
FILES={'gym.py':HERE/'gym.py','model.py':HERE/'model.py',
       'experiment.py':HERE/'experiment.py','verify.py':HERE/'verify.py',
       'aggregate.py':HERE/'aggregate.py','RuleSoundness.lean':HERE/'RuleSoundness.lean',
       'PROTOCOL.json':PROTOCOL,'fresh_manifest.json':MANIFEST,
       'original_Af05_gym.py':ROOT/'experiments/activation_funnel_af05/gym.py',
       'original_TypedLoop.py':ROOT/'experiments/lean_reasoner/train.py'}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def hashes():return {k:sha(p) for k,p in FILES.items()}

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n')

def write_jsonl(path,rows):
    with Path(path).open('w') as f:
        for row in rows:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')

def check_manifest():
    expected=json.loads(MANIFEST.read_text())
    if expected!=gym.manifest():raise ValueError('AF-15 frozen graph corpus mismatch')
    return sha(MANIFEST)


def reference_states():
    cases=gym.cases_for('train')
    return [(c,state,targets) for c in cases for state,targets in gym.reference_episode(c)]


def rollout(model,cases,baseline=None,seed=17):
    """Host takes exactly one enabled rule per action; it verifies every proof edge."""
    model.eval() if model is not None else None
    initial=[frozenset(base.initial_facts(c)) for c in cases]
    states=list(initial);done=[False]*len(cases);hist=[[] for _ in cases]
    optimal=[gym.optimal_remaining(c,s) for c,s in zip(cases,initial)]
    budget=[(v+1 if v<10**6 else 2) for v in optimal]
    generator=torch.Generator().manual_seed(1000+seed)
    for t in range(max(budget)):
        active=[i for i in range(len(cases)) if not done[i] and t<budget[i]]
        if not active:break
        for begin in range(0,len(active),32):
            ix=active[begin:begin+32]
            subcase=[cases[i] for i in ix];known=[states[i] for i in ix]
            x,bits,gate=policy.observations(subcase,known)
            if model is not None:
                with torch.no_grad():
                    raw=model(x,bits)
                    scores=raw.masked_fill(~gate,-1e4)
                    actions=scores.argmax(-1).tolist()
                    rawlogits=raw.tolist()
            else:
                actions=[];rawlogits=[None]*len(ix)
                for j,c in enumerate(subcase):
                    legal=list(gym.enabled(c,known[j]))
                    if baseline=='uniform':
                        if legal:
                            choice=int(torch.randint(len(legal),(1,),generator=generator).item())
                            actions.append(legal[choice])
                        else:actions.append(gym.STOP)
                    elif baseline=='fixed':actions.append(legal[0] if legal else gym.STOP)
                    elif baseline=='shortest':actions.append(gym.optimal_actions(c,known[j])[0])
                    else:raise ValueError('unknown AF-15 baseline')
            for j,i in enumerate(ix):
                old=states[i];action=actions[j];legal=list(gym.enabled(cases[i],old))
                new,proof=gym.sound_fire(cases[i],old,action)
                hist[i].append({'before':sorted(old),'enabled':legal,'action':action,
                   'proof':proof,'after':sorted(new),'raw_logits':rawlogits[j]})
                states[i]=new
                if action==gym.STOP or base.GOAL in new:done[i]=True
    rows=[]
    for i,c in enumerate(cases):
        provable=base.oracle(c)[0][2]==1
        solved=base.GOAL in states[i]
        assert (not solved) or provable
        rows.append({'case_id':c.id,'split':c.split,'skeleton':[c.depth,list(c.kinds),c.gate,c.permutation],
         'initial':sorted(initial[i]),'final':sorted(states[i]),
         'goal_truth':int(provable),'optimal_steps':None if optimal[i]>=10**6 else optimal[i],
         'budget':budget[i],'actions':hist[i],'solved':bool(solved),
         'stopped':bool(done[i]),'used_actions':len(hist[i]),'host_accepted_false_proof':False})
    return rows


def summarize(rows):
    positives=[r for r in rows if r['goal_truth']]
    negative=[r for r in rows if not r['goal_truth']]
    if not positives or not negative:raise ValueError('need both positive and negative original Horn cases')
    return {'cases':len(rows),'provable':len(positives),
      'solved':sum(r['solved'] for r in positives),
      'proof_success':sum(r['solved'] for r in positives)/len(positives),
      'negative_safe':sum(not r['host_accepted_false_proof'] for r in negative)/len(negative),
      'actions_per_positive':sum(r['used_actions'] for r in positives)/len(positives),
      'budget':sum(r['budget'] for r in positives)/len(positives),
      'false_accepted':sum(r['host_accepted_false_proof'] for r in rows)}


def train(seed,out,steps=STEPS,smoke=False):
    if seed not in SEEDS or steps<2 or (not smoke and steps!=STEPS):
        raise ValueError('AF-15 training identity/budget differs from registration')
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('AF-15 requires fresh training destination')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    manifest=check_manifest();torch.manual_seed(81500+seed)
    prototype=policy.TypedHornPolicy()
    initial={k:v.detach().clone() for k,v in prototype.state_dict().items()}
    training=reference_states()
    fitting={};weight_files={};losses={}
    for arm in ARMS:
        model=policy.TypedHornPolicy();model.load_state_dict(initial)
        opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
        rng=torch.Generator().manual_seed(815010+seed)
        logs=[];start=time.perf_counter();first_grad=None
        for step in range(steps):
            ix=torch.randint(len(training),(BATCH,),generator=rng).tolist()
            selected=[training[i] for i in ix]
            x,bits,gate,target=policy.state_inputs([a for a,_,_ in selected],
                                                    [b for _,b,_ in selected])
            model.train();opt.zero_grad(set_to_none=True)
            raw=model(x,bits)
            loss,parts=policy.objective(raw,gate,target,arm)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite AF-15 loss')
            loss.backward()
            if first_grad is None:
                first_grad=sum(float(p.grad.square().sum()) for p in model.cell.parameters() if p.grad is not None)**.5
                assert first_grad>0
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            if any(not bool(torch.isfinite(v).all()) for v in model.parameters()):raise FloatingPointError('nonfinite AF-15 weights')
            if step%60==0 or step==steps-1:logs.append({'step':step,'loss':float(loss.detach()),**parts})
        model.eval()
        for name,ts in model.state_dict().items():
            if not bool(torch.isfinite(ts).all()):raise FloatingPointError('nonfinite source state')
        torch.save({'experiment':'AF-15','seed':seed,'arm':arm,'state_dict':model.state_dict()},out/f'{arm}.pt')
        np.savez(out/f'{arm}.npz',**{k:v.detach().numpy() for k,v in model.state_dict().items()})
        weight_files[arm]={ext:sha(out/f'{arm}.{ext}') for ext in ('pt','npz')}
        dev_rows=rollout(model,gym.cases_for('dev'),seed=seed)
        train_rows=rollout(model,gym.cases_for('train'),seed=seed)
        fitting[arm]={'train':summarize(train_rows),'dev':summarize(dev_rows)}
        losses[arm]={'seconds':time.perf_counter()-start,'native_grad_first_step_l2':first_grad,'losses':logs}
        print(json.dumps({'stage':'train','seed':seed,'arm':arm,'dev':fitting[arm]['dev'],'seconds':losses[arm]['seconds']}),flush=True)
    receipt={'experiment':'AF-15','seed':seed,'steps':steps,'smoke':smoke,
      'source_hashes':hashes(),'manifest_sha256':manifest,'reference_training_states':len(training),
      'fit':fitting,'weights':weight_files,'training':losses,
      'torch':torch.__version__,'threads':2,'model_parameters':sum(x.numel() for x in prototype.parameters())}
    write(out/'train_receipt.json',receipt)
    (out/'train_receipt.sha256').write_text(sha(out/'train_receipt.json')+'\n')
    return receipt


def load_models(path,seed,smoke=False):
    path=Path(path);recpath=path/'train_receipt.json'
    if sha(recpath)!=(path/'train_receipt.sha256').read_text().strip():raise ValueError('training receipt hash changed')
    rec=json.loads(recpath.read_text())
    if (rec['experiment'],rec['seed'],rec['smoke'])!=('AF-15',seed,smoke):raise ValueError('training identity mismatch')
    if rec['manifest_sha256']!=check_manifest() or rec['source_hashes']!=hashes():
        raise ValueError('AF-15 frozen source changed')
    models={}
    for arm in ARMS:
        for ext in ('pt','npz'):
            if sha(path/f'{arm}.{ext}')!=rec['weights'][arm][ext]:raise ValueError('checkpoint bytes changed')
        p=torch.load(path/f'{arm}.pt',map_location='cpu',weights_only=True)
        if (p['experiment'],p['seed'],p['arm'])!=('AF-15',seed,arm):raise ValueError('checkpoint metadata changed')
        model=policy.TypedHornPolicy();model.load_state_dict(p['state_dict']);model.eval()
        with np.load(path/f'{arm}.npz',allow_pickle=False) as z:
            if set(z.files)!=set(p['state_dict']) or any(not np.array_equal(z[k],v.numpy()) for k,v in p['state_dict'].items()):
                raise ValueError('NumPy and Torch checkpoint disagree')
        models[arm]=model
    return rec,models


def evaluate(seed,train,out,smoke=False):
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('AF-15 evaluation must be fresh')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    receipt,models=load_models(train,seed,smoke)
    splits=('dev',) if smoke else gym.FRESH
    rows=[];scores={}
    for arm,m in {**models,'fixed':None,'uniform':None,'shortest':None}.items():
        scores[arm]={}
        for split in splits:
            output=rollout(m,gym.cases_for(split),baseline=arm if m is None else None,seed=seed)
            for row in output:rows.append({'seed':seed,'arm':arm,**row})
            scores[arm][split]=summarize(output)
            print(json.dumps({'stage':'evaluation','seed':seed,'arm':arm,'split':split,'score':scores[arm][split]}),flush=True)
    write_jsonl(out/'episodes.jsonl',rows)
    summary={'experiment':'AF-15','seed':seed,'smoke':smoke,'split_names':list(splits),
        'source_hashes':hashes(),'manifest_sha256':check_manifest(),
        'training_sha256':sha(Path(train)/'train_receipt.json'),
        'episodes_sha256':sha(out/'episodes.jsonl'),'rows':len(rows),'scores':scores,'fit':receipt['fit']}
    write(out/'evaluation.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('train','evaluate'))
    p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--training',type=Path);p.add_argument('--steps',type=int,default=STEPS)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.command=='train':train(a.seed,a.out,a.steps,a.smoke)
    else:
        if a.training is None:p.error('evaluation requires --training')
        evaluate(a.seed,a.training,a.out,a.smoke)
if __name__=='__main__':main()

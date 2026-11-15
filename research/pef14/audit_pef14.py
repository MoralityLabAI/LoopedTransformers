"""Independent PEF-14 receipt audit, including pure-Python phase-mixture replay.

Reads frozen JSON and checkpoint bytes; it does not call the experiment's
fitters, projector, quantizer or scoring implementation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch

REG='e63ad7c93109713decdd69ad662c52e916d0a8b3'
SEEDS=(211,223,239)
TASKS=('coalition','threshold')
TANGENTS=('random','pair01')
ANGLES=(.25,4.)
STEPS=(.009,.018,.045,.07)
N=768


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def canonical(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()

def weight_sha(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode())
        h.update(bytes(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).tolist()))
    return h.hexdigest()


def independent_forecast(cond,scale,relative):
    f=cond['margin'];j=cond['jacobian'];y=cond['labels'];q=cond['q']
    if not len(f)==len(j)==len(y)==len(q)==768:raise AssertionError('probe arrays not full')
    delta=scale*relative
    d=[delta*round(v/delta)-v for v in q]
    exact=0.;uniform=0.;cross=0.
    for c in range(48):
        for m in range(16):
            k=16*c+m
            pos=(f[k]+j[k]*d[k]>=0)
            exact+=(int(pos)==int(y[k]))
            if abs(j[k])<=1e-14:
                up=float((int(f[k]>=0)==int(y[k])))
            else:
                signed=(2*y[k]-1)*f[k]
                up=max(0.,min(1.,.5+signed/(delta*abs(j[k]))))
            uniform+=up
            for other in range(48):
                if other==c:continue
                value=f[k]+j[k]*d[16*other+m]
                cross+=(int(value>=0)==int(y[k]))
    exact/=768
    uniform/=768
    cross/=768*47
    mixture=(47*cross+8*uniform)/55
    return dict(exact_probe_phase=exact,uniform_phase=uniform,
                conditional_cross=cross,distribution_mixture=mixture)


def audit(root):
    root=Path(root)
    paths={name:root/name for name in ('fresh_source_split.json','fresh_training_ledger.json',
                                         'sealed_probe_forecasts.json','summary.json')}
    src,train,probe,res=(json.loads(paths[k].read_text()) for k in paths)
    if hashlib.sha256(canonical({k:v for k,v in src.items() if k!='sha256'})).hexdigest()!=src['sha256']:
        raise AssertionError('source manifest hash failed')
    if probe['source_sha256']!=train['source_sha256'] or src['sha256']!=train['source_sha256']:
        raise AssertionError('source chain mismatch')
    if sha(paths['fresh_training_ledger.json'])!=probe['training_ledger_sha256']:
        raise AssertionError('training ledger hash changed')
    if sha(paths['sealed_probe_forecasts.json'])!=res['sealed_probe_forecast_sha256']:
        raise AssertionError('forecast frozen hash changed')
    if res['registered_protocol_commit']!=REG or not probe['forecast_sealed_before_heldout']:
        raise AssertionError('preregistration order missing')
    if src['context_counts']!={'train':160,'probe':48,'heldout':48}:
        raise AssertionError('context assignment sizes altered')
    if len(set(src['sorted_contexts']))!=256 or len(src['context_assignment'])!=256:
        raise AssertionError('context identity collision')
    assert len(train['checkpoints'])==len(probe['models'])==len(res['models'])==6
    entries=[]
    for rec in train['checkpoints']:
        seed,task=rec['seed'],rec['task']
        path=root/rec['file']
        if sha(path)!=rec['checkpoint_sha256']:
            raise AssertionError('frozen bytes changed')
        payload=torch.load(path,weights_only=True,map_location='cpu')
        if (payload['seed'],payload['task'],payload['steps'])!=(seed,task,450):
            raise AssertionError('wrong checkpoint metadata')
        if weight_sha(payload['model'])!=rec['frozen_weights_sha256']:
            raise AssertionError('trained parameter digest altered')
        if not all(int(v['step'].item())==450 for v in payload['optimizer']['state'].values()):
            raise AssertionError('incomplete training run')
        prow=next(p for p in probe['models'] if p['seed']==seed and p['task']==task)
        rrow=next(r for r in res['models'] if r['seed']==seed and r['task']==task)
        if len(prow['conditions'])!=4 or len(rrow['conditions'])!=16:
            raise AssertionError('condition coverage incomplete')
        if rrow['native_h12_correct']!=768:
            raise AssertionError('new-seed native benchmark score changed')
        per=[]
        for cond in prow['conditions']:
            if (cond['tangent'],cond['angle_deg']) not in [(t,a) for t in TANGENTS for a in ANGLES]:
                raise AssertionError('unknown preregistered condition')
            if len(cond['forecasts'])!=4:raise AssertionError('partial step grid')
            for pred in cond['forecasts']:
                rel=pred['relative_step']
                if rel not in STEPS:raise AssertionError('unspecified quantizer step')
                computed=independent_forecast(cond,prow['pc_scale'],rel)
                for name,value in computed.items():
                    if abs(value-pred[name])>1e-9:
                        raise AssertionError(f'{seed} {task} {name} mismatch {value} vs {pred[name]}')
                actual=next(x for x in rrow['conditions'] if
                            (x['tangent'],x['angle_deg'],x['relative_step'])==
                            (cond['tangent'],cond['angle_deg'],rel))
                if abs(actual['accuracy']-actual['correct']/768)>1e-12:
                    raise AssertionError('heldout numerator/denominator mismatch')
                per.append((computed,actual['accuracy']))
        values={name:sum(abs(a[name]-b) for a,b in per)/len(per) for name in
                ('distribution_mixture','uniform_phase','exact_probe_phase','conditional_cross')}
        decision=next(d for d in res['decisions'] if (d['seed'],d['task'])==(seed,task))
        for name,err in values.items():
            if abs(err-decision['mae'][name])>1e-9:
                raise AssertionError('registered decision MAE mismatch')
        success=values['distribution_mixture']<=.030 and values['distribution_mixture']<=.90*values['uniform_phase']
        if success!=decision['registered_seed_criterion']:
            raise AssertionError('registered condition reclassification')
        entries.append({'seed':seed,'task':task,'model_digest':rec['frozen_weights_sha256'],
                        'conditions':len(per),'mae':values,'criterion':success})
    got=sum(r['criterion'] for r in entries if r['task']=='coalition')
    expected='criterion_met' if got>=2 else 'criterion_not_met'
    if got!=res['registered_successful_coalition_seeds'] or expected!=res['primary_decision']:
        raise AssertionError('cohort decision mismatch')
    return {'campaign':'PEF-14','independent_audit':'passed',
            'model_checkpoints_verified':len(entries),
            'condition_forecasts_recomputed':sum(x['conditions'] for x in entries),
            'optimizer_updates_per_model_verified':450,
            'registered_coalition_pass_count':got,'primary_decision':expected,
            'sha256':{k:sha(v) for k,v in paths.items()},
            'seed_results':entries}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    result=audit(args.out)
    (args.out/'independent_audit.json').write_bytes(canonical(result))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()

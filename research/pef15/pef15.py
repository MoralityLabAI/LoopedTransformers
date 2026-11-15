"""PEF-15: new-seed conformal intervals for quantized recursive representations.

Registration: 20e72699aa9717bc7de3bb395302d48f9e5a0043.
All model/checkpoint/fit/calibration receipts precede heldout model scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from research.pef14 import pef14 as p14
from research.pef07 import pef07 as p7

REGISTRATION = "20e72699aa9717bc7de3bb395302d48f9e5a0043"
SALT = b"pef15-splitconformal-v1-20261115:"
COUNTS = {"train":160,"fit":32,"calibration":16,"heldout":48}
SEEDS = (251,263,277)
TASKS = ("coalition","threshold")
TANGENTS = ("random","pair01")
ANGLES = (0.25,4.0)
STEPS = (0.011,0.025,0.04,0.065)
K = 16
ALPHA = 0.10
LAMBDA = 8


def canonical(x):
    return p14.canonical(x)


def sha(path):
    return p14.sha_file(path)


def split_contexts():
    words = [tuple((i >> j) & 1 for j in range(8)) for i in range(256)]
    words.sort(key=lambda c: (hashlib.sha256(SALT + bytes(c)).digest(), bytes(c)))
    catalog, owners = {}, {}
    offset = 0
    for section, count in COUNTS.items():
        selected = words[offset:offset + count]
        offset += count
        for ctx in selected:
            owners["".join(map(str,ctx))] = section
        catalog[section] = torch.tensor(
            [tuple((mask >> j) & 1 for j in range(4)) + ctx
             for ctx in selected for mask in range(16)],dtype=torch.long)
    assert offset == 256
    seen=set()
    for name,values in catalog.items():
        assert values.shape == (COUNTS[name] * 16,12)
        cube=values.reshape(-1,16,12)
        for group in cube:
            assert len({tuple(x[4:].tolist()) for x in group})==1
            assert [tuple(x[:4].tolist()) for x in group] == [
                tuple((mask >> j) & 1 for j in range(4)) for mask in range(16)]
        for bits in values.tolist():
            word=tuple(bits)
            assert word not in seen
            seen.add(word)
    assert len(seen) == 4096
    manifest={"campaign":"PEF-15","registration_commit":REGISTRATION,
              "salt":SALT.decode(),"context_counts":COUNTS,
              "history_counts":{k:len(v) for k,v in catalog.items()},
              "context_assignment":owners,
              "sorted_contexts":["".join(map(str,x)) for x in words],
              "full_focal_cube_per_context":True}
    manifest["sha256"] = p14.sha_bytes(canonical(manifest))
    return catalog,manifest


def conformal_rank(n,alpha=ALPHA):
    if n < 1 or not 0 < alpha < 1:
        raise ValueError('invalid conformal parameters')
    return min(n + 1,math.ceil((n+1)*(1-alpha)))


def interval_from_calibration(point, sample, alpha=ALPHA):
    """Split conformal for one future *context*, with all 16 masks kept together."""
    if not 0 <= point <= 1 or not sample:
        raise ValueError('invalid point or sample')
    residuals = sorted(abs(float(x)-point) for x in sample)
    n=len(residuals)
    rank=conformal_rank(n,alpha)
    radius = residuals[rank-1] if rank<=n else 1.0
    return {"point_forecast":float(point),"rank":rank,"n_calibration_contexts":n,
            "radius":radius,"low":max(0.,point-radius),
            "high":min(1.,point+radius),
            "calibration_residuals_sorted":residuals}


def serfling_heldout_interval(calibration_rates,n_heldout=48,alpha=ALPHA):
    """Bounded finite-population 90% CI for mean of the other contexts."""
    n=len(calibration_rates)
    if n < 1 or n_heldout < 1 or not 0 < alpha < 1:
        raise ValueError('invalid population or confidence')
    if any(not 0 <= float(z) <= 1 for z in calibration_rates):
        raise ValueError('context accuracy must lie in [0,1]')
    N=n+n_heldout
    mean=sum(float(z) for z in calibration_rates)/n
    radius=(N/n_heldout)*math.sqrt((1-(n-1)/N)*math.log(2/alpha)/(2*n))
    return {"center":mean,"radius":radius,"low":max(0,mean-radius),
            "high":min(1,mean+radius),"n_calibration":n,"n_heldout":n_heldout}


def phase_forecast(f,j,y,q,step,ncontexts=32,pseudocount=LAMBDA):
    if len(f)!=16*ncontexts or not (len(f)==len(j)==len(y)==len(q)):
        raise ValueError('complete context hypercubes required')
    residual=p14.quant_error(q,step).reshape(ncontexts,16)
    a=f.double().reshape(ncontexts,16)
    grad=j.double().reshape(ncontexts,16)
    truth=y.long().reshape(ncontexts,16)
    unif=p14.uniform_probability(f,j,y,step).reshape(ncontexts,16)
    own=((a+grad*residual>=0).long()==truth).double()
    mask=~torch.eye(ncontexts,dtype=torch.bool)
    cross=torch.empty_like(a)
    for k in range(16):
        predicted=a[:,k,None]+grad[:,k,None]*residual[None,:,k]
        good=((predicted>=0).long()==truth[:,k,None]).double()
        cross[:,k]=good[mask].reshape(ncontexts,ncontexts-1).mean(1)
    mixture=((ncontexts-1)*cross+pseudocount*unif)/(ncontexts-1+pseudocount)
    return {"distribution_mixture":float(mixture.mean()),
            "uniform_phase":float(unif.mean()),
            "exact_fit_phase":float(own.mean()),
            "cross_context":float(cross.mean()),
            "clean_fit":float(((a>=0).long()==truth).double().mean())}


def torch_ridge(fit):
    return tuple(torch.tensor(fit[k],dtype=torch.float64) for k in ('mean','std','beta'))


def train_and_seal(catalog,manifest,root):
    rows=[]
    for seed in SEEDS:
        for task in TASKS:
            rec=p14.train_single(seed,task,catalog['train'],root)
            rec['source_sha256']=manifest['sha256']
            rows.append(rec)
            print('PEF15_CHECKPOINT_SEALED',seed,task,rec['checkpoint_sha256'],flush=True)
    ledger={"campaign":"PEF-15","registered_commit":REGISTRATION,
            "source_sha256":manifest['sha256'],"all_sealed_before_fit":True,
            "seeds":list(SEEDS),"tasks":list(TASKS),"steps":450,
            "torch_version":str(torch.__version__),"cpu_capability":torch.backends.cpu.get_cpu_capability(),
            "checkpoint_records":rows}
    (root/'training_ledger.json').write_bytes(canonical(ledger))
    return ledger


def freeze_fit(catalog,manifest,root):
    training=json.loads((root/'training_ledger.json').read_text())
    assert len(training['checkpoint_records'])==6 and training['source_sha256']==manifest['sha256']
    raw=catalog['fit']
    labels={task:p14.class_truth(raw,task) for task in TASKS}
    models=[]
    for rec in training['checkpoint_records']:
        model=p14.checked_model(root,rec)
        seed,task=rec['seed'],rec['task']
        beginning=p14.model_digest(model)
        h8=p14.at_eight(model,raw)
        geo=p14.geometry(h8,seed,task)
        mu,g=geo['mu'],geo['g']
        conditions=[]
        for tangent in TANGENTS:
            for angle in ANGLES:
                v,w=p14.rotate(g,geo['tangents'][tangent],angle)
                z=p14.project(h8,mu,v)
                with torch.no_grad():
                    h12=p14.after_eight(model,raw,z)
                    fit=p7.fit_ridge(h12,labels[task])
                f,j=p14.probe_margin_j(model,raw,z,fit,w)
                q=(z.double()-mu)@w
                if not (torch.isfinite(f).all() and torch.isfinite(j).all() and torch.isfinite(q).all()):
                    raise AssertionError('nonfinite fit context features')
                predicted=[]
                for relative in STEPS:
                    stats=phase_forecast(f,j,labels[task],q,relative*geo['pc_scale'])
                    predicted.append({'relative_step':relative,**stats})
                conditions.append({'tangent':tangent,'angle':angle,'v':v.tolist(),
                    'w':w.tolist(), 'ridge':{'mean':fit[0].tolist(),
                        'std':fit[1].tolist(),'beta':fit[2].tolist()},
                    'ridge_sha256':p14.ridge_digest(fit),
                    'fit_features':{'margin':f.tolist(),'jacobian':j.tolist(),
                                     'q':q.tolist(),'label':labels[task].tolist()},
                    'forecast':predicted})
        assert p14.model_digest(model)==beginning
        models.append({'seed':seed,'task':task,'checkpoint_sha256':rec['checkpoint_sha256'],
            'center':mu.tolist(),'global_pc':g.tolist(),
            'pc_scale':geo['pc_scale'],'pair_degenerate':geo['pair_degenerate'],
            'conditions':conditions})
        print('PEF15_FIT_READY',seed,task,flush=True)
    ledger={'campaign':'PEF-15','registration_commit':REGISTRATION,
        'training_ledger_sha256':sha(root/'training_ledger.json'),
        'source_sha256':manifest['sha256'], 'fit_contexts':COUNTS['fit'],
        'calibration_contexts':COUNTS['calibration'],'heldout_contexts':COUNTS['heldout'],
        'fit_sealed_before_calibration':True,'angles':list(ANGLES),
        'tangents':list(TANGENTS),'relative_steps':list(STEPS),'models':models}
    (root/'sealed_fit_ledger.json').write_bytes(canonical(ledger))
    print('PEF15_ALL_FIT_FORECASTS_FROZEN',sha(root/'sealed_fit_ledger.json'),flush=True)
    return ledger


@torch.no_grad()
def score_group(model,bits,h8,truth,mu,pc_scale,cond,rel):
    v=torch.tensor(cond['v'],dtype=torch.float64)
    w=torch.tensor(cond['w'],dtype=torch.float64)
    fit=torch_ridge(cond['ridge'])
    if p14.ridge_digest(fit)!=cond['ridge_sha256']:
        raise AssertionError('readout SHA256 differs')
    z=p14.project(h8,mu,v)
    q=(z.double()-mu)@w
    d=p14.quant_error(q,pc_scale*rel)
    quantized=(z.double()+d[:,None]*w[None,:]).float()
    m=p14.ridge_margin(p14.after_eight(model,bits,quantized),fit)
    labels=p14.class_truth(bits,cond['task']) if 'task' in cond else None
    if labels is None:
        raise AssertionError('condition task absent')
    return (m>=0).long(),labels


def validated_chains(root,manifest, require_calibration=False):
    training=json.loads((root/'training_ledger.json').read_text())
    fit=json.loads((root/'sealed_fit_ledger.json').read_text())
    if not (training['all_sealed_before_fit'] and fit['fit_sealed_before_calibration']):
        raise AssertionError('missing freeze receipts')
    if (training['source_sha256']!=fit['source_sha256'] or
            fit['source_sha256']!=manifest['sha256'] or
            fit['training_ledger_sha256']!=sha(root/'training_ledger.json')):
        raise AssertionError('wrong source or parent fit ledger')
    for rec in training['checkpoint_records']:
        p14.checked_model(root,rec)
        row=next((x for x in fit['models'] if (x['seed'],x['task'])==(rec['seed'],rec['task'])),None)
        if row is None or row['checkpoint_sha256']!=rec['checkpoint_sha256'] or len(row['conditions'])!=4:
            raise AssertionError('missing matching fit condition')
    if not require_calibration:
        return training,fit
    calibration=json.loads((root/'sealed_calibration_ledger.json').read_text())
    if (calibration['fit_ledger_sha256']!=sha(root/'sealed_fit_ledger.json') or
        calibration['source_sha256']!=manifest['sha256'] or
        not calibration['all_calibrated_before_heldout']):
        raise AssertionError('calibration ledger mismatch')
    return training,fit,calibration


def freeze_calibration(catalog,manifest,root):
    training,fit=validated_chains(root,manifest)
    bits=catalog['calibration']
    out=[]
    with torch.no_grad():
        for rec in training['checkpoint_records']:
            seed,task=rec['seed'],rec['task']
            model=p14.checked_model(root,rec)
            begin=p14.model_digest(model)
            h8=p14.at_eight(model,bits)
            prow=next(x for x in fit['models'] if (x['seed'],x['task'])==(seed,task))
            mu=torch.tensor(prow['center'],dtype=torch.float64)
            cells=[]
            for cond in prow['conditions']:
                cond_task={**cond,'task':task}
                for forecast in cond['forecast']:
                    step=forecast['relative_step']
                    pred,truth=score_group(model,bits,h8,None,mu,prow['pc_scale'],cond_task,step)
                    correct=(pred==truth).reshape(16,16).sum(1).tolist()
                    rates=[int(x)/16 for x in correct]
                    point=forecast['distribution_mixture']
                    interval=interval_from_calibration(point,rates)
                    serfling=serfling_heldout_interval(rates)
                    cells.append({'tangent':cond['tangent'],'angle':cond['angle'],
                        'relative_step':step,'fit_prediction':forecast,
                        'calibration_correct_by_context':list(map(int,correct)),
                        'group_interval':interval,'aggregate_serfling':serfling})
            assert p14.model_digest(model)==begin
            out.append({'seed':seed,'task':task,'checkpoint_sha256':rec['checkpoint_sha256'],
                        'conditions':cells})
            print('PEF15_CALIBRATED',seed,task,flush=True)
    ledger={'campaign':'PEF-15','registration_commit':REGISTRATION,
            'fit_ledger_sha256':sha(root/'sealed_fit_ledger.json'),
            'source_sha256':manifest['sha256'],'calibration_contexts':16,
            'heldout_contexts':48,'all_calibrated_before_heldout':True,'models':out}
    (root/'sealed_calibration_ledger.json').write_bytes(canonical(ledger))
    print('PEF15_ALL_INTERVALS_FROZEN',sha(root/'sealed_calibration_ledger.json'),flush=True)
    return ledger


def heldout_scoring(catalog,manifest,root):
    training,fit,calibration=validated_chains(root,manifest,True)
    bits=catalog['heldout']
    models=[]
    for rec in training['checkpoint_records']:
        seed,task=rec['seed'],rec['task']
        model=p14.checked_model(root,rec)
        beginning=p14.model_digest(model)
        before_sha=sha(root/rec['file'])
        with torch.no_grad():
            h8=p14.at_eight(model,bits)
            native=model.head(p14.after_eight(model,bits,h8)).argmax(1)
            truth=p14.class_truth(bits,task)
            native_correct=int((native==truth).sum())
        prow=next(x for x in fit['models'] if (x['seed'],x['task'])==(seed,task))
        crow=next(x for x in calibration['models'] if (x['seed'],x['task'])==(seed,task))
        mu=torch.tensor(prow['center'],dtype=torch.float64)
        results=[]
        for cond in prow['conditions']:
            cond_task={**cond,'task':task}
            for forecast in cond['forecast']:
                step=forecast['relative_step']
                baseline=next(x for x in crow['conditions'] if (x['tangent'],x['angle'],x['relative_step'])==(
                    cond['tangent'],cond['angle'],step))
                with torch.no_grad():
                    pred,truth=score_group(model,bits,h8,None,mu,prow['pc_scale'],cond_task,step)
                correct_group=(pred==truth).reshape(48,16).sum(1).tolist()
                rate=[int(x)/16 for x in correct_group]
                correct=int(sum(correct_group))
                group_interval=baseline['group_interval']
                serfling=baseline['aggregate_serfling']
                covered=sum(group_interval['low']<=r<=group_interval['high'] for r in rate)
                heldout_avg=correct/768
                results.append({'tangent':cond['tangent'],'angle':cond['angle'],
                    'relative_step':step,'correct':correct,'total':768,
                    'mean_heldout_accuracy':heldout_avg,'correct_by_context':list(map(int,correct_group)),
                    'conformal_covered_contexts':covered,'conformal_total_contexts':48,
                    'conformal_width':group_interval['high']-group_interval['low'],
                    'conformal_interval':{'low':group_interval['low'],'high':group_interval['high']},
                    'aggregate_serfling_covered':bool(serfling['low']<=heldout_avg<=serfling['high']),
                    'aggregate_serfling_interval':{'low':serfling['low'],'high':serfling['high']},
                    'fit_prediction':forecast['distribution_mixture'],
                    'absolute_point_error':abs(forecast['distribution_mixture']-heldout_avg),
                    'class0_recall':float((pred[truth==0]==0).float().mean()),
                    'class1_recall':float((pred[truth==1]==1).float().mean())})
        if p14.model_digest(model)!=beginning or sha(root/rec['file'])!=before_sha:
            raise AssertionError('checkpoint changed after scoring')
        coverage=sum(x['conformal_covered_contexts'] for x in results)/(48*16)
        width=sum(x['conformal_width'] for x in results)/16
        success=coverage>=.88 and width<=.85
        models.append({'seed':seed,'task':task,'checkpoint_sha256':before_sha,
            'native_h12_correct':native_correct,'conformal_group_coverage':coverage,
            'conformal_mean_interval_width':width,
            'aggregate_serfling_covered':sum(x['aggregate_serfling_covered'] for x in results),
            'group_coverage_count':sum(x['conformal_covered_contexts'] for x in results),
            'registered_seed_success':bool(success),'conditions':results})
        print('PEF15_HELDOUT',seed,task,'native',native_correct,'coverage',coverage,
              'width',width,'serfling',models[-1]['aggregate_serfling_covered'],flush=True)
    coalition=sum(x['registered_seed_success'] for x in models if x['task']=='coalition')
    report={'campaign':'PEF-15','registration_commit':REGISTRATION,
        'source_sha256':manifest['sha256'],
        'training_ledger_sha256':sha(root/'training_ledger.json'),
        'fit_ledger_sha256':sha(root/'sealed_fit_ledger.json'),
        'calibration_ledger_sha256':sha(root/'sealed_calibration_ledger.json'),
        'all_models_frozen_before_fit':True,'fit_frozen_before_calibration':True,
        'calibration_frozen_before_heldout':True,'fresh_training_seeds':list(SEEDS),
        'registered_successful_coalition_seeds':coalition,
        'primary_decision':'criterion_met' if coalition>=2 else 'criterion_not_met',
        'models':models}
    (root/'summary.json').write_bytes(canonical(report))
    print('PEF15_RESULT',report['primary_decision'],coalition,flush=True)
    return report


def run(root):
    os.environ.setdefault('DNNL_MAX_CPU_ISA','AVX2')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    catalog,manifest=split_contexts()
    manifest_path=root/'source_manifest.json'
    if manifest_path.exists() and sha(manifest_path)!=p14.sha_bytes(canonical(manifest)):
        raise AssertionError('source split changed on resume')
    manifest_path.write_bytes(canonical(manifest))
    if not (root/'training_ledger.json').exists():
        train_and_seal(catalog,manifest,root)
    if not (root/'sealed_fit_ledger.json').exists():
        validated_training=json.loads((root/'training_ledger.json').read_text())
        for rec in validated_training['checkpoint_records']:
            p14.checked_model(root,rec)
        freeze_fit(catalog,manifest,root)
    if not (root/'sealed_calibration_ledger.json').exists():
        freeze_calibration(catalog,manifest,root)
    return heldout_scoring(catalog,manifest,root)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    run(args.out)


if __name__=='__main__':
    main()

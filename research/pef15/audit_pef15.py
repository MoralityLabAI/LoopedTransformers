"""PEF-15 independent arithmetic/receipt auditor.

Recomputes all 96 probe forecasts using scalar Python, all 96 calibration
intervals, and all context-level coverage decisions. Does not run neural
forward passes, refit heads or consult PEF-15 scoring functions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

SEEDS = (251,263,277)
TASKS = ("coalition","threshold")
STEPS = (0.011,0.025,0.04,0.065)
COUNTS = {"train":160,"fit":32,"calibration":16,"heldout":48}
SALT = b"pef15-splitconformal-v1-20261115:"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(x,y,tol=2e-12):
    if abs(float(x)-float(y))>tol:
        raise AssertionError(f"numeric receipt mismatch: {x} != {y}")


def read(path):
    return json.loads(Path(path).read_text())


def model_state_digest(state):
    h=hashlib.sha256()
    for name,t in sorted(state.items()):
        h.update(name.encode())
        h.update(bytes(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).tolist()))
    return h.hexdigest()


def recalc_probe_forecast(raw,step):
    # 32 complete context groups * 16 focal patterns; Python round is tie-to-even.
    f,j,q,y=[raw[k] for k in ("margin","jacobian","q","label")]
    assert len(f)==len(j)==len(q)==len(y)==512
    delta=[step*round(v/step)-v for v in q]
    cross_sum=uniform_sum=exact_sum=clean_sum=0.0
    for context in range(32):
        for mask in range(16):
            i=16*context+mask
            target=y[i]
            signed=(2*target-1)*f[i]
            if abs(j[i]) <= 1e-14:
                u=float((f[i]>=0)==bool(target))
            else:
                u=max(0.,min(1.,.5 + signed/(step*abs(j[i]))))
            matches=sum(int((f[i]+j[i]*delta[16*other+mask]>=0)==bool(target))
                        for other in range(32) if other!=context)
            cross=matches/31
            cross_sum+=cross
            uniform_sum+=u
            exact_sum+=int((f[i]+j[i]*delta[i]>=0)==bool(target))
            clean_sum+=int((f[i]>=0)==bool(target))
    return {"distribution_mixture":(31*cross_sum+8*uniform_sum)/(39*512),
            "uniform_phase":uniform_sum/512,
            "exact_fit_phase":exact_sum/512,
            "cross_context":cross_sum/512,
            "clean_fit":clean_sum/512}


def run(out):
    root=Path(out)
    manifest=read(root/'source_manifest.json')
    training=read(root/'training_ledger.json')
    fit=read(root/'sealed_fit_ledger.json')
    cal=read(root/'sealed_calibration_ledger.json')
    report=read(root/'summary.json')
    assert manifest['context_counts']==COUNTS
    assert manifest['salt']==SALT.decode()
    assert training['source_sha256']==fit['source_sha256']==cal['source_sha256']==manifest['sha256']
    assert sha(root/'training_ledger.json')==fit['training_ledger_sha256']==report['training_ledger_sha256']
    assert sha(root/'sealed_fit_ledger.json')==cal['fit_ledger_sha256']==report['fit_ledger_sha256']
    assert sha(root/'sealed_calibration_ledger.json')==report['calibration_ledger_sha256']
    assert report['all_models_frozen_before_fit'] and report['fit_frozen_before_calibration'] and report['calibration_frozen_before_heldout']
    assert len(manifest['context_assignment'])==256
    contexts=[tuple((i>>j)&1 for j in range(8)) for i in range(256)]
    contexts.sort(key=lambda c:(hashlib.sha256(SALT+bytes(c)).digest(),bytes(c)))
    assert ["".join(map(str,c)) for c in contexts]==manifest['sorted_contexts']
    flat=sum((["".join(map(str,c)) for c in contexts[sum(list(COUNTS.values())[:i]):sum(list(COUNTS.values())[:i+1])]]
            for i in range(len(COUNTS))),[])
    assert len(set(flat))==256
    chk=training['checkpoint_records']
    assert len(chk)==6 and {(r['seed'],r['task']) for r in chk}=={(s,t) for s in SEEDS for t in TASKS}
    for rec in chk:
        filename=root/rec['file']
        assert sha(filename)==rec['checkpoint_sha256']
        payload=torch.load(filename,map_location='cpu',weights_only=True)
        assert payload['steps']==450 and payload['seed']==rec['seed'] and payload['task']==rec['task']
        assert model_state_digest(payload['model'])==rec['frozen_weights_sha256']
        assert all(int(v['step'].item())==450 for v in payload['optimizer']['state'].values())
    score_rows=0
    seed_results=[]
    for row in report['models']:
        seed,task=row['seed'],row['task']
        fr=next(x for x in fit['models'] if x['seed']==seed and x['task']==task)
        cr=next(x for x in cal['models'] if x['seed']==seed and x['task']==task)
        assert row['checkpoint_sha256']==fr['checkpoint_sha256']==cr['checkpoint_sha256']
        assert len(row['conditions'])==len(cr['conditions'])==16
        group_covered=0;width=0;agg_covered=0
        for actual in row['conditions']:
            tang,ang,step=actual['tangent'],actual['angle'],actual['relative_step']
            fc=next(x for x in fr['conditions'] if (x['tangent'],x['angle'])==(tang,ang))
            forecast=next(x for x in fc['forecast'] if x['relative_step']==step)
            reference=next(x for x in cr['conditions'] if (x['tangent'],x['angle'],x['relative_step'])==(tang,ang,step))
            prediction=recalc_probe_forecast(fc['fit_features'],step*fr['pc_scale'])
            for name in prediction:close(prediction[name],forecast[name])
            rates=[c/16 for c in reference['calibration_correct_by_context']]
            assert len(rates)==16 and all(0<=c<=16 for c in reference['calibration_correct_by_context'])
            p=forecast['distribution_mixture']
            maxres=max(abs(c-p) for c in rates)
            low,high=max(0,p-maxres),min(1,p+maxres)
            ci=reference['group_interval']
            close(ci['low'],low);close(ci['high'],high);close(ci['radius'],maxres)
            assert ci['rank']==16
            held=[c/16 for c in actual['correct_by_context']]
            assert len(held)==48 and sum(actual['correct_by_context'])==actual['correct']
            covered=sum(low<=c<=high for c in held)
            assert covered==actual['conformal_covered_contexts']
            close(actual['conformal_width'],high-low)
            radius=64/48*math.sqrt((1-15/64)*math.log(20)/32)
            avg=sum(rates)/16
            b_low,b_high=max(0,avg-radius),min(1,avg+radius)
            b=reference['aggregate_serfling']
            close(b['low'],b_low);close(b['high'],b_high)
            aggregate=bool(b_low<=actual['mean_heldout_accuracy']<=b_high)
            assert aggregate==actual['aggregate_serfling_covered']
            group_covered+=covered;width+=high-low;agg_covered+=aggregate
            score_rows+=1
        coverage=group_covered/(48*16)
        mean_width=width/16
        success=coverage>=.88 and mean_width<=.85
        close(coverage,row['conformal_group_coverage'])
        close(mean_width,row['conformal_mean_interval_width'])
        assert row['registered_seed_success']==success
        assert row['group_coverage_count']==group_covered
        assert row['aggregate_serfling_covered']==agg_covered
        seed_results.append({'seed':seed,'task':task,'coverage':coverage,
            'width':mean_width,'covered_context_conditions':group_covered,
            'serfling_covered_conditions':agg_covered,'registered_success':success})
    count=sum(z['registered_success'] for z in seed_results if z['task']=='coalition')
    assert count==report['registered_successful_coalition_seeds']
    assert report['primary_decision']==('criterion_met' if count>=2 else 'criterion_not_met')
    assert score_rows==96
    result={'campaign':'PEF-15','audit':'passed',
            'fresh_checkpoint_bytes_and_optimizer_steps_verified':6,
            'fit_phase_forecasts_recomputed':96,
            'conformal_context_intervals_recomputed':96,
            'registered_coalition_seeds':count,'decision':report['primary_decision'],
            'seed_results':seed_results,
            'sha256':{file:sha(root/file) for file in ['source_manifest.json','training_ledger.json',
                'sealed_fit_ledger.json','sealed_calibration_ledger.json','summary.json']}}
    (root/'independent_audit.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    run(p.parse_args().out)

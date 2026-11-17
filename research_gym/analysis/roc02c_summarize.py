"""ROC-02C pure stdlib source-only audit of calibration-only 80-step observer fits."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

SEEDS=(2026111601,2026111603,2026111607,2026111609,2026111613,2026111619)
DEPTHS=(8,16)
RANKS=(7,8)
GAMMAS=(0.,.05,.2,1.)
STATES=tuple(range(6))

def audit(receipt_dir:Path):
    rows=[];nulls=[]
    for d in DEPTHS:
        for s in SEEDS:
            f=receipt_dir/f'roc02c_R{d}_S{s}.json'
            payload=json.loads(f.read_text())
            if payload.get('source_split','').find('sealed withheld')<0 or payload['model_seed']!=s or payload['depth']!=d:
                raise ValueError('unregistered source provenance '+f.name)
            if len(payload['rows'])!=len(GAMMAS)*len(RANKS) or len(payload['null_medians'])!=len(RANKS)*len(STATES):
                raise ValueError('incomplete rank/grid receipt '+f.name)
            sha=payload['checkpoint_sha256']
            if len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha):
                raise ValueError('bad frozen model sha')
            nulls.extend(payload['null_medians'])
            for k in RANKS:
                arms=[v for v in payload['rows'] if v['rank']==k]
                if sorted(v['gamma'] for v in arms)!=list(GAMMAS):raise ValueError('gamma grid drift')
                selected=[r for r in arms if r['selected']]
                if len(selected)!=1:raise ValueError('missing winner selection')
                task=next(r for r in arms if r['gamma']==0)
                for r in arms:
                    if r['model_checkpoint_sha256']!=sha or not r['calibration_only']:
                        raise ValueError('frozen checkpoint mismatch / unsealed result')
                    for met in ('selection_calibration_MSE','selection_B_null_squared','selection_H_future_logit_capture','selection_calibration_accuracy'):
                        if not math.isfinite(r[met]):raise ValueError('invalid '+met)
                best=selected[0]
                if best['gamma']!=0 and best['selection_calibration_MSE']>1.10*task['selection_calibration_MSE']+1e-12:
                    raise AssertionError('task-only anchored 1.10 gate violated')
                rows.extend(arms)
    keyset={(n['seed'],n['depth'],n['rank'],n['state_index']) for n in nulls}
    if len(rows)!=96 or len(nulls)!=144 or len(keyset)!=144:
        raise ValueError('bad cohort cardinality')
    if any(n['median_b']<=0 or not n['null_128_distinct'] for n in nulls):
        raise ValueError('bad null medians')
    selected=[r for r in rows if r['selected']]
    bydepth=[]
    for d in DEPTHS:
        for k in RANKS:
            x=[r for r in selected if r['depth']==d and r['rank']==k]
            t={(r['seed'],r['depth'],r['rank']):r for r in rows if r['gamma']==0}
            ts=[t[(r['seed'],d,k)] for r in x]
            m=lambda group,key:mean(r[key] for r in group)
            bydepth.append({
                'depth':d,'rank':k,'model_count':len(x),
                'selected_gamma_by_seed':{str(r['seed']):r['gamma'] for r in x},
                'mean_task_only_MSE':m(ts,'selection_calibration_MSE'),
                'mean_joint_MSE':m(x,'selection_calibration_MSE'),
                'mean_task_only_normalized_B_squared':m(ts,'selection_B_null_squared'),
                'mean_joint_normalized_B_squared':m(x,'selection_B_null_squared'),
                'mean_task_only_future_logit_capture':m(ts,'selection_H_future_logit_capture'),
                'mean_joint_future_logit_capture':m(x,'selection_H_future_logit_capture'),
                'mean_task_only_stress_B_squared':m(x,'task_only_stress_B_null_squared'),
                'mean_joint_stress_B_squared':m(x,'stress_B_null_squared'),
                'joint_MSE_better_count':sum(r['selection_calibration_MSE']<t[(r['seed'],d,k)]['selection_calibration_MSE'] for r in x),
                'joint_J_better_count':sum(r['selection_J_lambda_one']<t[(r['seed'],d,k)]['selection_J_lambda_one'] for r in x)
            })
    return rows,nulls,{'study':'ROC-02C-calibration-only-equal-budget-v1',
        'description':'Train-only ridge readout + 80-step task-only or null-normalized computed-change B regularizer',
        'prereg':'configs/roc02c_calibration_equal_budget_v1.json',
        'n_original_checkpoints':12,'n_observer_model_rank_cells':24,'n_candidate_arms':96,
        'n_rank_state_null_medians':144,'seeds':list(SEEDS),'depths':list(DEPTHS),
        'rank_grid':list(RANKS),'gamma_grid':list(GAMMAS),'calibration_only':True,
        'sealed_test_evaluated':False,'equal_budget_updates':80,
        'fit_null_states':[0,1],'gamma_selection_null_states':[2,3],
        'stress_null_states':[4,5],
        'task_MSE_anchor':'80-step task-only calibration MSE, eligibility ratio 1.10',
        'warning':'Gamma selected on same calibration source-family labels used to report MSE; independent task generalization untested. Output H is original weak future-logit readout, not true rule recall.',
        'by_depth_rank':bydepth}

def emit_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),extrasaction='ignore')
        w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt-dir',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    args=p.parse_args()
    rows,nulls,summary=audit(args.receipt_dir)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    emit_csv(args.out_dir/'roc02c_all_arms.csv',rows)
    emit_csv(args.out_dir/'roc02c_null_medians.csv',nulls)
    (args.out_dir/'roc02c_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()

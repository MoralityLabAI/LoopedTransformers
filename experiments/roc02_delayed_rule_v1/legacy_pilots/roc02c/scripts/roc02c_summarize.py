"""ROC-02C public calibration-only audit, paired seed summaries and SHA checks.

Does not load ROC-02 sealed heldout examples or labels.
"""
from __future__ import annotations
import csv
import hashlib
import json
import statistics as stats
from collections import Counter
from pathlib import Path

SEEDS=(2026111601,2026111603,2026111607,2026111609,2026111613,2026111619)
DEPTHS=(8,16)
GAMMAS=(0.,.05,.2,1.,2.)
STUDY='ROC-02C-rank8-equal-budget-null-normalized-calibration-v1'

def sha(p:Path):return hashlib.sha256(p.read_bytes()).hexdigest()

def gather(root:Path, *, check_adapters:bool=True):
    rows=[];out=[];provenance=[]
    for depth in DEPTHS:
        for seed in SEEDS:
            path=root/f'roc02c_R{depth}_S{seed}.json'
            v=json.loads(path.read_text())
            assert v['study']==STUDY and v['seed']==seed and v['depth']==depth
            assert v['calibration_only'] and 'no heldout' in v['scope']
            assert v['updates_per_fitted_arm']==80 and v['gamma_grid']==list(GAMMAS)
            assert len(v['null_medians'])==2 and all(x>0 for x in v['null_medians'])
            assert set(v['arms'])=={'baseline',*(str(g) for g in GAMMAS)}
            path_adapter=root/'adapters'/v['observer_adapter_filename']
            if check_adapters:
                assert path_adapter.is_file()
                assert sha(path_adapter)==v['observer_adapter_sha256']
            ref=v['arms']['0.0']
            eligible=[a for a in v['arms'].values() if a['gamma']!='baseline' and a['calibration_mse']<=1.1*ref['calibration_mse']]
            assert eligible
            chosen=min(eligible,key=lambda x:(x['calibration_J_lambda1'],x['gamma']))
            if chosen['calibration_J_lambda1']>=ref['calibration_J_lambda1']:
                chosen=ref
            assert abs(chosen['gamma']-v['selected_gamma'])<1e-12
            assert v['selected_task_budget_passes']
            assert v['selected_is_positive']==(chosen['gamma']>0)
            assert v['selected_beats_task_only_J']==(chosen['calibration_J_lambda1']<ref['calibration_J_lambda1'])
            for key,a in v['arms'].items():
                assert a['orthogonality_error']<1e-9
                assert abs(a['null_normalized_penalty']-sum(b*b for b in a['b_relative'])/2)<1e-11
                assert abs(a['calibration_J_lambda1']-a['calibration_mse']-a['null_normalized_penalty'])<1e-10
                assert len(a['states'])==2
                for st in a['states']:
                    assert len(st['holm_q_six'])==len(st['rank_p_six'])==6
                rows.append({'depth':depth,'seed':seed,'arm':key,'gamma':a['gamma'],
                             'train_mse':a['train_mse'],
                             'calibration_mse':a['calibration_mse'],
                             'calibration_accuracy':a['calibration_accuracy'],
                             'B_to_null':a['mean_B_to_rank_null'],
                             'B2_null_penalty':a['null_normalized_penalty'],
                             'source_logit_H_capture':a['source_logit_Gram_capture'],
                             'calibration_J_lambda1':a['calibration_J_lambda1'],
                             'orthogonality_error':a['orthogonality_error'],
                             'last_optimizer_step':(a['trace'][-1]['step'] if a['trace'] else 0),
                             'is_selected':key==str(v['selected_gamma']),
                             'checkpoint_sha256':v['source_checkpoint_sha256'],
                             'adapter_sha256':v['observer_adapter_sha256']})
            out.append({'depth':depth,'seed':seed,'selected_gamma':v['selected_gamma'],
                        'task_only_mse':ref['calibration_mse'],
                        'joint_mse':chosen['calibration_mse'],
                        'task_only_accuracy':ref['calibration_accuracy'],
                        'joint_accuracy':chosen['calibration_accuracy'],
                        'task_only_B_to_null':ref['mean_B_to_rank_null'],
                        'joint_B_to_null':chosen['mean_B_to_rank_null'],
                        'task_only_H_capture':ref['source_logit_Gram_capture'],
                        'joint_H_capture':chosen['source_logit_Gram_capture'],
                        'task_only_J':ref['calibration_J_lambda1'],
                        'joint_J':chosen['calibration_J_lambda1'],
                        'joint_win':v['selected_beats_task_only_J'],
                        'joint_task_budget':v['selected_task_budget_passes'],
                        'checkpoint_sha256':v['source_checkpoint_sha256'],
                        'adapter_sha256':v['observer_adapter_sha256']})
            provenance.append({'seed':seed,'depth':depth,
                               'receipt_file':path.name,'receipt_sha256':sha(path),
                               'adapter_file':path_adapter.name,'adapter_sha256':v['observer_adapter_sha256'],
                               'source_checkpoint_sha256':v['source_checkpoint_sha256'],
                               'source_roc02b_receipt_sha256':v['source_roc02b_receipt_sha256']})
    return rows,out,provenance

def summaries(rows,paired,provenance):
    summary={'study':STUDY,'calibration_only':True,'sealed_test_opened':False,
        'source_model_seeds':list(SEEDS),'depths':list(DEPTHS),'rank':8,
        'trained_model_steps':240,'trained_observer_steps_per_arm':80,
        'penalty_grid':list(GAMMAS),'n_model_cells':len(paired),
        'n_candidate_rows':len(rows),'adapter_receipts':provenance,
        'selection':'same calibration families; optimistic development outcome',
        'by_depth':{}}
    for depth in DEPTHS:
        sub=[r for r in paired if r['depth']==depth]
        mean=lambda key:stats.mean(x[key] for x in sub)
        summary['by_depth'][str(depth)]={
            'selected_gamma_counts':dict(sorted(Counter(str(x['selected_gamma']) for x in sub).items())),
            'positive_gamma_selected':sum(x['selected_gamma']>0 for x in sub),
            'joint_objective_wins':sum(x['joint_win'] for x in sub),
            'strict_task_budget_passes':sum(x['joint_task_budget'] for x in sub),
            'MSE_improves':sum(x['joint_mse']<x['task_only_mse'] for x in sub),
            'accuracy_improves':sum(x['joint_accuracy']>x['task_only_accuracy'] for x in sub),
            'H_capture_improves':sum(x['joint_H_capture']>x['task_only_H_capture'] for x in sub),
            'means':{k:mean(k) for k in ('task_only_mse','joint_mse','task_only_accuracy',
                    'joint_accuracy','task_only_B_to_null','joint_B_to_null','task_only_H_capture',
                    'joint_H_capture','task_only_J','joint_J')}
        }
    summary['registered_calibration_gate_passed']=(summary['by_depth']['16']['positive_gamma_selected']>=5
            and summary['by_depth']['16']['joint_objective_wins']>=5
            and summary['by_depth']['16']['strict_task_budget_passes']>=5)
    return summary

def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--receipt-dir',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    a=p.parse_args()
    cand,paired,prov=gather(a.receipt_dir)
    summary=summaries(cand,paired,prov)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    write_csv(a.output_dir/'roc02c_candidates.csv',cand)
    write_csv(a.output_dir/'roc02c_paired_by_seed.csv',paired)
    (a.output_dir/'roc02c_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k in ('n_model_cells','n_candidate_rows','registered_calibration_gate_passed','by_depth')},indent=2))

if __name__=='__main__':main()

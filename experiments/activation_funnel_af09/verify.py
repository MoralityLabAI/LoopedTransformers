"""Independent AF-09 audit: immutable source, Horn truth, code patch semantics.

Recompute every reported metric from per-case raw logits, not a cached summary.
Additionally replay sampled original Torch checkpoints after rehashing files;
changing predictions and their outer SHA is therefore detectable.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from collections import defaultdict
import numpy as np

import torch

from experiments.activation_funnel_af05 import gym
from experiments.activation_funnel_af07.core import paired_traces
from . import experiment as ex

TOL=2e-5


def read_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def close(a,b,tol=TOL):
    if isinstance(a,(tuple,list)) and isinstance(b,(tuple,list)):
        return len(a)==len(b) and all(close(x,y,tol) for x,y in zip(a,b))
    return isinstance(a,(int,float)) and isinstance(b,(int,float)) and math.isfinite(a) and math.isfinite(b) and abs(a-b)<=tol



def verify_margin_records(margins,stage,models,head,ex,smoke,tol):
    """Independent NumPy reconstruction of every affine code-margin bound."""
    if len(margins)!=len(stage) or head['margin_rows']!=len(stage):
        raise ValueError('AF-09 margin roster is incomplete')
    stages={(r['arm'],r['split'],r['recipient_id'],r['selected'],r['post_loops']):r
            for r in stage}
    seen=set()
    for r in margins:
        key=(r['arm'],r['split'],r['recipient_id'],r['selected'],r['post_loops'])
        if key in seen or key not in stages:raise ValueError('AF-09 unexpected margin certificate')
        seen.add(key)
        parent=stages[key]
        if any(r[k]!=parent[k] for k in ('donor_id','pre_loops','oracle','recipient_truth')):
            raise ValueError('AF-09 changed certificate donor identity or Horn truth')
        if r['truth_protected']!=r['recipient_truth'][1-r['selected']]:
            raise ValueError('AF-09 protected-Q class is not the recipient truth')
        model=models[r['arm']]
        decoder=model.codec.atom.decoder.weight.detach().numpy().astype('float64')
        head_w=model.subgoal_head.weight.detach().numpy().astype('float64')
        y=r['truth_protected'];v=head_w[y]-head_w[1-y]
        kval=float(np.linalg.norm(v @ decoder))
        if abs(kval-r['exact_sensitivity'])>tol:
            raise ValueError('AF-09 affine readout sensitivity changed')
        for k in range(r['post_loops']):
            z=np.asarray(r['clean_code'][k],dtype=np.float64)
            zprime=np.asarray(r['patched_code'][k],dtype=np.float64)
            if z.shape!=(ex.WIDTH,) or zprime.shape!=(ex.WIDTH,):
                raise ValueError('AF-09 incomplete actual carried code')
            drift=float(np.linalg.norm(zprime-z))
            if abs(drift-r['code_distance'][k])>tol or \
               abs(drift-parent['protected_code_l2_by_step'][k])>tol:
                raise ValueError('AF-09 carried-code drift mismatch')
            l=parent['clean_q_steps'][k][1-r['selected']]
            p=parent['patched_q_steps'][k][1-r['selected']]
            clean=float(l[y]-l[1-y]);patch=float(p[y]-p[1-y])
            bound=clean-kval*drift
            cert=bound>tol
            correct=patch>0
            if abs(clean-r['clean_true_margin'][k])>tol or \
               abs(patch-r['patched_true_margin'][k])>tol or \
               abs(bound-r['certified_lower_bound'][k])>tol:
                raise ValueError('AF-09 true-class signed margin mismatch')
            if abs(patch-clean)>kval*drift+tol or (cert and not correct):
                raise ValueError('AF-09 false mathematical certificate')
            if r['certificate_positive'][k]!=cert or r['patched_correct'][k]!=correct or r['clean_correct'][k]!=(clean>0):
                raise ValueError('AF-09 certificate predicates inconsistent')
    if set(stages)!=seen:raise ValueError('AF-09 certificate coverage mismatch')
    for arm in ex.ARMS:
        for split in (('dev',) if smoke else ('held_composition','held_layout','held_depth4','held_depth5')):
            for steps,summary in head['certificates'][arm][split].items():
                subset=[r for r in margins if r['arm']==arm and r['split']==split and str(r['post_loops'])==steps]
                if summary['pairs']!=len(subset) or summary['false_certificates']!=0:
                    raise ValueError('AF-09 invalid margin summary roster')
                for k in range(int(steps)):
                    rate=sum(r['certificate_positive'][k] for r in subset)/len(subset)
                    if abs(rate-summary['positive_certificate_rate_by_step'][k])>1e-6:
                        raise ValueError('AF-09 signed margin certificate rate mismatch')
    return True


def verify_one(training,evaluation,seed,smoke=False,replay_cases=2):
    training,evaluation=Path(training),Path(evaluation)
    rec,models=ex.load_models(training,seed,smoke)
    head=json.loads((evaluation/'evaluation.json').read_text())
    if (evaluation/'evaluation.sha256').read_text().strip()!=ex.sha_file(evaluation/'evaluation.json'):
        raise ValueError('AF-09 evaluation header mutated')
    if (head['experiment'],head['seed'],head['width'],head['smoke'])!=('AF-09',seed,ex.WIDTH,smoke):
        raise ValueError('AF-09 evaluation run mismatch')
    if head['train_receipt_sha256']!=ex.sha_file(training/'train_receipt.json') or head['source_hashes']!=ex.source_receipt():
        raise ValueError('AF-09 changed evidence source')
    for name,h in head['files'].items():
        if ex.sha_file(evaluation/name)!=h:raise ValueError('AF-09 corrupted scored rows: '+name)
    clean=read_jsonl(evaluation/'clean_predictions.jsonl')
    patch=read_jsonl(evaluation/'patch_predictions.jsonl')
    stage=read_jsonl(evaluation/'stage_predictions.jsonl')
    margins=read_jsonl(evaluation/'margin_predictions.jsonl')
    splits=('dev',) if smoke else ('held_composition','held_layout','held_depth4','held_depth5')
    datasets={s:ex.Dataset(s) for s in splits}
    cases={s:datasets[s].cases for s in splits}
    case_indices={s:{c.id:i for i,c in enumerate(cases[s])} for s in splits}
    valid={(arm,s,c.id):c for arm in ex.ARMS for s in splits for c in cases[s]}
    ncases=sum(map(len,cases.values()))
    if (len(clean),len(patch),head['clean_rows'],head['patch_rows'])!=(3*ncases,6*ncases,3*ncases,6*ncases):
        raise ValueError('AF-09 clean/patch roster count mismatch')
    cindex={};pindex={};sindex={}
    for r in clean:
        k=(r['arm'],r['split'],r['id'])
        if k in cindex or k not in valid or (r['seed'],r['width'])!=(seed,ex.WIDTH):
            raise ValueError('AF-09 duplicate/unknown clean prediction')
        truth=gym.oracle(valid[k])[0]
        if r['target']!=list(truth) or r['pre_loops']!=ex.EVAL_LOOPS[r['split']]:
            raise ValueError('AF-09 original Horn goal changed')
        if not (len(r['logits'])==3 and all(len(x)==2 for x in r['logits'])):
            raise ValueError('AF-09 incomplete original model logits')
        cindex[k]=r
    for arm in ex.ARMS:
        for split in splits:
            logits=torch.tensor([cindex[(arm,split,c.id)]['logits'] for c in cases[split]])
            labels=torch.tensor([gym.oracle(c)[0] for c in cases[split]],dtype=torch.long)
            scores=ex.metrics(logits,labels)
            for key in ('goal_accuracy','subgoal_exact','all_exact','false_goal_attempts'):
                if abs(scores[key]-head['scores'][arm][split][key])>1e-6:
                    raise ValueError('AF-09 clean score differs from logits: '+key)
    for r in patch:
        k=(r['arm'],r['split'],r['recipient_id'],r['selected'])
        if k in pindex or (r['seed'],r['width'])!=(seed,ex.WIDTH) or r['selected'] not in (0,1):
            raise ValueError('AF-09 duplicate/foreign patch identity')
        case=valid.get(k[:3]);split=r['split']
        if case is None:raise ValueError('AF-09 unknown recipient')
        i=case_indices[split][case.id]
        donor_i,truth=datasets[split].donors[i,r['selected']]
        if r['donor_id']!=cases[split][donor_i].id or r['oracle']!=list(truth):
            raise ValueError('AF-09 changed exact matched donor or Horn composition')
        if r['recipient_truth']!=list(gym.oracle(case)[0]) or r['pre_loops']!=ex.EVAL_LOOPS[split]:
            raise ValueError('AF-09 counterfactual parent binding changed')
        for fld in ('addressed_logits','other_logits','whole_logits'):
            if not(len(r[fld])==3 and all(len(x)==2 for x in r[fld])):
                raise ValueError('AF-09 incomplete counterfactual logits')
        if r['whole_replay_error_max']>TOL:raise ValueError('whole-state/direct-donor tolerance failed')
        pindex[k]=r
    for arm in ex.ARMS:
        for split in splits:
            ps=[pindex[(arm,split,c.id,j)] for c in cases[split] for j in (0,1)]
            changed=[p['oracle'][2]!=p['recipient_truth'][2] for p in ps]
            if not any(changed):raise ValueError('AF-09 missing informative goal flips')
            pred=lambda a: [int(x[1]>x[0]) for x in a]
            q=[pred(p['addressed_logits']) for p in ps]
            other=[pred(p['other_logits']) for p in ps]
            metrics={
              'informative_goal_accuracy':sum(q[i][2]==ps[i]['oracle'][2] for i,v in enumerate(changed) if v)/sum(changed),
              'untouched_subgoal_accuracy':sum(q[i][1-p['selected']]==p['oracle'][1-p['selected']] for i,p in enumerate(ps))/len(ps),
              'other_patch_goal_retention':sum(other[i][2]==p['recipient_truth'][2] for i,p in enumerate(ps))/len(ps),
              'addressed_subgoal_accuracy':sum(q[i][p['selected']]==p['oracle'][p['selected']] for i,p in enumerate(ps))/len(ps),
            }
            reported=head['patches'][arm][split]
            for name,v in metrics.items():
                if abs(v-reported[name])>1e-6:raise ValueError('AF-09 patched score differs from Horn oracle: '+name)
    for r in stage:
        k=(r['arm'],r['split'],r['recipient_id'],r['selected'],r['post_loops'])
        if k in sindex or k[:4] not in pindex:
            raise ValueError('AF-09 unexpected or duplicate staged patch')
        p=pindex[k[:4]]
        steps=r['post_loops']
        expected=(3,5) if r['split'] in ('held_depth4','held_depth5') and not smoke else (3,)
        if steps not in expected or (r['seed'],r['width'])!=(seed,ex.WIDTH):
            raise ValueError('AF-09 unregistered recurrence duration')
        if r['donor_id']!=p['donor_id'] or r['oracle']!=p['oracle'] or r['recipient_truth']!=p['recipient_truth'] or r['pre_loops']!=p['pre_loops']:
            raise ValueError('AF-09 recurrence donor/oracle mismatch')
        for fld in ('clean_q_steps','patched_q_steps','patched_goal_steps',
                    'protected_q_correct_by_step','patched_q_correct_by_step','protected_code_l2_by_step'):
            if len(r[fld])!=steps:raise ValueError('AF-09 missing recurrence step: '+fld)
        untouched=1-r['selected']
        for t in range(steps):
            logit=r['patched_q_steps'][t]
            if len(logit)!=2 or any(len(q)!=2 for q in logit):raise ValueError('AF-09 bad Q logit shape')
            corr=[int(x[1]>x[0])==r['oracle'][j] for j,x in enumerate(logit)]
            if r['patched_q_correct_by_step'][t]!=corr or r['protected_q_correct_by_step'][t]!=corr[untouched]:
                raise ValueError('AF-09 staged semantic metric tampered')
            distance=r['protected_code_l2_by_step'][t]
            if not isinstance(distance,(int,float)) or not math.isfinite(distance) or distance<0:
                raise ValueError('AF-09 invalid code displacement')
        if steps==3:
            if not close(r['patched_q_steps'][-1]+[r['patched_goal_steps'][-1]],p['addressed_logits']):
                raise ValueError('AF-09 stage-3 patch differs from original scorer')
            if not close(r['clean_q_steps'][-1],cindex[k[:3]]['logits'][:2]):
                raise ValueError('AF-09 stage-3 clean differs from original scorer')
        sindex[k]=r
    expected_stage=sum(3*len(cases[split])*2*(2 if split in ('held_depth4','held_depth5') and not smoke else 1)
                       for split in splits)
    if len(sindex)!=expected_stage or head['stage_rows']!=expected_stage:
        raise ValueError('AF-09 recurrence roster incomplete')
    for arm in ex.ARMS:
        for split in splits:
            for steps,summary in head['trajectories'][arm][split].items():
                subset=[r for r in stage if r['arm']==arm and r['split']==split and str(r['post_loops'])==steps]
                if summary['pairs']!=len(subset):raise ValueError('AF-09 stage summary count changed')
                for i,v in enumerate(summary['protected_q_step_accuracy']):
                    check=sum(r['protected_q_correct_by_step'][i] for r in subset)/len(subset)
                    if abs(v-check)>1e-6:raise ValueError('AF-09 staged accuracy score mismatch')
    # Mutating both row hashes and aggregate values is insufficient: replay the
    # saved checkpoint on independently selected held-out recipient/donor rows.
    for arm,model in models.items():
        model.eval()
        for split in splits:
            ds=datasets[split]
            for i in sorted({0,len(ds.cases)-1})[:replay_cases]:
                receiver=ds.get([i,i])
                donor,hybrid=ds.pair([i,i],[0,1])
                selected=torch.tensor([0,1],dtype=torch.long)
                addressed=torch.ones(2,dtype=torch.bool)
                with torch.no_grad():
                    clean_trace,patched_trace=paired_traces(model,receiver,donor,selected,addressed,ex.EVAL_LOOPS[split])
                for j in (0,1):
                    key=(arm,split,ds.cases[i].id,j)
                    logits=patched_trace.q[j].tolist()+[patched_trace.goal[j].tolist()]
                    if not close(logits,pindex[key]['addressed_logits']):
                        raise ValueError('AF-09 independent checkpoint patch replay mismatch')
                    key3=(*key,3)
                    if not close(patched_trace.q_steps[j].tolist(),sindex[key3]['patched_q_steps']):
                        raise ValueError('AF-09 independent staged logits replay mismatch')
    verify_margin_records(margins,stage,models,head,ex,smoke,TOL)
    return {'experiment':'AF-09','seed':seed,'verified':True,'clean_rows':len(clean),
            'patch_rows':len(patch),'stage_rows':len(stage),
            'checkpoint_replay_cases_per_arm_split':replay_cases,'margin_rows':len(margins),
            'evaluation_sha256':ex.sha_file(evaluation/'evaluation.json'),
            'training_sha256':ex.sha_file(training/'train_receipt.json'),
            'source':ex.source_receipt()}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--training',type=Path,required=True)
    p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--seed',type=int,choices=ex.SEEDS,required=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    result=verify_one(a.training,a.evaluation,a.seed,a.smoke)
    ex.json_write(a.out,result)
    print(json.dumps({k:result[k] for k in ('seed','verified','clean_rows','patch_rows','stage_rows')},sort_keys=True),flush=True)

if __name__=='__main__':main()

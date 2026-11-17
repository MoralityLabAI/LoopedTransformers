"""ROC-02C: calibrated, equal-budget rank-eight observer pilot.

Frozen registration: configs/roc02c_equal_budget_rank8_v1.json.
Reads train/calibration source families and frozen ROC-02 checkpoints only.
Never accesses heldout ROC-02 source-family test inputs or labels.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from research_gym.analysis.roc02b_calibration_card import (
        SEEDS, DEPTHS, RANKS, load_and_verify, stage_model, observer_bases,
        tangent, null_calibrate, jacobian_metrics, causal_interventions, sha,
        rows, tensors, suffix,
    )
except ImportError:
    from roc02b_calibration_card import (
        SEEDS, DEPTHS, RANKS, load_and_verify, stage_model, observer_bases,
        tangent, null_calibrate, jacobian_metrics, causal_interventions, sha,
        rows, tensors, suffix,
    )

GAMMAS = (0., 0.05, 0.2, 1., 2.)
RANK = 8
UPDATES = 80
LR = 0.02
RADIUS = 0.2
SHADOW_INDICES = (0, 1)
STUDY = 'ROC-02C-rank8-equal-budget-null-normalized-calibration-v1'


def load_roc02b_receipt(path: Path, seed: int, depth: int, checkpoint_sha: str,
                        original_manifest: Path) -> dict:
    index = json.loads(original_manifest.read_text())
    expected = next(row for row in index['original_checkpoint_receipts']
                    if row['seed'] == seed and row['depth'] == depth)
    if sha(path) != expected['receipt_sha256'] or expected['checkpoint_sha256'] != checkpoint_sha:
        raise RuntimeError('ROC-02B source receipt / checkpoint hash mismatch')
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        data = json.load(f)
    if data['checkpoint_sha256'] != checkpoint_sha or (data['seed'], data['depth']) != (seed, depth):
        raise RuntimeError('ROC-02B receipt identity mismatch')
    return data


def null_metric_sets(receipt: dict):
    out = {}
    for idx in SHADOW_INDICES:
        group = [r for r in receipt['nulls'] if r['rank'] == RANK and r['state_index'] == idx]
        if len(group) != 128 or len({x['subspace_sha256'] for x in group}) != 128:
            raise RuntimeError(f'Invalid ROC02B rank-8 null group {idx}')
        med = sorted(x['b'] for x in group)
        median = (med[63] + med[64])/2
        if not median > 0 or not math.isfinite(median):
            raise ValueError('Undefined rank-relative null median')
        out[idx] = {'nulls': group, 'median_b': median}
    return out


def ridge_for_basis(x_train, y_train, x_cal, y_cal, U, label_prior=None):
    k = U.shape[1]
    tr_x = x_train @ U
    cal_x = x_cal @ U
    ZTZ = tr_x.T@tr_x + torch.eye(k, dtype=torch.float64, device=x_train.device)
    head = torch.linalg.solve(ZTZ, tr_x.T@y_train)
    tr_logits = tr_x @ head
    cal_logits = cal_x @ head
    E_train = (tr_logits - y_train).square().mean()
    E_cal = (cal_logits - y_cal).square().mean()
    # Ridge losses use centered labels; class decisions restore the train prior.
    if label_prior is None:
        label_prior = torch.zeros(y_train.shape[1],dtype=torch.float64,device=y_train.device)
    acc_cal = ((cal_logits+label_prior).argmax(-1) == y_cal.argmax(-1)).double().mean()
    return head, E_train, E_cal, acc_cal


def b_relative_squared(U, D, d_norms, null_medians):
    # D = true suffix Jacobian minus identity. Each state has its own 128-null median.
    Utd = torch.matmul(U.T.unsqueeze(0), D)
    Utdu = torch.matmul(Utd, U)
    bsq = (Utd.square().sum(dim=(1,2)) - Utdu.square().sum(dim=(1,2))).clamp_min(0.)
    b_normalized = torch.sqrt(bsq)/d_norms
    b_relative = b_normalized / null_medians
    return b_relative.square().mean(), b_relative


def fit_one(U0, x_train, y_train, x_dev, y_dev, D, d_norms, medians,
            gamma: float, *, updates: int = UPDATES, label_prior=None) -> dict:
    if gamma not in GAMMAS or updates != UPDATES:
        raise ValueError('Not in registered ROC02C arm or update budget')
    raw = torch.nn.Parameter(U0.clone().double())
    opt = torch.optim.Adam([raw], lr=LR)
    states = []
    for step in range(updates):
        opt.zero_grad(set_to_none=True)
        U = torch.linalg.qr(raw, mode='reduced')[0]
        _, E_train, _, _ = ridge_for_basis(x_train,y_train,x_dev,y_dev,U)
        b_penalty, _ = b_relative_squared(U,D,d_norms,medians)
        loss = E_train + gamma*b_penalty
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite observer training objective')
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw],10.)
        opt.step()
        if step in (0,39,79):
            states.append({'step':step+1,'loss':float(loss.detach())})
    with torch.no_grad():
        U = torch.linalg.qr(raw, mode='reduced')[0].detach()
    return evaluate(U,x_train,y_train,x_dev,y_dev,D,d_norms,medians,gamma,states,label_prior=label_prior)


def evaluate(U, x_train, y_train, x_dev, y_dev, D, d_norms, medians,
             gamma, training_trace=None, label_prior=None):
    with torch.no_grad():
        head, Et, Ecal, acc = ridge_for_basis(x_train,y_train,x_dev,y_dev,U,label_prior=label_prior)
        pen, relative = b_relative_squared(U,D,d_norms,medians)
        orth_err = float(torch.linalg.norm(U.T@U-torch.eye(RANK,dtype=torch.float64)))
        return {
            'gamma':gamma,
            'U':U.detach().clone(),
            'head':head.detach().clone(),
            'train_mse':float(Et),'calibration_mse':float(Ecal),
            'calibration_accuracy':float(acc),
            'b_relative':relative.tolist(),
            'null_normalized_penalty':float(pen),
            'calibration_J_lambda1':float(Ecal+pen),
            'orthogonality_error':orth_err,
            'trace':training_trace or [],
        }


def run(seed:int, depth:int, source_dir:Path, checkpoint_dir:Path,
        original_roc02b_receipts:Path, original_manifest:Path,
        out_dir:Path):
    if seed not in SEEDS or depth not in DEPTHS:
        raise ValueError('Outside registered seed/depth cohort')
    torch.set_num_threads(1)
    ck, cksha, model=load_and_verify(seed,depth,source_dir,checkpoint_dir)
    receipt=load_roc02b_receipt(
        original_roc02b_receipts/f'roc02b_R{depth}_S{seed}.json.gz',
        seed,depth,cksha,original_manifest)
    nulls=null_metric_sets(receipt)
    train=tensors(rows(Path(source_dir)/'data'/'train_inputs.jsonl'))
    dev=tensors(rows(Path(source_dir)/'data'/'calibration_inputs.jsonl'))
    with torch.no_grad():
        ht,_=stage_model(model,train[0],train[1])
        hd,_=stage_model(model,dev[0],dev[1])
        U0=observer_bases(ht,train[2])[('supervised',RANK)].detach().double()
        Dt = torch.stack([tangent(model,hd[idx])-torch.eye(128,dtype=torch.float64)
                          for idx in SHADOW_INDICES])
        Tt=Dt+torch.eye(128,dtype=torch.float64)
        d_norm=torch.linalg.norm(Dt,dim=(1,2))
        if torch.any(d_norm<=1e-12):raise ValueError('Undefined D norm')
        medians=torch.tensor([nulls[idx]['median_b'] for idx in SHADOW_INDICES],dtype=torch.float64)
        center=ht.double().mean(0)
        xt=ht.double()-center
        xd=hd.double()-center
        yt=F.one_hot(train[2],8).double()
        yd=F.one_hot(dev[2],8).double()
        ybar=yt.mean(0)
        yt=yt-ybar
        yd=yd-ybar
    baseline=evaluate(U0,xt,yt,xd,yd,Dt,d_norm,medians,'baseline',label_prior=ybar)
    fitted={str(gamma):fit_one(U0,xt,yt,xd,yd,Dt,d_norm,medians,gamma,label_prior=ybar) for gamma in GAMMAS}
    task=fitted[str(0.)]
    eligible=[c for c in fitted.values() if c['calibration_mse']<=1.10*task['calibration_mse']]
    if not eligible or task not in eligible:
        raise AssertionError('Missing task-only calibration eligibility')
    chosen=min(eligible,key=lambda v:(v['calibration_J_lambda1'],v['gamma']))
    if chosen['calibration_J_lambda1']>=task['calibration_J_lambda1']:
        chosen=task
    # Fixed train-only source readout defines output capture for both tuned bases.
    R=model.readout.weight.detach().double()
    def add_capture(c):
        with torch.no_grad():
            caps=[]
            for i in range(len(SHADOW_INDICES)):
                actual=jacobian_metrics(Dt[i],c['U'],R@Tt[i])
                rawnull=null_calibrate(actual,nulls[SHADOW_INDICES[i]]['nulls'])
                caps.append({
                    'state_index':SHADOW_INDICES[i],
                    'B_per_sqrtk':actual['B_per_sqrtk'],
                    'B_to_null':rawnull['ratios']['b'],
                    'H_capture':actual['s'],
                    'tags':rawnull['tags'],
                    'rank_p_six':rawnull['p'],
                    'holm_q_six':rawnull['q'],
                })
            c['source_logit_Gram_capture']=sum(x['H_capture'] for x in caps)/len(caps)
            c['mean_B_to_rank_null']=sum(x['B_to_null'] for x in caps)/len(caps)
            c['states']=caps
        return c
    for c in [baseline,*fitted.values()]:add_capture(c)
    # Physical perturbation of the actual model, on calibration states only.
    for name,c in [('task_only',task),('selected_joint',chosen)]:
        responses=[]
        for i in SHADOW_INDICES:
            responses.append(causal_interventions(
                model,hd[i],c['U'],cksha,i,RANK,Tt[i],int(dev[2][i])))
        c[f'calibration_physical_{name}']=responses
    out_dir.mkdir(parents=True,exist_ok=True)
    adapters=out_dir/'adapters';adapters.mkdir(exist_ok=True)
    path=adapters/f'roc02c_adapter_R{depth}_S{seed}.pt'
    # Tensor payloads are fixed before any heldout testing.
    torch.save({
        'study':STUDY,'seed':seed,'depth':depth,'source_checkpoint_sha256':cksha,
        'task_only_U':task['U'],'task_only_head':task['head'],
        'joint_U':chosen['U'],'joint_head':chosen['head'],
        'selected_gamma':chosen['gamma'],'U0':U0,'mean_h':center,'mean_y':ybar,
        'normalization_medians':medians,
        'source_data_train_sha256':sha(source_dir/'data'/'train_inputs.jsonl'),
        'source_data_calibration_sha256':sha(source_dir/'data'/'calibration_inputs.jsonl'),
        },path)
    checksum=sha(path)
    def strip(v):
        return {k:val for k,val in v.items() if k not in ('U','head')}
    arms={'baseline':strip(baseline),**{k:strip(v) for k,v in fitted.items()}}
    result={
        'study':STUDY,'scope':'calibration_only; no heldout source family opened',
        'seed':seed,'depth':depth,'source_checkpoint_sha256':cksha,
        'source_roc02b_receipt_sha256':sha(original_roc02b_receipts/f'roc02b_R{depth}_S{seed}.json.gz'),
        'observer_adapter_sha256':checksum,'observer_adapter_filename':path.name,
        'null_rank':RANK,'null_medians':medians.tolist(),'gamma_grid':list(GAMMAS),
        'same_initial_U0_for_all_arms':True,'updates_per_fitted_arm':UPDATES,
        'selected_gamma':chosen['gamma'],
        'selected_is_positive':chosen['gamma']>0,
        'selected_beats_task_only_J':chosen['calibration_J_lambda1']<task['calibration_J_lambda1'],
        'selected_task_budget_passes':chosen['calibration_mse']<=1.1*task['calibration_mse'],
        'arms':arms,'calibration_only':True,
        'caution':'Calibration optimistically selects gamma on same 256-case development split. No ROC02 prospective generalization or mechanism classification inferred.'}
    (out_dir/f'roc02c_R{depth}_S{seed}.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--depth',type=int,choices=DEPTHS,required=True)
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--checkpoint-dir',type=Path,required=True)
    p.add_argument('--prior-roc02b-dir',type=Path,required=True)
    p.add_argument('--prior-roc02b-manifest',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args()
    result=run(a.seed,a.depth,a.source_dir,a.checkpoint_dir,a.prior_roc02b_dir,
               a.prior_roc02b_manifest,a.out_dir)
    print(json.dumps({'depth':a.depth,'seed':a.seed,'gamma':result['selected_gamma'],
          'task_only_J':result['arms'][str(0.)]['calibration_J_lambda1'],
          'joint_J':result['arms'][str(result['selected_gamma'])]['calibration_J_lambda1'],
          'budget_ok':result['selected_task_budget_passes']},sort_keys=True),flush=True)

if __name__=='__main__':main()

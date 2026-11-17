"""ROC-02C calibration-only equal-budget rank-7/8 observer fitting.

Registered config: configs/roc02c_calibration_equal_budget_v1.json.
No sealed-test data is read. Source weights stay frozen. Evaluation is calibration
only, with held-out source family 4-5 reused across hyperparameter selection.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

try:
    from research_gym.analysis.roc02b_calibration_card import (
        load_and_verify,stage_model,observer_bases,tangent,random_U,jacobian_metrics,SEEDS,DEPTHS)
    from research_gym.analysis.roc02_tied_train import tensors,rows
except ImportError:
    from roc02b_calibration_card import (
        load_and_verify,stage_model,observer_bases,tangent,random_U,jacobian_metrics,SEEDS,DEPTHS)
    from roc02_tied_train import tensors,rows

RANKS=(7,8)
GAMMAS=(0.0,0.05,0.2,1.0)
STATES_PENALTY=(0,1)
STATES_SELECTION=(2,3)
STATES_DIAGNOSTIC=(4,5)
UPDATES=80
LR=.02
N_NULL=128


def score_components(U:torch.Tensor, Cxx:torch.Tensor, Cxy:torch.Tensor, yy:torch.Tensor,
                     n:int,noutput:int=8):
    """Differentiable centered train MSE; ridge refitted at every update."""
    C=U.T@Cxx@U
    G=U.T@Cxy
    W=torch.linalg.solve(C+torch.eye(U.shape[1],dtype=U.dtype,device=U.device),G)
    SSE=yy-2*(W*G).sum()+torch.sum(W*(C@W))
    return SSE/(n*noutput),W


def data_matrices(train_h:torch.Tensor, train_y:torch.Tensor):
    H=train_h.double(); hmean=H.mean(0);X=H-hmean
    Y=F.one_hot(train_y,8).double();ymean=Y.mean(0);Y=Y-ymean
    return X.T@X,X.T@Y,(Y*Y).sum(),hmean,ymean


def null_medians(Ts:list[torch.Tensor],checksum:str,rank:int,indices:tuple[int,...]):
    if len(Ts)!=len(indices):raise ValueError('state count mismatch')
    out=[]
    eye=torch.eye(Ts[0].shape[0],dtype=torch.float64)
    for T,idx in zip(Ts,indices):
        D=T-eye
        dn=torch.linalg.norm(D)
        if dn < 1e-10:raise ValueError('D zero-dynamics')
        vals=[];captures=[];digits=set()
        R=None
        for nullseed in range(941001,941129):
            U=random_U(checksum,idx,rank,nullseed)
            P=U.T@D
            A=P@U
            leak=torch.clamp(P.square().sum()-A.square().sum(),min=0).sqrt()/dn
            vals.append(float(leak))
            digits.add(hashlib.sha256((U[:16,:]@U[:16,:].T).float().numpy().tobytes()).hexdigest())
        if len(digits)!=128:raise AssertionError(f'distinct nulls only {len(digits)}')
        med=sorted(vals);median=(med[63]+med[64])/2
        if median <=0:raise ValueError('undefined null median B')
        out.append({'state_index':idx,'median_b':median,'null_128_distinct':True,'null_b_min':min(vals),'null_b_max':max(vals)})
    return out


def b_fraction_squared(U:torch.Tensor,D:torch.Tensor, null_b:float):
    V=U.T@D
    A=V@U
    B2=torch.clamp(V.square().sum()-A.square().sum(),min=0.)
    D2=D.square().sum()
    return B2/D2/(null_b**2)


def calibrate_objective(U:torch.Tensor, Ds:list[torch.Tensor], medians:list[dict], ids:tuple[int,...]):
    table={m['state_index']:m for m in medians}
    dtable={j:D for j,D in enumerate(Ds)}
    return torch.stack([b_fraction_squared(U,dtable[i],table[i]['median_b']) for i in ids]).mean()


def optimize(U0, Cxx,Cxy,yy,n,Ds,medians,gamma,updates=UPDATES,lr=LR):
    if gamma not in GAMMAS or updates!=UPDATES or abs(lr-LR)>1e-12:
        raise ValueError('outside fixed protocol')
    raw=torch.nn.Parameter(U0.clone())
    opt=torch.optim.Adam([raw],lr=lr)
    for it in range(updates):
        opt.zero_grad(set_to_none=True)
        U=torch.linalg.qr(raw,mode='reduced')[0]
        mse,_=score_components(U,Cxx,Cxy,yy,n)
        penalty=calibrate_objective(U,Ds,medians,STATES_PENALTY)
        loss=mse+gamma*penalty
        if not torch.isfinite(loss):raise RuntimeError('invalid observer loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw],10.)
        opt.step()
    with torch.no_grad():
        U=torch.linalg.qr(raw,mode='reduced')[0]
    return U.detach().clone()


def evaluate(U,train_h,train_y,dev_h,dev_y,matrices,Ds,medians,Tlist,model,indices):
    Cxx,Cxy,yy,hmean,ymean=matrices
    train_loss,W=score_components(U,Cxx,Cxy,yy,len(train_h))
    pred=ymean+(dev_h.double()-hmean)@U@W
    Ydev=F.one_hot(dev_y,8).double()
    mse=float(F.mse_loss(pred,Ydev))
    accuracy=float((pred.argmax(-1)==dev_y).double().mean())
    smoothed=calibrate_objective(U,Ds,medians,indices)
    R=model.readout.weight.detach().double()
    captures=[]
    for idx in indices:
        RT=R@Tlist[idx]
        x=(RT@U).square().sum();den=RT.square().sum()
        captures.append(float(x/den))
    return {'train_MSE':float(train_loss),'calibration_MSE':mse,'calibration_accuracy':accuracy,
            'B_null_squared':float(smoothed),'J_lambda_one':mse+float(smoothed),
            'H_future_logit_capture':sum(captures)/len(captures)}


def one(seed,depth,source_dir:Path,checkpoint_dir:Path,outdir:Path):
    torch.set_num_threads(1)
    checkpoint,checksum,model=load_and_verify(seed,depth,source_dir,checkpoint_dir)
    train=tensors(rows(source_dir/'data/train_inputs.jsonl'))
    cal=tensors(rows(source_dir/'data/calibration_inputs.jsonl'))
    with torch.no_grad():
        train_h,_=stage_model(model,train[0],train[1]);dev_h,_=stage_model(model,cal[0],cal[1])
        basis=observer_bases(train_h,train[2])
        mats=data_matrices(train_h,train[2])
        Tlist=[tangent(model,dev_h[idx]) for idx in range(6)]
        Ds=[T-torch.eye(128,dtype=torch.float64) for T in Tlist]
    rowsOut=[];mlist=[]
    for k in RANKS:
        nulls=null_medians(Tlist,checksum,k,tuple(range(6)))
        mlist.extend([{'seed':seed,'depth':depth,'rank':k,**v} for v in nulls])
        U0=basis[('supervised',k)].double()
        baseline=evaluate(U0,train_h,train[2],dev_h,cal[2],mats,Ds,nulls,Tlist,model,STATES_SELECTION)
        arms=[]
        for gamma in GAMMAS:
            U=optimize(U0,*mats[:3],len(train_h),Ds,nulls,gamma)
            selected=evaluate(U,train_h,train[2],dev_h,cal[2],mats,Ds,nulls,Tlist,model,STATES_SELECTION)
            arms.append((gamma,U,selected))
        task_only=arms[0][2]
        eligible=[(g,U,s) for (g,U,s) in arms[1:] if s['calibration_MSE'] <= 1.10*task_only['calibration_MSE']]
        best=min(eligible,key=lambda x:(x[2]['J_lambda_one'],x[0])) if eligible else arms[0]
        chosen_gamma,Uchosen,chosen=best
        # Diagnostic reserved states are opened after gamma selection. They remain within calibration family.
        stress_task=evaluate(arms[0][1],train_h,train[2],dev_h,cal[2],mats,Ds,nulls,Tlist,model,STATES_DIAGNOSTIC)
        stress_joint=evaluate(Uchosen,train_h,train[2],dev_h,cal[2],mats,Ds,nulls,Tlist,model,STATES_DIAGNOSTIC)
        for g,U,s in arms:
            rowsOut.append({'study':'ROC-02C-calibration-only-equal-budget-v1','seed':seed,'depth':depth,
               'rank':k,'gamma':g,'selected':g==chosen_gamma,'eligible':g!=0 and s['calibration_MSE']<=1.10*task_only['calibration_MSE'],
               'model_checkpoint_sha256':checksum,'task_only_MSE_anchor':task_only['calibration_MSE'],
               'baseline_MSE_unoptimized':baseline['calibration_MSE'],
               **{f'selection_{a}':b for a,b in s.items()},
               'stress_MSE':float(stress_joint['calibration_MSE']) if g==chosen_gamma else None,
               'stress_B_null_squared':stress_joint['B_null_squared'] if g==chosen_gamma else None,
               'stress_H_capture':stress_joint['H_future_logit_capture'] if g==chosen_gamma else None,
               'task_only_stress_B_null_squared':stress_task['B_null_squared'] if g==chosen_gamma else None,
               'calibration_only':True})
    outdir.mkdir(parents=True,exist_ok=True)
    path=outdir/f'roc02c_R{depth}_S{seed}.json'
    result={'study':'ROC-02C-calibration-only-equal-budget-v1','checkpoint_sha256':checksum,
      'source_split':'train source family 0-3; calibration family 4-5; sealed withheld',
      'model_seed':seed,'depth':depth,'rows':rowsOut,'null_medians':mlist,
      'note':'Calibration-only: selection uses calibration task labels. Stress states are calibration, not independent held-out inputs.'}
    path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,required=True,choices=SEEDS)
    p.add_argument('--depth',type=int,required=True,choices=DEPTHS)
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--checkpoint-dir',type=Path,required=True)
    p.add_argument('--outdir',type=Path,required=True)
    a=p.parse_args();r=one(a.seed,a.depth,a.source_dir,a.checkpoint_dir,a.outdir)
    for rank in RANKS:
        chosen=[x for x in r['rows'] if x['rank']==rank and x['selected']][0]
        baseline=[x for x in r['rows'] if x['rank']==rank and x['gamma']==0][0]
        print(f"R{a.depth} S{a.seed} k{rank} gamma{chosen['gamma']:.2g} sel MSE={chosen['selection_calibration_MSE']:.5f} B2={chosen['selection_B_null_squared']:.5f} taskB2={baseline['selection_B_null_squared']:.5f} stressB2={chosen['stress_B_null_squared']:.5f}")

if __name__=='__main__': main()

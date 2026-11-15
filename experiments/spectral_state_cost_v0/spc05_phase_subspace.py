"""SPC-05B: train-only phase-difference subspace with causal held-out swaps.

Uses *fixed* SPC-04 trained weights. PCA is fit to answer-token phase
increments from distinct prompt seeds and multiple predeclared depths.
Ranged-rank projections use no labels. Held-out evaluation compares the
projected intervention with a full future-state transplant and random
orthonormal subspaces of identical dimension, both raw and norm-matched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_state_cost_v0.spc05_phase_interventions import (
    SEEDS, EVAL_SEEDS, PHASE_DEPTH, FOLLOW_LOOPS, MAX_PERIOD,
    model_from_receipt, pointer_periods, prepare_step, _norm_batch,
)
from experiments.spectral_state_cost_v0.spc04_trained_dynamics import N_NODES

BASIS_SEED = 904001
BASIS_PROMPTS = 256
BASIS_DEPTHS = (32,48,64)
RANKS = (2,4,7,8,12,16)
RANDOM_REPEATS = 8


def fit_phase_svd(model, seed: int=BASIS_SEED,
                  n_prompts: int=BASIS_PROMPTS, depths=BASIS_DEPTHS):
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from research_gym.scripts.loopscope_toy import make_batch
    if n_prompts<8 or not depths or min(depths)<1:
        raise ValueError('invalid phase SVD training config')
    tokens,_=make_batch(N_NODES,n_prompts,seed,max(depths)+2)
    with torch.no_grad(),sdpa_kernel([SDPBackend.MATH]):
        advance,h=prepare_step(model,tokens)
        increments=[]
        for depth in range(1,max(depths)+2):
            next_h=advance(h)
            if depth-1 in depths:
                increments.append((next_h[:,-1,:]-h[:,-1,:]).clone())
            h=next_h
        X=torch.cat(increments,dim=0)
        # No test labels or test trajectories are involved. Do not center away
        # the mean causal increment, which itself may carry phase changes.
        U,S,Vh=torch.linalg.svd(X,full_matrices=False)
        cumulative=(torch.cumsum(S.square(),dim=0)/S.square().sum()).cpu().numpy()
    return Vh.detach(),{
        'train_seed':seed,'train_prompts':n_prompts,'sample_depths':list(depths),
        'rows':int(X.shape[0]),'rank':int(Vh.shape[0]),
        'energy_capture_by_rank':{str(r):float(cumulative[r-1]) for r in (1,2,4,7,8,12,16,24,32,48) if r<=len(cumulative)},
        'singular_values':S.cpu().tolist(),
    }


def random_basis(width:int, rank:int, seed:int):
    import torch
    if not (1<=rank<=width):raise ValueError('bad subspace dimension')
    g=torch.Generator(device='cpu').manual_seed(seed)
    matrix=torch.randn((width,rank),dtype=torch.float64,generator=g)
    return torch.linalg.qr(matrix).Q.T


def project_increment(delta, basis, mode: str):
    """Orthogonal subspace projection, optionally with total L2 norm matched."""
    projected=delta@basis.T@basis
    if mode=='norm_matched':
        original_norm=_norm_batch(delta[:,None,:])
        projected_norm=_norm_batch(projected[:,None,:]).clamp_min(1e-30)
        projected=projected*(original_norm/projected_norm)[:,None]
    elif mode!='raw':
        raise ValueError('unknown projection mode')
    return projected


def evaluate_rank_batch(model, Vh, prompt_seed:int, batch:int=64,
                        ranks=RANKS, random_repeats:int=RANDOM_REPEATS,
                        depth:int=PHASE_DEPTH, follow:int=FOLLOW_LOOPS)->dict:
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from research_gym.scripts.loopscope_toy import make_batch
    if batch<1 or random_repeats<1 or not ranks or max(ranks)>Vh.shape[0]:
        raise ValueError('invalid rank evaluation config')
    if depth<2 or follow<1:raise ValueError('invalid phase-depth configuration')
    horizon=depth+follow+MAX_PERIOD+2
    tokens,hop=make_batch(N_NODES,batch,prompt_seed,horizon)
    periods=np.array(pointer_periods(tokens),dtype=int)
    with torch.no_grad(),sdpa_kernel([SDPBackend.MATH]):
        advance,h=prepare_step(model,tokens)
        trajectory=[h]
        for _ in range(horizon):
            h=advance(h);trajectory.append(h)
        x=trajectory[depth]; y=trajectory[depth+1];delta=y[:,-1]-x[:,-1]
        reference=trajectory[depth+follow]
        phase_plus=trajectory[depth+follow+1]
        true_period=np.array([float(_norm_batch((trajectory[depth+follow+int(p)]-reference)[i:i+1])/(_norm_batch(reference[i:i+1]).clamp_min(1e-30))) for i,p in enumerate(periods)])
        lagone=(_norm_batch(phase_plus-reference)/_norm_batch(reference).clamp_min(1e-30)).numpy()
        stable=(periods>1)&(true_period<1e-4)&(lagone>0.05)
        phase_gap=np.array([min(float(_norm_batch((trajectory[depth+j]-x)[i:i+1])) for j in range(1,int(p))) if p>1 else 0. for i,p in enumerate(periods)])
        stable &= phase_gap>1e-8
        refs=[trajectory[depth+follow+j] for j in range(MAX_PERIOD)]
        idx=np.flatnonzero(stable)
        future_target=hop[:,depth+follow].numpy()
        baseline_pred=model.readout(phase_plus[:,-1]).argmax(-1).numpy()
        def evaluate_basis(basis, mode):
            raw_projection=project_increment(delta,basis,'raw')
            capture=(_norm_batch(raw_projection[:,None,:]).square() / _norm_batch(delta[:,None,:]).square().clamp_min(1e-30)).numpy()
            projected=project_increment(delta,basis,mode)
            pert=x.clone();pert[:,-1]+=projected
            for _ in range(follow):pert=advance(pert)
            distances=torch.stack([_norm_batch(pert-ref) for ref in refs],dim=1).numpy()
            closest=np.asarray([int(np.argmin(distances[i,:p])) if p>1 else -1 for i,p in enumerate(periods)])
            preds=model.readout(pert[:,-1]).argmax(-1).numpy()
            return {
                'registered_shift_count':int(np.sum(closest[idx]==1)),
                'registered_evaluable':int(len(idx)),
                'registered_shift_fraction':float(np.mean(closest[idx]==1)) if len(idx) else None,
                'registered_readout_agree_shift_count':int(np.sum(preds[idx]==baseline_pred[idx])),
                'registered_correct_next_hop_count':int(np.sum(preds[idx]==future_target[idx])),
                'median_projection_energy_fraction_registered':float(np.median(capture[idx])) if len(idx) else None,
                'mean_projection_energy_fraction_registered':float(np.mean(capture[idx])) if len(idx) else None,
            }
        result={
            'prompt_seed':prompt_seed,'batch_size':batch,'registered_cycles':int(len(idx)),
            'pointer_nontrivial_count':int(np.count_nonzero(periods>1)),
            'true_period_relative_residual_median_nontrivial':float(np.median(true_period[periods>1])) if np.any(periods>1) else None,
            'rank_rows':{},
        }
        for rank in ranks:
            rows={}
            for mode in ('raw','norm_matched'):
                rows[f'pca_{mode}']=evaluate_basis(Vh[:rank],mode)
                random_rows=[]
                for trial in range(random_repeats):
                    Q=random_basis(delta.shape[1],rank,seed=771991+10007*rank+311*trial)
                    random_rows.append(evaluate_basis(Q,mode))
                rows[f'random_{mode}']=random_rows
            result['rank_rows'][str(rank)]=rows
    return result


def campaign(checkpoint_dir: str|Path, model_seeds=SEEDS, eval_seeds=EVAL_SEEDS,
             batch:int=64, ranks=RANKS, random_repeats:int=RANDOM_REPEATS)->dict:
    if BASIS_SEED in eval_seeds:raise ValueError('basis and evaluation prompts must be disjoint')
    allrows=[]
    for model_seed in model_seeds:
        model,sha=model_from_receipt(Path(checkpoint_dir)/f'spc04_seed{model_seed}.pt',model_seed)
        Vh,basis=fit_phase_svd(model)
        rows=[evaluate_rank_batch(model,Vh,seed,batch=batch,ranks=ranks,random_repeats=random_repeats) for seed in eval_seeds]
        allrows.append({'model_seed':model_seed,'checkpoint_weight_sha256':sha,
                        'basis_training':basis,'heldout':rows})
    return {'campaign':'SPC-05B','trained_checkpoint_source':'SPC-04','phase_depth':PHASE_DEPTH,
            'follow':FOLLOW_LOOPS,'rank_grid':list(ranks),
            'heldout_prompt_seeds':list(eval_seeds),'random_repeats':random_repeats,
            'method':'uncentered answer-token phase-displacement SVD; random orthonormal controls',
            'checkpoint_rows':allrows}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--batch',type=int,default=64)
    parser.add_argument('--random-repeats',type=int,default=RANDOM_REPEATS)
    args=parser.parse_args()
    result=campaign(args.checkpoint_dir,batch=args.batch,random_repeats=args.random_repeats)
    payload=json.dumps(result,indent=2,sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(payload+'\n')
    else:print(payload)

if __name__=='__main__':main()

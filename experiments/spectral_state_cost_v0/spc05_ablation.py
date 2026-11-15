"""SPC-05C: necessity test, suppressing an SVD subspace each recurrent step.

The rank-r basis is fitted to disjoint training prompts in SPC-05B and frozen.
Starting after supervised loop 3, this module subtracts the answer-token
increment projected onto that basis at EVERY later loop. Baseline and
random-subspace ablations use the same frozen native LoopedChase map.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_state_cost_v0.spc05_phase_interventions import (
    SEEDS, EVAL_SEEDS, model_from_receipt, prepare_step, _norm_batch,
)
from experiments.spectral_state_cost_v0.spc05_phase_subspace import (
    fit_phase_svd, random_basis, project_increment,
)
from experiments.spectral_state_cost_v0.spc04_trained_dynamics import N_NODES

START_AFTER_SUPERVISED = 3
ABLATION_HORIZONS=(6,12,24)
RANK_GRID=(4,7,8)
RANDOM_REPEATS=8


def suppress_answer_increment(current, following, basis, equal_norm_reference=None):
    """Remove a coordinate-subspace component of the *next* answer change.

    This never edits the context-token coordinates or prompt injection.
    `basis` is an orthonormal row basis. If equal_norm_reference is given,
    rescale the removed component to its reference norm as a separate
    out-of-distribution magnitude control; it ceases to be an orthogonal
    projection in this branch.
    """
    delta=following[:,-1,:]-current[:,-1,:]
    removal=project_increment(delta,basis,'raw')
    if equal_norm_reference is not None:
        mag=_norm_batch(equal_norm_reference[:,None,:])
        own=_norm_batch(removal[:,None,:]).clamp_min(1e-30)
        removal=removal * (mag/own)[:,None]
    updated=following.clone()
    updated[:,-1,:]=following[:,-1,:]-removal
    return updated


def run_ablation_batch(model, Vh, prompt_seed:int, batch:int=64,
                       random_repeats:int=RANDOM_REPEATS,
                       horizons=ABLATION_HORIZONS, ranks=RANK_GRID):
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from research_gym.scripts.loopscope_toy import make_batch
    if not horizons or min(horizons)<=START_AFTER_SUPERVISED or not ranks or max(ranks)>Vh.shape[0]:
        raise ValueError('invalid horizon or projector rank')
    if batch<1 or random_repeats<1:raise ValueError('invalid sample or random count')
    tokens,hops=make_batch(N_NODES,batch,prompt_seed,max(horizons))
    with torch.no_grad(),sdpa_kernel([SDPBackend.MATH]):
        step,h=prepare_step(model,tokens)
        for _ in range(START_AFTER_SUPERVISED):h=step(h)
        frozen_prediction=model.readout(h[:,-1,:]).argmax(-1).numpy()
        model_rows={}
        specs=[('baseline',None,None),('freeze_answer',torch.eye(h.shape[-1],dtype=h.dtype),None)]
        specs.extend((f'pca_rank{rank}',Vh[:rank],None) for rank in ranks)
        for trial in range(random_repeats):
            random_q=random_basis(h.shape[-1],8,seed=818009+97*trial)
            specs.append((f'random8_raw_trial{trial}',random_q,None))
            specs.append((f'random8_equal_removed_norm_trial{trial}',random_q,Vh[:8]))
        for name,basis,match_to in specs:
            state=h.clone()
            totals={}
            for time in range(START_AFTER_SUPERVISED+1,max(horizons)+1):
                following=step(state)
                if basis is not None:
                    ref_component=None
                    if match_to is not None:
                        d=following[:,-1,:]-state[:,-1,:]
                        ref_component=project_increment(d,match_to,'raw')
                    following=suppress_answer_increment(state,following,basis,ref_component)
                state=following
                if time in horizons:
                    pred=model.readout(state[:,-1,:]).argmax(-1).numpy()
                    correct=pred==hops[:,time-1].numpy()
                    totals[str(time)]={
                        'correct_current_hop':int(correct.sum()),
                        'matches_frozen_loop3_readout':int((pred==frozen_prediction).sum()),
                        'n_items':batch,
                    }
            model_rows[name]=totals
    return {'prompt_seed':prompt_seed,'items':batch,'horizons':list(horizons),'conditions':model_rows}


def ablation_campaign(checkpoint_dir:str|Path, model_seeds=SEEDS, eval_seeds=EVAL_SEEDS,
                      batch:int=64,random_repeats:int=RANDOM_REPEATS):
    rows=[]
    for seed in model_seeds:
        model,sha=model_from_receipt(Path(checkpoint_dir)/f'spc04_seed{seed}.pt',seed)
        Vh,basis=fit_phase_svd(model)
        x=[run_ablation_batch(model,Vh,prompt_seed, batch,random_repeats) for prompt_seed in eval_seeds]
        rows.append({'seed':seed,'checkpoint_sha256':sha,'basis_training':basis,'heldout':x})
    return {'campaign':'SPC-05C','model':'native trained LoopedChase',
            'supervised_hops':[1,2,3],'interventions_from_loop':4,
            'fixed_prompt':True,'rank_grid':list(RANK_GRID),
            'random_repeats':random_repeats,'eval_seeds':list(eval_seeds),
            'description':'Repeated answer-token projected-increment suppression; two matched random-projection ablations',
            'rows':rows}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint-dir',type=Path,required=True)
    p.add_argument('--out',type=Path)
    p.add_argument('--batch',type=int,default=64)
    p.add_argument('--random-repeats',type=int,default=RANDOM_REPEATS)
    args=p.parse_args()
    r=ablation_campaign(args.checkpoint_dir,batch=args.batch,random_repeats=args.random_repeats)
    z=json.dumps(r,indent=2,sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(z+'\n')
    else:print(z)

if __name__=='__main__':main()

"""SPC-08 held-out, local Jacobian follow-up for instantaneous/deferred bases.

This assay uses fresh prompt seeds unrelated to the train-only basis cohorts.
Report only estimates of finite-difference local output sensitivity.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path

from experiments.spectral_state_cost_v0.spc05_phase_interventions import SEEDS,model_from_receipt
from experiments.spectral_state_cost_v0.spc08_phase_readout import (
    fit_causal_decomposition,heldout_local_sensitivity,
)
from experiments.spectral_state_cost_v0.spc07_transfer import token_hashes

HOLDOUT_SEED=910709
HOLDOUT_PROMPTS=12

def run(checkpoint_dir:Path,holdout_seed=HOLDOUT_SEED,n_prompts=HOLDOUT_PROMPTS):
    import torch
    from research_gym.scripts.loopscope_toy import make_batch
    torch.set_num_threads(min(2,torch.get_num_threads()))
    rows={}
    tokens,_=make_batch(8,n_prompts,holdout_seed,41)
    heldout_hash=token_hashes(tokens)
    for seed in SEEDS:
        model,digest=model_from_receipt(checkpoint_dir/f'spc04_seed{seed}.pt',seed)
        bases,train_diag,train_hash=fit_causal_decomposition(model)
        if train_hash & heldout_hash:
            raise RuntimeError('training and heldout prompt overlap')
        record=heldout_local_sensitivity(model,bases,prompt_seed=holdout_seed,n_prompts=n_prompts)
        rows[str(seed)]={'checkpoint_weight_sha256':digest,'heldout_local_jacobian':record,
                         'training_gramian':'SPC-08 fixed training seed 920731'}
    return {'campaign':'SPC-08-heldout-Jacobian','seed':holdout_seed,
            'n_heldout':n_prompts,'rows':rows}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint-dir',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--n-heldout',type=int,default=HOLDOUT_PROMPTS)
    opts=ap.parse_args()
    obj=run(opts.checkpoint_dir,n_prompts=opts.n_heldout)
    opts.out.parent.mkdir(parents=True,exist_ok=True)
    opts.out.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'seed':obj['seed'],'checkpoints':list(obj['rows']),'path':str(opts.out)},indent=2))
if __name__=='__main__':main()

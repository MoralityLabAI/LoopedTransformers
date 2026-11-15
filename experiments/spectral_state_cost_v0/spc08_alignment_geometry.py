"""SPC-08A: held-out cross-checkpoint subspace overlap after frozen alignment.

This isolates the geometry readout from the causal phase intervention score.
Paired Procrustes is fitted on the SPC-07 unlabeled training prompts; raw
neuron coordinates and shuffled-pair alignment remain negative controls.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from experiments.spectral_state_cost_v0.spc05_phase_interventions import SEEDS,model_from_receipt
from experiments.spectral_state_cost_v0.spc08_phase_readout import fit_causal_decomposition,RANK
from experiments.spectral_state_cost_v0.spc07_transfer import train_alignment,projector_alignment


def overlap_summary(source,target,aligned,shuffled,rank=RANK):
    if sorted(source)!=sorted(target):
        raise ValueError('source and target basis groups differ')
    output={}
    for group in ('phase','immediate','temporal','delayed_null'):
        if group not in source:
            raise ValueError('required basis group missing')
        output[group]={
            'raw':projector_alignment(source[group],target[group],rank)['overlap_trace_fraction'],
            'paired':projector_alignment(source[group]@aligned,target[group],rank)['overlap_trace_fraction'],
            'shuffled':projector_alignment(source[group]@shuffled,target[group],rank)['overlap_trace_fraction'],
        }
    return output


def run(checkpoint_dir:Path):
    import torch
    torch.set_num_threads(min(2,torch.get_num_threads()))
    fitted={};models={};hashes={}
    for seed in SEEDS:
        model,digest=model_from_receipt(checkpoint_dir/f'spc04_seed{seed}.pt',seed)
        basis,diagnostics,_=fit_causal_decomposition(model)
        fitted[seed]=basis;models[seed]=model;hashes[str(seed)]=digest
    paired,shuffled,_,calibration=train_alignment(models[SEEDS[0]],models[SEEDS[1]])
    return {
        'campaign':'SPC-08A','checkpoint_sha256':hashes,
        'method':'top-eight subspace projector overlap = tr(P_source P_target)/8',
        'paired_alignment_frozen':'SPC-07 depth/phase paired Procrustes, 256 training prompts',
        'shuffled_alignment_frozen':'same source with target row permutation',
        'fit_basis':'SPC-08 current/deferred Gramians from 16 training-only prompts',
        'alignment_calibration':calibration,
        'subspace_overlap':overlap_summary(fitted[SEEDS[0]],fitted[SEEDS[1]],paired,shuffled),
    }


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--checkpoint-dir',type=Path,required=True)
    a.add_argument('--out',type=Path,required=True)
    opts=a.parse_args()
    obj=run(opts.checkpoint_dir)
    opts.out.parent.mkdir(parents=True,exist_ok=True)
    opts.out.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
    print(str(opts.out))
if __name__=='__main__':main()

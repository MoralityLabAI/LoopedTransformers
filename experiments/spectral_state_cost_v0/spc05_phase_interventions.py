"""SPC-05: causal phase-state interventions on saved SPC-04 LoopedChase models.

This is a frozen, synthetic pointer-chase research assay, not a language-model
safety evaluation. All interventions reuse *the same prompt* and compare to
native LoopedChase forward states. A full-state phase transplant is an exact
semigroup control. Only token-partial transplants probe phase localization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.spectral_state_cost_v0.spc04_trained_dynamics import (
    N_NODES, D_MODEL, new_model, weight_sha256,
)

SEEDS = (41522, 41523)
EVAL_SEEDS = (905501, 905502, 905503)
NOISE_LEVELS = (0.05, 0.25, 0.75)
PHASE_DEPTH = 64
FOLLOW_LOOPS = 12
MAX_PERIOD = N_NODES


def model_from_receipt(path: str | Path, expected_seed: int):
    """Load the trusted campaign's weights-only checkpoint and verify hash."""
    import torch
    rec = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(rec, dict) or rec.get('seed') != expected_seed:
        raise ValueError('checkpoint seed mismatch')
    if rec.get('hyperparameters', {}).get('n_nodes') != N_NODES:
        raise ValueError('checkpoint architecture mismatch')
    if rec.get('hyperparameters', {}).get('d_model') != D_MODEL:
        raise ValueError('checkpoint width mismatch')
    m = new_model(expected_seed)
    m.load_state_dict(rec['model'], strict=True)
    digest = weight_sha256(m)
    if digest != rec.get('weight_sha256'):
        raise ValueError('checkpoint digest mismatch')
    return m.eval().double(), digest


def prepare_step(model, tokens):
    """Reconstruct the full native LoopedChase map, including prompt injection."""
    import torch
    n = N_NODES
    if tokens.ndim != 2 or tokens.shape[1] != 2*n+1:
        raise ValueError('no-depth-token pointer prompt expected')
    edges = tokens[:, :2*n].reshape(tokens.shape[0],n,2)
    original = torch.cat((model.tok(edges[...,0])+model.dst(edges[...,1]),
                          model.tok(tokens[:,2*n:])), dim=1)
    e = original + model.pos(torch.arange(original.shape[1], device=tokens.device))
    def advance(h):
        v = model.inject(h) + e
        for core in model.core:
            v = core(v)
        return v
    return advance, torch.zeros_like(e)


def pointer_periods(tokens):
    """Exact graph orbit length of the start vertex for each sample."""
    from experiments.spectral_state_cost_v0.spc04_floquet_cycle import pointer_cycle_length
    return [pointer_cycle_length(tokens[i:i+1]) for i in range(len(tokens))]


def _norm_batch(a):
    import torch
    return torch.linalg.vector_norm(a.reshape(a.shape[0],-1),dim=1)


def _normalized_noise(reference, mask, seed: int):
    import torch
    g = torch.Generator(device='cpu').manual_seed(seed)
    z = torch.randn(reference.shape, dtype=reference.dtype, generator=g)
    z = z * mask
    magnitude = _norm_batch(z).clamp_min(1e-30)
    return z / magnitude[:,None,None]


def causal_phase_batch(model, seed: int, batch: int = 32,
                       phase_depth: int = PHASE_DEPTH, follow: int = FOLLOW_LOOPS,
                       noise_levels: tuple[float,...] = NOISE_LEVELS) -> dict[str, Any]:
    """Score all-token and isolated-token interventions at matched input.

    The nearest phase is selected using the *entire* hidden state against
    held-out reference phases. This is an oracle diagnostic, not a learned
    monitor. For p=1, phase attribution is intentionally excluded.
    """
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from research_gym.scripts.loopscope_toy import make_batch
    if batch < 1 or phase_depth < 2 or follow < 1:
        raise ValueError('invalid evaluation shape')
    if any(z <= 0 or z > 2 for z in noise_levels):
        raise ValueError('noise must be in (0,2] units of phase separation')
    model = model.eval().double()
    torch.set_num_threads(min(2,torch.get_num_threads()))
    nstates = phase_depth + follow + MAX_PERIOD + 3
    tokens, hops = make_batch(N_NODES,batch,seed,nstates)
    periods = np.asarray(pointer_periods(tokens),dtype=np.int64)
    with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
        advance, h0 = prepare_step(model,tokens)
        h = h0
        states=[h]
        for _ in range(nstates):
            h=advance(h)
            states.append(h)
        original=model(tokens, min(phase_depth+1,5))
        native_gap=float((torch.stack([s[:,-1] for s in states[1:len(original)+1]])-original).abs().max().item())
        if native_gap > 1e-10:
            raise AssertionError('full-state map and native forward disagree')

        start=states[phase_depth]
        plus=states[phase_depth+1]
        phase_gap=torch.zeros((batch,),dtype=torch.float64)
        for i,p in enumerate(periods):
            if p>1:
                phase_gap[i]=min(float(torch.linalg.vector_norm((states[phase_depth+j][i]-start[i]).reshape(-1)))
                                 for j in range(1,int(p)))
        nontrivial=periods>1
        # Account for which coordinate group carries the phase-step displacement.
        jump=plus-start
        jump_sq=_norm_batch(jump).square().clamp_min(1e-30)
        answer_fraction=(_norm_batch(jump[:,-1:,:]).square()/jump_sq).numpy()
        if np.any(nontrivial & (phase_gap.cpu().numpy() < 1e-10)):
            # Such points have no numerically distinguishable states and get
            # explicit phase-invalid flags rather than spurious phase labels.
            pass
        active=np.asarray(nontrivial & (phase_gap.cpu().numpy() > 1e-8))
        delta=phase_gap[:,None,None]
        allmask=torch.ones_like(start)
        ansmask=torch.zeros_like(start); ansmask[:,-1,:]=1
        ctxmask=torch.zeros_like(start); ctxmask[:,:-1,:]=1
        candidates={
            'unperturbed':start,
            'full_phase_plus1':plus,
            'answer_phase_plus1':torch.cat((start[:,:-1,:],plus[:,-1:,:]),dim=1),
            'context_phase_plus1':torch.cat((plus[:,:-1,:],start[:,-1:,:]),dim=1),
        }
        for j,alpha in enumerate(noise_levels):
            for kind,mask in [('full',allmask),('answer',ansmask),('context',ctxmask)]:
                direction=_normalized_noise(start,mask,seed*1009+j*131+({'full':1,'answer':2,'context':3}[kind]))
                candidates[f'noise_{kind}_{alpha:g}']=start + (alpha*delta)*direction
        target_now=hops[:,phase_depth+follow-1].numpy()
        target_plus=hops[:,phase_depth+follow].numpy()
        reference=states[phase_depth+follow]
        # Only claim task-cycle phase registration for numerically closed,
        # nonfixed orbits with the *actual integer pointer period*.
        relative_cycle=[]
        relative_one=[]
        for i,p in enumerate(periods):
            ref_i=reference[i]
            scale=max(float(torch.linalg.vector_norm(ref_i)),1e-30)
            relative_cycle.append(float(torch.linalg.vector_norm(states[phase_depth+follow+int(p)][i]-ref_i))/scale)
            relative_one.append(float(torch.linalg.vector_norm(states[phase_depth+follow+1][i]-ref_i))/scale)
        registered=np.asarray(nontrivial & (np.array(relative_cycle)<1e-4)
                              & (np.array(relative_one)>0.05) & active)
        comparison=[states[phase_depth+follow+j] for j in range(MAX_PERIOD)]
        baseline_pred=model.readout(reference[:,-1]).argmax(-1).numpy()
        response={
            'prompt_seed':seed,'items':batch,'phase_depth':phase_depth,'follow':follow,
            'n_nontrivial_graph_period':int(nontrivial.sum()),
            'n_distinguishable_phase_reference':int(active.sum()),
            'n_registered_nonfixed_task_cycles':int(registered.sum()),
            'median_task_cycle_relative_residual_nontrivial':float(np.median(np.array(relative_cycle)[nontrivial])) if nontrivial.any() else None,
            'answer_token_phase_jump_fraction_median_nontrivial':float(np.median(answer_fraction[nontrivial])) if nontrivial.any() else None,
            'native_forward_gap':native_gap,
            'reference_current_hop_accuracy':float(np.mean(baseline_pred==target_now)),
            'reference_current_hop_accuracy_nontrivial':float(np.mean(baseline_pred[nontrivial]==target_now[nontrivial])) if nontrivial.any() else None,
            'phase_separation_median':float(torch.median(phase_gap[torch.from_numpy(active)])) if active.any() else None,
            'models':{},
        }
        for name, pert in candidates.items():
            shift=(name=='full_phase_plus1')
            x=pert
            for _ in range(follow):
                x=advance(x)
            pred=model.readout(x[:,-1]).argmax(-1).numpy()
            distances=torch.stack([_norm_batch(x-z) for z in comparison],dim=1).numpy()
            phase_predictions=[]
            margin=[]
            for i,p in enumerate(periods):
                if not active[i]:
                    phase_predictions.append(None);margin.append(None);continue
                choices=distances[i,:p]
                order=np.argsort(choices)
                phase_predictions.append(int(order[0]))
                margin.append(float((choices[order[1]]-choices[order[0]])/float(phase_gap[i])))
            first_diff=_norm_batch(pert-start)
            last_diff=_norm_batch(x-reference)
            norm_gap=phase_gap.clamp_min(1e-12)
            valid=np.flatnonzero(active)
            correct_shift=np.asarray([phase_predictions[i]==1 for i in valid])
            reg_valid=np.flatnonzero(registered)
            correct_reg_shift=np.asarray([phase_predictions[i]==1 for i in reg_valid])
            correct_reg_zero=np.asarray([phase_predictions[i]==0 for i in reg_valid])
            correct_zero=np.asarray([phase_predictions[i]==0 for i in valid])
            row={
                'n_phase_evaluable':int(len(valid)),
                'n_nearest_reference_zero':int(correct_zero.sum()),
                'n_nearest_reference_plus_one':int(correct_shift.sum()),
                'n_registered_task_cycles':int(len(reg_valid)),
                'n_registered_nearest_zero':int(correct_reg_zero.sum()),
                'n_registered_nearest_plus_one':int(correct_reg_shift.sum()),
                'n_nontrivial_correct_current':int(np.count_nonzero(pred[nontrivial]==target_now[nontrivial])),
                'n_nontrivial_correct_shifted':int(np.count_nonzero(pred[nontrivial]==target_plus[nontrivial])),
                'initial_displacement_relative_to_phase_gap_median':float(np.median((first_diff/norm_gap).numpy()[active])) if active.any() else None,
                'nearest_reference_phase_zero_rate':float(np.mean(correct_zero)) if len(valid) else None,
                'nearest_reference_phase_plus_one_rate':float(np.mean(correct_shift)) if len(valid) else None,
                'min_phase_margin_median':float(np.median([margin[i] for i in valid])) if len(valid) else None,
                'current_hop_accuracy':float(np.mean(pred==target_now)),
                'shifted_hop_accuracy':float(np.mean(pred==target_plus)),
                'current_hop_accuracy_nontrivial':float(np.mean(pred[nontrivial]==target_now[nontrivial])) if nontrivial.any() else None,
                'shifted_hop_accuracy_nontrivial':float(np.mean(pred[nontrivial]==target_plus[nontrivial])) if nontrivial.any() else None,
                'prediction_agreement_with_unperturbed':float(np.mean(pred==baseline_pred)),
                'normalized_terminal_state_error_median':float(np.median((last_diff/norm_gap).numpy()[active])) if active.any() else None,
                'normalized_terminal_shifted_state_error_median':float(np.median((_norm_batch(x-states[phase_depth+follow+1])/norm_gap).numpy()[active])) if active.any() else None,
                'initial_to_terminal_error_ratio_median':float(np.median((last_diff / first_diff.clamp_min(1e-20)).numpy()[active])) if active.any() else None,
                'full_shift_exact_replay_gap':float(_norm_batch(x-states[phase_depth+follow+1]).max()) if shift else None,
            }
            response['models'][name]=row
    return response


def run_campaign(checkpoint_dir: str|Path, seeds=SEEDS,
                 eval_seeds=EVAL_SEEDS, batch:int=32, phase_depth:int=PHASE_DEPTH,
                 follow:int=FOLLOW_LOOPS, noise_levels=NOISE_LEVELS)->dict:
    from research_gym.scripts.loopscope_toy import LoopedChase
    result={'campaign':'SPC-05', 'task':'8-node permutation pointer chasing',
            'source_checkpoints':'SPC-04 offline saved weights (verified SHA256)',
            'train_hops':[1,2,3], 'eval_seeds':list(eval_seeds),'seeds':list(seeds),
            'protocol':{'batch':batch,'depth':phase_depth,'follow':follow,
                        'noise_separation_fractions':list(noise_levels),
                        'phase_attribution':'minimum full-state Euclidean distance to all reference phases',
                        'control':'full-phase transplant exact by semigroup',
                        'precision':'float64 weights from float32 trained checkpoints'},
            'checkpoints':[]}
    for seed in seeds:
        p=Path(checkpoint_dir)/f'spc04_seed{seed}.pt'
        model,sha=model_from_receipt(p,seed)
        rows=[causal_phase_batch(model,eval_seed,batch=batch,phase_depth=phase_depth,follow=follow,noise_levels=tuple(noise_levels)) for eval_seed in eval_seeds]
        result['checkpoints'].append({'seed':seed,'weight_sha256':sha,'batches':rows})
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--batch',type=int,default=32)
    parser.add_argument('--phase-depth',type=int,default=PHASE_DEPTH)
    parser.add_argument('--follow',type=int,default=FOLLOW_LOOPS)
    args=parser.parse_args()
    output=run_campaign(args.checkpoint_dir,batch=args.batch,phase_depth=args.phase_depth,follow=args.follow)
    txt=json.dumps(output,indent=2,sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(txt+'\n')
    else:print(txt)

if __name__=='__main__':
    main()

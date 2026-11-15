"""SPC-06B: latent phase intervention invisible to the *linearized* readout.

Two fixed, previously trained SPC-04 models; exact same native prompt and
full recurrent update as SPC-05. The gradient of each of seven independent
logit margins is computed analytically for the native LayerNorm+linear head.
Its row/null orthogonal decomposition of a one-loop phase displacement is
used as a causal intervention. Tests check J*delta_null ~ 0 explicitly.

The nonlinear readout may respond at second order. We record this, including
whether the phase shift survives 12 further steps. No exact global readout
invisibility or control over arbitrary language models is claimed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_state_cost_v0.spc05_phase_interventions import (
    SEEDS, EVAL_SEEDS, PHASE_DEPTH, FOLLOW_LOOPS, MAX_PERIOD,
    model_from_receipt, prepare_step, pointer_periods, _norm_batch,
)
from experiments.spectral_state_cost_v0.spc04_trained_dynamics import N_NODES


def logit_margin_jacobian(model, answer):
    """Batched exact derivative of the model's logit *differences* wrt answer.

    answer: [B,d] float64, LayerNorm and head parameters belong to model.
    Result [B,n_classes-1,d] and its Jacobian SVD; all on CPU.
    """
    import torch
    if answer.ndim != 2 or not answer.is_floating_point():
        raise ValueError('answer must have shape [batch, hidden dimension]')
    d = answer.shape[1]
    ln = model.ln_f
    head = model.head
    if ln.normalized_shape != (d,):
        raise ValueError('LayerNorm channel contract changed')
    u = answer - answer.mean(dim=1, keepdim=True)
    variance = u.square().mean(dim=1, keepdim=True)
    inv = torch.rsqrt(variance + ln.eps)
    # D[(h-mu)/sqrt(var+eps)] = inv*C - inv^3/d * uu^T.
    centered = torch.eye(d, dtype=answer.dtype, device=answer.device) - (1.0/d)
    derivative = inv[:,:,None] * centered[None,:,:] \
        - inv.pow(3)[:,:,None] * (u[:,:,None] * u[:,None,:])/d
    d_logits = (head.weight * ln.weight[None,:]) @ derivative
    # Softmax/argmax only sees differences from an arbitrary reference logit.
    margins = d_logits[:,:-1,:] - d_logits[:,-1:,:]
    return margins


def split_readout_directions(model, answer, delta):
    """Euclidean projection of delta into Jacobian row space and nullspace."""
    import torch
    if answer.shape != delta.shape:
        raise ValueError('answer and displacement shape mismatch')
    margins = logit_margin_jacobian(model, answer)
    _,s,vh = torch.linalg.svd(margins, full_matrices=False)
    rank = (s > s.max(dim=1,keepdim=True).values * 1e-10).sum(dim=1)
    if torch.any(rank != margins.shape[1]):
        raise ValueError('readout margin Jacobian lacks full row rank')
    row = (delta[:,None,:] @ vh.transpose(1,2)) @ vh
    row = row[:,0,:]
    null = delta - row
    derivative_residual = torch.linalg.vector_norm(
        (margins @ null[:,:,None]).squeeze(-1),dim=-1)
    return row, null, {
        'max_null_first_order_margin_change':float(derivative_residual.max()),
        'min_readout_margin_rank':int(rank.min()),
        'median_row_energy_fraction':float(torch.median(
            row.square().sum(dim=-1)/delta.square().sum(dim=-1).clamp_min(1e-30))),
    }


def exact_readout_fiber_rotate(model, answer, phase_delta, scale:float=1.0, preserve:str="logits"):
    """Move within the EXACT LayerNorm readout-margin fiber.

    The original centered vector u splits into its component in the
    relevant readout row space and the orthogonal nullspace. Rotating only
    the null component while preserving its norm leaves the centered norm
    unchanged, including the effect of LayerNorm eps. preserve='logits'
    leaves all eight logits unchanged; preserve='margins' leaves their seven
    independent differences unchanged. The mean also stays fixed.
    """
    import torch
    if not (0 <= scale <= 16) or answer.shape != phase_delta.shape:
        raise ValueError('invalid readout fiber rotation')
    d=answer.shape[1]
    ln=model.ln_f; head=model.head
    centered=torch.eye(d,dtype=answer.dtype,device=answer.device) - 1.0/d
    W=head.weight * ln.weight[None,:]
    if preserve == 'logits':
        A = W @ centered
    elif preserve == 'margins':
        A = (W[:-1] - W[-1:]) @ centered
    else:
        raise ValueError('preserve must be logits or margins')
    _,s,Vh=torch.linalg.svd(A,full_matrices=False)
    if torch.sum(s>s.max()*1e-10)<A.shape[0]:
        raise ValueError('readout margin matrix rank changed')
    mean=answer.mean(dim=1,keepdim=True)
    u=answer-mean
    row=(u@Vh.T)@Vh
    w=u-row
    dc=phase_delta-phase_delta.mean(dim=1,keepdim=True)
    direction=dc-(dc@Vh.T)@Vh
    trial=w+scale*direction
    scale_factor=torch.linalg.vector_norm(w,dim=-1)/torch.linalg.vector_norm(trial,dim=-1).clamp_min(1e-30)
    rotated=mean+row+trial*scale_factor[:,None]
    return rotated


def random_equal_chord_fiber(model, answer, target, seed:int):
    """Uniform-ish tangent control at the target's EXACT readout-fiber chord.

    Projects Gaussian noise into the nullspace of the full logit linear map,
    then removes its component along the current nullspace activation. A
    spherical rotation with the target's polar angle preserves every logit,
    as well as the exact target intervention norm per sample. This control
    uses the same target *magnitude* and a random tangent direction.
    """
    import torch
    if answer.shape != target.shape:
        raise ValueError('target shape mismatch')
    d=answer.shape[1]
    ln=model.ln_f
    W=model.head.weight*ln.weight[None,:]
    C=torch.eye(d,dtype=answer.dtype,device=answer.device)-1.0/d
    A=W@C
    _,_,Vh=torch.linalg.svd(A,full_matrices=False)
    mean=answer.mean(dim=-1,keepdim=True)
    u=answer-mean
    row=(u@Vh.T)@Vh
    w=u-row
    w_target=(target-mean)-row
    radius=torch.linalg.vector_norm(w,dim=-1,keepdim=True)
    cos=((w*w_target).sum(dim=-1,keepdim=True)/radius.square().clamp_min(1e-30)).clamp(-1,1)
    sin=torch.sqrt((1-cos.square()).clamp_min(0))
    generator=torch.Generator(device='cpu').manual_seed(seed)
    noise=torch.randn(answer.shape,generator=generator,dtype=answer.dtype)
    noise=noise-noise.mean(dim=-1,keepdim=True)
    g=noise-(noise@Vh.T)@Vh
    g=g-w*((g*w).sum(dim=-1,keepdim=True)/radius.square().clamp_min(1e-30))
    g=g/torch.linalg.vector_norm(g,dim=-1,keepdim=True).clamp_min(1e-30)
    rotated=mean+row+cos*w+sin*radius*g
    return rotated


def evaluate_readout_null(model, prompt_seed:int, batch:int=64,
                          depth:int=PHASE_DEPTH, follow:int=FOLLOW_LOOPS)->dict:
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from research_gym.scripts.loopscope_toy import make_batch
    if batch < 1 or depth < 3 or follow < 1:
        raise ValueError('invalid phase monitor configuration')
    horizon = depth + follow + MAX_PERIOD + 2
    tokens, hops = make_batch(N_NODES, batch, prompt_seed, horizon)
    periods = np.array(pointer_periods(tokens),dtype=int)
    with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
        step,h=prepare_step(model,tokens)
        states=[h]
        for _ in range(horizon):
            h=step(h);states.append(h)
        x=states[depth]
        y=states[depth+1]
        delta=y[:,-1,:]-x[:,-1,:]
        row,null,derivative=split_readout_directions(model,x[:,-1,:],delta)
        current=states[depth+follow]
        forward=states[depth+follow+1]
        cycle_residual=np.asarray([
            float(_norm_batch((states[depth+follow+int(p)]-current)[i:i+1])/
                  _norm_batch(current[i:i+1]).clamp_min(1e-30))
            for i,p in enumerate(periods)
        ])
        one_residual=(_norm_batch(forward-current)/_norm_batch(current).clamp_min(1e-30)).numpy()
        stable=(periods>1)&(cycle_residual<1e-4)&(one_residual>0.05)
        reference=[states[depth+follow+j] for j in range(MAX_PERIOD)]
        immediate_logits=model.readout(x[:,-1,:])
        m0=immediate_logits[:,:-1]-immediate_logits[:,-1:]
        target_now=hops[:,depth+follow-1].numpy()
        target_next=hops[:,depth+follow].numpy()
        terms={'original':torch.zeros_like(delta),'full_answer':delta,
               'readout_row':row,'readout_null':null}
        # A conservative full-displacement size match for the null component,
        # recorded separately because it can be out-of-distribution.
        norms=torch.linalg.vector_norm(delta,dim=-1)
        null_norm=torch.linalg.vector_norm(null,dim=-1).clamp_min(1e-30)
        terms['readout_null_normmatched']=null*(norms/null_norm)[:,None]
        # The following transformations preserve every logit MARGIN
        # exactly at intervention time, beyond a first-order approximation.
        for alpha in (0.5,1.0,2.0,4.0,8.0):
            target=exact_readout_fiber_rotate(model,x[:,-1,:],delta,alpha,
                                               preserve='logits')
            terms[f'exact_logits_{alpha:g}']=target-x[:,-1,:]
            if alpha in (4.0,8.0):
                for trial in range(8):
                    ctrl=random_equal_chord_fiber(model,x[:,-1,:],target,
                                  seed=701120+prompt_seed*71+int(alpha)*101+trial)
                    terms[f'random_fiber_{alpha:g}_trial{trial}']=ctrl-x[:,-1,:]
        output={'prompt_seed':prompt_seed,'batch':batch,
                'registered_cycles':int(np.sum(stable)),
                'readout_jacobian':derivative,
                'condition':{}}
        idx=np.flatnonzero(stable)
        for name,change in terms.items():
            pt=x.clone();pt[:,-1,:]+=change
            mt=model.readout(pt[:,-1,:]);m=mt[:,:-1]-mt[:,-1:]
            immediate_margin_change=torch.linalg.vector_norm(m-m0,dim=-1).numpy()
            immediate_logits_change=torch.linalg.vector_norm(mt-immediate_logits,dim=-1).numpy()
            immediate_same=(mt.argmax(-1)==immediate_logits.argmax(-1)).numpy()
            for _ in range(follow):pt=step(pt)
            distances=torch.stack([_norm_batch(pt-z) for z in reference],dim=1).numpy()
            phase=np.array([int(np.argmin(distances[i,:p])) if p>1 else -1
                            for i,p in enumerate(periods)])
            pred=model.readout(pt[:,-1,:]).argmax(-1).numpy()
            output['condition'][name]={
                'n_registered':int(len(idx)),
                'registered_next_phase':int(np.sum(phase[idx]==1)),
                'registered_original_phase':int(np.sum(phase[idx]==0)),
                'registered_next_phase_with_same_initial_label':int(
                    np.sum((phase[idx]==1)&immediate_same[idx])),
                'registered_next_phase_and_correct_next_hop_with_same_initial_label':int(
                    np.sum((phase[idx]==1)&immediate_same[idx]&(pred[idx]==target_next[idx]))),
                'immediate_readout_label_unchanged_all':int(np.sum(immediate_same)),
                'max_abs_logit_margin_change':float((m-m0).abs().max()),
                'immediate_readout_label_unchanged_registered':int(np.sum(immediate_same[idx])),
                'median_margin_change_registered':float(np.median(immediate_margin_change[idx])) if len(idx) else None,
                'max_abs_immediate_logit_change':float((mt-immediate_logits).abs().max()),
                'next_hop_correct_registered':int(np.sum(pred[idx]==target_next[idx])),
                'current_hop_correct_registered':int(np.sum(pred[idx]==target_now[idx])),
                'median_intervention_norm_registered':float(np.median(
                    torch.linalg.vector_norm(change,dim=-1).numpy()[idx])) if len(idx) else None,
            }
    return output


def campaign(checkpoint_dir:str|Path, model_seeds=SEEDS,eval_seeds=EVAL_SEEDS,batch:int=64):
    rows=[]
    for seed in model_seeds:
        model,digest=model_from_receipt(Path(checkpoint_dir)/f'spc04_seed{seed}.pt',seed)
        runs=[evaluate_readout_null(model,prompt_seed, batch=batch) for prompt_seed in eval_seeds]
        rows.append({'seed':seed,'checkpoint_sha256':digest,'heldout':runs})
    return {'campaign':'SPC-06B','description':'Native held-out phase interventions decomposed by immediate readout Jacobian',
            'model_seed_values':list(model_seeds),'heldout_prompt_seeds':list(eval_seeds),
            'frozen_phase_depth':PHASE_DEPTH,'follow_loops':FOLLOW_LOOPS,'rows':rows}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint-dir',type=Path,required=True)
    p.add_argument('--batch',type=int,default=64)
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    r=campaign(a.checkpoint_dir,batch=a.batch)
    s=json.dumps(r,indent=2,sort_keys=True)
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(s+'\n')
    else:print(s)

if __name__=='__main__':main()

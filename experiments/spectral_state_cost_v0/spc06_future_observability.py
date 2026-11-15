"""SPC-06C: finite-difference local observability of trained loop-state fibers.

For a frozen prompt and full hidden state at a chosen recurrent loop, probe
positive/negative perturbations in all 48 answer-token coordinates in parallel.
For k future steps, build the Jacobian of the stacked readout *margins* wrt
the original answer token. The singular spectrum detects hidden directions
visible only after recurrence. Computation is fully native, CPU and float64;
central difference evidence is checked at two step sizes.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from experiments.spectral_state_cost_v0.spc05_phase_interventions import (
    SEEDS, model_from_receipt, prepare_step, pointer_periods,
)
from experiments.spectral_state_cost_v0.spc04_trained_dynamics import N_NODES, D_MODEL, new_model
from experiments.spectral_state_cost_v0.spc06_readout_observability import logit_margin_jacobian


def native_observability(model, prompt_seed: int, depth: int=32, horizon: int=10,
                         eps:float=2e-4):
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from research_gym.scripts.loopscope_toy import make_batch
    if depth<2 or horizon<1 or not (1e-6<=eps<=1e-2):
        raise ValueError('invalid observation configuration')
    tokens,_=make_batch(N_NODES,1,prompt_seed,depth+horizon+1)
    model=model.eval().double()
    with torch.no_grad(),sdpa_kernel([SDPBackend.MATH]):
        F,h=prepare_step(model,tokens)
        for _ in range(depth):h=F(h)
        baseline=h.clone()
        dim=baseline.shape[-1]
        repeated_tokens=tokens.repeat(2*dim,1)
        step,_=prepare_step(model,repeated_tokens)
        perturb=torch.cat((torch.eye(dim,dtype=h.dtype),
                           -torch.eye(dim,dtype=h.dtype)),dim=0)*eps
        states=h.repeat(2*dim,1,1)
        states[:,-1,:]+=perturb
        def margins(z):
            logits=model.readout(z[:,-1,:])
            return logits[:,:-1]-logits[:,-1:]
        def diff_jac(z):
            y=margins(z)
            return ((y[:dim]-y[dim:])/(2*eps)).T.detach().clone()
        k=0
        jacobians=[diff_jac(states)]
        for k in range(horizon):
            states=step(states)
            jacobians.append(diff_jac(states))
        instant=logit_margin_jacobian(model,baseline[:,-1,:])[0]
        fd_error=float(torch.linalg.vector_norm(jacobians[0]-instant)/torch.linalg.vector_norm(instant).clamp_min(1e-30))
        snapshots={}
        for t in (0,1,2,4,8,horizon):
            if t>horizon:continue
            O=torch.cat(jacobians[:t+1],dim=0)
            sigma=torch.linalg.svdvals(O)
            if not bool(torch.isfinite(sigma).all()):
                raise RuntimeError('nonfinite finite-difference observability matrix')
            s=sigma.cpu().numpy()
            max_s=float(s[0]) if len(s) else 0
            snapshots[str(t)]={
                'measurements':t+1,
                'rank_rel_1e-6':int(np.count_nonzero(s>max_s*1e-6)),
                'rank_rel_1e-3':int(np.count_nonzero(s>max_s*1e-3)),
                'largest_sigma':max_s,
                'smallest_sigma':float(s[-1]) if len(s) else None,
                'singular_values':s.tolist(),
            }
    return {'prompt_seed':prompt_seed,'checkpoint_kind':getattr(model,'spc06_kind','unspecified'),
            'true_pointer_cycle_length':int(pointer_periods(tokens)[0]),
            'depth':depth,'future_steps':horizon,'eps':eps,
            'output_margins':N_NODES-1,'answer_hidden_channels':D_MODEL,
            'instant_analytic_vs_fd_relative_error':fd_error,
            'steps':snapshots}


def campaign(checkpoint_dir:str|Path, seeds=SEEDS,
             prompts=(905501,905502),depth:int=32,horizon:int=10,
             eps=(2e-4,1e-4)):
    rows=[]
    for seed in seeds:
        trained,digest=model_from_receipt(Path(checkpoint_dir)/f'spc04_seed{seed}.pt',seed)
        untrained=new_model(seed).eval().double()
        for kind,model in [('trained',trained),('untrained',untrained)]:
            for prompt in prompts:
                checks=[native_observability(model,prompt,depth,horizon,e) for e in eps]
                rows.append({'model_seed':seed,'model_kind':kind,'checkpoint_sha256':digest if kind=='trained' else None,
                             'prompt_seed':prompt,'checks':checks})
    return {'campaign':'SPC-06C','registered_source':'SPC-04 trained checkpoints',
            'note':'central differences estimate local Jacobian; no nonlinear global injectivity claim',
            'rows':rows}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint-dir',type=Path,required=True)
    p.add_argument('--out',type=Path)
    p.add_argument('--depth',type=int,default=32)
    p.add_argument('--horizon',type=int,default=10)
    a=p.parse_args();s=json.dumps(campaign(a.checkpoint_dir,depth=a.depth,horizon=a.horizon),indent=2)
    if a.out:a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(s+'\n')
    else:print(s)

if __name__=='__main__':main()

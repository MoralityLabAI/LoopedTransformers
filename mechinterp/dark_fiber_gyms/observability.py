"""Sampled readout-kernel observability Gramian for a stationary TRM outer step.

Finite angular pairs are *exactly matched at t* in the full lm_head readout
and in per-cell L2 norm. Their subsequent readout differences approximate
columns of D(q o F^j) restricted to head-orthogonal, sphere-tangent directions.
The returned spectrum is a sampled local diagnostic, not a global proof of
dark-fiber dimensionality or path-dependent reasoning.
"""
from __future__ import annotations

import math
import torch

from .core import project_visible, row_basis


@torch.no_grad()
def sphere_tangent_pairs(
    zh: torch.Tensor, head: torch.Tensor, *,
    board_start: int, probes: int = 8, angle: float = 0.04, seed: int = 20261115,
    atol: float = 1e-3,
) -> tuple[torch.Tensor, dict]:
    """Return [2*probes,T,D] paired states (+angle,-angle) on matched readout fibers."""
    if zh.shape[0]!=1 or zh.ndim!=3 or zh.dtype!=torch.float32:
        raise ValueError("requires one float32 [1,T,D] source state")
    if probes<1 or not (0<angle<0.25):
        raise ValueError("invalid local tangent sampling")
    basis=row_basis(head).to(zh.device)
    board=zh[:,board_start:,:]
    vis=project_visible(board,basis)
    orth=board-vis
    radius=orth.norm(dim=-1,keepdim=True)
    if torch.any(radius<1e-5):
        raise ValueError("orthogonal carrier radius too small")
    gen=torch.Generator(device="cpu").manual_seed(seed)
    all_states=[]
    for _ in range(probes):
        rand=torch.randn(tuple(board.shape),generator=gen,dtype=torch.float32).to(zh.device)
        tangent=rand-project_visible(rand,basis)
        tangent=tangent-((tangent*orth).sum(-1,keepdim=True)/radius.square())*orth
        tangent_norm=tangent.norm(dim=-1,keepdim=True)
        if torch.any(tangent_norm<1e-6):
            raise ValueError("tangent collapses; insufficient kernel")
        tangent=tangent/tangent_norm*radius
        for sign in (1,-1):
            out=zh.clone()
            out[:,board_start:,:]=vis+math.cos(angle)*orth+sign*math.sin(angle)*tangent
            all_states.append(out)
    states=torch.cat(all_states,dim=0)
    w=head.float().to(zh.device)
    full_read=states[:,board_start:,:]@w.T
    ref=board@w.T
    leak=float((full_read-ref).abs().max())
    norm_error=float((states[:,board_start:,:].norm(dim=-1)-board.norm(dim=-1)).abs().max())
    if max(leak,norm_error)>atol:
        raise AssertionError(f"gramian tangent pre-readout parity failed: {leak}, {norm_error}")
    return states, {"max_readout_error":leak, "max_l2_error":norm_error,
                    "probes":probes, "angle":angle,"kernel_dim":int(zh.shape[-1]-basis.shape[1])}


@torch.no_grad()
def sampled_trm_gramian(model, zh: torch.Tensor, zl: torch.Tensor, x_emb: torch.Tensor,
                       *, board_start: int, probes: int = 8, angle: float = 0.04,
                       horizon: int = 4, seed: int = 20261115) -> dict:
    """Gramian over fixed random angular tangent directions at one clean state.

    The matrix is probes x probes and its units are squared readout response
    per radian, since all directions rotate each board-cell orthogonal radius.
    """
    if horizon<1:
        raise ValueError("horizon must be >=1")
    states,certificate=sphere_tangent_pairs(
        zh,model.lm_head.weight,board_start=board_start,probes=probes,
        angle=angle,seed=seed)
    batch=2*probes
    zl_batch=zl.expand(batch,-1,-1).clone()
    x_batch=x_emb.expand(batch,-1,-1)
    gram=torch.zeros((probes,probes),dtype=torch.float64,device=zh.device)
    responses=[]
    for j in range(1,horizon+1):
        states,zl_batch=model.act_step(states,zl_batch,x_batch)
        readout=model.lm_head(states)[:,board_start:,:].float()
        # Central differences in the angular fiber chart.
        delta=(readout[0::2]-readout[1::2])/(2*angle)
        flat=delta.reshape(probes,-1).double()
        gram+=flat@flat.T
        responses.append(float(delta.norm(dim=(-1,-2)).mean()))
    gram=(gram+gram.T)/2
    values=torch.linalg.eigvalsh(gram).clamp_min(0).cpu().tolist()[::-1]
    top=values[0] if values else 0.0
    return {
        "schema":"darkfibers.sampled_gramian.v1",
        "certificate":certificate,"horizon":horizon,
        "eigenvalues_descending":[float(v) for v in values],
        "sampled_rank_relative_1e-3":sum(v>top*1e-3 for v in values) if top>0 else 0,
        "per_lag_mean_response":[float(v) for v in responses],
        "frobenius_trace":float(gram.trace()),
    }

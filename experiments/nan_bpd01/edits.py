"""Measured physical-dose edits, keeping dark-arm scored positions byte-identical."""
from __future__ import annotations
import torch

@torch.inference_mode()
def dose_edit(h,direction,relative_dose=.02,tolerance=.005):
    target=float(h.float().norm())*relative_dose
    norm=float(direction.float().norm())
    if norm==0 or target==0:
        return h.clone(),{'target_l2':target,'actual_l2':0.,'relative_error':None,'eligible':False,'zero_dose':True}
    unit=direction.float()/norm;lo=0.;hi=target*2.;best=None
    for _ in range(22):
        scale=(lo+hi)/2
        edited=(h.float()+scale*unit).to(h.dtype)
        actual=float((edited.float()-h.float()).norm());err=abs(actual-target)/target
        if best is None or err<best[0]:best=(err,edited,actual)
        if actual<target:lo=scale
        else:hi=scale
    err,edited,actual=best
    return edited,{'target_l2':target,'actual_l2':actual,'relative_error':err,'eligible':err<=tolerance,'zero_dose':actual==0,
                  'whole_state_norm_ratio':float(edited.float().norm()/h.float().norm()),
                  'scored_tail_identical':torch.equal(edited[:,-3:,:],h[:,-3:,:])}

def phase_direction(h,channel,*,context_width=None,replacement=0.,pattern=None):
    d=torch.zeros_like(h,dtype=torch.float32)
    sl=slice(None) if context_width is None else slice(-3-context_width,-3)
    values=replacement-h[:,sl,channel].float() if pattern is None else pattern
    d[:,sl,channel]=values
    return d

def routed_mask(reached,read_factors,k):
    """Always observes the argument supplied AFTER any intervention."""
    activity=(reached@read_factors).square().sum(tuple(range(reached.ndim-1)))
    result=torch.zeros_like(activity);result[activity.topk(k).indices]=1
    return result

def iter_phase_directions(h,cfg,root,replacement):
    """Same registered directions/order/seeds, with one full tensor live at a time."""
    phase=cfg['phase_channel'];width=cfg['context_width']
    global_pattern=-h[:,:,phase].float();dark_pattern=-h[:,-3-width:-3,phase].float()
    yield 'phase_global_suppress',phase_direction(h,phase)
    yield 'phase_context_suppress',phase_direction(h,phase,context_width=width)
    yield 'phase_context_replace',phase_direction(h,phase,context_width=width,replacement=replacement)
    for c in cfg['neighbor_channels']+cfg['random_channels']:
        yield f'channel{c}_global',phase_direction(h,c,pattern=global_pattern)
    for c in cfg['random_channels']:
        yield f'random_dark_{c}',phase_direction(h,c,context_width=width,pattern=dark_pattern)
    for seed in [741,947]:
        generator=torch.Generator(device=h.device).manual_seed(root+seed)
        dense=torch.zeros_like(h,dtype=torch.float32)
        dense[:,-3-width:-3,:]=torch.randn((16,width,h.shape[-1]),generator=generator,device=h.device)
        yield f'random_dense_dark_{seed}',dense

def shared_random_dark(h,*,width,seed):
    """One context write shared by every candidate row; no future-prefix noise."""
    generator=torch.Generator(device=h.device).manual_seed(seed)
    noise=torch.randn((1,width,h.shape[-1]),generator=generator,device=h.device)
    direction=torch.zeros_like(h,dtype=torch.float32)
    direction[:,-3-width:-3,:]=noise.expand(h.shape[0],-1,-1)
    return direction

def prefix_hidden_error(h):
    one=float((h[:,-3,:].float()-h[0:1,-3,:].float()).abs().max())
    two=max(float((h[4*i:4*i+4,-2,:].float()-h[4*i:4*i+1,-2,:].float()).abs().max()) for i in range(4))
    return max(one,two)

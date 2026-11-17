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

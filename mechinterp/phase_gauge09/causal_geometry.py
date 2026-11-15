"""COE-09 norm-preserving rotations for selected and matched-random interventions.

A dark-fiber intervention preserves the output-head row-space projection
exactly (up to floating point arithmetic). All interventions preserve the
chosen subspace's norm, so injected vectors stay on the model's RMS sphere.
No labels enter the rotation rule.
"""
from __future__ import annotations
from math import cos,sin
import torch


def row_basis(w:torch.Tensor,*,relative_cutoff:float=1e-8)->torch.Tensor:
    if w.ndim!=2 or not torch.isfinite(w).all() or w.shape[0]>=w.shape[1]:
        raise ValueError('invalid output head')
    u,s,vh=torch.linalg.svd(w.to(torch.float64),full_matrices=False)
    rank=int((s>relative_cutoff*s[0]).sum())
    if rank<1:raise ValueError('zero rank head')
    return vh[:rank].T.to(dtype=w.dtype,device=w.device).contiguous()


def project(v:torch.Tensor,basis:torch.Tensor)->torch.Tensor:
    return basis@(basis.T@v)


def rotate_vector(x:torch.Tensor,direction:torch.Tensor,basis:torch.Tensor|None,
                  *,sector:str,angle:float)->torch.Tensor:
    if x.ndim!=1 or direction.shape!=x.shape or not torch.isfinite(x).all() or not torch.isfinite(direction).all():
        raise ValueError('wrong direction shape or nonfinite state')
    if sector not in ('full','visible','dark','scratch') or not 0.0<abs(angle)<=0.20:
        raise ValueError('invalid intervention sector or angle')
    if sector in ('visible','dark') and (basis is None or basis.shape[0]!=x.numel()):
        raise ValueError('projection basis required')
    if sector=='dark':
        p=project(x,basis)
        component=x-p
        u=direction-project(direction,basis)
        remainder=p
    elif sector=='visible':
        component=project(x,basis)
        u=project(direction,basis)
        remainder=x-component
    else:
        component=x
        remainder=torch.zeros_like(x)
        u=direction
    r=component.norm()
    if r<1.e-7:raise ValueError('cannot rotate a zero-norm sector')
    tangent=u-(u@component)/(r*r)*component
    mag=tangent.norm()
    if mag<1.e-8:raise ValueError('selected direction lacks transverse component')
    tangent=tangent/mag
    result=remainder+float(cos(angle))*component+float(sin(angle))*r*tangent
    if not torch.isfinite(result).all():raise ValueError('nonfinite rotation')
    return result


def prepare_direction(vec:torch.Tensor,model_head:torch.Tensor,sector:str)->torch.Tensor:
    if sector=='visible':
        if vec.numel()!=model_head.shape[0]:raise ValueError('visible readout is in logit space')
        # Minimum Euclidean norm preimage in the model-head row space.
        return torch.linalg.pinv(model_head.to(torch.float64)).to(vec.dtype)@vec
    if vec.numel()!=model_head.shape[1]:raise ValueError('latent direction mismatch')
    return vec


def geometry_certificate(old:torch.Tensor,new:torch.Tensor,head:torch.Tensor,sector:str)->dict:
    oldnorm=float(old.norm());newnorm=float(new.norm())
    logits=float((head@(new-old)).abs().max()) if sector in ('dark','visible','full') else None
    return {'abs_norm_change':abs(newnorm-oldnorm),
            'relative_norm_change':abs(newnorm-oldnorm)/max(oldnorm,1e-8),
            'max_head_logit_change':logits,
            'displacement_l2':float((new-old).norm())}

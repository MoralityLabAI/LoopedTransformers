"""LGA-10 source-only curvature-matched shams in future dark pair fibers.

Protocol 95d9c6ab precedes implementation. No original models or seed 31064.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import math
from typing import Callable
import torch
from torch import Tensor

NEW_SEED=31064
EXCLUDED_SEED=61064
GOALS=(0,10,20,30,40,50,60,70)
N_SHAMS=8
MAX_ATTEMPTS=32
N_ANGLES=64
ANGLE_COS_MAX=.85
CURVATURE_RTOL=1e-8


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(
        ("LGA10|"+"|".join(map(str,parts))).encode()).digest()[:8],
        "big")&((1<<63)-1)


def finite_pair(t):
    x=t.detach().double().reshape(-1)
    if x.numel()!=96 or not bool(torch.isfinite(x).all()):
        raise ValueError("invalid 96D Euclidean physical pair")
    return x


@dataclass
class CurvaturePlan:
    eligible: bool
    reason: str | None
    shams: list[Tensor]
    angles: list[float]
    attempts_used: list[int]
    selected_curvature: float
    null_radius: float
    max_curvature_error: float
    max_jacobian_error: float
    max_dose_error: float


def quadratic_at(theta,z0,q0,v,hz0,hq0,hv,radius):
    c,s=math.cos(theta),math.sin(theta)
    z=z0+c*q0+(radius*s)*v
    hz=hz0+c*hq0+(radius*s)*hv
    return float(z@hz)


def root_angle(z0,q0,v,hz0,hq0,hv,r,target):
    step=2*math.pi/N_ANGLES
    points=[(step*i,quadratic_at(step*i,z0,q0,v,hz0,hq0,hv,r)-target)
            for i in range(N_ANGLES+1)]
    for (lo,fl),(hi,fh) in zip(points[:-1],points[1:]):
        if abs(math.cos(lo))>ANGLE_COS_MAX or abs(math.cos(hi))>ANGLE_COS_MAX:
            continue
        if fl*fh>0:
            continue
        if fl==0:
            angle=lo
        elif fh==0:
            angle=hi
        else:
            for _ in range(64):
                mid=(lo+hi)/2
                fm=quadratic_at(mid,z0,q0,v,hz0,hq0,hv,r)-target
                if fl*fm<=0:
                    hi,fh=mid,fm
                else:
                    lo,fl=mid,fm
            angle=(lo+hi)/2
        if abs(math.cos(angle))<=ANGLE_COS_MAX:
            return angle
    return None


def make_curvature_shams(selected:Tensor,row_basis:Tensor,
                         project_K:Callable[[Tensor],Tensor],
                         hvp:Callable[[Tensor],Tensor],*,
                         teacher_seed:int,base_index:int,goal:int,
                         n_shams=N_SHAMS,max_attempts=MAX_ATTEMPTS):
    """Preserve K, radius, all ten first derivatives, and signed margin curvature."""
    z=finite_pair(selected)
    R=row_basis.detach().double()
    if R.ndim!=2 or R.shape[0]!=96 or not 0<=R.shape[1]<=10:
        raise ValueError("full future Jacobian rowspace basis must be 96xrank<=10")
    if not bool(torch.isfinite(R).all()):
        raise ValueError("nonfinite future rowspace basis")
    if R.shape[1] and float(torch.linalg.vector_norm(R.T@R-
              torch.eye(R.shape[1],dtype=torch.float64)))>1e-7:
        raise ValueError("future rowspace basis not orthonormal")
    radius=float(torch.linalg.vector_norm(z))
    z0=R@(R.T@z)
    q0=z-z0
    r=float(torch.linalg.vector_norm(q0))
    if r<=1e-10 or radius<=1e-10:
        return CurvaturePlan(False,"DEGENERATE_FIRST_ORDER_DARK_RADIUS",
                             [],[],[],0.,r,0.,0.,0.)
    if float(torch.linalg.vector_norm(finite_pair(project_K(z))-z))>1e-7*max(1.,radius):
        raise ValueError("selected pair direction outside K")
    hz0=finite_pair(hvp(z0))
    hq0=finite_pair(hvp(q0))
    target=float(z@(hz0+hq0))
    shams=[];angles=[];attempts=[]
    maxq=maxj=maxdose=0.
    for i in range(n_shams):
        found=False
        for attempt in range(max_attempts):
            seed=stable_seed("curvature",teacher_seed,base_index,goal,i,attempt)
            draw=torch.randn((96,),dtype=torch.float64,
                  generator=torch.Generator().manual_seed(seed))
            v=finite_pair(project_K(draw))
            v=v-R@(R.T@v)
            v=v-q0*(float(q0@v)/(r*r))
            vn=float(torch.linalg.vector_norm(v))
            if vn<=1e-11:
                continue
            v/=vn
            hv=finite_pair(hvp(v))
            angle=root_angle(z0,q0,v,hz0,hq0,hv,r,target)
            if angle is None:
                continue
            c,s=math.cos(angle),math.sin(angle)
            trial=z0+c*q0+(r*s)*v
            curv=float(trial@(hz0+c*hq0+(r*s)*hv))
            qerr=abs(curv-target)
            dose_err=abs(float(torch.linalg.vector_norm(trial))-radius)
            jerr=float(torch.linalg.vector_norm(R.T@(trial-z)))
            kerr=float(torch.linalg.vector_norm(finite_pair(project_K(trial))-trial))
            if (qerr>CURVATURE_RTOL*max(1.,abs(target),abs(curv)) or
                dose_err>1e-7*max(1.,radius) or
                jerr>1e-8*max(1.,radius) or
                kerr>1e-7*max(1.,radius) or
                abs(c)>ANGLE_COS_MAX):
                raise ArithmeticError("curvature sham violates source-only numerical constraints")
            shams.append(trial.reshape(2,48))
            angles.append(angle)
            attempts.append(attempt+1)
            maxq=max(maxq,qerr)
            maxj=max(maxj,jerr)
            maxdose=max(maxdose,dose_err)
            found=True
            break
        if not found:
            return CurvaturePlan(False,"INELIGIBLE_HESSIAN_BRACKET",
                      [],angles,attempts,target,r,maxq,maxj,maxdose)
    return CurvaturePlan(True,None,shams,angles,attempts,
                         target,r,maxq,maxj,maxdose)


def native_margin_hvp(teacher,reached_h,goal,donor_class,original_class):
    """HVP for native two-block signed donor margin at clean physical h2.

    Uses the original model dtype; source-only matching uses FP64 arithmetic
    on returned HVP vectors. The teacher weights and input state are frozen.
    """
    if (reached_h.shape!=(1,81,48) or goal not in GOALS or
        donor_class==original_class or not 0<=donor_class<10 or
        not 0<=original_class<10):
        raise ValueError("legal native source/margin shape invalid")
    p=(goal-1)%81
    base=reached_h.detach()
    def margin(edit):
        state=base.clone()
        state[0,p]=state[0,p]+edit[0]
        state[0,goal]=state[0,goal]+edit[1]
        out=teacher.head(teacher.block(teacher.block(state)))[0,goal]
        return out[donor_class]-out[original_class]
    def hvp(v):
        x=torch.zeros((2,48),dtype=base.dtype,device=base.device)
        w=finite_pair(v).reshape(2,48).to(dtype=base.dtype,device=base.device)
        with torch.enable_grad():
            _,res=torch.autograd.functional.hvp(margin,x,w,strict=False)
        return finite_pair(res)
    return hvp

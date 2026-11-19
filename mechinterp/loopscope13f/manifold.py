"""LGA-10F: fixed-budget curvature-level-set donors in the source tangent fiber.

No model or source initialization occurs here. FP64 dense Hessian is used as
a *construction* oracle, so this mechanism is explicitly mixed-precision.
"""
from __future__ import annotations

import hashlib
import math

import torch

SEED=61069
SHAPES=(96,64)
S_VALUES=(-0.12,-0.06,-0.03,-0.015,0.015,0.03,0.06,0.12)
N_DIRECTIONS=4
T_BOUND=0.12
BISECTION_STEPS=48
LEVEL_TOL=1e-11


def stable_seed(*fields):
    raw=("LGA10F|"+"|".join(map(str,fields))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8],"big")&((1<<63)-1)


def l2(a):
    return float(torch.linalg.vector_norm(a.double()))


def curvature(x,H):
    return float(x@(H@x))


def prepare_levelset(z,R,V,H):
    z=z.detach().double().reshape(96)
    R=R.detach().double()
    V=V.detach().double()
    H=H.detach().double()
    if (R.ndim!=2 or R.shape[0]!=96 or R.shape[1]>10 or
        V.shape!=SHAPES or H.shape!=(96,96)):
        raise ValueError("curvature source dimension or complement changed")
    if not all(bool(torch.isfinite(x).all()) for x in (z,R,V,H)):
        raise ValueError("nonfinite curvature source")
    if l2(H-H.T)>1e-8*max(1.,l2(H)):
        raise ValueError("unvalidated nonsymmetric dense Hessian")
    if l2(R.T@R-torch.eye(R.shape[1],dtype=torch.float64))>1e-8:
        raise ValueError("invalid row basis")
    if l2(V.T@V-torch.eye(V.shape[1],dtype=torch.float64))>1e-8:
        raise ValueError("invalid dark complement")
    if l2(R.T@V)>1e-8:
        raise ValueError("dark complement overlaps first-order row space")
    fixed=R@(R.T@z)
    q=z-fixed
    r=l2(q)
    if r<=1e-12:
        raise ValueError("source dark radius collapsed")
    u=q/r
    if l2(V.T@u)>1e-8:
        raise ValueError("dark complement overlaps natural donor dark axis")
    g=V@(V.T@(H@z))
    gn=l2(g)
    if gn>1e-14:
        e=g/gn
    else:
        e=None
    return {"z":z,"R":R,"V":V,"H":H,"fixed":fixed,
            "u":u,"r":r,"e":e,"grad_norm":gn,
            "target_dense":curvature(z,H)}


def propose(s,t,geom,w):
    r=geom["r"]
    direction=geom["u"]+s*w+t*geom["e"]
    return geom["fixed"]+r*direction/math.sqrt(1.+s*s+t*t)


def solve_offset(s,geom,w):
    """Fixed 48-step bisection on a physically specified 2-plane circle."""
    if geom["e"] is None:
        return None,{"reason":"DEGENERATE_CURVATURE_TANGENT"}
    target=geom["target_dense"]
    def f(t):
        x=propose(s,t,geom,w)
        return curvature(x,geom["H"])-target
    f0=f(0.)
    tol=LEVEL_TOL*max(1.,abs(target))
    # If a numerically flat local level set already meets the contract,
    # t=0 is a deterministic valid root.
    if abs(f0)<=tol:
        return 0.,{"reason":None,"residual":abs(f0),"bisections":0,
                  "bracket":(0.,0.)}
    lo=-T_BOUND if f0>0 else 0.
    hi=0. if f0>0 else T_BOUND
    flo,fhi=f(lo),f(hi)
    if not (math.isfinite(flo) and math.isfinite(fhi)):
        return None,{"reason":"NONFINITE_ROOT_BRACKET","f0":f0,
                     "flo":flo,"fhi":fhi}
    if flo*fhi>0:
        return None,{"reason":"CURVATURE_ROOT_UNBRACKETED",
                     "f0":f0,"flo":flo,"fhi":fhi}
    for _ in range(BISECTION_STEPS):
        mid=(lo+hi)/2.
        fm=f(mid)
        if not math.isfinite(fm):
            return None,{"reason":"NONFINITE_ROOT_VALUE"}
        if flo*fm<=0:
            hi=mid
        else:
            lo,flo=mid,fm
    t=(lo+hi)/2.
    residual=abs(f(t))
    if residual>tol:
        return None,{"reason":"CURVATURE_ROOT_RESIDUAL","residual":residual}
    return t,{"reason":None,"residual":residual,"bisections":BISECTION_STEPS,
              "bracket":(-T_BOUND if f0>0 else 0.,
                         0. if f0>0 else T_BOUND)}


def levelset_candidates(z,R,V,H,*,base,goal,seed=SEED):
    """Yield candidate zero plus all 32 registered offsets or failed-root rows.

    Each yielded item has index, optional candidate, and the full root receipt.
    Candidate indices are stable across changed feasibility outcomes.
    """
    geo=prepare_levelset(z,R,V,H)
    yield 0,geo["z"].clone(),{
        "kind":"natural","direction":None,"s":0.,"t":0.,
        "root_reason":None,"root_residual":0.,
        "curvature_gradient_norm":geo["grad_norm"],
    }
    for j in range(N_DIRECTIONS):
        direction_seed=stable_seed(seed,base,goal,j)
        gen=torch.Generator().manual_seed(direction_seed)
        ambient=torch.randn(96,dtype=torch.float64,generator=gen)
        w=geo["V"]@(geo["V"].T@ambient)
        reason=None
        if geo["e"] is None:
            reason="DEGENERATE_CURVATURE_TANGENT"
        else:
            w=w-geo["e"]*float(geo["e"]@w)
            wn=l2(w)
            if wn<=1e-12:
                reason="DEGENERATE_LEVELSET_DIRECTION"
            else:
                w=w/wn
        for k,s in enumerate(S_VALUES):
            idx=1+j*len(S_VALUES)+k
            header={"kind":"levelset","direction":j,"seed":direction_seed,
                    "s":s,"curvature_gradient_norm":geo["grad_norm"]}
            if reason is not None:
                yield idx,None,{**header,"root_reason":reason}
                continue
            t,receipt=solve_offset(s,geo,w)
            if t is None:
                yield idx,None,{**header,"t":None,
                                "root_reason":receipt["reason"],
                                "root_receipt":receipt}
                continue
            candidate=propose(s,t,geo,w)
            dense_error=abs(curvature(candidate,H)-geo["target_dense"])
            yield idx,candidate,{**header,"t":t,
                "root_reason":None,"root_residual":dense_error,
                "root_receipt":receipt}

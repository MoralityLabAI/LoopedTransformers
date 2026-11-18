"""LGA-09 source-only full ten-logit future Jacobian null, before 31063.

Pair K and LGA-08 source-only candidate are frozen. The extra control matches
every FIRST-ORDER future digit logit at natural source donor pair radius.
"""
from __future__ import annotations
import hashlib
import math
from dataclasses import dataclass
import torch

from mechinterp.loopscope11.pair_fiber import (
    GOALS,SHAMS as OLD_SHAMS,pair_project,project_into_k,
    plan_pair,normalized,norm
)

NEW_SEED=31063
SMOKE_SEED=61063
N_BASE=64
TEACHERS=(103,211,307)
SHAMS=8
RTOL=1e-9
ATOL=1e-12
ARMS=(
    "pair_donor_k",
    *(f"full_j_sham_{i}" for i in range(8)),
    *(f"margin_sham_{i}" for i in range(8)),
    *(f"random_k_sham_{i}" for i in range(8)),
    "pair_head_only","natural_pair",
    "goal_only_natural","predecessor_only_natural",
    "pair_quarter","full_donor_state"
)


def stable_seed(*parts):
    payload=("LGA09|"+"|".join(map(str,parts))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8],"big") & ((1<<63)-1)


def future_ten_jacobian(teacher,h,goal):
    """Exact ten class gradients wrt two current source tokens, native h2→h4."""
    if h.shape!=(1,81,48) or goal not in GOALS:
        raise ValueError("registered source h2 and goal required")
    state=h.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        z=teacher.block(teacher.block(state))
        logits=teacher.head(z)[0,goal]
        p=(goal-1)%81
        gradients=[]
        for c in range(10):
            whole,=torch.autograd.grad(logits[c],state,retain_graph=c<9)
            gradients.append(whole[0,[p,goal]].detach().double())
    j=torch.stack(gradients)  # [10,2,48]
    if j.shape!=(10,2,48) or not bool(torch.isfinite(j).all()):
        raise ArithmeticError("full native future Jacobian invalid")
    return j,logits.detach().double()


@dataclass
class FullFutureFiber:
    full_jacobian:torch.Tensor
    restricted_jacobian:torch.Tensor
    row_basis:torch.Tensor
    singular_values:torch.Tensor
    rank:int
    cutoff:float
    future_null_dimension:int
    minimal_rowspace:torch.Tensor
    orthogonal_radius:float
    selected_full_logit_derivative:torch.Tensor
    max_physical_residual:float


def measure_full_response(teacher,h,goal,base_plan,head_projector,
                          *,rtol=RTOL,atol=ATOL):
    """Compute full ten-logit output derivative and source-donor rowspace."""
    if not base_plan.get("eligible"):
        raise ValueError("pair source-only geometry not admitted")
    sel=base_plan["directions"]["pair_energy_tangent"].detach().double()
    f=base_plan["fiber"]
    j,logits=future_ten_jacobian(teacher,h,goal)
    jr=torch.stack([project_into_k(j[c],f,head_projector).reshape(-1)
                    for c in range(10)])
    _,sv,vh=torch.linalg.svd(jr,full_matrices=False)
    cutoff=max(atol,rtol*float(sv[0]))
    rank=int((sv>cutoff).sum())
    if rank<1:
        raise ValueError("future Jacobian has no identifiable physical K row")
    basis=vh[:rank].T.contiguous()
    flat=sel.reshape(-1)
    z0=basis@(basis.T@flat)
    radius=base_plan["radius"]
    left=radius*radius-float((z0*z0).sum())
    if left<0 and abs(left)<=1e-7*max(1.,radius*radius):
        left=0.
    if left<=1e-10*radius*radius:
        raise ValueError("full-J matching has degenerate orthogonal radius")
    ref=jr@flat
    physical_res=max(
        norm(project_into_k(sel,f,head_projector)-sel),
        float(torch.linalg.vector_norm(jr@z0-ref))
    )
    if physical_res>1e-6*max(1.,radius,float(ref.norm())):
        raise ArithmeticError("minimal full-J rowspace source response failed")
    return FullFutureFiber(
        full_jacobian=j,restricted_jacobian=jr,
        row_basis=basis,singular_values=sv,
        rank=rank,cutoff=cutoff,
        future_null_dimension=f.dimension-rank,
        minimal_rowspace=z0.reshape(2,48),
        orthogonal_radius=math.sqrt(left),
        selected_full_logit_derivative=ref,
        max_physical_residual=physical_res
    )


def full_j_matched_sham(future:FullFutureFiber,base_plan,head_projector,*,seed):
    """Same K, L2, and ALL TEN future-logit first derivatives as source donor."""
    f=base_plan["fiber"]
    q=torch.randn((2,48),generator=torch.Generator().manual_seed(seed),
                  dtype=torch.float64)
    q=project_into_k(q,f,head_projector).reshape(-1)
    basis=future.row_basis
    q=q-basis@(basis.T@q)
    qnorm=float(torch.linalg.vector_norm(q))
    if qnorm<=1e-10:
        raise ArithmeticError("full-J null Gaussian direction collapsed")
    z=future.minimal_rowspace.reshape(-1)+q*(future.orthogonal_radius/qnorm)
    original=base_plan["directions"]["pair_energy_tangent"].reshape(-1).double()
    d=torch.linalg.vector_norm(future.restricted_jacobian@(z-original))
    denom=max(1.,float(torch.linalg.vector_norm(future.selected_full_logit_derivative)))
    if float(d)>1e-8*denom:
        raise ArithmeticError("sham does not preserve the entire 10-logit first-order response")
    if abs(float(z.norm())-base_plan["radius"])>1e-7*max(1.,base_plan["radius"]):
        raise ArithmeticError("full-J sham physical dose mismatch")
    restored=project_into_k(z.reshape(2,48),f,head_projector).reshape(-1)
    if float((restored-z).norm())>1e-7*max(1.,base_plan["radius"]):
        raise ArithmeticError("full-J sham not in head-and-energy physical pair K")
    return z.reshape(2,48)


def plan_full_j(model,energy,hrec,hdon,mu,std,*,goal,original_class,
                donor_class,teacher_seed,base_id,head_projector,cached_egrad=None):
    """All label-blind full-J and target-aware margin control plans, pre-scoring."""
    parent=plan_pair(
        model,energy,hrec,hdon,mu,std,goal=goal,
        old_class=original_class,donor_class=donor_class,
        teacher_seed=teacher_seed,base_index=base_id,
        head_projector=head_projector,cached_egrad=cached_egrad
    )
    eligible=bool(parent["eligible"])
    reason=parent["reason"]
    future=None
    if eligible:
        try:
            future=measure_full_response(model,hrec,goal,parent,head_projector)
            full_shams={
                f"full_j_sham_{i}":full_j_matched_sham(
                    future,parent,head_projector,
                    seed=stable_seed("full",teacher_seed,base_id,goal,i))
                for i in range(SHAMS)
            }
        except (ValueError,ArithmeticError,RuntimeError) as error:
            eligible=False
            reason="INELIGIBLE_FULL_J_MATCH"
            future=None
    directions={
        "natural_pair":parent["directions"]["pair_natural"],
        "goal_only_natural":parent["directions"]["goal_natural"],
        "predecessor_only_natural":parent["directions"]["predecessor_natural"],
        "full_donor_state":parent["directions"]["full_donor_state"],
    }
    if eligible:
        directions["pair_donor_k"]=parent["directions"]["pair_energy_tangent"]
        directions.update(full_shams)
        directions.update({f"margin_sham_{i}":parent["directions"][f"pair_gradient_match_{i}"]
                           for i in range(SHAMS)})
        directions.update({f"random_k_sham_{i}":parent["directions"][f"pair_random_k_{i}"]
                           for i in range(SHAMS)})
        directions["pair_head_only"]=parent["directions"]["pair_head_only"]
        directions["pair_quarter"]=parent["directions"]["pair_tangent_quarter"]
    else:
        for arm in ARMS:
            if arm not in directions:
                directions[arm]=torch.zeros((2,48),dtype=torch.float64)
    if set(directions)!=set(ARMS):
        raise ArithmeticError("LGA09 frozen full-J arm roster changed")
    return {
        "eligible":eligible,"reason":reason,
        "pair_plan":parent,"future":future,
        "directions":directions,
        "natural_pair_radius":parent["radius"],
        "rank_full_j":future.rank if future is not None else None,
        "future_null_dimension":future.future_null_dimension if future is not None else None,
    }

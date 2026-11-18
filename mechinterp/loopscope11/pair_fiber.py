"""LGA-08 source-only pair-fiber planning, before opening reserved 31062.

Registered in PROTOCOL.md at commit 538f01ad before implementation. Actual
outcome evaluation lives elsewhere. Original model weights never mutate.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import math
import torch

from mechinterp.loopscope10.global_energy_fiber import full_head_kernel
from mechinterp.loopscope10.energy_fiber import source_energy_gradient
from mechinterp.loopscope10.donor_source import legal_predecessor_donor

BASE_COUNT = 64
RESERVED_SEED = 31062
EXCLUDED_SEED = 61062
GOALS = (0, 10, 20, 30, 40, 50, 60, 70)
TEACHERS = (103, 211, 307)
RTOL, ATOL = 1e-9, 1e-12
SHAMS = 8
ARMS = (
    "pair_energy_tangent",
    *(f"pair_random_k_{i}" for i in range(SHAMS)),
    *(f"pair_gradient_match_{i}" for i in range(SHAMS)),
    "pair_head_only",
    "pair_natural",
    "goal_natural",
    "predecessor_natural",
    "pair_tangent_quarter",
    "full_donor_state",
)


def stable_seed(*parts):
    x = ("LGA08|" + "|".join(map(str,parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(x).digest()[:8], "big") & ((1<<63)-1)


def norm(t):
    return float(torch.linalg.vector_norm(t))


def normalized(t, radius, eps=1e-11):
    n=norm(t)
    if n<=eps or radius<=eps or not math.isfinite(radius):
        raise ValueError("nonpositive physical edit dose")
    z=t * (radius/n)
    if not bool(torch.isfinite(z).all()):
        raise ArithmeticError("nonfinite normalized pair edit")
    return z


@dataclass
class PairFiber:
    donor: torch.Tensor
    egrad: torch.Tensor
    head_only: torch.Tensor
    k_donor: torch.Tensor
    egrad_head: torch.Tensor
    energy_active: bool
    current_head_rank: int
    dimension: int
    natural_radius: float
    tangent_fraction: float
    head_fraction: float
    tangent_residual: float
    head_residual: float


def pair_project(u, g, p, *, rtol=RTOL, atol=ATOL):
    """Real two-token Euclidean right kernel intersection."""
    if u.shape!=(2,48) or g.shape!=u.shape or p.shape!=(48,48):
        raise ValueError("pair and full-head projector have incompatible physical shape")
    a=u.detach().double()
    b=g.detach().double()
    p=p.detach().double()
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all() and torch.isfinite(p).all()):
        raise ValueError("nonfinite pair physical projection")
    uhead=a @ p
    ghead=b @ p
    gnorm=norm(ghead)
    energy_active=bool(gnorm>max(atol,rtol*norm(b)))
    if energy_active:
        uk=uhead-((uhead*ghead).sum()/(ghead*ghead).sum())*ghead
    else:
        uk=uhead
    head_res=norm(uk@(torch.eye(48,dtype=torch.float64)-p))
    e_res=float(torch.sum(uk*b))
    if head_res>1e-7*max(1.,norm(uk)) or abs(e_res)>1e-7*max(1.,norm(uk)*norm(b)):
        raise ArithmeticError("pair head/energy tangent residual too large")
    radius=norm(a)
    return PairFiber(
        donor=a,egrad=b,head_only=uhead,k_donor=uk,egrad_head=ghead,
        energy_active=energy_active,current_head_rank=10,
        dimension=2*(48-10)-int(energy_active),
        natural_radius=radius,
        tangent_fraction=norm(uk)/max(radius,1e-30),
        head_fraction=norm(uhead)/max(radius,1e-30),
        tangent_residual=e_res,head_residual=head_res,
    )


def project_into_k(x, f:PairFiber, p):
    """Use the same current-head and energy constraints as source donor."""
    y=x.detach().double()@p.double()
    if f.energy_active:
        g=f.egrad_head
        y=y-((y*g).sum()/(g*g).sum())*g
    return y


def future_legal_margin_gradient(model, h, goal, old_class, donor_class):
    """Target-aware first-order ORACLE, using only legal input source classes.

    This never reads intervention outcomes or picks the rival from edited logits.
    """
    if (h.shape!=(1,81,48) or goal not in GOALS or
            old_class==donor_class or not 0<=old_class<10 or
            not 0<=donor_class<10):
        raise ValueError("future-margin gradient requires a legal class pair")
    state=h.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        z=model.block(model.block(state))
        logits=model.head(z)[0,goal]
        margin=logits[donor_class]-logits[old_class]
        grad,=torch.autograd.grad(margin,state)
    pair=(goal-1)%81,goal
    return grad.detach()[0,list(pair)].double(),float(margin.detach())


def matched_future_margin_sham(selected, future_gradient, f:PairFiber, projector,
                               radius, *, seed):
    """Same K, L2, and signed future-margin directional derivative as selected."""
    gf=project_into_k(future_gradient,f,projector)
    gn2=float((gf*gf).sum())
    if gn2<=1e-20:
        raise ValueError("degenerate signed future gradient in K")
    dot=float((selected*gf).sum())
    fixed=gf*(dot/gn2)
    free2=max(0.,radius*radius-float((fixed*fixed).sum()))
    if free2<=1e-10*radius*radius:
        raise ValueError("degenerate radius after signed-gradient matching")
    r=torch.randn((2,48),generator=torch.Generator().manual_seed(seed),
                  dtype=torch.float64)
    perp=project_into_k(r,f,projector)
    perp=perp-((perp*gf).sum()/gn2)*gf
    if norm(perp)<=1e-10:
        raise ArithmeticError("random signed-gradient-matched null collapsed")
    z=fixed+perp*(math.sqrt(free2)/norm(perp))
    if abs(norm(z)-radius)>1e-7*max(1.,radius):
        raise ArithmeticError("signed-gradient sham physical dose incorrect")
    if abs(float((z*gf).sum())-dot)>1e-8*max(1.,abs(dot),norm(gf)*radius):
        raise ArithmeticError("signed future margin gradient matching incorrect")
    return z


def plan_pair(model, energy, hrec, hdon, mu, std, *, goal, old_class,
              donor_class, teacher_seed, base_index, head_projector,
              cached_egrad=None):
    """Fixed source-only candidate and target-aware oracle controls before scoring."""
    if hrec.shape!=(1,81,48) or hdon.shape!=hrec.shape or goal not in GOALS:
        raise ValueError("one legal recipient/donor reached-state pair required")
    pair=((goal-1)%81,goal)
    difference=(hdon-hrec)[0,list(pair)].detach().double()
    gradient=(source_energy_gradient(energy,hrec,mu,std) if cached_egrad is None
              else cached_egrad)[0,list(pair)].detach().double()
    f=pair_project(difference,gradient,head_projector)
    selected_norm=norm(f.k_donor)
    eligible=bool(f.natural_radius>1e-10 and selected_norm>1e-10
                  and norm(f.head_only)>1e-10 and f.energy_active
                  and f.current_head_rank==10 and f.dimension==75)
    directions={}
    if eligible:
        sel=normalized(f.k_donor,f.natural_radius)
        dirs_random={}
        for i in range(SHAMS):
            q=torch.randn((2,48),generator=torch.Generator().manual_seed(
                stable_seed("random",teacher_seed,base_index,goal,i)),dtype=torch.float64)
            dirs_random[f"pair_random_k_{i}"]=normalized(
                project_into_k(q,f,head_projector),f.natural_radius)
        # Freeze model-blind candidate and random-K before target-aware oracle.
        future,margin=future_legal_margin_gradient(
            model,hrec,goal,old_class,donor_class)
        gf=project_into_k(future,f,head_projector)
        if norm(gf)<=1e-10:
            eligible=False
            reason="DEGENERATE_GRADIENT_MATCH"
        else:
            dirs_matched={}
            try:
                for i in range(SHAMS):
                    dirs_matched[f"pair_gradient_match_{i}"]=matched_future_margin_sham(
                        sel,future,f,head_projector,f.natural_radius,
                        seed=stable_seed("match",teacher_seed,base_index,goal,i))
            except (ValueError, ArithmeticError):
                eligible=False
                reason="DEGENERATE_GRADIENT_MATCH"
    else:
        future=torch.zeros_like(difference)
        gf=torch.zeros_like(difference)
        margin=None
        reason="DEGENERATE_PAIR_TANGENT_OR_RADIUS"
    if eligible:
        directions["pair_energy_tangent"]=sel
        directions.update(dirs_random)
        directions.update(dirs_matched)
        directions["pair_head_only"]=normalized(f.head_only,f.natural_radius)
        directions["pair_tangent_quarter"]=sel*0.25
    else:
        for name in ("pair_energy_tangent",
                     *(f"pair_random_k_{i}" for i in range(SHAMS)),
                     *(f"pair_gradient_match_{i}" for i in range(SHAMS)),
                     "pair_head_only","pair_tangent_quarter"):
            directions[name]=torch.zeros((2,48),dtype=torch.float64)
    directions["pair_natural"]=f.donor.clone()
    directions["goal_natural"]=torch.stack((torch.zeros(48,dtype=torch.float64),
                                            f.donor[1]))
    directions["predecessor_natural"]=torch.stack((f.donor[0],
                                                   torch.zeros(48,dtype=torch.float64)))
    directions["full_donor_state"]=f.donor.clone()  # separate WHOLE-STATE control in scorer
    if set(directions)!=set(ARMS):
        raise ArithmeticError("registered pair arm roster changed")
    if eligible:
        target=float((sel*gf).sum())
        for name,z in directions.items():
            if name.startswith("pair_gradient_match_"):
                if abs(float((z*gf).sum())-target)>1e-7*max(1.,abs(target)):
                    raise ArithmeticError("signed future margin sham mismatch")
    return {
        "eligible":bool(eligible),
        "reason":None if eligible else reason,
        "fiber":f,
        "directions":directions,
        "future_gradient":future,
        "future_k_gradient":gf,
        "clean_future_margin":margin,
        "selected_signed_derivative":float((directions["pair_energy_tangent"]*gf).sum()),
        "pair_positions":list(pair),
        "radius":f.natural_radius,
    }

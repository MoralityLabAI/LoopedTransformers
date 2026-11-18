"""LGA-07: first-order energy-tangent dark-channel instrument on native states.

This file performs source-only math or label-blind native planning. Stage-B
reserved seed 31059 is never generated here. No causal science verdict is
produced. See prospectively committed PROTOCOL.md.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import math
import torch
from torch import Tensor

RANK_RTOL = 1e-9
RANK_ATOL = 1e-12
N_SHAMS = 8


@dataclass
class TokenFiber:
    basis: Tensor
    projector: Tensor
    rank_current: int
    rank_joint: int
    null_dimension: int
    lower_bound: int
    singular_values: Tensor
    cutoff: float
    null_residual: float
    projector_error: float


def _rank_from_svd(s: Tensor, *, rtol=RANK_RTOL, atol=RANK_ATOL):
    if s.numel() == 0:
        return 0, atol
    cutoff = max(float(atol), float(rtol) * float(s[0]))
    return int((s > cutoff).sum()), cutoff


def token_energy_fiber(head_weight: Tensor, energy_gradient: Tensor, *,
                       rtol=RANK_RTOL, atol=RANK_ATOL) -> TokenFiber:
    """Head-logit-null AND frozen energy-tangent directions for one token."""
    if (head_weight.ndim != 2 or energy_gradient.ndim != 1 or
        head_weight.shape[1] != energy_gradient.numel() or
        energy_gradient.numel() < 1):
        raise ValueError("expected [classes,width] and [width] physical inputs")
    a = head_weight.detach().double()
    g = energy_gradient.detach().double()
    if not bool(torch.isfinite(a).all() and torch.isfinite(g).all()):
        raise ValueError("nonfinite physical constraints")
    # A raw SVD of [head; gradient] can erase head constraints when the
    # gradient has far larger units. First find the head-null PHYSICAL space,
    # after individual nonzero head-row normalization, then intersect g.
    row_norms = torch.linalg.vector_norm(a, dim=1, keepdim=True)
    active = row_norms > atol
    a_unit = torch.where(active, a / row_norms.clamp_min(atol),
                         torch.zeros_like(a))
    _, head_s, head_vh = torch.linalg.svd(a_unit, full_matrices=True)
    rank_current, _ = _rank_from_svd(head_s, rtol=rtol, atol=atol)
    head_null = head_vh[rank_current:].T.contiguous()
    restricted_g = head_null.T @ g
    restricted_norm = float(restricted_g.norm())
    energy_cutoff = max(float(atol), float(rtol) * float(g.norm()))
    if restricted_norm > energy_cutoff:
        _, _, restricted_vh = torch.linalg.svd(
            (restricted_g / restricted_norm)[None, :],
            full_matrices=True)
        basis = head_null @ restricted_vh[1:].T
        joint_rank = rank_current + 1
    else:
        basis = head_null
        joint_rank = rank_current
    c = torch.cat((a, g[None, :]), dim=0)
    # Raw singular spectrum remains descriptive telemetry; combined-scale
    # thresholding must not define the intersection's effective rank.
    singular = torch.linalg.svdvals(c)
    cutoff = energy_cutoff
    projector = basis @ basis.T
    width = g.numel()
    dim = width - joint_rank
    lower_bound = width - rank_current - 1
    if dim < lower_bound:
        raise ArithmeticError("rank-nullity violated")
    return TokenFiber(
        basis=basis, projector=projector,
        rank_current=rank_current, rank_joint=joint_rank,
        null_dimension=dim, lower_bound=lower_bound,
        singular_values=singular, cutoff=cutoff,
        null_residual=float(torch.linalg.vector_norm(c @ basis)),
        projector_error=float(torch.linalg.vector_norm(projector @ projector - projector))
    )


def native_decomposition(v: Tensor, g: Tensor, *, atol=RANK_ATOL):
    """Whole-physical-state v=-alpha*g+r; the zero-gradient case is unclassified."""
    if v.shape != g.shape or v.numel() == 0:
        raise ValueError("native update and energy gradient shapes differ")
    vv = v.detach().double().flatten()
    gg = g.detach().double().flatten()
    if not bool(torch.isfinite(vv).all() and torch.isfinite(gg).all()):
        raise ValueError("nonfinite native field")
    norm2 = float(gg @ gg)
    if norm2 <= atol * atol:
        return dict(identified=False, alpha=None,
                    normal=torch.zeros_like(v.detach().double()),
                    tangent=v.detach().double().clone(),
                    energy_rate=float(gg @ vv),
                    tangent_dot_gradient=None)
    alpha = -float(gg @ vv) / norm2
    normal = -alpha * gg
    tangent = vv - normal
    return dict(identified=True, alpha=alpha, normal=normal.view(v.shape),
                tangent=tangent.view(v.shape), energy_rate=float(gg @ vv),
                tangent_dot_gradient=float(gg @ tangent))


def future_selfscore_gradient(teacher, current_state: Tensor, goal: int, remaining: int):
    """Future native head scalar selected by its own clean argmax (label blind)."""
    if current_state.ndim != 3 or current_state.shape[0] != 1:
        raise ValueError("one native [1,tokens,width] state expected")
    if remaining < 1 or not 0 <= goal < current_state.shape[1]:
        raise ValueError("invalid true native continuation horizon or goal")
    h = current_state.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        next_state = h
        for _ in range(remaining):
            next_state = teacher.block(next_state)
        logits = teacher.head(next_state)[0, goal]
        selected_class = int(logits.detach().argmax())
        grad, = torch.autograd.grad(logits[selected_class], h)
    return grad.detach()[0, goal].double(), selected_class, logits.detach()


def future_goal_jacobian(teacher, current_state: Tensor, goal: int,
                         remaining: int) -> Tensor:
    """Exact full future-goal logit derivative wrt ONE present physical token.

    Every row differentiates through the true number of remaining tied blocks.
    This is a model-only observer; external correct target labels are excluded.
    """
    if (current_state.ndim != 3 or current_state.shape[0] != 1 or
        not 0 <= goal < current_state.shape[1] or remaining < 1):
        raise ValueError("expected one native source, legal goal and positive horizon")
    h = current_state.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        state = h
        for _ in range(remaining):
            state = teacher.block(state)
        logits = teacher.head(state)[0,goal]
        gradients = []
        for class_id in range(logits.numel()):
            gradient, = torch.autograd.grad(
                logits[class_id], h,
                retain_graph=(class_id + 1 < logits.numel()))
            gradients.append(gradient[0,goal].detach().double())
    result = torch.stack(gradients)
    if not bool(torch.isfinite(result).all()):
        raise ArithmeticError("nonfinite native future Jacobian")
    return result


def native_delayed_rank(teacher, current_state: Tensor, fiber: TokenFiber, *,
                        goal: int, remaining: int,
                        rtol=RANK_RTOL, atol=RANK_ATOL):
    """Rank of future native-goal logits restricted to the physical fiber K.

    Derivative-only geometric witness, NOT a corrected-answer causal result.
    """
    jac = future_goal_jacobian(teacher,current_state,goal,remaining)
    if fiber.basis.ndim != 2 or fiber.basis.shape[0] != jac.shape[1]:
        raise ValueError("future Jacobian and fiber physical widths differ")
    restricted = jac @ fiber.basis
    if fiber.null_dimension == 0:
        singular = jac.new_zeros((0,))
        rank, cutoff = 0, float(atol)
    else:
        singular = torch.linalg.svdvals(restricted)
        rank, cutoff = _rank_from_svd(singular,rtol=rtol,atol=atol)
    return {"rank":rank,
            "fiber_dimension":fiber.null_dimension,
            "future_jacobian":jac,
            "restricted_operator":restricted,
            "restricted_singular_values":singular,
            "rank_cutoff":cutoff,
            "restricted_frobenius":float(torch.linalg.vector_norm(restricted)),
            "ambient_future_rank":_rank_from_svd(torch.linalg.svdvals(jac),
                                                rtol=rtol,atol=atol)[0]}


def source_energy_gradient(potential, state, mu, std):
    h = state.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        scalar = potential(h, 0, mu, std).sum()
        grad, = torch.autograd.grad(scalar, h)
    return grad.detach()


def plan_source_only(teacher, potential, h: Tensor, mu: Tensor, std: Tensor, *,
                     goal: int, remaining: int, seed: int, dose_fraction=0.25):
    """Fix candidate and tangent shams before task labels are ever consulted."""
    if h.shape[0] != 1 or h.ndim != 3:
        raise ValueError("single-source hidden state expected")
    with torch.no_grad():
        native = teacher.block(h) - h
    g_e = source_energy_gradient(potential, h, mu, std)
    f = token_energy_fiber(teacher.head.weight, g_e[0,goal])
    g_future, selected_class, _ = future_selfscore_gradient(teacher, h, goal,
                                                             remaining)
    proposal = f.projector @ g_future
    proposed_norm = float(proposal.norm())
    radius = dose_fraction * float(native[0,goal].norm())
    if f.null_dimension < 1 or proposed_norm <= 1e-12 or radius <= 1e-12:
        return {"eligible":False,"reason":"degenerate_fiber_or_selected_direction",
                "fiber":f,"radius":radius,"future_selected_class":selected_class}
    selected = proposal / proposed_norm * radius
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    shams=[]
    for _ in range(N_SHAMS):
        coeff=torch.randn((f.null_dimension,),generator=generator,dtype=torch.float64)
        coeff_norm=float(coeff.norm())
        if coeff_norm<=1e-12:
            raise ArithmeticError("zero coefficient for tangent sham")
        shams.append((f.basis @ (coeff / coeff_norm * radius)))
    directions=torch.stack([selected,*shams])
    errors=(directions @ torch.cat(
        (teacher.head.weight.detach().double(),g_e[0,goal].double()[None,:]),dim=0
    ).T).norm(dim=-1)
    norms=directions.norm(dim=1)
    if float(errors.max())>1e-7 or float((norms-radius).abs().max())>1e-7:
        raise ArithmeticError("planned physical edits violate tangent/norm constraints")
    return {
        "eligible":True,"fiber":f,"radius":radius,
        "future_selected_class":selected_class,
        "projected_future_gradient_norm":proposed_norm,
        "current_energy_gradient":g_e.detach(),
        "native_goal_update":native[0,goal].detach(),
        "directions":directions,
        "max_linear_constraint_error":float(errors.max()),
        "max_dose_error":float((norms-radius).abs().max()),
    }


@torch.no_grad()
def native_finite_edit(teacher, potential, h, mu, std, *, goal, remaining, edit):
    """Continue actual tied blocks after a single-token physical additive edit."""
    if edit.shape != (h.shape[-1],):
        raise ValueError("wrong physical single-token edit width")
    clean_e = float(potential(h,0,mu,std).sum())
    base_logits = teacher.head(h)[0,goal].clone()
    mutated = h.clone()
    mutated[0,goal] += edit.to(dtype=h.dtype,device=h.device)
    immediate_logits = teacher.head(mutated)[0,goal]
    finite_energy = float(potential(mutated,0,mu,std).sum()) - clean_e
    s = mutated
    for _ in range(remaining):
        s = teacher.block(s)
    final_logits = teacher.head(s)
    return {
        "finite_energy_change":finite_energy,
        "finite_current_head_l2":float((immediate_logits-base_logits).norm()),
        "realized_edit_norm":float((mutated-h)[0,goal].norm()),
        "goal_logits":final_logits[0,goal].detach().clone(),
        "all_logits":final_logits.detach().clone(),
    }


def stable_seed(*parts):
    raw=("LGA07|"+ "|".join(str(x) for x in parts)).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8],"big") & ((1<<63)-1)

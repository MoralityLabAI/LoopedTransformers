"""LGA-07C source-only 81x48 physical head+energy tangent projection.

Registration: LGA07C_GLOBAL_FIBER_PROTOCOL.md committed before this file.
No reserved scientific data or old model weights are generated or loaded here.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
import math

import torch
from torch import Tensor

from mechinterp.loopscope10.energy_fiber import (
    RANK_RTOL, RANK_ATOL, source_energy_gradient,
)

GOALS = (0, 10, 20, 30, 40, 50, 60, 70)
RESERVED = 31061
EXCLUDED_SMOKE = 61061
BASE_COUNT = 64
SHAMS = 8
ARMS = (
    "global_energy_tangent",
    *(f"same_k_sham_{i}" for i in range(SHAMS)),
    "global_head_only",
    "full_donor_state",
    "goal_token_only",
    "predecessor_goal_pair",
    "global_energy_tangent_quarter",
)
ORIGINAL_SEEDS = (103, 211, 307)


def stable_seed(*parts):
    key = ("LGA07C|" + "|".join(map(str, parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") & ((1 << 63) - 1)


@dataclass
class HeadFiber:
    projector: Tensor
    rank: int
    dimension: int
    max_head_residual: float


def full_head_kernel(head_weight: Tensor, *, rtol=RANK_RTOL, atol=RANK_ATOL) -> HeadFiber:
    """Physical 48D right-kernel; normalize each nonzero class row first."""
    if head_weight.ndim != 2 or head_weight.shape[1] < 1:
        raise ValueError("expected a nonempty [classes,width] head")
    w = head_weight.detach().double()
    if not bool(torch.isfinite(w).all()):
        raise ValueError("nonfinite head")
    norms = torch.linalg.vector_norm(w, dim=1, keepdim=True)
    normalized = torch.where(norms > atol, w / norms.clamp_min(atol),
                             torch.zeros_like(w))
    _, s, vh = torch.linalg.svd(normalized, full_matrices=True)
    threshold = max(float(atol), float(rtol) * float(s[0])) if s.numel() else float(atol)
    rank = int((s > threshold).sum())
    n = vh[rank:].T.contiguous()
    p = n @ n.T
    residual = float(torch.linalg.vector_norm(w @ p))
    if (float(torch.linalg.vector_norm(p @ p - p)) > 1e-7
            or residual > 1e-6 * max(1., float(torch.linalg.vector_norm(w)))):
        raise ArithmeticError("readout physical projector fails")
    return HeadFiber(projector=p, rank=rank, dimension=w.shape[1] - rank,
                     max_head_residual=residual)


@dataclass
class GlobalFiber:
    delta: Tensor
    gradient: Tensor
    head_only: Tensor
    tangent: Tensor
    tangent_gradient: Tensor
    rank_head: int
    dimension: int
    energy_row_active: bool
    donor_norm: float
    head_fraction: float
    tangent_fraction: float
    head_residual: float
    tangent_derivative: float


def project_global_donor(delta: Tensor, gradient: Tensor, head_fiber: HeadFiber,
                         *, rtol=RANK_RTOL, atol=RANK_ATOL) -> GlobalFiber:
    """Project full-state donor difference onto head-null then global energy-null."""
    if delta.ndim != 2 or gradient.shape != delta.shape:
        raise ValueError("expected two identically shaped [tokens,width] matrices")
    u = delta.detach().double()
    g = gradient.detach().double()
    p = head_fiber.projector
    if u.shape[-1] != p.shape[0] or not bool(torch.isfinite(u).all() and torch.isfinite(g).all()):
        raise ValueError("invalid global physical coordinates")
    full = float(torch.linalg.vector_norm(u))
    uh = u @ p
    gh = g @ p
    gn = float(torch.linalg.vector_norm(gh))
    active = bool(gn > max(float(atol), float(rtol) * float(torch.linalg.vector_norm(g))))
    uk = uh - (torch.sum(uh * gh) / (gh * gh).sum()) * gh if active else uh
    d = u.shape[0] * head_fiber.dimension - int(active)
    head_res = float(torch.linalg.vector_norm(uk @ (torch.eye(p.shape[0],dtype=p.dtype) - p)))
    derivative = float(torch.sum(uk * g))
    if (d < 0 or head_res > 2e-7 * max(1., float(torch.linalg.vector_norm(uk)))
            or abs(derivative) > 2e-7 * max(1., float(torch.linalg.vector_norm(uk)) * float(torch.linalg.vector_norm(g)))):
        raise ArithmeticError("global tangent is outside registered physical kernel")
    return GlobalFiber(
        delta=u, gradient=g, head_only=uh, tangent=uk,
        tangent_gradient=gh, rank_head=head_fiber.rank, dimension=d,
        energy_row_active=active, donor_norm=full,
        head_fraction=float(torch.linalg.vector_norm(uh))/max(full,1e-30),
        tangent_fraction=float(torch.linalg.vector_norm(uk))/max(full,1e-30),
        head_residual=head_res, tangent_derivative=derivative,
    )


def unit_scale(z: Tensor, radius: float) -> Tensor:
    n = float(torch.linalg.vector_norm(z))
    if n <= 1e-12 or radius <= 1e-12 or not math.isfinite(radius):
        raise ValueError("zero or invalid physical edit dose")
    out = z * (radius / n)
    if not bool(torch.isfinite(out).all()):
        raise ArithmeticError("nonfinite physical edit")
    return out


def label_blind_global_plan(teacher, potential, recipient_h: Tensor,
                            donor_h: Tensor, mu: Tensor, std: Tensor, *,
                            goal: int, head_fiber: HeadFiber, seed: int):
    """Source/model-only plan. Correct donor labels never enter geometry."""
    if (recipient_h.shape != (1,81,48) or donor_h.shape != recipient_h.shape
            or goal not in GOALS):
        raise ValueError("registered single-source 81x48 legal pair required")
    grad = source_energy_gradient(potential,recipient_h,mu,std)[0].detach().double()
    delta = (donor_h - recipient_h)[0].detach().double()
    f = project_global_donor(delta, grad, head_fiber)
    selected_norm = float(torch.linalg.vector_norm(f.tangent))
    head_norm = float(torch.linalg.vector_norm(f.head_only))
    eligible = bool(selected_norm > 1e-10 and head_norm > 1e-10 and f.donor_norm > 1e-10)
    natural = f.donor_norm
    directions = {
        "full_donor_state":f.delta.clone(),
    }
    goal_only = torch.zeros_like(f.delta)
    goal_only[goal] = f.delta[goal]
    pair_only = goal_only.clone()
    pair_only[(goal-1) % 81] = f.delta[(goal-1) % 81]
    directions["goal_token_only"] = goal_only
    directions["predecessor_goal_pair"] = pair_only
    if eligible:
        directions["global_energy_tangent"] = unit_scale(f.tangent,natural)
        directions["global_head_only"] = unit_scale(f.head_only,natural)
        directions["global_energy_tangent_quarter"] = directions["global_energy_tangent"]*.25
        for i in range(SHAMS):
            generator = torch.Generator(device="cpu").manual_seed(
                stable_seed("sham",seed,i))
            random = torch.randn(f.delta.shape, generator=generator, dtype=torch.float64)
            draw = project_global_donor(random,f.gradient,head_fiber)
            directions[f"same_k_sham_{i}"] = unit_scale(draw.tangent,natural)
    else:
        for name in ("global_energy_tangent","global_head_only",
                     "global_energy_tangent_quarter",
                     *(f"same_k_sham_{i}" for i in range(SHAMS))):
            directions[name] = torch.zeros_like(f.delta)
    if tuple(directions.keys()) != ("full_donor_state","goal_token_only",
                                     "predecessor_goal_pair",
                                     *(x for x in ARMS if x not in (
                                         "full_donor_state","goal_token_only","predecessor_goal_pair"))):
        # The ordering of the dictionary is irrelevant; the public ARMS tuple
        # determines the exact execution/archival order.
        if set(directions) != set(ARMS):
            raise ArithmeticError("selected physical arm roster changed")
    return {"eligible":eligible, "fiber":f, "directions":directions,
            "radius":natural,"head_rank":f.rank_head,
            "tangent_dimension":f.dimension, "energy_row_active":f.energy_row_active,
            "head_fraction":f.head_fraction,"tangent_fraction":f.tangent_fraction,
            "head_residual":f.head_residual,"tangent_gradient_dot":f.tangent_derivative}

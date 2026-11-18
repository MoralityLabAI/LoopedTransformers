"""LOOPSCOPE-10B / LGA-07B: legal donor source and label-blind tangent plan.

Registration is DONOR_PROTOCOL.md, FIRST commit 94cba712. This module never
creates the reserved development source itself. All source generation must be
a separate explicit post-lock action; test only excluded source seed 61060.
"""
from __future__ import annotations
import hashlib
import torch
from torch import Tensor

from mechinterp.loopscope10.energy_fiber import (
    N_SHAMS, source_energy_gradient, token_energy_fiber, native_finite_edit
)

GOALS = (0, 10, 20, 30, 40, 50, 60, 70)
RESERVED_SEED = 31060
EXCLUDED_SMOKE_SEED = 61060
SOURCE_COUNT = 64
DOSE_FRACTION = 0.25


def legal_predecessor_donor(x: Tensor, goal: int):
    """Change one predecessor digit within 1..9; target changes at goal only."""
    if x.shape != (81,) or x.dtype != torch.int64:
        raise ValueError("single legal 81-token int64 source required")
    if not bool(((x >= 1) & (x <= 9)).all()) or goal not in GOALS:
        raise ValueError("out-of-support digit or unregistered goal site")
    position = (goal - 1) % 81
    donor = x.clone()
    donor[position] = int(x[position]) % 9 + 1
    base_label = x.roll(1, dims=0)
    donor_label = donor.roll(1, dims=0)
    changes = (donor_label != base_label).nonzero(as_tuple=False).flatten().tolist()
    if changes != [goal]:
        raise ArithmeticError("donor target change is not isolated to goal")
    return donor, {"predecessor_position":position,
                   "old_digit":int(x[position]),
                   "new_digit":int(donor[position]),
                   "original_goal_label":int(base_label[goal]),
                   "donor_goal_label":int(donor_label[goal]),
                   "changed_targets":changes}


def stable_seed(*parts):
    raw=("LGA07B|" + "|".join(map(str,parts))).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & ((1<<63)-1)


def label_blind_donor_plan(teacher, potential, recipient_state: Tensor,
                           donor_state: Tensor, mu: Tensor, std: Tensor, *,
                           goal: int, seed: int):
    """All physical directions come solely from legal source states and E/head."""
    if (goal not in GOALS or recipient_state.shape != donor_state.shape or
        recipient_state.ndim != 3 or recipient_state.shape[0] != 1):
        raise ValueError("matched single-source physical recipient/donor states required")
    if not bool(torch.isfinite(recipient_state).all() and torch.isfinite(donor_state).all()):
        raise ValueError("nonfinite donor state")
    with torch.no_grad():
        native = teacher.block(recipient_state) - recipient_state
    g_e = source_energy_gradient(potential, recipient_state, mu, std)
    fiber = token_energy_fiber(teacher.head.weight, g_e[0,goal])
    u = (donor_state-recipient_state)[0,goal].detach().double()
    selected_raw = fiber.projector @ u
    selected_length = float(selected_raw.norm())
    source_length = float(u.norm())
    radius = DOSE_FRACTION * float(native[0,goal].norm())
    if (fiber.null_dimension < 1 or selected_length <= 1e-12 or
        source_length <= 1e-12 or radius <= 1e-12):
        return {"eligible":False,"reason":"degenerate_projected_donor_or_dose",
                "fiber":fiber,"radius":radius}
    selected = selected_raw / selected_length * radius
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    shams = []
    for _ in range(N_SHAMS):
        r=torch.randn((fiber.null_dimension,),generator=generator,dtype=torch.float64)
        norm=float(r.norm())
        if norm<=1e-12:
            raise ArithmeticError("zero tangent sham coefficient")
        shams.append((fiber.basis @ (r/norm*radius)))
    directions = {
        "projected_donor":selected,
        "inverse_projected_donor":-selected,
        "unprojected_donor":u/source_length*radius,
        "full_donor_token":u,
    }
    for i,sham in enumerate(shams):
        directions[f"same_k_sham_{i}"] = sham
    constraints=torch.cat((teacher.head.weight.detach().double(),
                           g_e[0,goal][None,:].double()),dim=0)
    for name in ("projected_donor","inverse_projected_donor",*[f"same_k_sham_{i}" for i in range(N_SHAMS)]):
        z=directions[name]
        if float(torch.linalg.vector_norm(constraints@z))>1e-7:
            raise ArithmeticError("same-K physical dose failed nullness")
        if abs(float(z.norm())-radius)>1e-7:
            raise ArithmeticError("same-K physical edit dose mismatch")
    return {
        "eligible":True, "fiber":fiber,
        "radius":radius, "source_donor_goal_delta_l2":source_length,
        "selected_projected_norm":selected_length,
        "directions":directions,
        "native_update_norm":float(native[0,goal].norm()),
        "current_energy_gradient":g_e.detach(),
    }


def native_donor_arm_results(teacher, potential, recipient_state, mu, std,
                             *, goal, remaining, plan):
    """Source-only model readout. Correct donor labels are scored elsewhere."""
    if not plan.get("eligible"):
        return {}
    scored={}
    for arm,edit in plan["directions"].items():
        scored[arm] = native_finite_edit(teacher,potential,recipient_state,mu,std,
                                         goal=goal,remaining=remaining,edit=edit)
    return scored

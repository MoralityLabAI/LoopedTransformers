"""A first-gradient Richardson witness for a promoted FP64 continuation.

This helper creates no model or source. It measures fresh directional HVPs
inside the call and compares them with first gradients at fixed displaced
states. The result is a numerical consistency witness, not an interval
certificate. Candidate selection must finish before this helper is called.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import torch

from mechinterp.loopscope13c.precision_oracle import native_margin_hvp


FD_STEPS = (2.0 ** -8, 2.0 ** -9, 2.0 ** -10)
CURVATURE_RTOL = 1e-8
ORACLE_FRACTION = 0.1
N_VECTORS = 9


@dataclass
class PromotedPrecisionResult:
    receipt: dict
    raw: dict[str, torch.Tensor]


def _finite64(value, name):
    if (not isinstance(value, torch.Tensor) or value.dtype != torch.float64
            or not bool(torch.isfinite(value).all())):
        raise ValueError(name + " must be a finite FP64 tensor")


def _pair_gradient(model, base, edit, *, goal, donor_class, original_class):
    """One first derivative at a supplied displacement from the clean base."""
    predecessor = (goal - 1) % 81
    with torch.enable_grad():
        variable = edit.detach().reshape(2, 48).clone().requires_grad_(True)
        state = base.detach().clone()
        state[0, predecessor] = state[0, predecessor] + variable[0]
        state[0, goal] = state[0, goal] + variable[1]
        logits = model.head(model.block(model.block(state)))[0, goal]
        if logits.shape != (10,) or logits.dtype != torch.float64:
            raise ValueError("the promoted continuation must produce ten FP64 logits")
        margin = logits[donor_class] - logits[original_class]
        gradient, = torch.autograd.grad(margin, variable, create_graph=False)
        actual = (state[0, [predecessor, goal]]
                  - base[0, [predecessor, goal]]).reshape(96)
    _finite64(gradient, "displaced pair gradient")
    _finite64(margin, "displaced signed margin")
    _finite64(actual, "realized FP64 gradient displacement")
    return gradient.detach().reshape(96), actual.detach(), margin.detach()


@torch.no_grad()
def gradient_richardson_check(
    model64, base_state64: torch.Tensor, actual_edits64: torch.Tensor, *,
    goal: int, donor_class: int, original_class: int, target_curvature: float,
) -> PromotedPrecisionResult:
    """Measure a fixed numerical witness on a donor and eight applied shams.

    ``base_state64`` is the exact promoted clean execution state, while
    ``actual_edits64`` contains the exact physical deltas in donor-first
    order. The helper never applies those deltas through an FP32 cast.

    For u=delta/||delta||, first gradients are measured at +/- h*u using
    the three registered physical step sizes. The two Richardson estimates
    each use adjacent scales. All scalar reductions remain in FP64.

    Fresh HVPs are measured here on the exact deltas; the API accepts no
    precomputed HVP or Hessian. Physical constraints and dense-Hessian
    agreement belong to the caller's separate final validation.
    """
    started = time.perf_counter()
    _finite64(base_state64, "base_state64")
    _finite64(actual_edits64, "actual_edits64")
    if base_state64.shape != (1, 81, 48):
        raise ValueError("base_state64 must have shape (1,81,48)")
    if actual_edits64.shape not in ((N_VECTORS, 96), (N_VECTORS, 2, 48)):
        raise ValueError("the witness requires one donor and eight shams")
    if actual_edits64.device != base_state64.device:
        raise ValueError("the clean base and physical deltas must share a device")
    if (not isinstance(goal, int) or not 0 <= goal < 81
            or not isinstance(donor_class, int) or not 0 <= donor_class < 10
            or not isinstance(original_class, int) or not 0 <= original_class < 10
            or donor_class == original_class):
        raise ValueError("invalid goal or signed-margin classes")
    if not math.isfinite(float(target_curvature)):
        raise ValueError("target_curvature must be finite")
    if model64.training:
        raise ValueError("the promoted continuation must be in eval mode")
    for name, value in model64.state_dict().items():
        if value.is_floating_point():
            _finite64(value, "model tensor " + name)
        if value.device != base_state64.device:
            raise ValueError("model tensors and clean base must share a device")

    base = base_state64.detach().clone()
    deltas = actual_edits64.detach().reshape(N_VECTORS, 96).clone()
    norms = torch.linalg.vector_norm(deltas, dim=1)
    if not bool(torch.isfinite(norms).all()) or bool((norms <= 0).any()):
        raise ValueError("every physical delta must have a positive finite norm")
    units = deltas / norms[:, None]
    steps = torch.tensor(FD_STEPS, dtype=torch.float64, device=base.device)
    signs = torch.tensor((1., -1.), dtype=torch.float64, device=base.device)
    probe_edits = units[:, None, None, :] * steps[None, :, None, None] * signs[None, None, :, None]

    tick = time.perf_counter()
    # This fresh factory and all nine directional calls are deliberately
    # inside the helper, independent of the proposal/dense-Hessian caches.
    oracle = native_margin_hvp(model64, base, goal, donor_class, original_class)
    fresh_hvps = torch.stack([oracle(delta) for delta in deltas])
    _finite64(fresh_hvps, "fresh directional HVPs")
    if fresh_hvps.shape != deltas.shape:
        raise ValueError("fresh HVP shape differs from the physical directions")
    fresh_seconds = time.perf_counter() - tick

    tick = time.perf_counter()
    gradients = torch.empty_like(probe_edits)
    actual_probes = torch.empty_like(probe_edits)
    margins = torch.empty(probe_edits.shape[:3], dtype=torch.float64, device=base.device)
    for vector in range(N_VECTORS):
        for scale in range(len(FD_STEPS)):
            for sign in range(2):
                gradient, actual, margin = _pair_gradient(
                    model64, base, probe_edits[vector, scale, sign], goal=goal,
                    donor_class=donor_class, original_class=original_class)
                gradients[vector, scale, sign] = gradient
                actual_probes[vector, scale, sign] = actual
                margins[vector, scale, sign] = margin
    first_gradient_seconds = time.perf_counter() - tick

    central = (gradients[:, :, 0] - gradients[:, :, 1]) / (2 * steps[None, :, None])
    richardson = (4 * central[:, 1:] - central[:, :-1]) / 3
    q_fresh = (deltas * fresh_hvps).sum(dim=1)
    q_richardson = norms.square()[:, None] * (richardson * units[:, None, :]).sum(dim=2)
    q_coarse, q_fine = q_richardson[:, 0], q_richardson[:, 1]
    coarse_errors = (q_coarse - q_fresh).abs()
    fine_errors = (q_fine - q_fresh).abs()
    scale_discrepancies = (q_coarse - q_fine).abs()
    fd_parity = q_fine - q_fine[0]
    q_tolerance = CURVATURE_RTOL * max(1., abs(float(target_curvature)))
    oracle_tolerance = ORACLE_FRACTION * q_tolerance
    consistency = ((coarse_errors <= oracle_tolerance)
                   & (fine_errors <= oracle_tolerance)
                   & (scale_discrepancies <= oracle_tolerance))
    parity_passes = fd_parity.abs() <= q_tolerance
    for value, name in ((central, "central gradient estimates"),
                        (richardson, "Richardson estimates"),
                        (q_fresh, "fresh curvatures"),
                        (q_richardson, "Richardson curvatures")):
        _finite64(value, name)
    per_vector = []
    for index in range(N_VECTORS):
        per_vector.append({
            "index": index, "label": "selected" if index == 0 else f"sham_{index - 1}",
            "physical_norm": float(norms[index]),
            "q_fresh": float(q_fresh[index]),
            "q_richardson_coarse": float(q_coarse[index]),
            "q_richardson_fine": float(q_fine[index]),
            "coarse_error": float(coarse_errors[index]),
            "fine_error": float(fine_errors[index]),
            "scale_discrepancy": float(scale_discrepancies[index]),
            "fd_parity": float(fd_parity[index]),
            "coarse_vs_fresh_passed": bool(coarse_errors[index] <= oracle_tolerance),
            "fine_vs_fresh_passed": bool(fine_errors[index] <= oracle_tolerance),
            "coarse_vs_fine_passed": bool(scale_discrepancies[index] <= oracle_tolerance),
            "oracle_consistency_passed": bool(consistency[index]),
            "fd_parity_passed": bool(parity_passes[index]),
        })
    oracle_passed = bool(consistency.all())
    parity_passed = bool(parity_passes.all())
    receipt = {
        "schema": "LGA10D/gradient-richardson-witness-v1",
        "evidence_class": "NUMERICAL_CONSISTENCY_WITNESS",
        "is_interval_certificate": False,
        "oracle_consistency_passed": oracle_passed,
        "fd_parity_passed": parity_passed,
        "coarse_vs_fresh_passed": bool((coarse_errors <= oracle_tolerance).all()),
        "fine_vs_fresh_passed": bool((fine_errors <= oracle_tolerance).all()),
        "coarse_vs_fine_passed": bool((scale_discrepancies <= oracle_tolerance).all()),
        "passed": bool(oracle_passed and parity_passed),
        "goal": goal, "donor_class": donor_class, "original_class": original_class,
        "n_vectors": N_VECTORS, "n_shams": N_VECTORS - 1,
        "target_curvature": float(target_curvature),
        "q_tolerance": q_tolerance, "oracle_tolerance": oracle_tolerance,
        "steps": list(FD_STEPS), "probe_sign_order": [1, -1],
        "richardson_order": ["coarse", "fine"],
        "step_semantics": "fixed Euclidean physical displacement along each unit direction",
        "fresh_hvp_count": N_VECTORS, "first_gradient_count": 6 * N_VECTORS,
        "fresh_hvp_measured_inside_helper": True,
        "selection_feedback": False,
        "max_errors": {"coarse_vs_fresh": float(coarse_errors.max()),
                       "fine_vs_fresh": float(fine_errors.max()),
                       "coarse_vs_fine": float(scale_discrepancies.max()),
                       "fd_parity": float(fd_parity.abs().max())},
        "per_vector": per_vector,
        "seconds": {"fresh_hvps": fresh_seconds,
                    "first_gradients": first_gradient_seconds,
                    "total": time.perf_counter() - started},
    }
    raw = {
        "base_state64": base, "actual_edits64": deltas, "norms": norms,
        "unit_directions": units, "steps": steps, "probe_edits": probe_edits,
        "probe_actual_deltas": actual_probes, "pair_gradients": gradients,
        "probe_margins": margins, "fresh_hvps": fresh_hvps,
        "central_hvp_estimates": central, "richardson_hvp_estimates": richardson,
        "q_fresh": q_fresh, "q_richardson_coarse": q_coarse,
        "q_richardson_fine": q_fine, "coarse_errors": coarse_errors,
        "fine_errors": fine_errors, "scale_discrepancies": scale_discrepancies,
        "fd_parity": fd_parity,
    }
    return PromotedPrecisionResult(receipt, {key: value.detach().clone() for key, value in raw.items()})

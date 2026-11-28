"""Visit-indexed tangent observability, independent of gradient-sharing kappa_g.

All matrices use column-vector derivatives and a declared intervention support.
This is a local diagnostic, not an exact nonlinear quotient.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch


def stack_transport(step_jacobians: Sequence[torch.Tensor],
                    readout_jacobians: Sequence[torch.Tensor],
                    injection: torch.Tensor) -> torch.Tensor:
    """Stack D(q_j o suffix_j) E for visits strictly after a cut.

    E embeds the allowed recipient coordinates into the full cut state.
    Step matrices may have rectangular typed interfaces.
    """
    if not step_jacobians or len(step_jacobians) != len(readout_jacobians):
        raise ValueError("one readout per nonempty suffix visit required")
    if injection.ndim != 2 or injection.shape[1] == 0:
        raise ValueError("invalid intervention support")
    tangent = injection
    outputs = []
    for step, readout in zip(step_jacobians, readout_jacobians, strict=True):
        if step.ndim != 2 or readout.ndim != 2 or step.shape[1] != tangent.shape[0]:
            raise ValueError("typed step mismatch")
        if readout.shape[1] != step.shape[0]:
            raise ValueError("typed readout mismatch")
        tangent = step @ tangent
        outputs.append(readout @ tangent)
    result = torch.cat(outputs, dim=0)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("nonfinite observability matrix")
    return result


def spectrum(o: torch.Tensor, relative: float = 1e-6, absolute: float = 1e-10) -> dict:
    if o.ndim != 2 or not o.numel() or relative < 0 or absolute < 0:
        raise ValueError("invalid observability or threshold")
    if not bool(torch.isfinite(o).all()):
        raise ValueError("nonfinite observability")
    values = torch.linalg.svdvals(o.detach().double())
    threshold = max(absolute, relative * float(values[0]))
    rank = int((values > threshold).sum())
    return {"observations": o.shape[0], "support_dimension": o.shape[1],
            "d_obs": rank, "null_dimension": o.shape[1] - rank,
            "relative_threshold": relative, "absolute_threshold": absolute,
            "threshold": threshold, "singular_values": values.tolist(),
            "gramian_trace": float(values.square().sum())}


def matched_sham(delta: torch.Tensor, base: torch.Tensor, o: torch.Tensor,
                 generator: torch.Generator, *, tolerance: float = 1e-8) -> tuple[torch.Tensor, dict]:
    """Match full signed O d, displacement norm, and endpoint radius.

    Use the support coordinates as an orthonormal Euclidean chart. The sham
    rotates the residual in ker([O; base^T]); a one-dimensional nullspace
    admits reflection. No query labels or outcome-dependent search.
    """
    if delta.ndim != 1 or base.shape != delta.shape or o.ndim != 2 or o.shape[1] != len(delta):
        raise ValueError("shape mismatch")
    d, z, o = delta.double(), base.double(), o.double()
    if not all(bool(torch.isfinite(a).all()) for a in (d, z, o)):
        raise ValueError("nonfinite match inputs")
    constraints = torch.cat((o, z[None]), dim=0)
    norms = constraints.norm(dim=1)
    valid = norms > 1e-14
    rows = constraints[valid] / norms[valid, None]
    if rows.numel():
        _, values, vh = torch.linalg.svd(rows, full_matrices=False)
        basis = vh[values > 1e-11]
    else:
        basis = d.new_zeros((0, len(d)))
    projection = basis.T @ (basis @ d)
    residual = d - projection
    dose = float(d.norm())
    radius = float(residual.norm())
    if dose <= 1e-12:
        sham, status = d.clone(), "zero_dose"
    elif radius <= 1e-10 * max(1.0, dose):
        sham, status = torch.zeros_like(d), "no_distinct_match"
    else:
        u = torch.randn(len(d), generator=generator, dtype=d.dtype, device=d.device)
        u -= basis.T @ (basis @ u)
        u -= (u @ residual) * residual / radius**2
        if float(u.norm()) <= 1e-10:
            sham = projection - residual
        else:
            sham = projection + radius * u / u.norm()
        status = "matched"
    receipt = {
        "status": status, "constraint_rank": len(basis), "residual_radius": radius,
        "target_norm": dose, "norm_error": abs(float(sham.norm()) - dose),
        "signed_error": float((o @ (sham - d)).norm()),
        "radius_error": abs(float((z + sham).norm()) - float((z + d).norm())),
        "direction_distance": float((sham - d).norm()),
    }
    if status == "matched" and (
            max(receipt[k] for k in ("norm_error", "signed_error", "radius_error")) > tolerance
            or receipt["direction_distance"] <= 1e-10):
        receipt["status"] = "invalid_match"
    return sham, receipt

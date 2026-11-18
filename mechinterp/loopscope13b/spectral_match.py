"""LGA-10B fixed-grid, source-only spectral curvature matching.

The frozen seven slices are exhaustive only within their individual spheres.
Floating point residual checks are numerical evidence, not interval proofs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import time
from typing import Callable

import torch
from torch import Tensor

from mechinterp.loopscope13.curvature_match import finite_pair
from mechinterp.loopscope13b.sphere_solver import sphere_extrema

EXCLUDED_SEED = 61065
RESERVED_SCIENCE_SEED = 31065
GOALS = (0, 10, 20, 30, 40, 50, 60, 70)
COSINES = (-.85, -.65, -.35, 0., .35, .65, .85)
N_SHAMS = 8
MAX_ATTEMPTS = 32
BISECTION_STEPS = 64
PROJECTOR_CUTOFF = .5
GEOMETRY_TOL = 1e-8
CURVATURE_RTOL = 1e-8
PAIRWISE_COS_MAX = .98
ANGLE_TOL = 1e-12


def stable_seed(*parts) -> int:
    text = "LGA10B|" + "|".join(map(str, parts))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") & ((1 << 63) - 1)


def norm(x: Tensor) -> float:
    return float(torch.linalg.vector_norm(x))


@dataclass
class SpectralPlan:
    eligible: bool
    reason: str | None
    shams: list[Tensor]
    receipt: dict
    raw: dict[str, Tensor]


def _serial_extremum(extremum) -> dict:
    return {key: value for key, value in asdict(extremum).items() if key != "vector"}


def sphere_value(v: Tensor, A: Tensor, b: Tensor, constant: float) -> float:
    return float(v @ A @ v + 2 * b @ v) + constant


def sphere_path(start: Tensor, end: Tensor, t: float) -> Tensor:
    """Shortest great-circle path; deterministic tangent for an antipodal tie."""
    if t == 0.:
        return start.clone()
    if t == 1.:
        return end.clone()
    dot = max(-1., min(1., float(start @ end)))
    tangent = end - dot * start
    length = norm(tangent)
    if length < 1e-14:
        if dot > 0:
            result = (1 - t) * start + t * end
            return result / result.norm()
        axis = int(start.abs().argmin())
        tangent = torch.zeros_like(start)
        tangent[axis] = 1.
        tangent -= start[axis] * start
        tangent /= tangent.norm()
        theta = math.pi
    else:
        tangent /= length
        theta = math.atan2(length, dot)
    result = math.cos(t * theta) * start + math.sin(t * theta) * tangent
    return result / result.norm()


def bracketed_crossing(start: Tensor, endpoint: Tensor, A: Tensor,
                       b: Tensor, constant: float, target: float,
                       tolerance: float) -> tuple[Tensor | None, float | None]:
    """Find a crossing; sign bisection does not promise the earliest crossing."""
    left = sphere_value(start, A, b, constant) - target
    right = sphere_value(endpoint, A, b, constant) - target
    if abs(left) <= tolerance:
        return start, 0.
    if abs(right) <= tolerance:
        return endpoint, 1.
    if left * right > 0:
        return None, None
    lo, hi = 0., 1.
    for _ in range(BISECTION_STEPS):
        mid = (lo + hi) / 2
        value = sphere_value(sphere_path(start, endpoint, mid), A, b, constant) - target
        if left * value <= 0:
            hi = mid
        else:
            lo, left = mid, value
    t = (lo + hi) / 2
    vector = sphere_path(start, endpoint, t)
    if abs(sphere_value(vector, A, b, constant) - target) > tolerance:
        return None, None
    return vector, t


def make_spectral_shams(selected: Tensor, row_basis: Tensor,
                        project_K: Callable[[Tensor], Tensor],
                        hvp: Callable[[Tensor], Tensor], *,
                        teacher_seed: int, base_index: int, goal: int,
                        expected_complement_dim: int | None = 64,
                        n_shams: int = N_SHAMS,
                        max_attempts: int = MAX_ATTEMPTS) -> SpectralPlan:
    """Construct separately seeded, separated controls at frozen cosine slices.

    HVP cost is dimension(V)+2. All slices are recorded before sampling.
    Independent validation belongs to source_feasibility and audit_receipt.
    """
    started = time.perf_counter()
    z = finite_pair(selected)
    R = row_basis.detach().double()
    if (R.ndim != 2 or R.shape[0] != 96 or not 0 <= R.shape[1] <= 10
            or not bool(torch.isfinite(R).all())):
        raise ValueError("row_basis must be a finite 96 x rank<=10 matrix")
    if n_shams < 1 or max_attempts < 1:
        raise ValueError("positive sham and attempt counts required")
    identity = torch.eye(96, dtype=torch.float64)
    K = torch.stack([finite_pair(project_K(axis)) for axis in identity], dim=1)
    radius = norm(z)
    z0 = R @ (R.T @ z)
    q0 = z - z0
    r = norm(q0)
    receipt = {
        "schema": "LGA10B/spectral-plan-v1", "teacher_seed": teacher_seed,
        "base_index": base_index, "goal": goal, "radius": radius,
        "null_radius": r, "null_to_full_radius": r / radius if radius else 0.,
        "construction_hvp_count": 0, "slices": [], "attempts": [],
        "n_requested_shams": n_shams, "n_shams": 0,
        "projector_cutoff": PROJECTOR_CUTOFF,
        "numerical_certificates_are_interval_proofs": False,
    }
    raw = {"selected": z, "z0": z0, "q0": q0,
           "row_basis": R, "k_projector": K}
    shams: list[Tensor] = []

    def finish(reason: str | None) -> SpectralPlan:
        receipt["reason"] = reason
        receipt["eligible"] = reason is None
        receipt["n_shams"] = len(shams)
        receipt["construction_seconds"] = time.perf_counter() - started
        raw["shams"] = (torch.stack([s.reshape(-1) for s in shams]) if shams
                         else torch.empty((0, 96), dtype=torch.float64))
        return SpectralPlan(reason is None, reason, shams, receipt, raw)

    geometry = {
        "k_symmetry": norm(K - K.T), "k_idempotence": norm(K @ K - K),
        "row_orthogonality": norm(R.T @ R - torch.eye(R.shape[1], dtype=torch.float64)),
        "row_in_k": norm(K @ R - R), "selected_in_k": norm(K @ z - z),
    }
    receipt["geometry_residuals"] = geometry
    if radius <= 1e-10 or r <= 1e-10:
        return finish("DEGENERATE_FIRST_ORDER_DARK_RADIUS")
    if (max(geometry[k] for k in geometry if k != "selected_in_k") > GEOMETRY_TOL
            or geometry["selected_in_k"] > GEOMETRY_TOL * max(1., radius)):
        return finish("INVALID_FIRST_ORDER_GEOMETRY")
    u = q0 / r
    projector = (K + K.T) / 2 - R @ R.T - torch.outer(u, u)
    values, vectors = torch.linalg.eigh((projector + projector.T) / 2)
    V = vectors[:, values > PROJECTOR_CUTOFF].contiguous()
    for j in range(V.shape[1]):
        pivot = int(V[:, j].abs().argmax())
        if V[pivot, j] < 0:
            V[:, j] *= -1
    raw["complement_basis"] = V
    raw["projector_eigenvalues"] = values
    receipt["complement_dimension"] = V.shape[1]
    receipt["k_dimension"] = int(round(float(torch.trace(K))))
    receipt["future_rank"] = R.shape[1]
    spectrum_error = float(torch.minimum(values.abs(), (values - 1).abs()).max())
    basis_error = max(norm(V.T @ V - torch.eye(V.shape[1], dtype=torch.float64)),
                      norm(K @ V - V), norm(R.T @ V), norm(u @ V))
    receipt["projector_spectrum_error"] = spectrum_error
    receipt["basis_residual"] = basis_error
    if (V.shape[1] < 2 or (expected_complement_dim is not None
                          and V.shape[1] != expected_complement_dim)
            or spectrum_error > GEOMETRY_TOL or basis_error > GEOMETRY_TOL):
        return finish("INVALID_COMPLEMENT_GEOMETRY")

    hvp_started = time.perf_counter()
    hz0 = finite_pair(hvp(z0))
    hq0 = finite_pair(hvp(q0))
    HV = torch.stack([finite_pair(hvp(v)) for v in V.T], dim=1)
    receipt["construction_hvp_count"] = V.shape[1] + 2
    receipt["construction_hvp_seconds"] = time.perf_counter() - hvp_started
    projected_raw = V.T @ HV
    M = (projected_raw + projected_raw.T) / 2
    raw.update({"hz0": hz0, "hq0": hq0, "hv": HV,
                "projected_hessian": M,
                "projected_hessian_eigenvalues": torch.linalg.eigvalsh(M)})
    target = float(z @ (hz0 + hq0))
    receipt["selected_curvature"] = target
    qtolerance = CURVATURE_RTOL * max(1., abs(target))
    receipt["curvature_tolerance"] = qtolerance
    symmetry = max(norm(projected_raw - projected_raw.T),
                   norm(z0 @ HV - hz0 @ V), norm(q0 @ HV - hq0 @ V),
                   abs(float(z0 @ hq0 - q0 @ hz0)))
    receipt["hvp_symmetry_residual"] = symmetry
    hscale = max(1., norm(M), norm(hz0), norm(hq0), norm(HV))
    if symmetry > GEOMETRY_TOL * hscale:
        return finish("HVP_SYMMETRY_FAILURE")

    solved = []
    min_vectors, max_vectors = [], []
    for c in COSINES:
        s = math.sqrt(1 - c * c)
        a, ha = z0 + c * q0, hz0 + c * hq0
        constant = float(a @ ha)
        A, b = r * r * s * s * M, r * s * (V.T @ ha)
        extrema = sphere_extrema(A, b)
        low, high = extrema.minimum, extrema.maximum
        qmin, qmax = constant + low.value, constant + high.value
        attained = qmin - qtolerance <= target <= qmax + qtolerance
        entry = {
            "cosine": c, "constant": constant, "q_min": qmin,
            "q_max": qmax, "target": target, "attainable": attained,
            "minimum": _serial_extremum(low), "maximum": _serial_extremum(high),
            "outer_lower_bound": constant + low.dual_bound,
            "outer_upper_bound": constant + high.dual_bound,
        }
        receipt["slices"].append(entry)
        min_vectors.append(low.vector)
        max_vectors.append(high.vector)
        solved.append((c, s, a, ha, A, b, constant, extrema))
    raw["min_vectors"] = torch.stack(min_vectors)
    raw["max_vectors"] = torch.stack(max_vectors)
    if not all(e["minimum"]["certificate_ok"] and e["maximum"]["certificate_ok"]
               for e in receipt["slices"]):
        return finish("SPHERE_SOLVER_RESIDUAL_FAILURE")
    if not any(e["attainable"] for e in receipt["slices"]):
        unresolved = any(e["outer_lower_bound"] - qtolerance <= target
                         <= e["outer_upper_bound"] + qtolerance
                         for e in receipt["slices"])
        return finish("SPHERE_INTERVAL_UNRESOLVED" if unresolved
                      else "GRID_NO_ATTAINABLE_TARGET")

    accepted_q = []
    checks = {"max_curvature_error": 0., "max_rowspace_error": 0.,
              "max_k_error": 0., "max_dose_error": 0.,
              "max_abs_source_cosine": 0., "max_pairwise_cosine": -1.}
    for sham_index in range(n_shams):
        accepted = False
        for slice_index, (c, s, a, ha, A, b, constant, extrema) in enumerate(solved):
            if not receipt["slices"][slice_index]["attainable"]:
                continue
            for attempt in range(max_attempts):
                seed = stable_seed("ambient", teacher_seed, base_index, goal,
                                   sham_index, slice_index, attempt)
                draw = torch.randn(96, dtype=torch.float64,
                                   generator=torch.Generator().manual_seed(seed))
                start = V.T @ draw
                start /= start.norm()
                start_value = sphere_value(start, A, b, constant)
                endpoint = (extrema.minimum.vector if start_value >= target
                            else extrema.maximum.vector)
                root, interpolation = bracketed_crossing(start, endpoint, A, b,
                                                         constant, target, qtolerance)
                record = {"sham_index": sham_index, "slice_index": slice_index,
                          "attempt": attempt, "seed": seed, "cosine": c,
                          "accepted": False, "interpolation": interpolation}
                receipt["attempts"].append(record)
                if root is None:
                    record["reason"] = "NO_NUMERICAL_PATH_CROSSING"
                    continue
                q = c * q0 + r * s * (V @ root)
                ztrial = z0 + q
                pairwise = max((float(q @ previous) / (r * r)
                                for previous in accepted_q), default=-1.)
                if pairwise > PAIRWISE_COS_MAX + ANGLE_TOL:
                    record["reason"] = "PAIRWISE_DIVERSITY_REJECTION"
                    record["max_pairwise_cosine"] = pairwise
                    continue
                htrial = ha + r * s * (HV @ root)
                qerror = abs(float(ztrial @ htrial) - target)
                jerror, kerror = norm(R.T @ (ztrial - z)), norm(K @ ztrial - ztrial)
                dose_error = abs(norm(ztrial) - radius)
                source_cosine = float(q @ q0) / (r * r)
                if (qerror > qtolerance or jerror > 1e-8 * max(1., radius)
                        or kerror > 1e-7 * max(1., radius)
                        or dose_error > 1e-7 * max(1., radius)
                        or abs(source_cosine) > .85 + ANGLE_TOL):
                    record["reason"] = "NUMERICAL_CONSTRAINT_FAILURE"
                    return finish("NUMERICAL_CONSTRAINT_FAILURE")
                checks["max_curvature_error"] = max(checks["max_curvature_error"], qerror)
                checks["max_rowspace_error"] = max(checks["max_rowspace_error"], jerror)
                checks["max_k_error"] = max(checks["max_k_error"], kerror)
                checks["max_dose_error"] = max(checks["max_dose_error"], dose_error)
                checks["max_abs_source_cosine"] = max(checks["max_abs_source_cosine"], abs(source_cosine))
                checks["max_pairwise_cosine"] = max(checks["max_pairwise_cosine"], pairwise)
                record["accepted"] = True
                record["reason"] = None
                record["max_pairwise_cosine"] = pairwise
                shams.append(ztrial.reshape(2, 48))
                accepted_q.append(q)
                accepted = True
                break
            if accepted:
                break
        receipt["constraint_residuals"] = checks
        if not accepted:
            return finish("INSUFFICIENT_DISTINCT_SHAMS")
    return finish(None)

"""LGA-10C: accept controls only after their physical FP32 round trip.

The proposal basis, extrema and seeded stream come from an untouched LGA-10B
plan. The full dense Hessian is used only by the new admission check. Native
FP32 model validation occurs after selection in a separate module.
"""
from __future__ import annotations

from collections import Counter
import copy
import math
import time

import torch
from torch import Tensor

from mechinterp.loopscope13.curvature_match import finite_pair
from mechinterp.loopscope13b.source_feasibility import _physical_checks
from mechinterp.loopscope13b.spectral_match import (
    ANGLE_TOL, COSINES, MAX_ATTEMPTS, N_SHAMS, PAIRWISE_COS_MAX,
    SpectralPlan, bracketed_crossing, norm, sphere_value, stable_seed,
)


class PhysicalGate:
    """Check the physical applied candidate against donor and accepted shams."""

    def __init__(self, selected, hpair, hessian, k_projector, jacobian,
                 head, egrad, row_basis, q0, target_curvature):
        self.selected = finite_pair(selected)
        self.hpair = finite_pair(hpair)
        self.hessian = hessian.detach().double()
        self.k_projector = k_projector.detach().double()
        self.jacobian = jacobian.detach().double()
        self.head = head.detach().double()
        self.egrad = finite_pair(egrad)
        self.row_basis = row_basis.detach().double()
        self.q0 = finite_pair(q0)
        self.target_curvature = float(target_curvature)
        expected = ((self.hessian, (96, 96)), (self.k_projector, (96, 96)),
                    (self.jacobian, (10, 96)), (self.head, (10, 48)))
        if any(t.shape != shape or not bool(torch.isfinite(t).all())
               for t, shape in expected):
            raise ValueError("invalid physical-gate matrix")
        if (self.row_basis.ndim != 2 or self.row_basis.shape[0] != 96
                or not bool(torch.isfinite(self.row_basis).all())
                or not math.isfinite(self.target_curvature)):
            raise ValueError("invalid physical-gate rowspace or target")
        if norm(self.hessian - self.hessian.T) > 1e-10 * max(1., norm(self.hessian)):
            raise ValueError("physical gate requires the validated symmetric Hessian")
        if norm(self.selected) <= 1e-10 or norm(self.q0) <= 1e-10:
            raise ValueError("physical gate requires a nonzero donor and dark radius")

    def roundtrip(self, intended: Tensor) -> Tensor:
        intended = intended.detach().double()
        shape = intended.shape
        intended = intended.reshape(-1, 96)
        result = ((self.hpair.float() + intended.float()).double()
                  - self.hpair.float().double())
        return result.reshape(shape)

    def __call__(self, candidate: Tensor, accepted_shams: list[Tensor]):
        intended = torch.stack([self.selected,
                                *[finite_pair(z) for z in accepted_shams],
                                finite_pair(candidate)])
        applied = self.roundtrip(intended)
        hactions = applied @ self.hessian.T
        check = _physical_checks(
            applied, hactions, self.selected, self.target_curvature,
            self.k_projector, self.jacobian, self.head, self.egrad,
            self.row_basis, self.q0, fp32=True)
        failed = [key for key, value in check["max_errors"].items()
                  if not math.isfinite(value) or value > check["limits"][key]]
        if check["zero_actual_dark_radius"]:
            failed.append("zero_actual_dark_radius")
        receipt = {
            "passed": bool(check["passed"]), "failed_checks": failed,
            "max_errors": check["max_errors"], "limits": check["limits"],
            "trial_curvature": check["all_applied_curvatures"][-1],
            "selected_curvature": check["all_applied_curvatures"][0],
            "trial_source_cosine": check["source_cosines"][-1],
            "trial_dark_radius": check["dark_radii"][-1],
            "zero_actual_dark_radius": check["zero_actual_dark_radius"],
        }
        raw = {"intended": intended[-1], "actual32": applied[-1],
               "dense_hvp": hactions[-1]}
        return receipt, raw


def make_roundtrip_shams(reference_plan: SpectralPlan, physical_gate: PhysicalGate,
                         *, teacher_seed: int, base_index: int, goal: int,
                         n_shams: int = N_SHAMS,
                         max_attempts: int = MAX_ATTEMPTS) -> SpectralPlan:
    """Replay the frozen proposals with physical acceptance inside the loop.

    Every rounded rejection consumes its registered attempt. The native model
    oracle has no callback here, keeping that later gate outside selection.
    """
    if n_shams < 1 or max_attempts < 1:
        raise ValueError("positive sham and attempt counts required")
    started = time.perf_counter()
    reference = reference_plan.receipt
    for key, expected in (("teacher_seed", teacher_seed), ("base_index", base_index),
                          ("goal", goal)):
        if reference.get(key) != expected:
            raise ValueError("paired proposal identity differs: " + key)
    receipt = copy.deepcopy(reference)
    receipt.update({
        "schema": "LGA10C/roundtrip-plan-v1", "attempts": [],
        "reference_eligible": reference_plan.eligible,
        "reference_reason": reference_plan.reason,
        "proposal_geometry_shared": True, "seed_namespace": "LGA10B",
        "construction_hvp_count": 0, "n_requested_shams": n_shams,
        "registered_attempts_per_sham_per_slice": max_attempts,
        "n_shams": 0, "candidate_evaluations": 0,
    })
    # Values are read-only shared tensors; every new output uses a new key or
    # replaces a dictionary value. No original receipt or tensor is modified.
    raw = dict(reference_plan.raw)
    raw["admission_hessian"] = physical_gate.hessian
    shams: list[Tensor] = []
    accepted_q = []
    attempt_starts, starts, candidates, actuals, hactions = [], [], [], [], []
    dimension = raw.get("complement_basis", torch.empty((96, 0))).shape[1]

    def finish(reason):
        receipt.update({"eligible": reason is None, "reason": reason,
                        "n_shams": len(shams), "candidate_evaluations": len(candidates),
                        "construction_seconds": time.perf_counter() - started,
                        "rejection_counts": dict(Counter(
                            a["reason"] for a in receipt["attempts"] if not a["accepted"]))})
        raw["shams"] = (torch.stack([s.reshape(-1) for s in shams]) if shams
                         else torch.empty((0, 96), dtype=torch.float64))
        for key, values, width in (("attempt_start", attempt_starts, dimension),
                                   ("candidate_start", starts, dimension),
                                   ("candidate_intended", candidates, 96),
                                   ("candidate_actual32", actuals, 96),
                                   ("candidate_hvp_dense", hactions, 96)):
            raw[key] = (torch.stack(values) if values
                        else torch.empty((0, width), dtype=torch.float64))
        return SpectralPlan(reason is None, reason, shams, receipt, raw)

    slices = receipt.get("slices", [])
    if len(slices) != len(COSINES) or "min_vectors" not in raw or "max_vectors" not in raw:
        return finish(reference_plan.reason or "SHARED_GEOMETRY_INCOMPLETE")
    if not all(e["minimum"]["certificate_ok"] and e["maximum"]["certificate_ok"] for e in slices):
        return finish("SPHERE_SOLVER_RESIDUAL_FAILURE")
    if not any(e["attainable"] for e in slices):
        return finish(reference_plan.reason or "GRID_NO_ATTAINABLE_TARGET")

    z, z0, q0 = raw["selected"], raw["z0"], raw["q0"]
    R, K, V = raw["row_basis"], raw["k_projector"], raw["complement_basis"]
    hz0, hq0, HV, M = raw["hz0"], raw["hq0"], raw["hv"], raw["projected_hessian"]
    radius, r = receipt["radius"], receipt["null_radius"]
    target, tolerance = receipt["selected_curvature"], receipt["curvature_tolerance"]
    solved = []
    for index, c in enumerate(COSINES):
        s = math.sqrt(1 - c * c)
        a, ha = z0 + c * q0, hz0 + c * hq0
        constant = float(a @ ha)
        A, b = r * r * s * s * M, r * s * (V.T @ ha)
        solved.append((c, s, a, ha, A, b, constant,
                       raw["min_vectors"][index], raw["max_vectors"][index]))

    checks = {"max_curvature_error": 0., "max_rowspace_error": 0.,
              "max_k_error": 0., "max_dose_error": 0.,
              "max_abs_source_cosine": 0., "max_pairwise_cosine": -1.}
    for sham_index in range(n_shams):
        accepted = False
        for slice_index, (c, s, a, ha, A, b, constant, minimum, maximum) in enumerate(solved):
            if not slices[slice_index]["attainable"]:
                continue
            for attempt in range(max_attempts):
                seed = stable_seed("ambient", teacher_seed, base_index, goal,
                                   sham_index, slice_index, attempt)
                draw = torch.randn(96, dtype=torch.float64,
                                   generator=torch.Generator().manual_seed(seed))
                start = V.T @ draw
                start /= start.norm()
                start_index = len(attempt_starts)
                attempt_starts.append(start.clone())
                start_value = sphere_value(start, A, b, constant)
                endpoint_sense = "minimum" if start_value >= target else "maximum"
                endpoint = minimum if endpoint_sense == "minimum" else maximum
                root, interpolation = bracketed_crossing(start, endpoint, A, b,
                                                         constant, target, tolerance)
                record = {"sham_index": sham_index, "slice_index": slice_index,
                          "attempt": attempt, "seed": seed, "cosine": c,
                          "accepted": False, "interpolation": interpolation,
                          "endpoint_sense": endpoint_sense, "start_index": start_index,
                          "candidate_index": None,
                          "physical": None}
                receipt["attempts"].append(record)
                if root is None:
                    record["reason"] = "NO_NUMERICAL_PATH_CROSSING"
                    continue
                q = c * q0 + r * s * (V @ root)
                trial = z0 + q
                htrial = ha + r * s * (HV @ root)
                pairwise = max((float(q @ previous) / (r * r) for previous in accepted_q),
                               default=-1.)
                qerror = abs(float(trial @ htrial) - target)
                jerror, kerror = norm(R.T @ (trial - z)), norm(K @ trial - trial)
                dose_error = abs(norm(trial) - radius)
                source_cosine = float(q @ q0) / (r * r)
                physical, candidate_raw = physical_gate(trial, shams)
                record["candidate_index"] = len(candidates)
                record["physical"] = physical
                record["max_pairwise_cosine"] = pairwise
                record["intended_errors"] = {
                    "curvature": qerror, "rowspace": jerror, "k": kerror,
                    "dose": dose_error, "source_abs_cosine": abs(source_cosine),
                }
                starts.append(start.clone())
                candidates.append(candidate_raw["intended"].clone())
                actuals.append(candidate_raw["actual32"].clone())
                hactions.append(candidate_raw["dense_hvp"].clone())
                if pairwise > PAIRWISE_COS_MAX + ANGLE_TOL:
                    record["reason"] = "PAIRWISE_DIVERSITY_REJECTION"
                    continue
                if (qerror > tolerance or jerror > 1e-8 * max(1., radius)
                        or kerror > 1e-7 * max(1., radius)
                        or dose_error > 1e-7 * max(1., radius)
                        or abs(source_cosine) > .85 + ANGLE_TOL):
                    record["reason"] = "NUMERICAL_CONSTRAINT_FAILURE"
                    return finish("NUMERICAL_CONSTRAINT_FAILURE")
                if not physical["passed"]:
                    record["reason"] = ("FP32_CURVATURE_REJECTION"
                                        if "curvature" in physical["failed_checks"]
                                        else "FP32_PHYSICAL_REJECTION")
                    continue
                checks["max_curvature_error"] = max(checks["max_curvature_error"], qerror)
                checks["max_rowspace_error"] = max(checks["max_rowspace_error"], jerror)
                checks["max_k_error"] = max(checks["max_k_error"], kerror)
                checks["max_dose_error"] = max(checks["max_dose_error"], dose_error)
                checks["max_abs_source_cosine"] = max(checks["max_abs_source_cosine"], abs(source_cosine))
                checks["max_pairwise_cosine"] = max(checks["max_pairwise_cosine"], pairwise)
                record["accepted"] = True
                record["reason"] = None
                shams.append(trial.reshape(2, 48))
                accepted_q.append(q)
                accepted = True
                break
            if accepted:
                break
        receipt["constraint_residuals"] = checks
        if not accepted:
            return finish("ROUNDED_CANDIDATE_BUDGET_EXHAUSTED")
    return finish(None)

"""LGA-10G sham rescue with prospective, bounded precision-aware corrections.

The unchanged LGA-10C search is reproduced here, with a separate fallback
called only after a sham's original 7×32 attempts are exhausted. Model and
precision oracle execution occur in the injected callback, never this module.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import math
import time
import torch
from torch import Tensor

from mechinterp.loopscope13.curvature_match import finite_pair
from mechinterp.loopscope13b.spectral_match import (
    ANGLE_TOL, COSINES, MAX_ATTEMPTS, N_SHAMS, PAIRWISE_COS_MAX,
    SpectralPlan, bracketed_crossing, norm, sphere_value, stable_seed,
)
from mechinterp.loopscope13c.roundtrip_match import PhysicalGate

OFFSET_GRID = (-.010, -.005, -.0025, .0025, .005, .010)
FALLBACK_PARENTS = 3
LOCAL_DIRECTIONS = 2
LOCAL_BISECTIONS = 32
LOCAL_T_BOUND = .08
LOCAL_RTOL = 1e-10


def correction_seed(*fields):
    text="LGA10G|"+"|".join(map(str,fields))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],"big")&((1<<63)-1)


def _unit(x):
    n=norm(x)
    return x/n if n>1e-13 and math.isfinite(n) else None


def sham_corrections(parent, reference, H, *, teacher_seed, base_index, goal,
                     sham_index, parent_candidate_index):
    """Yield 12 local roots (or explicit failures) on the donor's dark sphere.

    The original cosine slice and null radius are held exactly at intended
    precision. A deterministic second tangent direction offsets the parent,
    and a first tangent direction corrects the FP64 dense curvature level.
    """
    raw=reference.raw
    z0, q0, V=(raw[k].detach().double() for k in
                ("z0","q0","complement_basis"))
    r=float(reference.receipt["null_radius"])
    target=float(reference.receipt["selected_curvature"])
    H=H.detach().double()
    p=parent.detach().double().reshape(96)
    if V.shape[0]!=96 or V.shape[1]!=64 or H.shape!=(96,96):
        raise ValueError("invalid frozen sham correction geometry")
    cosine=float((p-z0)@q0)/(r*r)
    scale=r*math.sqrt(max(0.,1.-cosine*cosine))
    a=z0+cosine*q0
    if scale<=1e-13 or not math.isfinite(scale):
        for j in range(LOCAL_DIRECTIONS):
            for s in OFFSET_GRID:
                yield None,{"j":j,"s":s,"t":None,
                            "reason":"DEGENERATE_LOCAL_SHAM_RADIUS"}
        return
    y=_unit(V.T@(p-a))
    if y is None:
        for j in range(LOCAL_DIRECTIONS):
            for s in OFFSET_GRID:
                yield None,{"j":j,"s":s,"t":None,
                            "reason":"DEGENERATE_LOCAL_SHAM_START"}
        return
    gradient=V.T@(H@(a+scale*(V@y)))
    e=_unit(gradient-y*float(y@gradient))
    if e is None:
        for j in range(LOCAL_DIRECTIONS):
            for s in OFFSET_GRID:
                yield None,{"j":j,"s":s,"t":None,
                            "reason":"DEGENERATE_LOCAL_CURVATURE_GRADIENT"}
        return

    def proposal(s,t,w):
        unit=(y+s*w+t*e)/math.sqrt(1.+s*s+t*t)
        return a+scale*(V@unit)

    tol=LOCAL_RTOL*max(1.,abs(target))
    for j in range(LOCAL_DIRECTIONS):
        seed=correction_seed(teacher_seed,base_index,goal,sham_index,
                             parent_candidate_index,j)
        gen=torch.Generator().manual_seed(seed)
        ambient=torch.randn((64,),dtype=torch.float64,generator=gen)
        w=_unit(ambient-y*float(y@ambient)-e*float(e@ambient))
        for s in OFFSET_GRID:
            fields={"j":j,"s":s,"seed":seed,"cosine":cosine,
                    "parent_candidate_index":parent_candidate_index}
            if w is None:
                yield None,{**fields,"t":None,"reason":"DEGENERATE_LOCAL_CORRECTION_AXIS"}
                continue
            def f(t):
                v=proposal(s,t,w)
                return float(v@(H@v))-target
            f0=f(0.)
            if not math.isfinite(f0):
                yield None,{**fields,"t":None,"reason":"NONFINITE_LOCAL_RESIDUAL"}
                continue
            if abs(f0)<=tol:
                t=0.
                residual=abs(f0)
                steps=0
            else:
                lo,hi=(-LOCAL_T_BOUND,0.) if f0>0 else (0.,LOCAL_T_BOUND)
                flo,fhi=f(lo),f(hi)
                if (not math.isfinite(flo) or not math.isfinite(fhi) or
                    flo*fhi>0):
                    yield None,{**fields,"t":None,"reason":"LOCAL_ROOT_UNBRACKETED",
                                "f0":f0,"flo":flo,"fhi":fhi}
                    continue
                for _ in range(LOCAL_BISECTIONS):
                    mid=(lo+hi)/2.
                    fm=f(mid)
                    if not math.isfinite(fm):
                        raise ArithmeticError("nonfinite local curvature root")
                    if flo*fm<=0:
                        hi=mid
                    else:
                        lo,flo=mid,fm
                t=(lo+hi)/2.
                residual=abs(f(t))
                steps=LOCAL_BISECTIONS
            if residual>tol:
                yield None,{**fields,"t":t,"reason":"LOCAL_ROOT_RESIDUAL",
                            "residual":residual}
                continue
            candidate=proposal(s,t,w)
            yield candidate,{**fields,"_axis":w.detach().double().clone(),
                             "t":t,"reason":None,
                             "residual":residual,"bisections":steps}


def _rank_original_gate(receipt, gate):
    if not all(key in receipt for key in (
        "trial_qnative","trial_q64_actual","trial_q32_probe","trial_q64_probe")):
        return None
    qn=float(receipt["trial_qnative"])
    qp=float(receipt["trial_q64_actual"])
    q32=float(receipt["trial_q32_probe"])
    q64=float(receipt["trial_q64_probe"])
    donor_n=float(gate.donor_q32)
    donor_p=float(gate.donor_q64)
    t=float(gate.tau)
    if t<=0 or not all(math.isfinite(v) for v in
            (qn,qp,q32,q64,donor_n,donor_p,t)):
        return None
    return max(abs(qn-donor_n), abs(qp-donor_p),
               abs((qn-donor_n)-(qp-donor_p)),abs(q32-q64))/t


def make_lattice_shams(reference_plan: SpectralPlan, physical_gate: PhysicalGate,
                         *, teacher_seed: int, base_index: int, goal: int,
                         n_shams: int = N_SHAMS,
                         max_attempts: int = MAX_ATTEMPTS) -> SpectralPlan:
    """Replay frozen proposals, then bounded precision rescue on exhaustion.

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
        "schema": "LGA10G/lattice-roundtrip-plan-v1", "attempts": [],
        "reference_eligible": reference_plan.eligible,
        "reference_reason": reference_plan.reason,
        "proposal_geometry_shared": True, "seed_namespace": "LGA10B",
        "construction_hvp_count": 0, "n_requested_shams": n_shams,
        "registered_attempts_per_sham_per_slice": max_attempts,
        "n_shams": 0, "candidate_evaluations": 0,
        "rescue_attempts": [], "n_rescued_shams": 0,
        "rescue_budget_per_sham": FALLBACK_PARENTS*LOCAL_DIRECTIONS*len(OFFSET_GRID),
    })
    # Values are read-only shared tensors; every new output uses a new key or
    # replaces a dictionary value. No original receipt or tensor is modified.
    raw = dict(reference_plan.raw)
    raw["admission_hessian"] = physical_gate.hessian
    shams: list[Tensor] = []
    accepted_q = []
    attempt_starts, starts, candidates, actuals, hactions = [], [], [], [], []
    rescue_intended, rescue_actual, rescue_hessian_hvp = [], [], []
    rescue_root_vectors, rescue_root_axes = [], []
    dimension = raw.get("complement_basis", torch.empty((96, 0))).shape[1]

    def finish(reason):
        receipt.update({"eligible": reason is None, "reason": reason,
                        "n_shams": len(shams), "candidate_evaluations": len(candidates),
                        "construction_seconds": time.perf_counter() - started,
                        "rejection_counts": dict(Counter(
                            a["reason"] for a in receipt["attempts"] if not a["accepted"]))})
        raw["rescue_root_vectors"] = torch.stack(rescue_root_vectors) if rescue_root_vectors else torch.empty((0,96), dtype=torch.float64)
        raw["rescue_root_axes"] = torch.stack(rescue_root_axes) if rescue_root_axes else torch.empty((0,64), dtype=torch.float64)
        raw["rescue_candidate_intended"] = torch.stack(rescue_intended) if rescue_intended else torch.empty((0,96), dtype=torch.float64)
        raw["rescue_candidate_actual"] = torch.stack(rescue_actual) if rescue_actual else torch.empty((0,96), dtype=torch.float64)
        raw["rescue_candidate_hvp_dense"] = torch.stack(rescue_hessian_hvp) if rescue_hessian_hvp else torch.empty((0,96), dtype=torch.float64)
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
        reject_pool = []
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
                    score=_rank_original_gate(physical,physical_gate)
                    if score is not None:
                        reject_pool.append((score,record["candidate_index"],
                                            trial.detach().clone()))
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
            # Exactly three ranked rejected original candidates are eligible
            # to seed a 2×6 local dark-sphere correction. Continue the frozen
            # sham sequence after the first accepted corrected candidate.
            reject_pool.sort(key=lambda entry: (entry[0],entry[1]))
            for parent_rank,(score,parent_index,parent) in enumerate(
                    reject_pool[:FALLBACK_PARENTS]):
                for candidate, meta in sham_corrections(
                        parent,reference_plan,physical_gate.physical.hessian,
                        teacher_seed=teacher_seed,base_index=base_index,
                        goal=goal,sham_index=sham_index,
                        parent_candidate_index=parent_index):
                    axis=meta.pop("_axis",None)
                    rr={"sham_index":sham_index,"parent_rank":parent_rank,
                        "parent_score":score,"parent_candidate_index":parent_index,
                        **meta,"accepted":False,"candidate_index":None,
                        "root_vector_index":None,"physical":None}
                    receipt["rescue_attempts"].append(rr)
                    if candidate is None:
                        continue
                    if axis is None:
                        raise ArithmeticError("local root omitted correction axis")
                    rr["root_vector_index"]=len(rescue_root_vectors)
                    rescue_root_vectors.append(candidate.detach().double().reshape(96).clone())
                    rescue_root_axes.append(axis)
                    trial=candidate.reshape(96)
                    q=trial-z0
                    qerror=abs(float(trial@(physical_gate.physical.hessian@trial))-target)
                    jerror=norm(R.T@(trial-z))
                    kerror=norm(K@trial-trial)
                    dose_error=abs(norm(trial)-radius)
                    cosine=float(q@q0)/(r*r)
                    pairwise=max((float(q@previous)/(r*r)
                                  for previous in accepted_q),default=-1.)
                    rr["intended_errors"]={"curvature":qerror,
                        "rowspace":jerror,"k":kerror,"dose":dose_error,
                        "source_abs_cosine":abs(cosine),
                        "max_pairwise_cosine":pairwise}
                    limit=LOCAL_RTOL*max(1.,abs(target))
                    if (qerror>limit or jerror>1e-8*max(1.,radius)
                        or kerror>1e-7*max(1.,radius)
                        or dose_error>1e-7*max(1.,radius)
                        or abs(cosine)>.85+ANGLE_TOL
                        or pairwise>PAIRWISE_COS_MAX+ANGLE_TOL):
                        rr["reason"]="LOCAL_GEOMETRY_REJECTION"
                        continue
                    physical,candidate_raw=physical_gate(trial,shams)
                    rr["candidate_index"]=len(rescue_intended)
                    rr["physical"]=physical
                    rescue_intended.append(candidate_raw["intended"].clone())
                    rescue_actual.append(candidate_raw["actual32"].clone())
                    rescue_hessian_hvp.append(candidate_raw["dense_hvp"].clone())
                    if not physical["passed"]:
                        rr["reason"]="LOCAL_NATIVE_PRECISION_REJECTION"
                        continue
                    rr["accepted"]=True
                    rr["reason"]=None
                    shams.append(trial.reshape(2,48))
                    accepted_q.append(q)
                    receipt["n_rescued_shams"]+=1
                    checks["max_curvature_error"]=max(checks["max_curvature_error"],qerror)
                    checks["max_rowspace_error"]=max(checks["max_rowspace_error"],jerror)
                    checks["max_k_error"]=max(checks["max_k_error"],kerror)
                    checks["max_dose_error"]=max(checks["max_dose_error"],dose_error)
                    checks["max_abs_source_cosine"]=max(
                        checks["max_abs_source_cosine"],abs(cosine))
                    checks["max_pairwise_cosine"]=max(
                        checks["max_pairwise_cosine"],pairwise)
                    accepted=True
                    break
                if accepted:
                    break
        if not accepted:
            return finish("ROUNDED_CANDIDATE_BUDGET_EXHAUSTED")
    return finish(None)

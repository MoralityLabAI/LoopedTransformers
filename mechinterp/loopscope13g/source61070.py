"""LGA-10G frozen-source curvature-level-set donor study / FP32 endpoint.

This module is never imported by the independent NumPy auditor. Source seed
61070 is initialized only after the runner validates its immutable git freeze.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import subprocess
import sys
import time

import numpy as np
import torch

from mechinterp.loopscope04.gradient_flow import TinyRecursiveModel, fixed_data, tensor_int64_sha
from mechinterp.loopscope10.donor_source import legal_predecessor_donor
from mechinterp.loopscope10.global_energy_fiber import full_head_kernel
from mechinterp.loopscope13c.precision_oracle import (
    native_margin_hvp, native_pair_jacobian, native_precision_check,
)
from mechinterp.loopscope13b.source_feasibility import QuadraticObserver, _physical_checks
from mechinterp.loopscope13b.spectral_match import make_spectral_shams, stable_seed
from mechinterp.loopscope13c.roundtrip_match import PhysicalGate, make_roundtrip_shams
from mechinterp.loopscope13d.geometry import build_common_native_parent, build_geometry
from mechinterp.loopscope13d.capture import write_capture
from mechinterp.loopscope13d.source_feasibility import REQUIRED_PATHS as PRIOR_CLOSURE
from mechinterp.loopscope13f.manifold import levelset_candidates, curvature
from mechinterp.loopscope13g.sham_repair import make_lattice_shams

SEED = 61070
GOALS = (0, 10, 20, 30, 40, 50, 60, 70)
ARMS = ("frozen_baseline", "fixed_donor_oracle_filter", "levelset_repair", "levelset_sham_repair")
BASES = 8
SHAMS = 8
THRESHOLD = 61
ANGLES = (-0.12, -0.06, 0.06, 0.12)
DIRECTIONS = 4
MAX_ATTEMPTS = 32
SCIENCE_SEEDS = (31059, 31064, 31065)
PROTO_PARENT = "c5fd486f2daa6bb0ae509833cdb220a9179305aa"
FILES = tuple(PRIOR_CLOSURE) + (
    "mechinterp/loopscope13f/manifold.py",
    "mechinterp/loopscope13g/PROTOCOL.md",
    "mechinterp/loopscope13g/IMPLEMENTATION_FREEZE.md",
    "mechinterp/loopscope13g/source61070.py",
    "mechinterp/loopscope13g/sham_repair.py",
    "mechinterp/loopscope13g/audit61070.py",
    "mechinterp/loopscope13g/tests/test_sham_repair.py",
    ".github/workflows/loopscope13g-61070.yml",
)

def norm(x):
    return float(torch.linalg.vector_norm(x.double()))

def num(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

def capture(arrays, prefix, items):
    for name, value in items.items():
        arrays[prefix + name] = num(value)

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def locked_source(freeze):
    if not re.fullmatch(r"[a-f0-9]{40}", freeze or ""):
        raise ValueError("full implementation SHA required")
    root = Path(__file__).resolve().parents[2]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != freeze:
        raise ValueError("source checkout differs from frozen commit")
    parent = subprocess.check_output(["git", "rev-parse", "HEAD^"], cwd=root, text=True).strip()
    if parent != PROTO_PARENT:
        raise ValueError("implementation commit must directly follow protocol")
    if os.environ.get("GITHUB_EVENT_NAME") != "push" or os.environ.get("GITHUB_RUN_ATTEMPT") != "1":
        raise ValueError("first frozen Actions push attempt only")
    subprocess.run(["git", "ls-files", "--error-unmatch", "--", *FILES],
                   cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *FILES],
                   cwd=root, check=True, capture_output=True)
    hashes = {}
    for name in sorted(set(FILES)):
        expected = subprocess.check_output(["git", "show", f"{freeze}:{name}"], cwd=root)
        current = (root / name).read_bytes()
        if expected != current:
            raise ValueError(f"changed frozen source: {name}")
        hashes[name] = hashlib.sha256(current).hexdigest()
    return {"freeze_commit": freeze, "protocol_parent": parent, "source_sha256": hashes}

def delta_at(h32, goal, intended):
    pair = h32[0, [(goal-1)%81,goal]].reshape(96)
    intended = intended.reshape(-1,96).double()
    return ((pair[None] + intended.float()).double() - pair.double()[None])

def donor_variations(selected, row_basis, complement, seed, base, goal):
    """Natural donor followed by 16 fixed-angle source-preserving dark rotations."""
    z = selected.reshape(96).double()
    R = row_basis.double()
    V = complement.double()
    fixed = R @ (R.T @ z)
    q = z - fixed
    r = norm(q)
    if r <= 1e-12 or V.ndim != 2 or V.shape[0] != 96 or V.shape[1] < 2:
        raise ValueError("invalid donor dark geometry")
    u = q / r
    yield 0, z.clone(), {"direction": None, "angle": 0.0}
    index=0
    for direction in range(DIRECTIONS):
        generator = torch.Generator().manual_seed(
            stable_seed("LGA10G",seed,base,goal,direction))
        ambient = torch.randn(96,dtype=torch.float64,generator=generator)
        w = V @ (V.T @ ambient)
        if norm(w) <= 1e-12:
            raise ArithmeticError("donor variation collapsed")
        w = w / norm(w)
        for angle in ANGLES:
            index += 1
            edit = fixed + r * (math.cos(angle)*u + math.sin(angle)*w)
            yield index, edit, {"direction": direction, "angle": angle}

class CurvatureSelectionFilter:
    """Frozen LGA-10C physical gate plus registered native precision filtering.

    Mode "fp32" uses only native HVP scalars as the additional selection
    signal; mode "mixed" uses FP64 HVP scalar/resolution checks as well.
    """

    def __init__(self, physical, oracle32, oracle64, h32, goal, selected,
                 qtarget, mode):
        if mode not in ("fp32", "mixed"):
            raise ValueError("unsupported filter")
        self.physical = physical
        self.oracle32 = oracle32
        self.oracle64 = oracle64
        self.h32 = h32
        self.goal = goal
        self.selected = selected.reshape(96).double()
        self.mode = mode
        self.tau = 1e-8 * max(1.,abs(float(qtarget)))
        self.calls = {"native32":0,"promoted_probe":0,"promoted_actual":0}
        self.statistics = Counter()
        self.donor_actual = delta_at(h32,goal,self.selected[None])[0]
        self.donor_probe32 = self.donor_actual.float()
        self.donor_h32 = oracle32(self.donor_probe32).reshape(96).double()
        self.calls["native32"] += 1
        self.donor_q32 = float(self.donor_actual @ self.donor_h32)
        self.donor_q64 = None
        if mode == "mixed":
            self.donor_h64_actual = oracle64(self.donor_actual).reshape(96)
            self.donor_h64_probe = oracle64(self.donor_probe32.double()).reshape(96)
            self.calls["promoted_actual"] += 1
            self.calls["promoted_probe"] += 1
            self.donor_q64 = float(self.donor_actual @ self.donor_h64_actual)

    def __getattr__(self, name):
        return getattr(self.physical,name)

    def __call__(self, candidate, accepted):
        receipt, raw = self.physical(candidate,accepted)
        if not receipt["passed"]:
            self.statistics["physical_rejected"] += 1
            return receipt, raw
        self.statistics["physical_passed"] += 1
        actual = self.physical.roundtrip(candidate.reshape(1,96))[0].reshape(96)
        probe32 = actual.float()
        reasons=[]
        try:
            H32=self.oracle32(probe32).reshape(96).double()
            self.calls["native32"]+=1
            q32=float(actual @ H32)
            if not math.isfinite(q32) or abs(q32-self.donor_q32) > self.tau:
                reasons.append("FP32_NATIVE_PARITY")
            receipt["trial_qnative"] = q32
            if self.mode == "mixed":
                H64a=self.oracle64(actual).reshape(96)
                H64p=self.oracle64(probe32.double()).reshape(96)
                self.calls["promoted_actual"] += 1
                self.calls["promoted_probe"] += 1
                q64a=float(actual @ H64a)
                q32p=float(probe32.double() @ H32)
                q64p=float(probe32.double() @ H64p)
                receipt.update({"trial_q64_actual":q64a,"trial_q32_probe":q32p,
                                "trial_q64_probe":q64p})
                if abs(q64a-self.donor_q64) > self.tau:
                    reasons.append("PROMOTED_PARITY")
                if abs((q32-self.donor_q32)-(q64a-self.donor_q64)) > self.tau:
                    reasons.append("PAIRED_RESOLUTION")
                if abs(q32p-q64p)>self.tau:
                    reasons.append("PROBE_RESOLUTION")
                if norm(H32-H64p) > 1e-6*max(1.,norm(H64p)):
                    reasons.append("HVP_VECTOR_AGREEMENT")
        except (RuntimeError, ValueError, ArithmeticError, FloatingPointError) as error:
            reasons.append("HVP_EXCEPTION_"+type(error).__name__)
        if reasons:
            self.statistics.update(reasons)
            receipt["passed"]=False
            receipt["failed_checks"].extend(reasons)
        else:
            self.statistics["native_filtered_accept"] += 1
        receipt["native_filter_mode"]=self.mode
        return receipt, raw

def resolve_parent(model32,model64,observer,h32,h64,x,hp,mu,std,base,goal,*,teacher_seed=SEED):
    donor, metadata = legal_predecessor_donor(x[0],goal)
    with torch.no_grad():
        hdon32=model32.reason_states(donor[None])[2]
    parent=build_common_native_parent(model64,observer,h32,hdon32,mu,std,
        goal=goal,original_class=metadata["original_goal_label"],
        donor_class=metadata["donor_goal_label"],teacher_seed=teacher_seed,
        base_index=base,head_projector=hp)
    selected=parent["directions"]["pair_energy_tangent"].reshape(96)
    return donor,metadata,parent,selected,hdon32

def _plan_h64(geometry, selected, model64, h32, model32, metadata,
              hp, base, goal, target, mode, *, teacher_seed=SEED, sham_rescue=False):
    """Build fresh LGA-10B spectral source family using frozen dense H64."""
    R = geometry.future.row_basis
    K = geometry.raw["k_projector"]
    mat=geometry.hessian
    def project(u):
        return (K @ u.reshape(96)).reshape(2,48)
    reference = make_spectral_shams(
        selected,R,project,
        lambda u: mat @ u.reshape(96),
        teacher_seed=teacher_seed,base_index=base,goal=goal,expected_complement_dim=64)
    if not reference.eligible:
        return reference,None,None
    pair=h32[0,[(goal-1)%81,goal]].double().reshape(96)
    gate=PhysicalGate(selected,pair,mat,K,
        geometry.future.full_jacobian.reshape(10,96),
        model64.head.weight.detach().double(),
        geometry.raw["energy_gradient"],R,
        reference.raw["q0"],float(reference.receipt["selected_curvature"]))
    filt=CurvatureSelectionFilter(
        gate,
        native_margin_hvp(model32,h32,goal,metadata["donor_goal_label"],
                          metadata["original_goal_label"]),
        native_margin_hvp(model64,h32.double(),goal,metadata["donor_goal_label"],
                          metadata["original_goal_label"]),
        h32,goal,selected,target,mode)
    selector=make_lattice_shams if sham_rescue else make_roundtrip_shams
    plan=selector(reference,filt,
        teacher_seed=teacher_seed,base_index=base,goal=goal,
        n_shams=SHAMS,max_attempts=MAX_ATTEMPTS)
    return plan,reference,filt

def find_repaired_donor(geometry,model32,model64,h32,selected,metadata,
                        base,goal,target,*,teacher_seed=SEED):
    """Search the registered curvature level set, preserving all donor receipts."""
    tau=1e-8*max(1.,abs(target))
    K=geometry.raw["k_projector"]
    J=geometry.future.full_jacobian.reshape(10,96)
    R=geometry.future.row_basis
    V=geometry.reference.raw["complement_basis"]
    H=geometry.hessian
    radius=norm(selected)
    qoriginal=J@selected
    state_oracle32=native_margin_hvp(model32,h32,goal,
        metadata["donor_goal_label"],metadata["original_goal_label"])
    state_oracle64=native_margin_hvp(model64,h32.double(),goal,
        metadata["donor_goal_label"],metadata["original_goal_label"])
    attempts=[]
    stored={}
    accepted=None
    hvp_counts={"native32":0,"promoted_probe":0,"promoted_actual":0}
    original_dense_curvature=curvature(selected,H)
    for idx,candidate,identity in levelset_candidates(
        selected,R,V,H,base=base,goal=goal,seed=teacher_seed):
        reasons=[]
        entry={"index":idx,**identity}
        if candidate is None:
            entry.update({"passed":False,"reasons":[
                identity.get("root_reason") or "NO_CURVATURE_ROOT"]})
            attempts.append(entry)
            continue
        stored[f"attempt_{idx:02d}__intended"]=candidate.detach().double().clone()
        source_distance=norm(candidate-selected)
        dose_error=abs(norm(candidate)-radius)
        k_error=norm(K@candidate-candidate)
        j_error=norm(J@(candidate-selected))
        intended_level_error=abs(curvature(candidate,H)-original_dense_curvature)
        entry.update({"relative_distance":source_distance/max(radius,1e-30),
                      "dose_error":dose_error,"k_error":k_error,
                      "signed_j_error":j_error,
                      "intended_levelset_error":intended_level_error})
        if dose_error>1e-7*max(1.,radius): reasons.append("DONOR_DOSE")
        if source_distance>0.2*radius: reasons.append("DONOR_SOURCE_DISTANCE")
        if k_error>1e-7*max(1.,radius): reasons.append("DONOR_TANGENT")
        if j_error>1e-8*max(1.,norm(qoriginal)): reasons.append("DONOR_MARGIN_DERIVATIVE")
        if intended_level_error>1e-11*max(1.,abs(original_dense_curvature)):
            reasons.append("DONOR_LEVELSET_ERROR")
        if reasons:
            entry.update({"passed":False,"reasons":reasons})
            attempts.append(entry)
            continue
        applied=delta_at(h32,goal,candidate[None])[0]
        probe32=applied.float()
        stored[f"attempt_{idx:02d}__actual"]=applied.detach().double().clone()
        stored[f"attempt_{idx:02d}__probe32"]=probe32.detach().clone()
        physical_dense=curvature(applied,H)
        entry["physical_dense_curvature_error"]=abs(physical_dense-target)
        if not math.isfinite(physical_dense) or abs(physical_dense-target)>tau:
            reasons.append("DONOR_PHYSICAL_DENSE_CURVATURE")
        if norm(applied)<=0:
            reasons.append("DONOR_ZERO_PHYSICAL")
        if reasons:
            entry.update({"passed":False,"reasons":reasons})
            attempts.append(entry)
            continue
        try:
            h32vec=state_oracle32(probe32).reshape(96).double()
            hvp_counts["native32"]+=1
            h64probe=state_oracle64(probe32.double()).reshape(96)
            hvp_counts["promoted_probe"]+=1
            h64actual=state_oracle64(applied).reshape(96)
            hvp_counts["promoted_actual"]+=1
            for name,value in (("hvp32",h32vec.float()),("hvp64_probe",h64probe),
                               ("hvp64_actual",h64actual)):
                stored[f"attempt_{idx:02d}__"+name]=value.detach().clone()
            q32p=float(probe32.double() @ h32vec)
            q64p=float(probe32.double() @ h64probe)
            q64a=float(applied @ h64actual)
            entry.update({"q32_probe":q32p,"q64_probe":q64p,
                          "q64_actual":q64a,
                          "probe_error":abs(q32p-q64p),
                          "target_error":abs(q64a-target),
                          "hvp_vector_error":norm(h32vec-h64probe),
                          "hvp_vector_limit":1e-6*max(1.,norm(h64probe))})
            if abs(q32p-q64p)>tau: reasons.append("DONOR_PROBE_RESOLUTION")
            if abs(q64a-target)>tau: reasons.append("DONOR_ORIGINAL_TARGET")
            if norm(h32vec-h64probe)>entry["hvp_vector_limit"]:
                reasons.append("DONOR_HVP_AGREEMENT")
        except (ArithmeticError,RuntimeError,ValueError,FloatingPointError) as error:
            reasons.append("DONOR_HVP_EXCEPTION_"+type(error).__name__)
        entry["passed"]=not reasons
        entry["reasons"]=reasons
        attempts.append(entry)
        if not reasons:
            accepted=(idx,candidate)
            break
    return accepted,attempts,stored,hvp_counts

def blank_arm():
    return {"n_rescued_shams":0,"n_rescue_trials":0,"complete":False,"source_preserved":False,
            "native_fp32_ready":False,"promoted_physical_only":False,
            "reason":"NOT_EVALUATED","n_shams":0,"filter_attempt_count":0}

def evaluate_case(model32,model64,h32,h64,x,hp,observer,mu,std,base,goal,arrays,*,teacher_seed=SEED):
    prefix=f"case_{base:02}_{goal:02}__"
    case={"base":base,"goal":goal,"source_eligible":False,
          "reason":None,"arms":{name:blank_arm() for name in ARMS},
          "donor_attempts":[]}
    try:
        donor,meta,parent,z,hdon32=resolve_parent(
            model32,model64,observer,h32,h64,x,hp,mu,std,base,goal,teacher_seed=teacher_seed)
        case["donor_metadata"]=meta
        case["source_eligible"]=bool(parent["eligible"])
        case["source_reason"]=parent.get("reason")
        case["source_signed_margin"]=parent["selected_signed_derivative"]
        case["natural_radius"]=parent["radius"]
        capture(arrays,prefix+"common__",{
            "native_donor_h32":hdon32,"natural_selected":z,
            "natural_difference":parent["fiber"].donor.reshape(96),
            "source_input":donor,
        })
        if not parent["eligible"]:
            case["reason"]="SOURCE_INELIGIBLE"
            return case
        target_hvp=native_margin_hvp(model64,h32.double(),goal,
            meta["donor_goal_label"],meta["original_goal_label"])(z)
        target=float(z@target_hvp)
        case["target_curvature_original"]=target
        capture(arrays,prefix+"common__",{"target_hvp64":target_hvp})
        geo=build_geometry(model64,h32.double(),h32,parent,hp,
            teacher_seed=teacher_seed,base_index=base,goal=goal,
            original_class=meta["original_goal_label"],
            donor_class=meta["donor_goal_label"])
        case["geometry_eligible"]=bool(geo.eligible)
        case["geometry_reason"]=geo.reason
        if geo.future is None or geo.reference is None:
            case["reason"]="GEOMETRY_INCOMPLETE"
            return case
        ref_j64,_=native_pair_jacobian(model64,h64,goal)
        capture(arrays,prefix+"common__",{
            "natural_k":geo.raw["k_projector"],
            "natural_row_basis":geo.future.row_basis,
            "natural_jacobian64":geo.future.full_jacobian.reshape(10,96),
            "reference_jacobian64":ref_j64,
            "natural_hessian":geo.hessian,
            "natural_complement":geo.reference.raw["complement_basis"],
        })
        case["n_geometry_hvp"]=sum(geo.receipt["hvp_counts"].values())
        candidates={}
        if geo.plan is not None:
            candidates["frozen_baseline"]=(z,geo.plan,None,None,True)
        if geo.hessian is not None and geo.dense_receipt and geo.dense_receipt["passed"]:
            plan,reference,filt=_plan_h64(geo,z,model64,h32,model32,meta,hp,
                        base,goal,target,"mixed",teacher_seed=teacher_seed)
            candidates["fixed_donor_oracle_filter"]=(z,plan,reference,filt,True)
            repair,attempts,donor_raw,donor_calls=find_repaired_donor(
                geo,model32,model64,h32,z,meta,base,goal,target,
                teacher_seed=teacher_seed)
            case["donor_attempts"]=attempts
            case["donor_hvp_counts"]=donor_calls
            capture(arrays,prefix+"repair__",donor_raw)
            if repair is not None:
                idx,repaired=repair
                case["repaired_donor_index"]=idx
                plan,reference,filt=_plan_h64(
                    geo,repaired,model64,h32,model32,meta,hp,base,goal,target,"mixed",
                    teacher_seed=teacher_seed)
                candidates["levelset_repair"]=(repaired,plan,reference,filt,True)
                repaired_plan,repaired_reference,repaired_filter=_plan_h64(
                    geo,repaired,model64,h32,model32,meta,hp,
                    base,goal,target,"mixed",teacher_seed=teacher_seed,
                    sham_rescue=True)
                candidates["levelset_sham_repair"]=(
                    repaired,repaired_plan,repaired_reference,repaired_filter,True)
            else:
                case["arms"]["levelset_repair"]["reason"]="DONOR_BUDGET_EXHAUSTED"
                case["arms"]["levelset_sham_repair"]["reason"]="DONOR_BUDGET_EXHAUSTED"
        for name,(selected,plan,reference,filt,source_ok) in candidates.items():
            arm=case["arms"][name]
            arm["source_preserved"]=bool(source_ok)
            arm["selection_eligible"]=bool(plan.eligible)
            arm["n_shams"]=len(plan.shams)
            arm["reason"]=plan.reason
            arm["filter_attempt_count"]=sum(filt.calls.values()) if filt else 0
            arm["filter_hvp_counts"]=dict(filt.calls) if filt else {}
            arm["filter_rejections"]=dict(filt.statistics) if filt else {}
            arm["rejection_counts"]=plan.receipt.get("rejection_counts",{})
            arm["n_rescued_shams"]=plan.receipt.get("n_rescued_shams",0)
            arm["n_rescue_trials"]=len(plan.receipt.get("rescue_attempts",[]))
            arm["n_candidate_evaluations"]=plan.receipt.get("candidate_evaluations",0)
            arm["selection_receipt"]=plan.receipt
            if name=="levelset_sham_repair" and reference is not None:
                needed=("z0","q0","complement_basis","row_basis")
                if all(k in reference.raw for k in needed):
                    capture(arrays,prefix+name+"__repair_geometry__",{
                        k:reference.raw[k] for k in needed})
            capture(arrays,prefix+name+"__selection__",{
                key:plan.raw[key] for key in
                ("attempt_start","candidate_start","candidate_intended",
                 "candidate_actual32","candidate_hvp_dense",
                 "rescue_candidate_intended","rescue_candidate_actual",
                 "rescue_candidate_hvp_dense",
                 "rescue_root_vectors","rescue_root_axes") if key in plan.raw
            })
            capture(arrays,prefix+name+"__",{
                "selected":selected,
                "shams":torch.stack([s.reshape(96) for s in plan.shams])
                        if plan.shams else torch.empty((0,96),dtype=torch.float64),
            })
            if len(plan.shams)!=SHAMS or not plan.eligible:
                arm["reason"]=plan.reason or "INCOMPLETE_PLAN"
                continue
            arm["complete"]=True
            try:
                evaluated=native_precision_check(model32,model64,h32,
                    selected,plan.shams,goal=goal,
                    donor_class=meta["donor_goal_label"],
                    original_class=meta["original_goal_label"],
                    reference_h64=h64,reference_j64=ref_j64,
                    head_projector=hp,target_curvature=target)
                arm["native"]=evaluated.receipt
                capture(arrays,prefix+name+"__native__",evaluated.raw)
                arm["native_fp32_ready"]=bool(
                    arm["source_preserved"] and evaluated.receipt["native_ready"])
                arm["promoted_physical_only"]=bool(evaluated.receipt["promoted_curvature"])
                arm["reason"]=None if arm["native_fp32_ready"] else "NATIVE_FP32_GATE"
                arm["hvp_count"]=sum(evaluated.receipt["hvp_counts"].values())
                # The original physical FP64 gate remains a separately named diagnostic.
                actual=evaluated.raw["actual_edits_native_fp64"]
                hvp=evaluated.raw["hvp64_actual_delta"]
                p=_physical_checks(actual,hvp,selected,target,
                    geo.raw["k_projector"],geo.future.full_jacobian.reshape(10,96),
                    model64.head.weight.detach().double(),
                    geo.raw["energy_gradient"],geo.future.row_basis,
                    selected-geo.future.row_basis@(geo.future.row_basis.T@selected),
                    fp32=True)
                arm["promoted_physical_only"]=bool(p["passed"])
                arm["promoted_physical_receipt"]=p
            except (ArithmeticError,RuntimeError,ValueError,KeyError) as error:
                arm["reason"]="NATIVE_ENDPOINT_EXCEPTION"
                arm["endpoint_exception"]={"type":type(error).__name__,"message":str(error)}
        return case
    except (ArithmeticError,RuntimeError,ValueError,KeyError,IndexError) as error:
        case["reason"]="COMMON_EXCEPTION"
        case["exception"]={"type":type(error).__name__,"message":str(error)}
        return case

def summaries(cases):
    out={}
    for name in ARMS:
        arms=[c["arms"][name] for c in cases]
        passed=sum(a["native_fp32_ready"] for a in arms)
        out[name]={
            "n_native_fp32_ready":passed,
            "status":"PASS" if passed>=THRESHOLD else "FAIL",
            "n_completed":sum(a["complete"] for a in arms),
            "n_promoted_physical_only":sum(a["promoted_physical_only"] for a in arms),
            "n_cases_rescued":sum(a.get("n_rescued_shams",0)>0 for a in arms),
            "n_rescued_shams":sum(a.get("n_rescued_shams",0) for a in arms),
            "n_local_correction_trials":sum(a.get("n_rescue_trials",0) for a in arms),
            "per_base":[sum(c["arms"][name]["native_fp32_ready"]
                            for c in cases if c["base"]==b) for b in range(BASES)],
            "failures":dict(Counter(a["reason"] for a in arms
                                   if not a["native_fp32_ready"])),
            "n_attempted_filter_hvps":sum(a["filter_attempt_count"] for a in arms),
        }
    return out

def run(out,raw,*,freeze):
    out,raw=Path(out),Path(raw)
    if any(p.exists() for p in (out,raw)):
        raise FileExistsError("original 61070 first run cannot be overwritten")
    provenance=locked_source(freeze)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)
    time0=time.perf_counter()
    cpu0=time.process_time()
    torch.manual_seed(SEED)
    model32=TinyRecursiveModel().float().eval().requires_grad_(False)
    model64=copy.deepcopy(model32).double()
    data,labels=fixed_data(BASES,SEED)
    observer=QuadraticObserver()
    hp=full_head_kernel(model64.head.weight).projector
    mu,std=torch.zeros(48,dtype=torch.float64),torch.ones(48,dtype=torch.float64)
    arrays={"source_inputs":num(data),"source_labels":num(labels),
            "head_weight":num(model64.head.weight),"head_projector":num(hp)}
    model_digest=hashlib.sha256()
    for key,v in model32.state_dict().items():
        model_digest.update(key.encode()+b"\x00")
        model_digest.update(np.asarray(num(v),dtype="<f4").tobytes())
        if not torch.equal(v.double(),model64.state_dict()[key]):
            raise ArithmeticError("model exact FP32 parameter promotion mismatch")
    cases=[]
    for b in range(BASES):
        x=data[b:b+1]
        with torch.no_grad():
            h32=model32.reason_states(x)[2]
            h64=model64.reason_states(x)[2]
        capture(arrays,f"base_{b:02d}__",{"hrec_fp32":h32,"hrec_fp64":h64})
        for g in GOALS:
            tic=time.perf_counter()
            case=evaluate_case(model32,model64,h32,h64,x,hp,observer,mu,std,b,g,arrays)
            case["elapsed_seconds"]=time.perf_counter()-tic
            cases.append(case)
        print("LGA10G_61070_BASE",b,{name:sum(
             c["arms"][name]["native_fp32_ready"] for c in cases if c["base"]==b)
             for name in ARMS},flush=True)
    raw.parent.mkdir(parents=True,exist_ok=True)
    saved=write_capture(raw,arrays)
    rec={"schema":"LGA10G/61070-native-fp32-prospective-v1",
         "evidence_class":"SOURCE_ONLY_UNTRAINED_FP32_ENGINEERING",
         "excluded_model_seed":SEED,"excluded_source_seed":SEED,
         "n_cases":len(cases),"n_independent_bases":BASES,"n_goals_per_base":len(GOALS),
         "n_model_initializations":1,
         "n_model_parameters":sum(p.numel() for p in model32.parameters()),
         "newly_trained_model_roots":0,"protected_science_seeds_unopened":list(SCIENCE_SEEDS),
         "registered_threshold":THRESHOLD,"arms":summaries(cases),
         "cases":cases,"raw_sha256":saved["raw_sha256"],"capture":saved,
         "source_inputs_sha256":tensor_int64_sha(data),
         "source_labels_sha256":tensor_int64_sha(labels),
         "model_fp32_sha256":model_digest.hexdigest(),
         "elapsed_seconds":time.perf_counter()-time0,
         "cpu_seconds":time.process_time()-cpu0,
         "rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         "environment":{"python":sys.version,"torch":torch.__version__,
              "numpy":np.__version__,"platform":platform.platform(),
              "torch_threads":torch.get_num_threads(),"github_run_id":os.environ.get("GITHUB_RUN_ID"),
              "github_run_attempt":os.environ.get("GITHUB_RUN_ATTEMPT")},
         "source_protocol":"LGA-10G/61070-native-state-levelset-sham-rescue",
         "model_training":False,"task_correctness_scored":False,
         "native_execution":"FP32 prefix, FP32 edit, FP32 continuation",
         "precision_construction_labels":{
             "frozen_baseline":"FP64 spectral matching; native-FP32 measured endpoint",
             "fixed_donor_oracle_filter":"natural donor; FP64/FP32 HVP-assisted sham selection",
             "levelset_repair":"FP64 dense curvature level-set donor and HVP-assisted sham selection; native-FP32 measured endpoint",
             "levelset_sham_repair":"same donor, bounded curvature-level-set sham correction and mixed-precision HVP checks; native-FP32 measured endpoint"},
         "provenance":provenance}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(rec,indent=2,sort_keys=True,allow_nan=False)+"\n")
    for path in (raw,out):
        path.with_suffix(path.suffix+".sha256").write_text(
            f"{digest(path)}  {path.name}\n")
    print("LGA10G_61070_RESULTS",json.dumps(rec["arms"],sort_keys=True),flush=True)
    return rec

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--raw",required=True,type=Path)
    parser.add_argument("--freeze-commit",required=True)
    a=parser.parse_args()
    run(a.out,a.raw,freeze=a.freeze_commit)

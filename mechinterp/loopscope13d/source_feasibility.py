"""Frozen LGA-10D source-only comparison of derivative anchor states.

One native FP32 donor defines both arms. A complete pre-generation commit
guard precedes every use of excluded seed 61067. Promoted continuation and
native FP32 compatibility remain separate measured execution contracts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
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
from mechinterp.loopscope11.pair_fiber import project_into_k
from mechinterp.loopscope13b.source_feasibility import QuadraticObserver, _physical_checks
from mechinterp.loopscope13c.precision_oracle import (
    native_margin_hvp, native_pair_jacobian, native_precision_check,
)
from mechinterp.loopscope13c.source_feasibility import REQUIRED_PATHS as PREVIOUS_REQUIRED_PATHS
from mechinterp.loopscope13d.capture import write_capture
from mechinterp.loopscope13d.geometry import build_common_native_parent, build_geometry
from mechinterp.loopscope13d.promoted_precision import gradient_richardson_check


EXCLUDED_SEED = 61067
GOALS = (0, 10, 20, 30, 40, 50, 60, 70)
ARMS = ("reference_derivatives", "native_derivatives")
N_BASES = 8
N_SHAMS = 8
MIN_ADMITTED = 61
RESERVED_UNOPENED_SEEDS = (31059, 31064, 31065)
REQUIRED_PATHS = tuple(PREVIOUS_REQUIRED_PATHS) + (
    "mechinterp/loopscope13d/PROTOCOL.md", "mechinterp/loopscope13d/IMPLEMENTATION_FREEZE.md",
    "mechinterp/loopscope13d/source_feasibility.py", "mechinterp/loopscope13d/audit_receipt.py",
    "mechinterp/loopscope13d/geometry.py", "mechinterp/loopscope13d/promoted_precision.py",
    "mechinterp/loopscope13d/capture.py", "mechinterp/loopscope13d/tests/test_geometry.py",
    "mechinterp/loopscope13d/tests/test_promoted_precision.py",
    "mechinterp/loopscope13d/tests/test_capture.py", "mechinterp/loopscope13d/tests/test_receipt.py",
    ".github/workflows/loopscope13d-native-state-source-only.yml",
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _capture(arrays, prefix, values):
    for name, value in values.items():
        arrays[prefix + name] = _array(value)


def _provenance(freeze_commit):
    if not re.fullmatch(r"[0-9a-f]{40}", freeze_commit or ""):
        raise ValueError("The full implementation commit SHA is required before generation.")
    root = Path(__file__).resolve().parents[2]
    discovered = tuple(str(p.relative_to(root)) for p in (root / "mechinterp/loopscope13d").rglob("*")
                       if p.suffix in (".py", ".md"))
    paths = sorted(set(REQUIRED_PATHS + discovered))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != freeze_commit:
        raise ValueError("Checkout HEAD differs from the implementation lock.")
    subprocess.run(["git", "ls-files", "--error-unmatch", "--", *paths], cwd=root,
                   check=True, capture_output=True)
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *paths], cwd=root,
                   check=True, capture_output=True)
    hashes = {}
    for name in paths:
        committed = subprocess.check_output(["git", "show", f"{freeze_commit}:{name}"], cwd=root)
        current = (root / name).read_bytes()
        if current != committed:
            raise ValueError("Source bytes differ from the implementation lock: " + name)
        hashes[name] = hashlib.sha256(current).hexdigest()
    return {"freeze_commit": freeze_commit, "source_sha256": hashes}


def _initial_arm():
    return {"construction_eligible": False, "selection_validated": False,
            "promoted_ready": False, "native_fp32_ready": False, "n_shams": 0,
            "selection_reason": "ARM_NOT_EVALUATED", "promoted_reason": "NO_COMPLETE_SHAM_SET",
            "native_fp32_reason": "NO_COMPLETE_SHAM_SET"}


def _summaries(cases):
    summaries = {}
    for arm in ARMS:
        rows = [c["arms"][arm] for c in cases]
        counts = {key: sum(bool(r[key]) for r in rows) for key in
                  ("construction_eligible", "selection_validated", "promoted_ready", "native_fp32_ready")}
        summaries[arm] = {
            "n_constructed": counts["construction_eligible"],
            "n_selection_validated": counts["selection_validated"],
            "n_promoted_ready": counts["promoted_ready"],
            "n_native_fp32_ready": counts["native_fp32_ready"],
            "promoted_status": "PASS" if counts["promoted_ready"] >= MIN_ADMITTED else "FAIL",
            "native_fp32_status": "PASS" if counts["native_fp32_ready"] >= MIN_ADMITTED else "FAIL",
            "per_base": {key: [sum(bool(c["arms"][arm][key]) for c in cases if c["base"] == b)
                                for b in range(N_BASES)]
                         for key in ("selection_validated", "promoted_ready", "native_fp32_ready")},
            "selection_failure_reasons": dict(Counter(r["selection_reason"] for r in rows if not r["selection_validated"])),
            "promoted_failure_reasons": dict(Counter(r["promoted_reason"] for r in rows if not r["promoted_ready"])),
            "native_fp32_failure_reasons": dict(Counter(r["native_fp32_reason"] for r in rows if not r["native_fp32_ready"])),
        }
    paired = {}
    for key in ("selection_validated", "promoted_ready", "native_fp32_ready"):
        differences = [int(c["arms"][ARMS[1]][key]) - int(c["arms"][ARMS[0]][key]) for c in cases]
        macro = [sum(d for d, c in zip(differences, cases) if c["base"] == b) / len(GOALS)
                 for b in range(N_BASES)]
        transitions = Counter((bool(c["arms"][ARMS[0]][key]), bool(c["arms"][ARMS[1]][key])) for c in cases)
        paired[key] = {"case_differences": differences, "case_difference_count": sum(differences),
                       "per_base_fraction_differences": macro,
                       "source_macro_fraction_difference": sum(macro) / N_BASES,
                       "both_pass": transitions[(True, True)], "reference_only": transitions[(True, False)],
                       "native_only": transitions[(False, True)], "both_fail": transitions[(False, False)],
                       "evidence_class": "DESCRIPTIVE_PAIRED_ENGINEERING",
                       "registered_role": "primary" if key == "promoted_ready" else "secondary_descriptive"}
    return summaries, paired


def _measurement_counts(cases):
    counts = Counter(common_target=sum(case["common_target_hvp_count"] for case in cases))
    gradients, incomplete = 0, []
    for arm in ARMS:
        for case in cases:
            row = case["arms"][arm]
            for key, value in row.get("geometry", {}).get("hvp_counts", {}).items():
                counts[arm + "__" + key] += value
            witness = row.get("promoted", {}).get("witness", {})
            counts[arm + "__promoted_fresh"] += witness.get("fresh_hvp_count", 0)
            gradients += witness.get("first_gradient_count", 0)
            for key, value in row.get("native", {}).get("hvp_counts", {}).items():
                counts[arm + "__native_" + key] += value
            for endpoint in ("promoted", "native"):
                if endpoint + "_exception" in row:
                    incomplete.append({"base": case["base"], "goal": case["goal"], "arm": arm, "endpoint": endpoint})
    return {"hvp_counts": dict(counts), "independent_first_gradient_count": gradients,
            "reference_drift_jacobian_fallback_count": sum(case.get("reference_drift_jacobian_fallback_count", 0) for case in cases),
            "call_counts_complete": not incomplete, "incomplete_call_count_endpoints": incomplete,
            "call_counts_interpretation": "exact completed calls" if not incomplete else
                "recorded completed-call lower bounds; failed helper internals are unmeasured"}


def _donor_precision_ceiling(cases):
    rows = []
    for case in cases:
        measured = [arm for arm in ARMS if "native" in case["arms"][arm]]
        native = case["arms"][measured[0]]["native"] if measured else None
        error = native["arms"][0]["probe_resolution_error"] if native is not None else None
        tolerance = native["q_tolerance"] if native is not None else None
        rows.append({"base": case["base"], "goal": case["goal"], "measured_arms": measured,
                     "probe_resolution_error": error, "q_tolerance": tolerance,
                     "probe_resolution_passed": error <= tolerance if native is not None else None})
    measured_count = sum(bool(row["measured_arms"]) for row in rows)
    failures = sum(row["probe_resolution_passed"] is False for row in rows)
    return {"n_measured_cases": measured_count, "n_unmeasured_cases": len(cases) - measured_count,
            "n_donor_probe_resolution_failures": failures,
            "native_fp32_case_upper_bound": len(cases) - failures,
            "coverage": "complete" if measured_count == len(cases) else "partial",
            "basis": "shared donor-only native versus promoted representable-probe scalar resolution",
            "per_case": rows}


def _actual_edits(h32, goal, selected, shams):
    intended = torch.stack([selected.reshape(-1), *[v.reshape(-1) for v in shams]])
    pair = h32[0, [(goal - 1) % 81, goal]].reshape(96)
    actual32 = (pair[None] + intended.float()).double() - pair.double()[None]
    actual64 = (pair.double()[None] + intended) - pair.double()[None]
    return intended, actual32, actual64


def _selection_validation(geometry, *, h32, goal, head):
    plan, future = geometry.plan, geometry.future
    intended, actual32, actual64 = _actual_edits(h32, goal, geometry.raw["selected"], plan.shams)
    hv64 = torch.stack([geometry.hvp(v, "selection_fp64") for v in actual64])
    hv32 = torch.stack([geometry.hvp(v, "selection_fp32") for v in actual32])
    target = geometry.reference.receipt["selected_curvature"]
    arguments = {"selected": intended[0], "target": target, "k": geometry.raw["k_projector"],
                 "jacobian": future.full_jacobian.reshape(10, 96), "head": head,
                 "egrad": geometry.raw["energy_gradient"], "row_basis": future.row_basis,
                 "q0": geometry.reference.raw["q0"]}
    checks64 = _physical_checks(actual64, hv64, fp32=False, **arguments)
    checks32 = _physical_checks(actual32, hv32, fp32=True, **arguments)
    dense_errors = {}
    for suffix, edits, hvps in (("fp64", actual64, hv64), ("fp32", actual32, hv32)):
        difference = edits @ geometry.hessian.T - hvps
        dense_errors["max_" + suffix + "_vector_error"] = float(torch.linalg.vector_norm(difference, dim=1).max())
        dense_errors["max_" + suffix + "_quadratic_error"] = float((edits * difference).sum(-1).abs().max())
    limit = 1e-10 * max(1., abs(target))
    dense_errors.update({"quadratic_limit": limit, "passed": bool(max(
        dense_errors["max_fp64_quadratic_error"], dense_errors["max_fp32_quadratic_error"]) <= limit)})
    passed = bool(plan.eligible and len(plan.shams) == N_SHAMS and checks64["passed"]
                  and checks32["passed"] and dense_errors["passed"] and geometry.dense_receipt["passed"])
    receipt = {"construction_eligible": bool(plan.eligible), "n_shams": len(plan.shams),
               "primary_fp64": checks64, "primary_fp32": checks32,
               "dense_vs_fresh_hvp": dense_errors, "primary_eligible": passed,
               "construction_target_curvature": target,
               "selection_space": "arm derivative anchor, common native application state"}
    raw = {"intended_edits": intended, "actual_edits_fp64": actual64, "actual_edits_fp32": actual32,
           "direct_hvps_fp64": hv64, "direct_hvps_fp32": hv32}
    return receipt, raw


def _promoted_validation(model64, h32, selected, shams, *, goal, metadata, target,
                         common_geometry, head):
    intended, actual, _ = _actual_edits(h32, goal, selected, shams)
    witness = gradient_richardson_check(model64, h32.double(), actual, goal=goal,
        donor_class=metadata["donor_goal_label"], original_class=metadata["original_goal_label"],
        target_curvature=target)
    receipt = {"witness": witness.receipt, "physical": None, "dense_vs_fresh": None,
               "passed": False, "reason": "NATIVE_GEOMETRY_UNAVAILABLE"}
    raw = dict(witness.raw)
    raw["intended_edits"] = intended
    if (common_geometry is None or common_geometry.future is None or common_geometry.hessian is None
            or common_geometry.dense_receipt is None):
        return receipt, raw
    future = common_geometry.future
    physical = _physical_checks(actual, witness.raw["fresh_hvps"], selected, target,
        common_geometry.raw["k_projector"], future.full_jacobian.reshape(10, 96), head,
        common_geometry.raw["energy_gradient"], future.row_basis,
        selected - future.row_basis @ (future.row_basis.T @ selected), fp32=True)
    discrepancy = actual @ common_geometry.hessian.T - witness.raw["fresh_hvps"]
    error = float((actual * discrepancy).sum(-1).abs().max())
    limit = 1e-10 * max(1., abs(target))
    dense = {"max_quadratic_error": error, "quadratic_limit": limit,
             "max_vector_error": float(torch.linalg.vector_norm(discrepancy, dim=1).max()),
             "coordinate_skew_passed": bool(common_geometry.dense_receipt["passed"]),
             "passed": bool(error <= limit and common_geometry.dense_receipt["passed"])}
    passed = bool(physical["passed"] and dense["passed"] and witness.receipt["passed"])
    receipt.update({"physical": physical, "dense_vs_fresh": dense, "passed": passed,
                    "reason": None if passed else "PROMOTED_CONTINUATION_GATE"})
    return receipt, raw


def _evaluate_selected_arms(geometries, *, case, arrays, model32, model64, h32, h64,
                             selected, target, head_projector, metadata):
    """Archive both selections, then isolate each measured endpoint."""
    prefix = case["raw_prefix"]
    for arm in ARMS:
        geometry = geometries.get(arm)
        if geometry is None:
            continue
        result = case["arms"][arm]
        result["geometry"] = geometry.receipt
        if geometry.dense_receipt is not None:
            result["dense_primary"] = geometry.dense_receipt
        _capture(arrays, prefix + arm + "__geometry__", geometry.raw)
        if geometry.reference is not None:
            result["reference"] = geometry.reference.receipt
            _capture(arrays, prefix + arm + "__reference__", geometry.reference.raw)
        if geometry.plan is not None:
            result.update({"selection": geometry.plan.receipt, "n_shams": len(geometry.plan.shams),
                           "construction_eligible": bool(geometry.plan.eligible),
                           "selection_reason": geometry.plan.reason})
            _capture(arrays, prefix + arm + "__selection__", geometry.plan.raw)
        else:
            result["selection_reason"] = geometry.reason
    reference = geometries.get(ARMS[0])
    reference_j = (reference.future.full_jacobian.reshape(10, 96)
                   if reference is not None and reference.future is not None else None)
    if reference_j is None and any(g is not None and g.plan is not None and len(g.plan.shams) == N_SHAMS
                                   for g in geometries.values()):
        try:
            reference_j, _ = native_pair_jacobian(model64, h64, case["goal"])
            case["reference_drift_jacobian_fallback"] = True
            case["reference_drift_jacobian_fallback_count"] = 10
        except (ArithmeticError, ValueError, RuntimeError) as error:
            case["reference_drift_jacobian_exception"] = {"type": type(error).__name__, "message": str(error)}
    if reference_j is not None:
        arrays[prefix + "reference_drift_jacobian"] = _array(reference_j)
    for arm in ARMS:
        geometry = geometries.get(arm)
        if geometry is None or geometry.plan is None:
            continue
        plan, result = geometry.plan, case["arms"][arm]
        arm_prefix = prefix + arm + "__"
        try:
            receipt, raw = _selection_validation(geometry, h32=h32, goal=case["goal"],
                                                  head=model64.head.weight.detach())
            result["selection_validation"] = receipt
            _capture(arrays, arm_prefix + "selection_validation__", raw)
            result["selection_validated"] = bool(receipt["primary_eligible"])
            result["selection_reason"] = None if result["selection_validated"] else (plan.reason or "SELECTION_SPACE_PHYSICAL_GATE")
        except (ArithmeticError, ValueError, RuntimeError, KeyError) as error:
            result["selection_exception"] = {"type": type(error).__name__, "message": str(error)}
            result["selection_reason"] = "EXPLICIT_SELECTION_VALIDATION_EXCEPTION"
        if len(plan.shams) != N_SHAMS:
            continue
        try:
            promoted, raw = _promoted_validation(model64, h32, selected, plan.shams,
                goal=case["goal"], metadata=metadata, target=target,
                common_geometry=geometries.get(ARMS[1]), head=model64.head.weight.detach())
            result["promoted"] = promoted
            _capture(arrays, arm_prefix + "promoted__", raw)
            result["promoted_ready"] = bool(plan.eligible and promoted["passed"])
            result["promoted_reason"] = None if result["promoted_ready"] else (promoted["reason"] or "INELIGIBLE_COMPLETE_PLAN")
        except (ArithmeticError, ValueError, RuntimeError, KeyError) as error:
            result["promoted_exception"] = {"type": type(error).__name__, "message": str(error), "call_counts_complete": False}
            result["promoted_reason"] = "EXPLICIT_PROMOTED_VALIDATION_EXCEPTION"
        try:
            if reference_j is None:
                raise ValueError("Reference-state Jacobian unavailable for declared drift diagnostics.")
            native = native_precision_check(model32, model64, h32, selected, plan.shams,
                goal=case["goal"], donor_class=metadata["donor_goal_label"],
                original_class=metadata["original_goal_label"], reference_h64=h64,
                reference_j64=reference_j, head_projector=head_projector, target_curvature=target)
            result["native"] = native.receipt
            _capture(arrays, arm_prefix + "native__", native.raw)
            result["native_fp32_ready"] = bool(native.receipt["native_ready"])
            result["native_fp32_reason"] = None if result["native_fp32_ready"] else "NATIVE_FP32_GATE"
        except (ArithmeticError, ValueError, RuntimeError, KeyError) as error:
            result["native_exception"] = {"type": type(error).__name__, "message": str(error), "call_counts_complete": False}
            result["native_fp32_reason"] = "EXPLICIT_NATIVE_VALIDATION_EXCEPTION"


def run(out: Path, raw: Path, *, freeze_commit: str):
    out, raw = Path(out), Path(raw)
    if any(path.exists() for path in (out, raw, out.with_suffix(out.suffix + ".sha256"), raw.with_suffix(raw.suffix + ".sha256"))):
        raise FileExistsError("First source receipts must not be overwritten.")
    provenance = _provenance(freeze_commit)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)
    wall, cpu = time.perf_counter(), time.process_time()
    torch.manual_seed(EXCLUDED_SEED)
    model32 = TinyRecursiveModel().float().eval().requires_grad_(False)
    model64 = copy.deepcopy(model32).double()
    data, labels = fixed_data(N_BASES, EXCLUDED_SEED)
    observer = QuadraticObserver()
    hp = full_head_kernel(model64.head.weight).projector
    mu, std = torch.zeros(48, dtype=torch.float64), torch.ones(48, dtype=torch.float64)
    arrays = {"source_inputs": _array(data), "source_labels": _array(labels),
              "head_weight": _array(model64.head.weight), "head_weight_fp32": _array(model32.head.weight),
              "head_projector": _array(hp)}
    model_digest = hashlib.sha256()
    for name, tensor in model32.state_dict().items():
        model_digest.update(name.encode() + b"\0")
        model_digest.update(np.asarray(_array(tensor), dtype="<f4").tobytes())
        if not torch.equal(tensor.double(), model64.state_dict()[name]):
            raise ArithmeticError("Promoted model tensors differ from the single FP32 initialization.")
    cases = []
    for base in range(N_BASES):
        x = data[base:base + 1]
        with torch.no_grad():
            h32, h64 = model32.reason_states(x)[2], model64.reason_states(x)[2]
        _capture(arrays, f"base_{base:02d}__", {"hrec_fp32": h32, "hrec_fp64": h64})
        for goal in GOALS:
            tick = time.perf_counter()
            prefix = f"case_{base:02d}_{goal:02d}__"
            case = {"base": base, "goal": goal, "raw_prefix": prefix, "case_reason": None,
                    "common_target_hvp_count": 0, "common_target_hvp_attempt_count": 0,
                    "arms": {arm: _initial_arm() for arm in ARMS}}
            geometries = {}
            try:
                donor, metadata = legal_predecessor_donor(x[0], goal)
                case["donor_metadata"] = metadata
                arrays[prefix + "donor_input"] = _array(donor)
                with torch.no_grad():
                    hdon32 = model32.reason_states(donor[None])[2]
                arrays[prefix + "hdon_fp32"] = _array(hdon32)
                parent = build_common_native_parent(model64, observer, h32, hdon32, mu, std,
                    goal=goal, original_class=metadata["original_goal_label"],
                    donor_class=metadata["donor_goal_label"], teacher_seed=EXCLUDED_SEED,
                    base_index=base, head_projector=hp)
                case["common_parent"] = {"eligible": bool(parent["eligible"]),
                    "reason": parent.get("reason"), "native_source": parent.get("native_source"),
                    "clean_future_margin": parent["clean_future_margin"],
                    "selected_signed_derivative": parent["selected_signed_derivative"],
                    "pair_positions": parent["pair_positions"], "natural_radius": parent["radius"]}
                fiber = parent["fiber"]
                selected = parent["directions"]["pair_energy_tangent"].reshape(96)
                K = torch.stack([project_into_k(e.reshape(2, 48), fiber, hp).reshape(96)
                                 for e in torch.eye(96, dtype=torch.float64)], dim=1)
                _capture(arrays, prefix + "common__", {"selected": selected, "natural_pair": fiber.donor.reshape(96),
                    "energy_gradient": fiber.egrad.reshape(96), "k_projector": K,
                    "application_pair64": h32[0, [(goal - 1) % 81, goal]].double().reshape(96),
                    "future_gradient": parent["future_gradient"].reshape(96),
                    "future_k_gradient": parent["future_k_gradient"].reshape(96)})
                _capture(arrays, prefix + "common__parent_direction__", parent["directions"])
                case.update({"radius": float(torch.linalg.vector_norm(selected)), "k_dimension": fiber.dimension})
                if not parent["eligible"]:
                    case["case_reason"] = "COMMON_NATIVE_PARENT_INELIGIBLE"
                    for arm in ARMS:
                        case["arms"][arm]["selection_reason"] = case["case_reason"]
                else:
                    case["common_target_hvp_attempt_count"] = 1
                    target_hvp = native_margin_hvp(model64, h32.double(), goal,
                        metadata["donor_goal_label"], metadata["original_goal_label"])(selected)
                    case["common_target_hvp_count"] = 1
                    target = float(selected @ target_hvp)
                    case["target_curvature_native"] = target
                    arrays[prefix + "common__target_hvp_native"] = _array(target_hvp)
                    for arm, state in ((ARMS[0], h64), (ARMS[1], h32.double())):
                        try:
                            geometry = build_geometry(model64, state, h32, parent, hp,
                                teacher_seed=EXCLUDED_SEED, base_index=base, goal=goal,
                                original_class=metadata["original_goal_label"], donor_class=metadata["donor_goal_label"])
                            geometries[arm] = geometry
                        except (ArithmeticError, ValueError, RuntimeError, KeyError) as error:
                            case["arms"][arm]["geometry_exception"] = {"type": type(error).__name__, "message": str(error)}
                            case["arms"][arm]["selection_reason"] = "EXPLICIT_GEOMETRY_EXCEPTION"
                    _evaluate_selected_arms(geometries, case=case, arrays=arrays, model32=model32,
                        model64=model64, h32=h32, h64=h64, selected=selected, target=target,
                        head_projector=hp, metadata=metadata)
            except (ArithmeticError, ValueError, RuntimeError, KeyError) as error:
                case["case_reason"] = "EXPLICIT_COMMON_CASE_EXCEPTION"
                case["exception"] = {"type": type(error).__name__, "message": str(error)}
                for arm in ARMS:
                    if case["arms"][arm]["selection_reason"] == "ARM_NOT_EVALUATED":
                        case["arms"][arm]["selection_reason"] = case["case_reason"]
            for arm, geometry in geometries.items():
                case["arms"][arm]["geometry"] = geometry.receipt
            case["elapsed_seconds"] = time.perf_counter() - tick
            cases.append(case)
        print("LGA10D_BASE", base, {arm: {key: sum(c["arms"][arm][key] for c in cases if c["base"] == base)
              for key in ("promoted_ready", "native_fp32_ready")} for arm in ARMS}, flush=True)
    raw.parent.mkdir(parents=True, exist_ok=True)
    capture = write_capture(raw, arrays)
    summaries, paired = _summaries(cases)
    counts = _measurement_counts(cases)
    record = {"schema": "LGA10D/excluded-source-61067-common-native-parent-v1",
        "evidence_class": "SOURCE_ONLY_UNTRAINED_PRECISION_ENGINEERING", "excluded_source_seed": EXCLUDED_SEED,
        "excluded_model_seed": EXCLUDED_SEED, "n_excluded_bases": N_BASES, "n_goals_per_base": len(GOALS),
        "n_cases": len(cases), "n_untrained_model_initializations": 1, "n_new_trained_model_roots": 0,
        "n_model_parameters": sum(p.numel() for p in model32.parameters()), "registered_min_admitted": MIN_ADMITTED,
        "reserved_science_seeds_unopened": list(RESERVED_UNOPENED_SEEDS), "trained_weights_loaded": False,
        "edited_task_outcomes_scored": False, "new_test_generated": False, "trained_release_authorized": False,
        "source_inputs_sha256": tensor_int64_sha(data), "source_labels_sha256": tensor_int64_sha(labels),
        "untrained_fp32_tensor_sha256": model_digest.hexdigest(), "paired_proposal_seed_namespace": "LGA10B",
        "precision_endpoints_used_in_proposal_search": False, "common_native_parent": True,
        "execution_contracts": {"promoted": "FP32 reached state and edit addition, followed by exact promotion and FP64 continuation",
                                "native_fp32": "FP32 reached state, edit addition and continuation with original LGA10C precision gate"},
        "primary_contrast": "native_derivatives minus reference_derivatives promoted_ready",
        "arms": summaries, "paired": paired, "cases": cases, **counts,
        "donor_precision_ceiling": _donor_precision_ceiling(cases),
        "elapsed_seconds": time.perf_counter() - wall, "cpu_seconds": time.process_time() - cpu,
        "process_peak_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "environment": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "device": "cpu", "torch_threads": torch.get_num_threads(),
            "deterministic_algorithms": True, "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT")},
        "provenance": provenance, "raw_file": raw.name, "raw_sha256": capture["raw_sha256"], "capture": capture}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    for path in (out, raw):
        path.with_suffix(path.suffix + ".sha256").write_text(f"{_sha(path)}  {path.name}\n")
    print("LGA10D_SOURCE_FEASIBILITY", json.dumps(summaries, sort_keys=True), flush=True)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--freeze-commit", required=True)
    options = parser.parse_args()
    run(options.out, options.raw, freeze_commit=options.freeze_commit)

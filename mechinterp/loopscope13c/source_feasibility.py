"""Prospectively frozen paired LGA-10C source-only engineering pilot.

The first excluded seed 61066 generation is guarded by a clean implementation
commit. One FP32 initialization is copied exactly into FP64. Both arms share
inputs, primary geometry, the original proposal seed stream, and final gates.
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
from mechinterp.loopscope11.pair_fiber import plan_pair, project_into_k
from mechinterp.loopscope12.full_jacobian import measure_full_response
from mechinterp.loopscope13.curvature_match import native_margin_hvp
from mechinterp.loopscope13b.source_feasibility import QuadraticObserver, _physical_checks
from mechinterp.loopscope13b.spectral_match import make_spectral_shams
from mechinterp.loopscope13c.precision_oracle import dense_margin_hessian, native_precision_check
from mechinterp.loopscope13c.roundtrip_match import PhysicalGate, make_roundtrip_shams

EXCLUDED_SEED = 61066
RESERVED_UNOPENED_SEEDS = (31059, 31064, 31065)
GOALS = (0, 10, 20, 30, 40, 50, 60, 70)
ARMS = ("baseline", "roundtrip")
N_BASES = 8
N_SHAMS = 8
MIN_ADMITTED = 61
REQUIRED_PATHS = (
    "AGENTS.md", "mechinterp/loopscope04/gradient_flow.py",
    "mechinterp/loopscope10/donor_source.py", "mechinterp/loopscope10/energy_fiber.py",
    "mechinterp/loopscope10/global_energy_fiber.py", "mechinterp/loopscope11/pair_fiber.py",
    "mechinterp/loopscope12/full_jacobian.py", "mechinterp/loopscope13/curvature_match.py",
    "mechinterp/loopscope13b/PROTOCOL.md", "mechinterp/loopscope13b/IMPLEMENTATION_FREEZE.md",
    "mechinterp/loopscope13b/source_feasibility.py", "mechinterp/loopscope13b/audit_receipt.py",
    "mechinterp/loopscope13b/spectral_match.py", "mechinterp/loopscope13b/sphere_solver.py",
    "mechinterp/loopscope13b/tests/test_sphere_solver.py",
    "mechinterp/loopscope13b/tests/test_spectral_match.py",
    "mechinterp/loopscope13b/tests/test_native_hvp.py",
    "mechinterp/loopscope13c/PROTOCOL.md", "mechinterp/loopscope13c/IMPLEMENTATION_FREEZE.md",
    "mechinterp/loopscope13c/source_feasibility.py", "mechinterp/loopscope13c/audit_receipt.py",
    "mechinterp/loopscope13c/precision_oracle.py", "mechinterp/loopscope13c/roundtrip_match.py",
    "mechinterp/loopscope13c/tests/test_precision_oracle.py",
    "mechinterp/loopscope13c/tests/test_roundtrip_match.py",
    "mechinterp/loopscope13c/tests/test_receipt.py",
    ".github/workflows/loopscope13c-roundtrip-source-only.yml",
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _provenance(freeze_commit):
    if not re.fullmatch(r"[0-9a-f]{40}", freeze_commit or ""):
        raise ValueError("the full pre-pilot implementation commit SHA is required")
    root = Path(__file__).resolve().parents[2]
    new_paths = tuple(str(path.relative_to(root))
                      for path in (root / "mechinterp/loopscope13c").rglob("*")
                      if path.suffix in (".py", ".md"))
    paths = sorted(set(REQUIRED_PATHS + new_paths))
    checked_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if checked_head != freeze_commit:
        raise ValueError("checkout HEAD differs from the supplied implementation lock")
    subprocess.run(["git", "ls-files", "--error-unmatch", "--", *paths],
                   cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *paths],
                   cwd=root, check=True, capture_output=True)
    hashes = {}
    for path in paths:
        committed = subprocess.check_output(["git", "show", f"{freeze_commit}:{path}"], cwd=root)
        working = (root / path).read_bytes()
        if working != committed:
            raise ValueError("source bytes differ from the implementation lock: " + path)
        hashes[path] = hashlib.sha256(working).hexdigest()
    return {"freeze_commit": freeze_commit, "source_sha256": hashes}


def _capture_raw(arrays, prefix, raw):
    for name, value in raw.items():
        arrays[prefix + name] = _array(value)


def _initial_arm():
    return {"construction_eligible": False, "primary_eligible": False,
            "native_ready": False, "overall_ready": False, "n_shams": 0,
            "primary_reason": "CASE_NOT_EVALUATED", "native_reason": "NO_COMPLETE_SHAM_SET"}


def _summaries(cases):
    summaries = {}
    for arm in ARMS:
        rows = [c["arms"][arm] for c in cases]
        counts = {key: sum(bool(r[key]) for r in rows)
                  for key in ("construction_eligible", "primary_eligible", "native_ready", "overall_ready")}
        per_base = {
            key: [sum(bool(c["arms"][arm][key]) for c in cases if c["base"] == b)
                  for b in range(N_BASES)]
            for key in ("primary_eligible", "native_ready", "overall_ready")
        }
        summaries[arm] = {
            "n_constructed": counts["construction_eligible"],
            "n_primary_admitted": counts["primary_eligible"],
            "n_native_ready": counts["native_ready"],
            "n_overall_ready": counts["overall_ready"],
            "primary_status": "PASS" if counts["primary_eligible"] >= MIN_ADMITTED else "FAIL",
            "native_status": "PASS" if counts["native_ready"] >= MIN_ADMITTED else "FAIL",
            "overall_status": "PASS" if counts["overall_ready"] >= MIN_ADMITTED else "FAIL",
            "per_base": per_base,
            "primary_failure_reasons": dict(Counter(r["primary_reason"] for r in rows if not r["primary_eligible"])),
            "native_failure_reasons": dict(Counter(r["native_reason"] for r in rows if not r["native_ready"])),
        }
    paired = {}
    for key in ("primary_eligible", "native_ready", "overall_ready"):
        differences = [int(c["arms"]["roundtrip"][key]) - int(c["arms"]["baseline"][key]) for c in cases]
        base_differences = [sum(d for d, c in zip(differences, cases) if c["base"] == b) / len(GOALS)
                            for b in range(N_BASES)]
        paired[key] = {"case_difference_count": sum(differences),
                       "case_differences": differences,
                       "per_base_fraction_differences": base_differences,
                       "source_macro_fraction_difference": sum(base_differences) / N_BASES,
                       "evidence_class": "DESCRIPTIVE_PAIRED_ENGINEERING"}
    return summaries, paired


def _validate_arm(plan, *, arm, shared, case, arrays, counted_hvp,
                  model32, model64, h32, h64, metadata, head_projector):
    result = case["arms"][arm]
    result["validation_stage"] = "primary"
    result.update({"construction_eligible": bool(plan.eligible), "n_shams": len(plan.shams),
                   "selection": plan.receipt, "primary_reason": plan.reason})
    prefix = case["raw_prefix"] + arm + "__"
    _capture_raw(arrays, prefix, plan.raw)
    selected = shared["selected"]
    intended = torch.stack([selected, *[v.reshape(-1) for v in plan.shams]])
    hpair = shared["hpair"]
    actual64 = (hpair[None] + intended) - hpair[None]
    actual32 = ((hpair.float()[None] + intended.float()).double()
                - hpair.float().double()[None])
    _capture_raw(arrays, prefix, {"intended_edits": intended,
                                 "actual_edits_fp64": actual64, "actual_edits_fp32": actual32})
    hv64 = torch.stack([counted_hvp(v, arm + "_final_fp64") for v in actual64])
    hv32 = torch.stack([counted_hvp(v, arm + "_final_fp32") for v in actual32])
    _capture_raw(arrays, prefix, {"intended_edits": intended,
                                 "actual_edits_fp64": actual64, "actual_edits_fp32": actual32,
                                 "direct_hvps_fp64": hv64, "direct_hvps_fp32": hv32})
    args = {"selected": selected, "target": shared["target"], "k": shared["k"],
            "jacobian": shared["jacobian"], "head": shared["head"],
            "egrad": shared["egrad"], "row_basis": shared["row_basis"], "q0": shared["q0"]}
    check64 = _physical_checks(actual64, hv64, fp32=False, **args)
    check32 = _physical_checks(actual32, hv32, fp32=True, **args)
    result["primary_fp64"] = check64
    result["primary_fp32"] = check32
    dense64 = actual64 @ shared["hessian"].T
    dense32 = actual32 @ shared["hessian"].T
    result["dense_vs_fresh_hvp"] = {
        "max_fp64_vector_error": float(torch.linalg.vector_norm(dense64 - hv64, dim=1).max()),
        "max_fp32_vector_error": float(torch.linalg.vector_norm(dense32 - hv32, dim=1).max()),
        "max_fp64_quadratic_error": float(((actual64 * (dense64 - hv64)).sum(-1)).abs().max()),
        "max_fp32_quadratic_error": float(((actual32 * (dense32 - hv32)).sum(-1)).abs().max()),
        "quadratic_limit": 1e-10 * max(1., abs(shared["target"])),
    }
    dense_check = result["dense_vs_fresh_hvp"]
    dense_check["passed"] = bool(max(dense_check["max_fp64_quadratic_error"],
                                     dense_check["max_fp32_quadratic_error"])
                                 <= dense_check["quadratic_limit"])
    result["primary_eligible"] = bool(plan.eligible and len(plan.shams) == N_SHAMS
                                      and check64["passed"] and check32["passed"]
                                      and dense_check["passed"] and shared["dense_passed"])
    if result["primary_eligible"]:
        result["primary_reason"] = None
    elif not shared["dense_passed"]:
        result["primary_reason"] = "DENSE_HESSIAN_SYMMETRY_FAILURE"
    elif plan.eligible and not check64["passed"]:
        result["primary_reason"] = "PRIMARY_FP64_APPLICATION_ACCURACY"
    elif plan.eligible and not check32["passed"]:
        result["primary_reason"] = "PRIMARY_FP32_APPLICATION_ACCURACY"
    elif plan.eligible and not dense_check["passed"]:
        result["primary_reason"] = "DENSE_FRESH_CURVATURE_DISAGREEMENT"
    elif not result["primary_reason"]:
        result["primary_reason"] = "INCOMPLETE_SHAM_SET"
    result["validation_stage"] = "primary_complete"


def _validate_native_arm(plan, *, arm, shared, case, arrays, model32, model64,
                         h32, h64, metadata, head_projector):
    result = case["arms"][arm]
    result["validation_stage"] = "native"
    prefix = case["raw_prefix"] + arm + "__"
    if len(plan.shams) == N_SHAMS:
        native = native_precision_check(
            model32, model64, h32, shared["selected"], plan.shams, goal=case["goal"],
            donor_class=metadata["donor_goal_label"], original_class=metadata["original_goal_label"],
            reference_h64=h64, reference_j64=shared["jacobian"],
            head_projector=head_projector, target_curvature=shared["target"])
        result["native"] = native.receipt
        _capture_raw(arrays, prefix + "native__", native.raw)
        result["native_ready"] = bool(native.receipt["native_ready"])
        result["native_reason"] = (None if result["native_ready"] else "NATIVE_PRECISION_GATE")
    else:
        result["native_ready"] = False
        result["native_reason"] = "NO_COMPLETE_SHAM_SET"
    result["overall_ready"] = bool(result["primary_eligible"] and result["native_ready"])
    result["validation_stage"] = "complete"


def _validate_captured_arms(plans, *, shared, case, arrays, counted_hvp,
                            model32, model64, h32, h64, metadata, head_projector):
    """Archive both selections before either independent validation can fail."""
    for arm, plan in plans.items():
        if plan is None:
            case["arms"][arm]["primary_reason"] = "DENSE_HESSIAN_SYMMETRY_FAILURE"
            continue
        case["arms"][arm].update({"construction_eligible": bool(plan.eligible),
                                 "n_shams": len(plan.shams), "selection": plan.receipt,
                                 "primary_reason": plan.reason})
        _capture_raw(arrays, case["raw_prefix"] + arm + "__", plan.raw)
    for arm, plan in plans.items():
        if plan is None:
            continue
        try:
            _validate_arm(plan, arm=arm, shared=shared, case=case, arrays=arrays,
                counted_hvp=counted_hvp, model32=model32, model64=model64, h32=h32, h64=h64,
                metadata=metadata, head_projector=head_projector)
        except (ArithmeticError, ValueError, RuntimeError, KeyError) as error:
            result = case["arms"][arm]
            result["primary_validation_exception"] = {"stage": "primary", "type": type(error).__name__, "message": str(error)}
            result["primary_eligible"] = False
            result["primary_reason"] = "EXPLICIT_PRIMARY_VALIDATION_EXCEPTION"
        # A failed primary measurement never skips a complete arm's native endpoint.
        try:
            _validate_native_arm(plan, arm=arm, shared=shared, case=case, arrays=arrays,
                model32=model32, model64=model64, h32=h32, h64=h64,
                metadata=metadata, head_projector=head_projector)
        except (ArithmeticError, ValueError, RuntimeError, KeyError) as error:
            result = case["arms"][arm]
            result["native_validation_exception"] = {"stage": "native", "type": type(error).__name__, "message": str(error)}
            result["native_ready"] = False
            result["native_reason"] = "EXPLICIT_NATIVE_VALIDATION_EXCEPTION"
            result["overall_ready"] = False


def run(out: Path, raw: Path, *, freeze_commit: str):
    out, raw = Path(out), Path(raw)
    if out.exists() or raw.exists():
        raise FileExistsError("first paired pilot receipts must not be overwritten")
    provenance = _provenance(freeze_commit)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    torch.manual_seed(EXCLUDED_SEED)
    model32 = TinyRecursiveModel().float().eval().requires_grad_(False)
    model64 = copy.deepcopy(model32).double()
    observer = QuadraticObserver()
    data, labels = fixed_data(N_BASES, EXCLUDED_SEED)
    head_projector = full_head_kernel(model64.head.weight).projector
    mu, std = torch.zeros(48, dtype=torch.float64), torch.ones(48, dtype=torch.float64)
    arrays = {"source_inputs": _array(data), "source_labels": _array(labels),
              "head_weight": _array(model64.head.weight), "head_weight_fp32": _array(model32.head.weight),
              "head_projector": _array(head_projector)}
    model_digest = hashlib.sha256()
    for name, value in model32.state_dict().items():
        model_digest.update(name.encode("utf-8") + b"\0")
        model_digest.update(np.asarray(_array(value), dtype="<f4").tobytes())
        if not torch.equal(value.double(), model64.state_dict()[name]):
            raise ArithmeticError("FP64 model does not exactly promote the single FP32 initialization")
    cases = []
    for base in range(N_BASES):
        x = data[base:base + 1]
        with torch.no_grad():
            h64, h32 = model64.reason_states(x)[2], model32.reason_states(x)[2]
        arrays[f"base_{base:02d}__hrec_fp64"] = _array(h64)
        arrays[f"base_{base:02d}__hrec_fp32"] = _array(h32)
        for goal in GOALS:
            started = time.perf_counter()
            prefix = f"case_{base:02d}_{goal:02d}__"
            case = {"base": base, "goal": goal, "raw_prefix": prefix,
                    "arms": {arm: _initial_arm() for arm in ARMS}, "case_reason": None}
            counts = {key: 0 for key in ("reference_construction", "dense_primary",
                      "baseline_final_fp64", "baseline_final_fp32", "roundtrip_final_fp64", "roundtrip_final_fp32")}
            seconds = {key: 0. for key in counts}
            try:
                donor, metadata = legal_predecessor_donor(x[0], goal)
                case["donor_metadata"] = metadata
                arrays[prefix + "donor_input"] = _array(donor)
                with torch.no_grad():
                    hdon64 = model64.reason_states(donor[None])[2]
                parent = plan_pair(
                    model64, observer, h64, hdon64, mu, std, goal=goal,
                    old_class=metadata["original_goal_label"], donor_class=metadata["donor_goal_label"],
                    teacher_seed=EXCLUDED_SEED, base_index=base, head_projector=head_projector)
                if not parent["eligible"]:
                    case["case_reason"] = "PAIR_SOURCE_DEGENERATE"
                    for arm in ARMS:
                        case["arms"][arm]["primary_reason"] = case["case_reason"]
                else:
                    fiber = parent["fiber"]
                    future = measure_full_response(model64, h64, goal, parent, head_projector)
                    case.update({"k_dimension": fiber.dimension, "rank_future_j": future.rank,
                                 "n_full_j_dark": future.future_null_dimension})
                    hvp = native_margin_hvp(model64, h64, goal, metadata["donor_goal_label"],
                                            metadata["original_goal_label"])

                    def counted_hvp(v, category="reference_construction"):
                        start = time.perf_counter()
                        result = hvp(v)
                        counts[category] += 1
                        seconds[category] += time.perf_counter() - start
                        return result

                    def project(v):
                        return project_into_k(v.reshape(2, 48), fiber, head_projector).reshape(-1)

                    selected = parent["directions"]["pair_energy_tangent"].reshape(-1)
                    reference = make_spectral_shams(selected, future.row_basis, project, counted_hvp,
                        teacher_seed=EXCLUDED_SEED, base_index=base, goal=goal, expected_complement_dim=64)
                    case["reference"] = reference.receipt
                    _capture_raw(arrays, prefix + "reference__", reference.raw)
                    _capture_raw(arrays, prefix, {"future_jacobian": future.full_jacobian.reshape(10, 96),
                        "restricted_jacobian": future.restricted_jacobian,
                        "energy_gradient": fiber.egrad.reshape(-1), "natural_pair": fiber.donor.reshape(-1),
                        "hrec_pair": h64[0, [(goal - 1) % 81, goal]].reshape(-1)})
                    if "selected_curvature" not in reference.receipt:
                        case["case_reason"] = reference.reason
                        for arm in ARMS:
                            case["arms"][arm]["primary_reason"] = reference.reason
                    else:
                        dense, dense_receipt, dense_raw = dense_margin_hessian(
                            lambda v: counted_hvp(v, "dense_primary"))
                        case["dense_primary"] = dense_receipt
                        _capture_raw(arrays, prefix, {"dense_hessian": dense, "dense_hessian_raw": dense_raw})
                        shared = {"selected": selected, "hpair": h64[0, [(goal - 1) % 81, goal]].reshape(-1),
                                  "hessian": dense, "k": reference.raw["k_projector"],
                                  "dense_passed": bool(dense_receipt["passed"]),
                                  "jacobian": future.full_jacobian.reshape(10, 96),
                                  "head": model64.head.weight.detach(), "egrad": fiber.egrad.reshape(-1),
                                  "row_basis": future.row_basis, "q0": reference.raw["q0"],
                                  "target": reference.receipt["selected_curvature"]}
                        gate = PhysicalGate(selected=selected, hpair=shared["hpair"], hessian=dense,
                            k_projector=shared["k"], jacobian=shared["jacobian"], head=shared["head"],
                            egrad=shared["egrad"], row_basis=shared["row_basis"], q0=shared["q0"],
                            target_curvature=shared["target"])
                        # Freeze both arm selections before looking at either native precision gate.
                        roundtrip = (make_roundtrip_shams(reference, gate, teacher_seed=EXCLUDED_SEED,
                                     base_index=base, goal=goal, n_shams=N_SHAMS, max_attempts=32)
                                     if dense_receipt["passed"] else None)
                        _validate_captured_arms({"baseline": reference, "roundtrip": roundtrip},
                            shared=shared, case=case, arrays=arrays, counted_hvp=counted_hvp,
                            model32=model32, model64=model64, h32=h32, h64=h64,
                            metadata=metadata, head_projector=head_projector)
            except (ArithmeticError, ValueError, RuntimeError, KeyError) as error:
                case["case_reason"] = "EXPLICIT_CASE_EXCEPTION"
                case["exception"] = {"type": type(error).__name__, "message": str(error)}
                for arm in ARMS:
                    if case["arms"][arm]["primary_reason"] == "CASE_NOT_EVALUATED":
                        case["arms"][arm]["primary_reason"] = "EXPLICIT_CASE_EXCEPTION"
            case["hvp_counts"] = counts
            case["hvp_seconds"] = seconds
            case["elapsed_seconds"] = time.perf_counter() - started
            cases.append(case)
        print("LGA10C_BASE", base, {arm: {key: sum(c["arms"][arm][key] for c in cases if c["base"] == base)
              for key in ("primary_eligible", "native_ready", "overall_ready")} for arm in ARMS}, flush=True)
    raw.parent.mkdir(parents=True, exist_ok=True)
    with raw.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    summaries, paired = _summaries(cases)
    native_counts = {key: sum(c["arms"][arm].get("native", {}).get("hvp_counts", {}).get(key, 0)
                             for c in cases for arm in ARMS)
                     for key in ("native32", "promoted_exact", "promoted_probe")}
    record = {
        "schema": "LGA10C/excluded-source-61066-paired-roundtrip-v1",
        "evidence_class": "SOURCE_ONLY_UNTRAINED_PAIRED_ENGINEERING",
        "experimental_unit": "independent source input with paired arms and eight correlated goals",
        "excluded_source_seed": EXCLUDED_SEED, "excluded_model_seed": EXCLUDED_SEED,
        "n_excluded_bases": N_BASES, "n_goals_per_base": len(GOALS), "n_cases": len(cases),
        "n_untrained_model_initializations": 1, "n_dtype_copies_of_same_model": 2,
        "n_new_trained_model_roots": 0, "reserved_science_seeds_unopened": list(RESERVED_UNOPENED_SEEDS),
        "trained_weights_loaded": False, "edited_task_outcomes_scored": False, "new_test_generated": False,
        "source_inputs_sha256": tensor_int64_sha(data), "source_labels_sha256": tensor_int64_sha(labels),
        "untrained_fp32_tensor_sha256": model_digest.hexdigest(),
        "n_model_parameters": sum(p.numel() for p in model32.parameters()),
        "registered_min_admitted": MIN_ADMITTED,
        "primary_gate": "all eight shams pass frozen10B FP64 and FP32-application checks with fresh primary HVPs",
        "native_gate": "all eight shams pass physical native, promoted curvature, resolution, and derivative agreement gates",
        "native_gate_used_in_proposal_search": False, "paired_proposal_seed_namespace": "LGA10B",
        "arms": summaries, "paired": paired, "cases": cases,
        "hvp_counts": {key: sum(c["hvp_counts"][key] for c in cases) for key in counts},
        "native_hvp_counts": native_counts,
        "hvp_seconds": {key: sum(c["hvp_seconds"][key] for c in cases) for key in seconds},
        "elapsed_seconds": time.perf_counter() - wall_start, "cpu_seconds": time.process_time() - cpu_start,
        "process_peak_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "environment": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                        "platform": platform.platform(), "device": "cpu", "torch_threads": torch.get_num_threads(),
                        "deterministic_algorithms": True, "primary_dtype": "float64", "native_dtype": "float32",
                        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
                        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT")},
        "provenance": provenance, "raw_file": raw.name, "raw_sha256": _sha(raw),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    for path in (out, raw):
        path.with_suffix(path.suffix + ".sha256").write_text(f"{_sha(path)}  {path.name}\n")
    print("LGA10C_PAIRED_SOURCE_FEASIBILITY", json.dumps(summaries, sort_keys=True), flush=True)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--freeze-commit", required=True)
    args = parser.parse_args()
    run(args.out, args.raw, freeze_commit=args.freeze_commit)

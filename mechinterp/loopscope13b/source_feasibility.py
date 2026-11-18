"""Frozen LGA-10B excluded-source engineering pilot, with complete raw receipts.

Only run after the implementation commit is frozen. This module never loads
trained tensors and never generates a reserved science source. All 64 outcomes
are retained; eight goal variants remain clustered within each of eight inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
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

from mechinterp.loopscope04.gradient_flow import (
    TinyRecursiveModel, fixed_data, tensor_int64_sha,
)
from mechinterp.loopscope10.donor_source import legal_predecessor_donor
from mechinterp.loopscope10.global_energy_fiber import full_head_kernel
from mechinterp.loopscope11.pair_fiber import plan_pair, project_into_k
from mechinterp.loopscope12.full_jacobian import measure_full_response
from mechinterp.loopscope13.curvature_match import native_margin_hvp
from mechinterp.loopscope13b.spectral_match import make_spectral_shams

EXCLUDED_SEED = 61065
RESERVED_UNOPENED_SEEDS = (31059, 31064, 31065)
GOALS = (0, 10, 20, 30, 40, 50, 60, 70)
N_BASES = 8
N_SHAMS = 8
MIN_ADMITTED = 61
CURVATURE_RTOL = 1e-8
TOLERANCES = {
    "curvature_rtol": CURVATURE_RTOL,
    "fp64_j_rtol": 1e-8,
    "fp64_k_dose_head_energy_rtol": 1e-7,
    "fp64_angle_atol": 1e-12,
    "fp32_j_k_dose_head_energy_rtol": 1e-6,
    "fp32_angle_atol": 2e-6,
    "max_source_abs_cosine": .85,
    "max_pairwise_signed_cosine": .98,
}


class QuadraticObserver(torch.nn.Module):
    def forward(self, h, k, mu, std):
        return .5 * ((h - mu) / std).square().sum(-1)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _frozen_provenance(freeze_commit: str):
    """Require a concrete clean checkout before the first excluded generation."""
    if not re.fullmatch(r"[0-9a-f]{40}", freeze_commit or ""):
        raise ValueError("--freeze-commit requires the full pre-pilot commit SHA")
    root = Path(__file__).resolve().parents[2]
    paths = (
        "AGENTS.md",
        "mechinterp/loopscope13b/PROTOCOL.md",
        "mechinterp/loopscope13b/IMPLEMENTATION_FREEZE.md",
        "mechinterp/loopscope13b/spectral_match.py",
        "mechinterp/loopscope13b/sphere_solver.py",
        "mechinterp/loopscope13b/source_feasibility.py",
        "mechinterp/loopscope13b/audit_receipt.py",
        "mechinterp/loopscope13b/tests/test_sphere_solver.py",
        "mechinterp/loopscope13b/tests/test_spectral_match.py",
        "mechinterp/loopscope13b/tests/test_native_hvp.py",
        ".github/workflows/loopscope13b-spectral-source-only.yml",
        "mechinterp/loopscope13/curvature_match.py",
        "mechinterp/loopscope12/full_jacobian.py",
        "mechinterp/loopscope11/pair_fiber.py",
        "mechinterp/loopscope10/global_energy_fiber.py",
        "mechinterp/loopscope10/energy_fiber.py",
        "mechinterp/loopscope10/donor_source.py",
        "mechinterp/loopscope04/gradient_flow.py",
    )
    hashes = {p: _sha_file(root / p) for p in paths}
    checked_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if checked_head != freeze_commit:
        raise ValueError("checkout HEAD differs from the supplied pre-pilot freeze")
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *paths],
                   cwd=root, check=True, capture_output=True)
    return {"freeze_commit": freeze_commit, "source_sha256": hashes}


def _physical_checks(edits, direct_hvps, selected, target, k, jacobian,
                     head, egrad, row_basis, q0, *, fp32):
    """Check the applied vectors; index zero is the applied donor reference."""
    radius = float(selected.norm())
    dark_radius = float(q0.norm())
    values = (edits * direct_hvps).sum(-1)
    reference_q = float(values[0])
    q_scale = max(1., abs(target))
    rtol = 1e-6 if fp32 else 1e-7
    jtol = 1e-6 if fp32 else 1e-8
    angle_tol = 2e-6 if fp32 else 1e-12
    ref = edits[0]
    # Curvature compares each physically applied sham with its applied donor.
    q_error = float((values[1:] - reference_q).abs().max()) if len(edits) > 1 else 0.
    intended_q_error = float((values - target).abs().max())
    j_errors = torch.linalg.vector_norm((edits - ref) @ jacobian.T, dim=1)
    k_errors = torch.linalg.vector_norm(edits @ k.T - edits, dim=1)
    dose_errors = (torch.linalg.vector_norm(edits, dim=1) - radius).abs()
    head_errors = torch.linalg.vector_norm(
        edits.reshape(-1, 2, 48) @ head.T, dim=(1, 2))
    energy_errors = (edits @ egrad).abs()
    dark = edits @ k.T - (edits @ row_basis) @ row_basis.T
    dark_norms = torch.linalg.vector_norm(dark, dim=1, keepdim=True)
    zero_dark = bool((dark_norms == 0).any())
    dark_units = dark / dark_norms.clamp_min(torch.finfo(torch.float64).tiny)
    source_cosines = dark_units[1:] @ dark_units[0]
    gram = dark_units[1:] @ dark_units[1:].T
    pairwise = (float(gram[torch.triu(torch.ones_like(gram, dtype=torch.bool),
                                     diagonal=1)].max()) if len(edits) > 2 else -1.)
    head_scale = max(1., float(torch.linalg.matrix_norm(head, ord=2)) * radius)
    energy_scale = max(1., float(egrad.norm()) * radius)
    limits = {
        "curvature": CURVATURE_RTOL * q_scale,
        "full_j": jtol * max(1., float((jacobian @ ref).norm())),
        "k": rtol * max(1., radius),
        "dose": rtol * max(1., radius),
        "head": rtol * head_scale,
        "energy": rtol * energy_scale,
        "source_abs_cosine": .85 + angle_tol,
        "pairwise_signed_cosine": .98 + angle_tol,
    }
    measured = {
        "curvature": q_error,
        "full_j": float(j_errors.max()),
        "k": float(k_errors.max()),
        "dose": float(dose_errors.max()),
        "head": float(head_errors.max()),
        "energy": float(energy_errors.max()),
        "source_abs_cosine": (float(source_cosines.abs().max())
                              if source_cosines.numel() else 0.),
        "pairwise_signed_cosine": pairwise,
    }
    finite = all(np.isfinite(v) for v in measured.values())
    passed = finite and not zero_dark and all(measured[key] <= bound for key, bound in limits.items())
    # FP64 must also agree with the construction target, including the donor.
    if not fp32:
        passed = passed and intended_q_error <= limits["curvature"]
    return {
        "passed": bool(passed), "max_errors": measured, "limits": limits,
        "applied_selected_curvature": reference_q,
        "intended_selected_curvature": target,
        "selected_curvature_drift": abs(reference_q - target),
        "max_curvature_error_to_intended": intended_q_error,
        "all_applied_curvatures": values.tolist(),
        "source_cosines": source_cosines.tolist(),
        "dark_radii": dark_norms.reshape(-1).tolist(),
        "zero_actual_dark_radius": zero_dark,
        "selected_null_to_full_radius": dark_radius / radius,
    }


def run(out: Path, raw: Path, *, freeze_commit: str):
    out, raw = Path(out), Path(raw)
    if out.exists() or raw.exists():
        raise FileExistsError("first pilot receipts must not be overwritten")
    provenance = _frozen_provenance(freeze_commit)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    torch.manual_seed(EXCLUDED_SEED)
    model = TinyRecursiveModel().double().eval().requires_grad_(False)
    observer = QuadraticObserver()
    data, labels = fixed_data(N_BASES, EXCLUDED_SEED)
    projector = full_head_kernel(model.head.weight).projector
    mu = torch.zeros(48, dtype=torch.float64)
    std = torch.ones(48, dtype=torch.float64)
    arrays = {
        "source_inputs": _numpy(data), "source_labels": _numpy(labels),
        "head_weight": _numpy(model.head.weight),
        "head_projector": _numpy(projector),
    }
    model_digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        model_digest.update(name.encode("utf-8") + b"\0")
        model_digest.update(np.asarray(_numpy(value), dtype="<f8").tobytes())
    cases = []
    for base in range(N_BASES):
        x = data[base:base + 1]
        with torch.no_grad():
            hrec = model.reason_states(x)[2]
        for goal in GOALS:
            case_start = time.perf_counter()
            prefix = f"case_{base:02d}_{goal:02d}__"
            counts = {"construction": 0, "validation_fp64": 0, "validation_fp32": 0}
            hvp_seconds = dict.fromkeys(counts, 0.)
            record = {
                "base": base, "goal": goal, "raw_prefix": prefix,
                "eligible": False, "fp64_eligible": False,
                "fp32_addition_eligible": False, "matcher_eligible": False,
                "reason": None, "n_curvature_shams": 0,
            }
            try:
                donor, metadata = legal_predecessor_donor(x[0], goal)
                record["donor_metadata"] = metadata
                arrays[prefix + "donor_input"] = _numpy(donor)
                with torch.no_grad():
                    hdon = model.reason_states(donor[None])[2]
                parent = plan_pair(
                    model, observer, hrec, hdon, mu, std, goal=goal,
                    old_class=metadata["original_goal_label"],
                    donor_class=metadata["donor_goal_label"],
                    teacher_seed=EXCLUDED_SEED, base_index=base,
                    head_projector=projector)
                if not parent["eligible"]:
                    record["reason"] = "PAIR_SOURCE_DEGENERATE"
                    record["pair_reason"] = parent.get("reason")
                else:
                    future = measure_full_response(model, hrec, goal, parent, projector)
                    fiber = parent["fiber"]
                    record.update({"k_dimension": fiber.dimension,
                                   "rank_future_j": future.rank,
                                   "n_full_j_dark": future.future_null_dimension})
                    native_hvp = native_margin_hvp(
                        model, hrec, goal, metadata["donor_goal_label"],
                        metadata["original_goal_label"])

                    def counted_hvp(v, category="construction"):
                        start = time.perf_counter()
                        result = native_hvp(v)
                        counts[category] += 1
                        hvp_seconds[category] += time.perf_counter() - start
                        return result

                    def project(v):
                        return project_into_k(v.reshape(2, 48), fiber, projector).reshape(-1)

                    selected = parent["directions"]["pair_energy_tangent"].reshape(-1)
                    plan = make_spectral_shams(
                        selected, future.row_basis, project, counted_hvp,
                        teacher_seed=EXCLUDED_SEED, base_index=base, goal=goal,
                        expected_complement_dim=64)
                    record.update({"matcher_eligible": bool(plan.eligible),
                                   "reason": plan.reason,
                                   "n_curvature_shams": len(plan.shams),
                                   "spectral": plan.receipt})
                    for name, value in plan.raw.items():
                        arrays[prefix + name] = _numpy(value)
                    arrays[prefix + "future_jacobian"] = _numpy(
                        future.full_jacobian.reshape(10, 96))
                    arrays[prefix + "restricted_jacobian"] = _numpy(future.restricted_jacobian)
                    arrays[prefix + "energy_gradient"] = _numpy(fiber.egrad.reshape(-1))
                    arrays[prefix + "natural_pair"] = _numpy(fiber.donor.reshape(-1))
                    arrays[prefix + "hrec_pair"] = _numpy(
                        hrec[0, [(goal - 1) % 81, goal]].reshape(-1))
                    arrays[prefix + "future_singular_values"] = _numpy(future.singular_values)
                    if "selected_curvature" in plan.receipt:
                        edits = torch.stack([selected, *[z.reshape(-1) for z in plan.shams]])
                        hpair = hrec[0, [(goal - 1) % 81, goal]].reshape(-1)
                        actual64 = (hpair[None] + edits) - hpair[None]
                        actual32 = ((hpair.float()[None] + edits.float()).double()
                                    - hpair.float().double()[None])
                        direct64 = torch.stack([counted_hvp(z, "validation_fp64") for z in actual64])
                        direct32 = torch.stack([counted_hvp(z, "validation_fp32") for z in actual32])
                        for name, value in (("intended_edits", edits), ("actual_edits_fp64", actual64),
                                            ("actual_edits_fp32", actual32), ("direct_hvps_fp64", direct64),
                                            ("direct_hvps_fp32", direct32)):
                            arrays[prefix + name] = _numpy(value)
                        target = float(plan.receipt["selected_curvature"])
                        kwargs = dict(selected=selected, target=target,
                                      k=plan.raw["k_projector"],
                                      jacobian=future.full_jacobian.reshape(10, 96),
                                      head=model.head.weight.detach(),
                                      egrad=fiber.egrad.reshape(-1),
                                      row_basis=plan.raw["row_basis"], q0=plan.raw["q0"])
                        record["fp64_application"] = _physical_checks(actual64, direct64, fp32=False, **kwargs)
                        record["fp32_application"] = _physical_checks(actual32, direct32, fp32=True, **kwargs)
                        record["fp64_eligible"] = bool(plan.eligible and len(plan.shams) == N_SHAMS
                                                        and record["fp64_application"]["passed"])
                        record["fp32_addition_eligible"] = bool(
                            plan.eligible and len(plan.shams) == N_SHAMS
                            and record["fp32_application"]["passed"])
                        record["eligible"] = bool(record["fp64_eligible"]
                                                  and record["fp32_addition_eligible"])
                        if plan.eligible and not record["fp64_eligible"]:
                            record["reason"] = "NATIVE_FP64_APPLICATION_ACCURACY"
                        elif record["fp64_eligible"] and not record["fp32_addition_eligible"]:
                            record["reason"] = "FP32_APPLICATION_ACCURACY"
                        elif record["eligible"]:
                            record["reason"] = None
            except (ArithmeticError, ValueError, RuntimeError, KeyError) as error:
                record["eligible"] = False
                record["reason"] = "EXPLICIT_CASE_EXCEPTION"
                record["exception"] = {"type": type(error).__name__, "message": str(error)}
            record["hvp_counts"] = counts
            record["hvp_seconds"] = hvp_seconds
            record["elapsed_seconds"] = time.perf_counter() - case_start
            cases.append(record)
        print("LGA10B_BASE", base, "admitted", sum(c["eligible"] for c in cases if c["base"] == base),
              "of", len(GOALS), flush=True)
    raw.parent.mkdir(parents=True, exist_ok=True)
    with raw.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    n_admitted = sum(c["eligible"] for c in cases)
    record = {
        "schema": "LGA10B/excluded-source-61065-spectral-feasibility-v1",
        "evidence_class": "SOURCE_ONLY_UNTRAINED_ENGINEERING",
        "experimental_unit": "independently generated input within one untrained initialization",
        "n_excluded_bases": N_BASES, "n_goals_per_base": len(GOALS), "n_cases": len(cases),
        "n_untrained_model_initializations": 1, "n_new_trained_model_roots": 0,
        "excluded_source_seed": EXCLUDED_SEED, "excluded_model_seed": EXCLUDED_SEED,
        "reserved_science_seeds_unopened": list(RESERVED_UNOPENED_SEEDS),
        "trained_weights_loaded": False, "edited_task_outcomes_scored": False,
        "new_test_generated": False, "source_inputs_sha256": tensor_int64_sha(data),
        "source_labels_sha256": tensor_int64_sha(labels),
        "untrained_model_tensor_sha256": model_digest.hexdigest(),
        "n_model_parameters": sum(p.numel() for p in model.parameters()),
        "n_admitted": n_admitted,
        "n_fp64_admitted": sum(c["fp64_eligible"] for c in cases),
        "n_fp32_addition_admitted": sum(c["fp32_addition_eligible"] for c in cases),
        "admission_fraction": n_admitted / len(cases),
        "registered_min_admitted": MIN_ADMITTED,
        "pilot_feasibility_status": ("PASS_SOURCE_FEASIBILITY" if n_admitted >= MIN_ADMITTED
                                      else "FAIL_SOURCE_FEASIBILITY"),
        "rejection_reasons": dict(Counter(c["reason"] for c in cases if not c["eligible"])),
        "per_base_admissions": [sum(c["eligible"] for c in cases if c["base"] == b)
                                for b in range(N_BASES)],
        "per_base_fp64_admissions": [sum(c["fp64_eligible"] for c in cases if c["base"] == b)
                                     for b in range(N_BASES)],
        "hvp_counts": {key: sum(c["hvp_counts"][key] for c in cases) for key in counts},
        "hvp_seconds": {key: sum(c["hvp_seconds"][key] for c in cases) for key in counts},
        "elapsed_seconds": time.perf_counter() - wall_start,
        "cpu_seconds": time.process_time() - cpu_start,
        "process_peak_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "environment": {"python": sys.version, "torch": torch.__version__,
                        "numpy": np.__version__, "platform": platform.platform(),
                        "torch_threads": torch.get_num_threads(), "device": "cpu",
                        "native_model_dtype": "float64", "deterministic_algorithms": True,
                        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
                        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT")},
        "provenance": provenance, "tolerances": TOLERANCES,
        "fp32_scope": "FP32 state-plus-edit rounding checked with frozen FP64 native derivatives",
        "raw_file": raw.name, "raw_sha256": _sha_file(raw), "cases": cases,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    out.with_suffix(out.suffix + ".sha256").write_text(f"{_sha_file(out)}  {out.name}\n")
    raw.with_suffix(raw.suffix + ".sha256").write_text(f"{_sha_file(raw)}  {raw.name}\n")
    print("LGA10B_SOURCE_ONLY_FEASIBILITY", record["pilot_feasibility_status"],
          n_admitted, len(cases), "FP64", record["n_fp64_admitted"],
          "reasons", record["rejection_reasons"], flush=True)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--freeze-commit", required=True)
    args = parser.parse_args()
    run(args.out, args.raw, freeze_commit=args.freeze_commit)

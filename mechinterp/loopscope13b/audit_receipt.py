"""Independent NumPy audit of frozen LGA-10B receipts, with no model access.

This file imports neither the matcher nor its sphere solver. It recomputes
physical constraints and numerical global-extremum conditions from raw arrays.
Floating point residual checks are evidence, rather than interval arithmetic.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

GOALS = (0, 10, 20, 30, 40, 50, 60, 70)
COSINES = (-.85, -.65, -.35, 0., .35, .65, .85)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _norm(value):
    return float(np.linalg.norm(value))


def _same_scalar(left, right, *, atol=1e-10, rtol=1e-9):
    return bool(np.isfinite(left) and np.isfinite(right)
                and abs(left - right) <= atol + rtol * max(abs(left), abs(right)))


def _checks(edits, hvps, selected, target, k, jacobian, head, egrad, row_basis,
            *, fp32):
    radius = _norm(selected)
    qvalues = np.einsum("ij,ij->i", edits, hvps)
    ref = edits[0]
    rtol, jtol = (1e-6, 1e-6) if fp32 else (1e-7, 1e-8)
    angle = 2e-6 if fp32 else 1e-12
    qdark = edits @ k.T - (edits @ row_basis) @ row_basis.T
    qnorm = np.linalg.norm(qdark, axis=1)
    zero_dark = bool(np.any(qnorm == 0))
    qunit = qdark / np.maximum(qnorm[:, None], np.finfo(np.float64).tiny)
    cosines = qunit[1:] @ qunit[0]
    gram = qunit[1:] @ qunit[1:].T
    pairwise = float(np.max(gram[np.triu_indices(len(gram), k=1)])) if len(gram) > 1 else -1.
    errors = {
        "curvature": float(np.max(np.abs(qvalues[1:] - qvalues[0]))) if len(edits) > 1 else 0.,
        "full_j": float(np.max(np.linalg.norm((edits - ref) @ jacobian.T, axis=1))),
        "k": float(np.max(np.linalg.norm(edits @ k.T - edits, axis=1))),
        "dose": float(np.max(np.abs(np.linalg.norm(edits, axis=1) - radius))),
        "head": float(np.max(np.linalg.norm(edits.reshape(-1, 2, 48) @ head.T, axis=(1, 2)))),
        "energy": float(np.max(np.abs(edits @ egrad))),
        "source_abs_cosine": float(np.max(np.abs(cosines))) if len(cosines) else 0.,
        "pairwise_signed_cosine": pairwise,
    }
    limits = {
        "curvature": 1e-8 * max(1., abs(target)),
        "full_j": jtol * max(1., _norm(jacobian @ ref)),
        "k": rtol * max(1., radius), "dose": rtol * max(1., radius),
        "head": rtol * max(1., float(np.linalg.norm(head, ord=2)) * radius),
        "energy": rtol * max(1., _norm(egrad) * radius),
        "source_abs_cosine": .85 + angle, "pairwise_signed_cosine": .98 + angle,
    }
    passed = not zero_dark and all(np.isfinite(v) and v <= limits[key] for key, v in errors.items())
    if not fp32:
        passed = passed and float(np.max(np.abs(qvalues - target))) <= limits["curvature"]
    return bool(passed), errors, limits


def audit(receipt: Path, raw: Path, out: Path | None = None):
    receipt, raw = Path(receipt), Path(raw)
    document = json.loads(receipt.read_text())
    failures = []
    cases_out = []

    def require(condition, message):
        if not condition:
            failures.append(message)

    require(document["raw_sha256"] == _sha(raw), "raw NPZ digest differs")
    sidecar = receipt.with_suffix(receipt.suffix + ".sha256")
    require(sidecar.exists() and sidecar.read_text().split()[0] == _sha(receipt),
            "receipt digest sidecar differs or is missing")
    require(document["excluded_source_seed"] == 61065, "wrong excluded source seed")
    require(document["n_excluded_bases"] == 8 and document["n_goals_per_base"] == 8,
            "independent source / goal counts differ")
    require(document["n_untrained_model_initializations"] == 1
            and document["n_new_trained_model_roots"] == 0, "wrong model evidence class")
    require(document["new_test_generated"] is False and document["trained_weights_loaded"] is False
            and document["edited_task_outcomes_scored"] is False, "source-only boundary differs")
    require(document["registered_min_admitted"] == 61, "admission gate changed")
    cases = document["cases"]
    require([(c["base"], c["goal"]) for c in cases] == [(b, g) for b in range(8) for g in GOALS],
            "64 ordered outcomes are missing, duplicated, or reordered")
    with np.load(raw, allow_pickle=False) as arrays:
        x, y = arrays["source_inputs"], arrays["source_labels"]
        require(x.shape == (8, 81) and y.shape == (8, 81), "raw source dimensions differ")
        require(np.all((x >= 1) & (x <= 9)) and np.array_equal(y, np.roll(x, 1, axis=1)),
                "source law differs")
        require(hashlib.sha256(np.asarray(x, dtype="<i8").tobytes()).hexdigest()
                == document["source_inputs_sha256"], "raw source SHA differs")
        require(hashlib.sha256(np.asarray(y, dtype="<i8").tobytes()).hexdigest()
                == document["source_labels_sha256"], "raw source label SHA differs")
        head, hp = arrays["head_weight"], arrays["head_projector"]
        require(head.shape == (10, 48) and hp.shape == (48, 48), "head shape differs")
        require(_norm(head @ hp) <= 1e-8 and _norm(hp @ hp - hp) <= 1e-8,
                "raw head projector is invalid")
        for case in cases:
            first_error = len(failures)
            base, goal, prefix = case["base"], case["goal"], case["raw_prefix"]
            label = f"base {base} goal {goal}: "

            def check(condition, message):
                require(condition, label + message)

            def a(key):
                return np.asarray(arrays[prefix + key], dtype=np.float64)

            outcome = {"base": base, "goal": goal, "recorded_eligible": case["eligible"],
                       "recorded_reason": case["reason"]}
            try:
                donor = arrays[prefix + "donor_input"]
                expected = x[base].copy()
                p = (goal - 1) % 81
                expected[p] = int(expected[p]) % 9 + 1
                check(np.array_equal(donor, expected), "legal donor differs")
                metadata = case["donor_metadata"]
                check(metadata["original_goal_label"] == int(x[base, p])
                      and metadata["donor_goal_label"] == int(expected[p])
                      and metadata["changed_targets"] == [goal], "source class metadata differs")
                if "spectral" not in case:
                    check(not case["eligible"] and bool(case["reason"]),
                          "missing matcher receipt on an admitted case")
                    outcome["geometry_evaluation"] = "PRESERVED_EXPLICIT_PREMATCH_FAILURE"
                else:
                    s = case["spectral"]
                    z, z0, q0 = a("selected"), a("z0"), a("q0")
                    R, K = a("row_basis"), a("k_projector")
                    rho, r = _norm(z), _norm(q0)
                    check(_norm(z - z0 - q0) <= 1e-9 * max(1., rho), "donor split differs")
                    check(_norm(z0 - R @ (R.T @ z)) <= 1e-9 * max(1., rho), "rowspace donor differs")
                    check(_norm(K - K.T) <= 1e-8 and _norm(K @ K - K) <= 1e-8,
                          "K projector is invalid")
                    check(_norm(R.T @ R - np.eye(R.shape[1])) <= 1e-8, "row basis is not orthonormal")
                    check(_norm(K @ R - R) <= 1e-8, "row basis lies outside K")
                    natural = a("natural_pair")
                    check(abs(_norm(natural) - rho) <= 1e-7 * max(1., rho), "natural dose differs")
                    check(case["hvp_counts"]["construction"] == s["construction_hvp_count"],
                          "construction HVP count differs")
                    if "selected_curvature" in s:
                        V, hv, hz0, hq0 = a("complement_basis"), a("hv"), a("hz0"), a("hq0")
                        M, J, g = a("projected_hessian"), a("future_jacobian"), a("energy_gradient")
                        target = float(z @ (hz0 + hq0))
                        check(_same_scalar(target, s["selected_curvature"]), "selected quadratic differs")
                        check(_same_scalar(rho, s["radius"]) and _same_scalar(r, s["null_radius"]),
                              "recorded radii differ")
                        check(V.shape == (96, 64) and case["k_dimension"] == 75
                              and case["rank_future_j"] == 10 and case["n_full_j_dark"] == 65,
                              "registered physical dimensions differ")
                        check(_norm(V.T @ V - np.eye(V.shape[1])) <= 1e-8
                              and _norm(V.T @ q0) <= 1e-8 * max(1., r)
                              and _norm(K @ V - V) <= 1e-8
                              and _norm(R.T @ V) <= 1e-8, "complement basis differs")
                        check(_norm(J @ K - (J @ K @ R) @ R.T) <= 1e-8,
                              "full-J response is not spanned by the stored rowspace")
                        check(_norm(M - .5 * (V.T @ hv + hv.T @ V)) <= 1e-9 * max(1., _norm(M)),
                              "projected Hessian differs")
                        check(_norm(np.linalg.eigvalsh(M) - a("projected_hessian_eigenvalues"))
                              <= 1e-8 * max(1., _norm(M)), "Hessian spectrum differs")
                        # Reconstruct K from physical current-head and scalar-energy data.
                        block_p = np.zeros((96, 96))
                        block_p[:48, :48] = hp
                        block_p[48:, 48:] = hp
                        gh = block_p @ g
                        expected_k = block_p - np.outer(gh, gh) / float(gh @ gh)
                        check(_norm(K - expected_k) <= 1e-8, "K differs from head plus energy constraints")
                        extrema_residuals = []
                        independent_outer_intervals = []
                        if s["slices"]:
                            check([entry["cosine"] for entry in s["slices"]] == list(COSINES),
                                  "frozen cosine slice order differs")
                            minima, maxima = a("min_vectors"), a("max_vectors")
                            for i, entry in enumerate(s["slices"]):
                                c = entry["cosine"]
                                sine = np.sqrt(1 - c * c)
                                A = r * r * sine * sine * M
                                b = r * sine * (V.T @ (hz0 + c * hq0))
                                constant = float((z0 + c * q0) @ (hz0 + c * hq0))
                                check(_same_scalar(constant, entry["constant"]), "slice constant differs")
                                scale = max(1., float(np.linalg.norm(A, ord=2)), _norm(b))
                                extrema_values = []
                                independent_bounds = []
                                for kind, vector in (("minimum", minima[i]), ("maximum", maxima[i])):
                                    lam = float(entry[kind]["multiplier"])
                                    residual = _norm(A @ vector + b - lam * vector)
                                    shifted = (A - lam * np.eye(len(b)) if kind == "minimum"
                                               else lam * np.eye(len(b)) - A)
                                    gap = float(np.linalg.eigvalsh(shifted)[0])
                                    value = constant + float(vector @ A @ vector + 2 * b @ vector)
                                    valid = (_norm(vector) - 1.) ** 2 <= 1e-18 and residual <= 1e-8 * scale and gap >= -1e-8 * scale
                                    if entry[kind]["certificate_ok"]:
                                        check(valid, f"{kind} numerical certificate invalid")
                                    else:
                                        check(not case["matcher_eligible"], "failed extremum certificate admitted")
                                    check(_same_scalar(value, entry["q_min" if kind == "minimum" else "q_max"]),
                                          "reported extremal quadratic differs")
                                    # A second outer bound follows directly from the quadratic
                                    # expansion around the stored unit vector. For all unit x,
                                    # ||x-v|| <= 1+||v||. This uses no secular solver or inverse.
                                    sign = 1. if kind == "minimum" else -1.
                                    length = _norm(vector)
                                    allowance = 256 * np.finfo(np.float64).eps * (len(b) + 1) * max(
                                        1., abs(value - constant), _norm(A), _norm(b), abs(lam))
                                    independent_lower = (
                                        sign * (value - constant)
                                        + sign * lam * (1. - float(vector @ vector))
                                        + min(0., gap) * (1. + length)**2
                                        - 2. * residual * (1. + length) - allowance)
                                    stored_bound = float(entry[kind]["dual_bound"])
                                    check(sign * stored_bound <= independent_lower + allowance,
                                          kind + " stored bound exceeds independent residual bound")
                                    outer_name = "outer_lower_bound" if kind == "minimum" else "outer_upper_bound"
                                    check(_same_scalar(entry[outer_name], constant + stored_bound),
                                          kind + " outer interval differs from dual bound")
                                    independent_bounds.append(constant + sign * independent_lower)
                                    extrema_residuals.append(residual)
                                    extrema_values.append(value)
                                tol = 1e-8 * max(1., abs(target))
                                attained = extrema_values[0] - tol <= target <= extrema_values[1] + tol
                                check(bool(entry["attainable"]) == bool(attained), "attained interval flag differs")
                                independent_outer_intervals.append(independent_bounds)
                            outcome["max_extremizer_stationarity_residual"] = max(extrema_residuals)
                            if s["reason"] in ("GRID_NO_ATTAINABLE_TARGET", "SPHERE_INTERVAL_UNRESOLVED"):
                                any_attained = any(entry["attainable"] for entry in s["slices"])
                                any_outer_overlap = any(
                                    entry["outer_lower_bound"] - tol <= target <= entry["outer_upper_bound"] + tol
                                    for entry in s["slices"])
                                check(not any_attained, "negative or unresolved grid contains an attained slice")
                                check(any_outer_overlap == (s["reason"] == "SPHERE_INTERVAL_UNRESOLVED"),
                                      "negative grid versus unresolved outer interval differs")
                                if s["reason"] == "GRID_NO_ATTAINABLE_TARGET":
                                    check(all(not (lo - tol <= target <= hi + tol)
                                              for lo, hi in independent_outer_intervals),
                                          "independent residual bound does not exclude the grid target")
                        intended, actual64, actual32 = a("intended_edits"), a("actual_edits_fp64"), a("actual_edits_fp32")
                        hrec = a("hrec_pair")
                        check(np.array_equal(actual64, (hrec[None] + intended) - hrec[None]),
                              "FP64 physical addition differs")
                        rounded = ((hrec.astype(np.float32)[None] + intended.astype(np.float32)).astype(np.float64)
                                   - hrec.astype(np.float32).astype(np.float64)[None])
                        check(np.array_equal(actual32, rounded), "FP32 physical addition differs")
                        check(_norm(intended[0] - z) <= 1e-12, "raw selected edit differs")
                        check(len(intended) == case["n_curvature_shams"] + 1, "sham count differs")
                        check(case["hvp_counts"]["validation_fp64"] == len(intended)
                              and case["hvp_counts"]["validation_fp32"] == len(intended),
                              "independent validation HVP count differs")
                        Hspan = np.column_stack((hz0, hq0, hv))
                        basis = np.column_stack((z0, q0, V))
                        coordinates, _, _, _ = np.linalg.lstsq(basis, intended.T, rcond=1e-12)
                        check(_norm(basis @ coordinates - intended.T) <= 1e-8 * max(1., _norm(intended)),
                              "constructed edits leave the measured Hessian span")
                        reconstructed_h = Hspan @ coordinates
                        check(_norm(reconstructed_h.T - a("direct_hvps_fp64"))
                              <= 1e-8 * max(1., _norm(reconstructed_h)),
                              "fresh native HVP differs from basis combination")
                        pass_by_dtype = []
                        for suffix, edits in (("fp64", actual64), ("fp32", actual32)):
                            passed, errors, limits = _checks(
                                edits, a("direct_hvps_" + suffix), z, target, K, J, head, g, R,
                                fp32=suffix == "fp32")
                            reported = case[suffix + "_application"]
                            check(passed == reported["passed"], suffix + " application decision differs")
                            for key in errors:
                                check(_same_scalar(errors[key], reported["max_errors"][key]),
                                      suffix + " measured " + key + " differs")
                                check(_same_scalar(limits[key], reported["limits"][key], atol=1e-14),
                                      suffix + " tolerance " + key + " differs")
                            outcome[suffix + "_max_errors"] = errors
                            pass_by_dtype.append(passed)
                        independent_fp64 = bool(case["matcher_eligible"] and len(intended) == 9 and pass_by_dtype[0])
                        independent_fp32 = bool(case["matcher_eligible"] and len(intended) == 9 and pass_by_dtype[1])
                        independent_eligible = independent_fp64 and independent_fp32
                        check(case["fp64_eligible"] == independent_fp64
                              and case["fp32_addition_eligible"] == independent_fp32
                              and case["eligible"] == independent_eligible,
                              "final admission differs from raw physical vectors")
                        if case["matcher_eligible"]:
                            check(len(s["slices"]) == 7 and len(intended) == 9,
                                  "admitted plan lacks seven extrema or eight shams")
                        outcome["independent_eligible"] = independent_eligible
                    else:
                        check(not case["eligible"] and bool(case["reason"]),
                              "pre-Hessian failure admitted")
            except (KeyError, ValueError, ArithmeticError, IndexError, np.linalg.LinAlgError) as error:
                check(False, f"audit could not read complete case: {type(error).__name__}: {error}")
            outcome["audit_passed"] = len(failures) == first_error
            cases_out.append(outcome)
    admitted = sum(bool(c["eligible"]) for c in cases)
    require(document["n_cases"] == len(cases) == 64 and document["n_admitted"] == admitted,
            "aggregate case counts differ")
    require(document["n_fp64_admitted"] == sum(bool(c["fp64_eligible"]) for c in cases),
            "FP64 aggregate count differs")
    require(document["n_fp32_addition_admitted"] == sum(bool(c["fp32_addition_eligible"]) for c in cases),
            "FP32 aggregate count differs")
    require(document["per_base_admissions"] == [sum(c["eligible"] for c in cases if c["base"] == b)
                                                for b in range(8)], "source grouping differs")
    require(document["rejection_reasons"] == dict(Counter(c["reason"] for c in cases if not c["eligible"])),
            "rejection summary differs")
    require(document["pilot_feasibility_status"] == ("PASS_SOURCE_FEASIBILITY" if admitted >= 61
                                                       else "FAIL_SOURCE_FEASIBILITY"), "gate verdict differs")
    for key in ("construction", "validation_fp64", "validation_fp32"):
        require(document["hvp_counts"][key] == sum(c["hvp_counts"][key] for c in cases),
                "aggregate HVP count differs: " + key)
    result = {
        "schema": "LGA10B/independent-numpy-raw-audit-v1",
        "audit_status": "PASS" if not failures else "FAIL",
        "receipt_sha256": _sha(receipt), "raw_sha256": _sha(raw),
        "numpy_version": np.__version__, "model_or_matcher_imported": False,
        "n_cases": len(cases_out), "n_cases_audit_passed": sum(c["audit_passed"] for c in cases_out),
        "n_recorded_admitted": admitted, "failures": failures, "cases": cases_out,
    }
    if out is not None:
        Path(out).write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print("LGA10B_INDEPENDENT_RAW_AUDIT", result["audit_status"],
          result["n_cases_audit_passed"], len(cases_out), flush=True)
    for failure in failures:
        print("AUDIT_FAILURE", failure, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.receipt, args.raw, args.out)
    raise SystemExit(0 if result["audit_status"] == "PASS" else 1)

"""Independent NumPy audit of the paired LGA-10C raw receipt.

The auditor imports neither Torch, either matcher, the precision oracle, nor
the model. Source-law, applied-vector, spectral, stream and native-arithmetic
claims are reconstructed from archived numeric arrays.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from mechinterp.loopscope13b.audit_receipt import _checks as physical_checks

GOALS = (0, 10, 20, 30, 40, 50, 60, 70)
COSINES = (-.85, -.65, -.35, 0., .35, .65, .85)
ARMS = ("baseline", "roundtrip")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _norm(value):
    return float(np.linalg.norm(value))


def _close(left, right, *, atol=1e-11, rtol=1e-8):
    return bool(np.isfinite(left) and np.isfinite(right)
                and abs(left - right) <= atol + rtol * max(abs(left), abs(right)))


def _seed(*parts):
    payload = "LGA10B|" + "|".join(map(str, parts))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") & ((1 << 63) - 1)


def _path(start, end, t):
    if t == 0.:
        return start.copy()
    if t == 1.:
        return end.copy()
    dot = np.clip(float(start @ end), -1., 1.)
    tangent = end - dot * start
    length = _norm(tangent)
    if length < 1e-14:
        if dot > 0:
            v = (1 - t) * start + t * end
            return v / _norm(v)
        axis = int(np.argmin(np.abs(start)))
        tangent = np.zeros_like(start)
        tangent[axis] = 1.
        tangent -= start[axis] * start
        tangent /= _norm(tangent)
        angle = math.pi
    else:
        tangent /= length
        angle = math.atan2(length, dot)
    v = math.cos(t * angle) * start + math.sin(t * angle) * tangent
    return v / _norm(v)


def audit_native(native, get, check, *, target, intended, h32, h64, reference_j,
                 head_projector, head):
    """Recompute all four native gates using only saved raw values."""
    raw = lambda name: np.asarray(get(name))
    a = lambda name: np.asarray(get(name), dtype=np.float64)
    m = len(intended)
    check(m == 9 and native["n_shams"] == 8, "native gate lacks eight shams")
    check(np.array_equal(a("intended_edits"), intended), "native intended vectors differ")
    check(np.array_equal(raw("native_h32"), h32)
          and np.array_equal(a("native_h32_lifted"), h32.astype(np.float64))
          and np.array_equal(a("reference_h64"), h64), "native common clean states differ")
    check(np.array_equal(a("reference_j64"), reference_j), "native primary Jacobian reference differs")
    goal = int(native["goal"])
    pair = h32[0, [(goal - 1) % 81, goal]]
    intended32 = intended.astype(np.float32)
    after = pair[None] + intended32.reshape(m, 2, 48)
    delta = (after.astype(np.float64) - pair.astype(np.float64)[None]).reshape(m, 96)
    probe32 = delta.astype(np.float32)
    probe64 = probe32.astype(np.float64)
    check(np.array_equal(raw("native_pair_base_fp32"), pair)
          and np.array_equal(raw("native_pair_after_fp32"), after), "native physical state addition differs")
    check(np.array_equal(a("actual_edits_native_fp64"), delta)
          and np.array_equal(raw("intended_edits_fp32"), intended32), "native exact physical delta differs")
    check(np.array_equal(raw("probe32"), probe32) and np.array_equal(a("probe64"), probe64),
          "native and promoted probes differ")
    check(np.array_equal(a("direction_cast_residual"), delta - probe64)
          and np.array_equal(a("application_rounding_residual"), delta - intended),
          "native rounding residual differs")
    J, J64 = a("jacobian32"), a("jacobian64_same_state")
    H, Hprobe, Hexact = a("hvp32"), a("hvp64_same_probe"), a("hvp64_actual_delta")
    logits, logits64 = a("clean_logits32"), a("clean_logits64_same_state")
    for key in ("jacobian32", "hvp32", "clean_logits32", "native_h32", "probe32"):
        check(raw(key).dtype == np.float32, "native raw dtype differs: " + key)
    for key in ("jacobian64_same_state", "hvp64_same_probe", "hvp64_actual_delta", "native_h32_lifted"):
        check(raw(key).dtype == np.float64, "promoted raw dtype differs: " + key)
    check(J.shape == (10, 96) and J64.shape == (10, 96)
          and H.shape == (m, 96) and Hprobe.shape == (m, 96) and Hexact.shape == (m, 96),
          "native derivative shapes differ")
    check(np.array_equal(a("head_projector"), head_projector)
          and np.array_equal(a("native_head_weight"), head), "native head data differ")
    energy = pair.astype(np.float64).reshape(96)
    check(np.array_equal(a("energy_gradient_native"), energy), "native quadratic-energy gradient differs")
    pair_p = np.zeros((96, 96))
    pair_p[:48, :48] = head_projector
    pair_p[48:, 48:] = head_projector
    gh = pair_p @ energy
    active = _norm(gh) > max(1e-12, 1e-9 * _norm(energy))
    K = pair_p - np.outer(gh, gh) / float(gh @ gh) if active else pair_p
    K = .5 * (K + K.T)
    restricted = J @ K
    _, sv, vh = np.linalg.svd(restricted, full_matrices=False)
    cutoff = max(1e-12, 1e-9 * float(sv[0]))
    rank = int(np.sum(sv > cutoff))
    R = a("row_basis_native")
    check(R.shape == (96, rank) and _norm(R.T @ R - np.eye(rank)) <= 1e-8,
          "native rowspace shape or orthogonality differs")
    check(_norm(R @ R.T - vh[:rank].T @ vh[:rank]) <= 1e-7, "native full-J rowspace differs")
    N = .5 * (K - R @ R.T + (K - R @ R.T).T)
    check(_norm(K - a("k_native")) <= 1e-8 and _norm(N - a("null_projector_native")) <= 1e-8,
          "native physical projectors differ")
    check(_norm(restricted - a("native_restricted_jacobian")) <= 1e-8
          and _norm(sv - a("native_jacobian_singular_values")) <= 1e-8,
          "native restricted Jacobian spectrum differs")
    qnative = np.einsum("ij,ij->i", delta, H)
    qexact = np.einsum("ij,ij->i", delta, Hexact)
    q32probe = np.einsum("ij,ij->i", probe64, H)
    q64probe = np.einsum("ij,ij->i", probe64, Hprobe)
    parity_native, parity_promoted = qnative - qnative[0], qexact - qexact[0]
    paired_resolution = np.abs(parity_native - parity_promoted)
    probe_resolution = np.abs(q32probe - q64probe)
    tol = 1e-8 * max(1., abs(target))
    check(_close(native["q_tolerance"], tol, atol=1e-16), "native curvature tolerance changed")
    hv_errors = np.linalg.norm(H - Hprobe, axis=1)
    hv_limits = 1e-6 * np.maximum(1., np.linalg.norm(Hprobe, axis=1))
    j_error, j_norm = _norm(J - J64), _norm(J64)
    j_limit = 1e-6 * max(1., j_norm)
    f_error = float(np.max(np.abs(logits - logits64)))
    f_norm = float(np.max(np.abs(logits64)))
    f_limit = 1e-6 * max(1., f_norm)
    derivative_flags = (hv_errors <= hv_limits) & (j_error <= j_limit) & (f_error <= f_limit)
    for name, error, refnorm, limit in (("jacobian_agreement", j_error, j_norm, j_limit),
                                       ("forward_agreement", f_error, f_norm, f_limit)):
        report = native[name]
        check(report["passed"] == bool(error <= limit), name + " flag differs")
        for key, value in (("error", error), ("reference_norm", refnorm), ("limit", limit)):
            check(_close(report[key], value, atol=1e-12), name + " " + key + " differs")
    physical, errors, limits = physical_checks(delta, H, intended[0], target, K, J, head, energy, R, fp32=True)
    for key in limits:
        check(_close(limits[key], native["physical_limits"][key], atol=1e-14), "native limit differs: " + key)
    gates = {"native_physical": physical,
             "promoted_curvature": bool(np.all(np.abs(parity_promoted) <= tol)),
             "curvature_resolution": bool(np.all(paired_resolution <= tol) and np.all(probe_resolution <= tol)),
             "derivative_agreement": bool(np.all(derivative_flags))}
    for key, value in gates.items():
        check(native[key] == value, "native aggregate " + key + " differs")
    ready = all(gates.values())
    check(native["native_ready"] == ready, "native conjunction differs")
    dark = delta @ N.T
    dark_norm = np.linalg.norm(dark, axis=1)
    dark_units = dark / np.maximum(dark_norm[:, None], np.finfo(float).tiny)
    cosines = dark_units[1:] @ dark_units[0]
    gram = dark_units[1:] @ dark_units[1:].T
    radius = _norm(intended[0])
    k_error = np.linalg.norm(delta @ K.T - delta, axis=1)
    jj_error = np.linalg.norm((delta - delta[0]) @ J.T, axis=1)
    dose_error = np.abs(np.linalg.norm(delta, axis=1) - radius)
    head_error = np.linalg.norm(delta.reshape(m, 2, 48) @ head.T, axis=(1, 2))
    energy_error = np.abs(delta @ energy)
    for index, report in enumerate(native["arms"]):
        pairwise = float(np.max(np.delete(gram[index - 1], index - 1))) if index and m > 2 else -1.
        arm_errors = {"curvature": abs(float(parity_native[index])), "full_j": float(jj_error[index]),
                      "k": float(k_error[index]), "dose": float(dose_error[index]),
                      "head": float(head_error[index]), "energy": float(energy_error[index]),
                      "source_abs_cosine": abs(float(cosines[index - 1])) if index else 0.,
                      "pairwise_signed_cosine": pairwise}
        arm_physical = bool(dark_norm[0] > 0 and dark_norm[index] > 0
                            and all(value <= limits[key] for key, value in arm_errors.items()))
        arm_gates = {"native_physical": arm_physical,
                     "promoted_curvature": bool(abs(parity_promoted[index]) <= tol),
                     "curvature_resolution": bool(paired_resolution[index] <= tol and probe_resolution[index] <= tol),
                     "derivative_agreement": bool(derivative_flags[index])}
        for key, value in arm_gates.items():
            check(report[key] == value, f"native vector {index} {key} differs")
        check(report["native_ready"] == all(arm_gates.values()), f"native vector {index} conjunction differs")
        for key, value in arm_errors.items():
            check(_close(value, report["errors"][key]), f"native vector {index} {key} differs")
        diagnostics = {"q_native": qnative[index], "q64_actual": qexact[index],
                       "q32_same_probe": q32probe[index], "q64_same_probe": q64probe[index],
                       "native_parity": parity_native[index], "promoted_parity": parity_promoted[index],
                       "paired_resolution_error": paired_resolution[index], "probe_resolution_error": probe_resolution[index],
                       "hvp_error": hv_errors[index], "hvp_limit": hv_limits[index],
                       "direction_cast_norm": _norm(delta[index] - probe64[index]),
                       "application_rounding_norm": _norm(delta[index] - intended[index])}
        for key, value in diagnostics.items():
            check(_close(float(value), report[key]), f"native vector {index} diagnostic {key} differs")
    state_diff = h32.astype(np.float64) - h64
    check(_close(_norm(state_diff), native["state_drift"]["full_l2"])
          and _close(_norm(J64 - reference_j), native["jacobian_state_drift"]["frobenius_norm"]),
          "native clean-state drift diagnostic differs")
    check(native["hvp_counts"] == {"native32": m, "promoted_exact": m, "promoted_probe": m},
          "native HVP count differs")
    check(native["jacobian_counts"] == {"native32": 10, "promoted_same_state": 10},
          "native Jacobian derivative count differs")
    return {**gates, "native_ready": ready, "max_native_curvature_error": float(np.max(np.abs(parity_native))),
            "max_promoted_curvature_error": float(np.max(np.abs(parity_promoted))),
            "max_paired_resolution_error": float(np.max(paired_resolution)),
            "max_probe_resolution_error": float(np.max(probe_resolution)), "physical_max_errors": errors}


def audit_geometry(case, get, check, *, head_projector, head):
    reference = case["reference"]
    a = lambda key: np.asarray(get("reference__" + key), dtype=np.float64)
    z, z0, q0, R, K = (a(key) for key in ("selected", "z0", "q0", "row_basis", "k_projector"))
    radius, r = _norm(z), _norm(q0)
    J, g = np.asarray(get("future_jacobian")), np.asarray(get("energy_gradient"))
    result = {"selected": z, "z0": z0, "q0": q0, "row_basis": R, "k": K,
              "jacobian": J, "energy_gradient": g, "radius": radius, "null_radius": r}
    reason = reference["reason"]
    first_errors = {"k_symmetry": _norm(K - K.T), "k_idempotence": _norm(K @ K - K),
                    "row_orthogonality": _norm(R.T @ R - np.eye(R.shape[1])),
                    "row_in_k": _norm(K @ R - R), "selected_in_k": _norm(K @ z - z)}
    if reason == "DEGENERATE_FIRST_ORDER_DARK_RADIUS":
        check(radius <= 1e-10 or r <= 1e-10, "reported radius degeneracy is absent")
        result["shared_geometry_failure_verified"] = reason
        return result
    if reason == "INVALID_FIRST_ORDER_GEOMETRY":
        check(max(first_errors[k] for k in first_errors if k != "selected_in_k") > 1e-8
              or first_errors["selected_in_k"] > 1e-8 * max(1., radius),
              "reported first-order geometry failure is absent")
        result["shared_geometry_failure_verified"] = reason
        return result
    check(_norm(z - z0 - q0) <= 1e-9 * max(1., radius)
          and _norm(z0 - R @ (R.T @ z)) <= 1e-9 * max(1., radius), "primary donor split differs")
    check(_norm(K - K.T) <= 1e-8 and _norm(K @ K - K) <= 1e-8, "primary K projector differs")
    check(_norm(R.T @ R - np.eye(R.shape[1])) <= 1e-8 and _norm(K @ R - R) <= 1e-8,
          "primary rowspace is invalid")
    pair_p = np.zeros((96, 96))
    pair_p[:48, :48] = head_projector
    pair_p[48:, 48:] = head_projector
    gh = pair_p @ g
    active = _norm(gh) > max(1e-12, 1e-9 * _norm(g))
    expected_k = pair_p - np.outer(gh, gh) / float(gh @ gh) if active else pair_p
    check(_norm(K - expected_k) <= 1e-8, "primary K differs from physical head/energy constraints")
    check(_norm(J @ K - (J @ K @ R) @ R.T) <= 1e-8, "primary full-J rowspace differs")
    check(abs(_norm(get("natural_pair")) - radius) <= 1e-7 * max(1., radius), "natural primary dose differs")
    if reason == "INVALID_COMPLEMENT_GEOMETRY":
        V, spectrum = a("complement_basis"), a("projector_eigenvalues")
        spectrum_error = float(np.minimum(np.abs(spectrum), np.abs(spectrum - 1)).max())
        basis_error = max(_norm(V.T @ V - np.eye(V.shape[1])), _norm(K @ V - V),
                          _norm(R.T @ V), _norm((q0 / r) @ V))
        check(V.shape[1] != 64 or spectrum_error > 1e-8 or basis_error > 1e-8,
              "reported complement geometry failure is absent")
        result["shared_geometry_failure_verified"] = reason
        return result
    if "selected_curvature" not in reference:
        check(False, "unexplained missing primary curvature oracle")
        return result
    V, hv, hz0, hq0, M = (a(key) for key in ("complement_basis", "hv", "hz0", "hq0", "projected_hessian"))
    target = float(z @ (hz0 + hq0))
    tol = 1e-8 * max(1., abs(target))
    result.update({"target": target, "basis": V, "hv": hv, "hz0": hz0, "hq0": hq0, "projected_hessian": M})
    check(_close(target, reference["selected_curvature"]), "primary target curvature differs")
    check(V.shape == (96, 64) and R.shape[1] == 10 and case["k_dimension"] == 75
          and case["n_full_j_dark"] == 65, "primary registered dimensions differ")
    check(_norm(V.T @ V - np.eye(64)) <= 1e-8 and _norm(V.T @ q0) <= 1e-8 * max(1., r)
          and _norm(K @ V - V) <= 1e-8 and _norm(R.T @ V) <= 1e-8, "primary complement differs")
    check(_norm(M - .5 * (V.T @ hv + hv.T @ V)) <= 1e-9 * max(1., _norm(M)),
          "primary projected Hessian differs")
    check(_norm(np.linalg.eigvalsh(M) - a("projected_hessian_eigenvalues")) <= 1e-8 * max(1., _norm(M)),
          "primary Hessian spectrum differs")
    symmetry = max(_norm(V.T @ hv - hv.T @ V), _norm(z0 @ hv - hz0 @ V),
                   _norm(q0 @ hv - hq0 @ V), abs(float(z0 @ hq0 - q0 @ hz0)))
    symmetry_limit = 1e-8 * max(1., _norm(M), _norm(hz0), _norm(hq0), _norm(hv))
    if reason == "HVP_SYMMETRY_FAILURE":
        check(symmetry > symmetry_limit, "reported HVP symmetry failure is absent")
        result["shared_geometry_failure_verified"] = reason
        return result
    check(symmetry <= symmetry_limit, "unreported HVP symmetry failure")
    if reference["slices"]:
        check([s["cosine"] for s in reference["slices"]] == list(COSINES), "primary cosine grid differs")
        minima, maxima = a("min_vectors"), a("max_vectors")
        result.update({"minima": minima, "maxima": maxima})
        independent_intervals = []
        for index, entry in enumerate(reference["slices"]):
            cosine = entry["cosine"]
            sine = math.sqrt(1 - cosine * cosine)
            A, b = r * r * sine * sine * M, r * sine * (V.T @ (hz0 + cosine * hq0))
            constant = float((z0 + cosine * q0) @ (hz0 + cosine * hq0))
            values, bounds = [], []
            for kind, vector in (("minimum", minima[index]), ("maximum", maxima[index])):
                extremum = entry[kind]
                sign = 1. if kind == "minimum" else -1.
                lam = float(extremum["multiplier"])
                residual = _norm(A @ vector + b - lam * vector)
                gap = float(np.linalg.eigvalsh(sign * (A - lam * np.eye(64)))[0])
                scale = max(1., float(np.linalg.norm(A, ord=2)), _norm(b))
                value = constant + float(vector @ A @ vector + 2 * b @ vector)
                if extremum["certificate_ok"]:
                    check(abs(_norm(vector) - 1.) <= 1e-9 and residual <= 1e-8 * scale
                          and gap >= -1e-8 * scale, "primary extremum certificate differs")
                check(_close(value, entry["q_min" if kind == "minimum" else "q_max"]),
                      "primary attained endpoint differs")
                allowance = 256 * np.finfo(float).eps * 65 * max(1., abs(value - constant), _norm(A), _norm(b), abs(lam))
                length = _norm(vector)
                lower = (sign * (value - constant) + sign * lam * (1 - float(vector @ vector))
                         + min(0., gap) * (1 + length)**2 - 2 * residual * (1 + length) - allowance)
                # Independent valid lower bounds have no required ordering.
                # Check the recorded bound's orientation and shifted PSD side;
                # use our own bound for any negative-grid exclusion below.
                dual_lam = sign * float(extremum["dual_multiplier"])
                dual_gap = float(np.linalg.eigvalsh(sign * A - dual_lam * np.eye(64))[0])
                oriented_value = sign * (value - constant)
                oriented_bound = sign * float(extremum["dual_bound"])
                if extremum["certificate_ok"]:
                    check(dual_gap >= -allowance and oriented_bound <= oriented_value + allowance,
                          "primary dual bound has an invalid orientation or multiplier")
                check(_close(extremum["bound_gap"], max(0., oriented_value - oriented_bound)),
                      "primary dual bound gap differs")
                outer_name = "outer_lower_bound" if kind == "minimum" else "outer_upper_bound"
                check(_close(entry[outer_name], constant + extremum["dual_bound"]), "primary outer bound differs")
                values.append(value)
                bounds.append(constant + sign * lower)
            check(bool(entry["attainable"]) == bool(values[0] - tol <= target <= values[1] + tol),
                  "primary attained flag differs")
            independent_intervals.append(bounds)
        bad_certificates = [e[k] for e in reference["slices"] for k in ("minimum", "maximum") if not e[k]["certificate_ok"]]
        if reason == "SPHERE_SOLVER_RESIDUAL_FAILURE":
            check(bool(bad_certificates), "reported solver certificate failure is absent")
            for bad in bad_certificates:
                threshold = bad["certificate_tolerance"]
                pscale = bad["scale"]
                normalized_psd = bad["psd_lower_bound"] / pscale if pscale else 0.
                check(bad["norm_error"] > threshold or bad["normalized_stationarity_residual"] > threshold
                      or normalized_psd < -threshold or bad["normalized_bound_gap"] > 32 * threshold,
                      "reported failed solver certificate satisfies every recorded gate")
            result["shared_geometry_failure_verified"] = reason
            return result
        check(not bad_certificates, "invalid solver certificate lacks a shared failure")
        if reference["reason"] in ("GRID_NO_ATTAINABLE_TARGET", "SPHERE_INTERVAL_UNRESOLVED"):
            result["shared_geometry_failure_verified"] = reference["reason"]
            overlap = any(e["outer_lower_bound"] - tol <= target <= e["outer_upper_bound"] + tol for e in reference["slices"])
            check(not any(e["attainable"] for e in reference["slices"]), "negative grid contains an attained slice")
            check(overlap == (reference["reason"] == "SPHERE_INTERVAL_UNRESOLVED"), "negative versus unresolved grid differs")
            if reference["reason"] == "GRID_NO_ATTAINABLE_TARGET":
                independent_exclusion = all(not (lo - tol <= target <= hi + tol) for lo, hi in independent_intervals)
                result["grid_exclusion_audit"] = ("INDEPENDENT_BOUND_EXCLUDES_REGISTERED_SLICES" if independent_exclusion
                                                   else "INDEPENDENT_BOUND_UNRESOLVED")
    return result


def audit_primary(arm, get, check, *, geometry, hpair, head, dense, dense_valid):
    intended = np.asarray(get("intended_edits"), dtype=np.float64)
    actual64, actual32 = (np.asarray(get("actual_edits_" + suffix), dtype=np.float64) for suffix in ("fp64", "fp32"))
    expected64 = (hpair[None] + intended) - hpair[None]
    expected32 = ((hpair.astype(np.float32)[None] + intended.astype(np.float32)).astype(np.float64)
                  - hpair.astype(np.float32).astype(np.float64)[None])
    check(np.array_equal(actual64, expected64) and np.array_equal(actual32, expected32), "primary actual addition differs")
    check(np.array_equal(intended[0], geometry["selected"]), "primary applied donor differs")
    check(len(intended) == arm["n_shams"] + 1, "primary sham count differs")
    flags = []
    dense_errors = {}
    for suffix, edits in (("fp64", actual64), ("fp32", actual32)):
        hvps = np.asarray(get("direct_hvps_" + suffix), dtype=np.float64)
        passed, errors, limits = physical_checks(edits, hvps, geometry["selected"], geometry["target"],
            geometry["k"], geometry["jacobian"], head, geometry["energy_gradient"], geometry["row_basis"], fp32=suffix == "fp32")
        report = arm["primary_" + suffix]
        check(report["passed"] == passed, "primary " + suffix + " decision differs")
        for key in errors:
            check(_close(errors[key], report["max_errors"][key]), "primary " + suffix + " error differs: " + key)
            check(_close(limits[key], report["limits"][key], atol=1e-14), "primary " + suffix + " limit differs: " + key)
        flags.append(passed)
        difference = edits @ dense.T - hvps
        dense_errors["max_" + suffix + "_vector_error"] = float(np.max(np.linalg.norm(difference, axis=1)))
        dense_errors["max_" + suffix + "_quadratic_error"] = float(np.max(np.abs(np.einsum("ij,ij->i", edits, difference))))
    limit = 1e-10 * max(1., abs(geometry["target"]))
    dense_agreement = max(dense_errors["max_fp64_quadratic_error"], dense_errors["max_fp32_quadratic_error"]) <= limit
    dense_report = arm["dense_vs_fresh_hvp"]
    for key, value in dense_errors.items():
        check(_close(value, dense_report[key]), "dense-versus-fresh diagnostic differs: " + key)
    check(_close(dense_report["quadratic_limit"], limit, atol=1e-16)
          and dense_report["passed"] == dense_agreement, "dense-versus-fresh gate differs")
    admitted = bool(arm["construction_eligible"] and len(intended) == 9 and all(flags)
                    and dense_valid and dense_agreement)
    check(arm["primary_eligible"] == admitted, "primary admission differs")
    return admitted, intended


def audit_candidates(case, get, check, *, geometry, hpair, head, dense, expected_seed):
    arm = case["arms"]["roundtrip"]
    selection = arm["selection"]
    reference = case["reference"]
    check(selection["n_requested_shams"] == 8 and selection["registered_attempts_per_sham_per_slice"] == 32,
          "candidate budget changed")
    attempts = selection["attempts"]
    attempt_starts = np.asarray(get("attempt_start"), dtype=np.float64)
    check(len(attempt_starts) == len(attempts), "attempt start capture differs")
    candidates = np.asarray(get("candidate_intended"), dtype=np.float64)
    starts = np.asarray(get("candidate_start"), dtype=np.float64)
    applied = np.asarray(get("candidate_actual32"), dtype=np.float64)
    hactions = np.asarray(get("candidate_hvp_dense"), dtype=np.float64)
    check(len(candidates) == selection["candidate_evaluations"]
          and len(starts) == len(candidates) == len(applied) == len(hactions), "candidate raw counts differ")
    z, z0, q0, V = (geometry[key] for key in ("selected", "z0", "q0", "basis"))
    radius, r, target = (geometry[key] for key in ("radius", "null_radius", "target"))
    R, K, J, g = (geometry[key] for key in ("row_basis", "k", "jacobian", "energy_gradient"))
    hv, hz0, hq0, M = (geometry[key] for key in ("hv", "hz0", "hq0", "projected_hessian"))
    baseline_shams = np.asarray(get("baseline_shams"), dtype=np.float64)
    baseline_keys = {}
    for attempt in reference["attempts"]:
        key = (attempt["sham_index"], attempt["slice_index"], attempt["attempt"])
        check(attempt["seed"] == _seed("ambient", expected_seed, case["base"], case["goal"], *key),
              "baseline proposal seed differs")
        if attempt["accepted"]:
            baseline_keys[key] = baseline_shams[attempt["sham_index"]]
    allowed_slices = [i for i, s in enumerate(reference["slices"]) if s["attainable"]]
    accepted, accepted_q = [], []
    expected_sham, expected_slice_index, expected_attempt = 0, 0, 0
    candidate_index = 0
    shared_count = 0
    for attempt_position, entry in enumerate(attempts):
        check(entry["start_index"] == attempt_position
              and abs(_norm(attempt_starts[attempt_position]) - 1.) <= 1e-10,
              "attempt start index or normalization differs")
        key = (entry["sham_index"], entry["slice_index"], entry["attempt"])
        check(bool(allowed_slices), "attempt recorded without an attainable slice")
        if not allowed_slices:
            break
        check(key == (expected_sham, allowed_slices[expected_slice_index], expected_attempt),
              "candidate attempt ordering or bounded budget differs")
        sham_index, slice_index, attempt = key
        check(0 <= sham_index < 8 and 0 <= attempt < 32, "candidate index outside frozen budget")
        check(entry["seed"] == _seed("ambient", expected_seed, case["base"], case["goal"], *key),
              "paired proposal seed differs")
        check(entry["cosine"] == COSINES[slice_index], "candidate cosine metadata differs")
        if entry["candidate_index"] is None:
            check(not entry["accepted"] and entry["reason"] == "NO_NUMERICAL_PATH_CROSSING"
                  and entry["physical"] is None, "missing candidate has an incompatible decision")
        else:
            check(entry["candidate_index"] == candidate_index, "candidate raw index order differs")
            trial, start, actual, ha_trial = candidates[candidate_index], starts[candidate_index], applied[candidate_index], hactions[candidate_index]
            check(np.array_equal(start, attempt_starts[entry["start_index"]]), "candidate and attempt starts differ")
            check(abs(_norm(start) - 1.) <= 1e-10, "stored candidate start leaves the sphere")
            c = COSINES[slice_index]
            sine = math.sqrt(1 - c * c)
            constant = float((z0 + c * q0) @ (hz0 + c * hq0))
            A, b = r * r * sine * sine * M, r * sine * (V.T @ (hz0 + c * hq0))
            start_value = float(start @ A @ start + 2 * b @ start) + constant
            sense = "minimum" if start_value >= target else "maximum"
            check(entry["endpoint_sense"] == sense, "opposite curvature endpoint differs")
            endpoint = geometry["minima" if sense == "minimum" else "maxima"][slice_index]
            t = float(entry["interpolation"])
            check(0 <= t <= 1., "spherical interpolation leaves its bracket")
            root = _path(start, endpoint, t)
            rebuilt = z0 + c * q0 + r * sine * (V @ root)
            check(_norm(trial - rebuilt) <= 1e-8 * max(1., radius), "candidate differs from recorded spherical path")
            expected_actual = ((hpair.astype(np.float32) + trial.astype(np.float32)).astype(np.float64)
                               - hpair.astype(np.float32).astype(np.float64))
            check(np.array_equal(actual, expected_actual), "candidate FP32 addition differs")
            check(_norm(ha_trial - dense @ actual) <= 1e-9 * max(1., _norm(ha_trial)), "candidate dense HVP differs")
            q = trial - z0
            intended_errors = {
                "curvature": abs(float(trial @ (hz0 + c * hq0 + r * sine * (hv @ root))) - target),
                "rowspace": _norm(R.T @ (trial - z)), "k": _norm(K @ trial - trial),
                "dose": abs(_norm(trial) - radius), "source_abs_cosine": abs(float(q @ q0) / (r * r)),
            }
            planned_limits = {"curvature": 1e-8 * max(1., abs(target)), "rowspace": 1e-8 * max(1., radius),
                              "k": 1e-7 * max(1., radius), "dose": 1e-7 * max(1., radius),
                              "source_abs_cosine": .85 + 1e-12}
            planned_ok = all(value <= planned_limits[key] for key, value in intended_errors.items())
            pairwise = max((float(q @ previous) / (r * r) for previous in accepted_q), default=-1.)
            for name, value in intended_errors.items():
                check(_close(value, entry["intended_errors"][name]), "candidate intended error differs: " + name)
            check(_close(pairwise, entry["max_pairwise_cosine"]), "candidate intended diversity differs")
            all_intended = np.stack([z, *accepted, trial])
            all_actual = ((hpair.astype(np.float32)[None] + all_intended.astype(np.float32)).astype(np.float64)
                          - hpair.astype(np.float32).astype(np.float64)[None])
            passed, errors, limits = physical_checks(all_actual, all_actual @ dense.T, z, target, K, J, head, g, R, fp32=True)
            physical = entry["physical"]
            check(physical["passed"] == passed, "candidate physical decision differs")
            for name, value in errors.items():
                check(_close(value, physical["max_errors"][name]), "candidate applied error differs: " + name)
                check(_close(limits[name], physical["limits"][name], atol=1e-14), "candidate applied limit differs: " + name)
            failed_checks = [name for name, value in errors.items() if not np.isfinite(value) or value > limits[name]]
            dark_norm = np.linalg.norm(all_actual @ K.T - (all_actual @ R) @ R.T, axis=1)
            if np.any(dark_norm == 0):
                failed_checks.append("zero_actual_dark_radius")
            check(set(physical["failed_checks"]) == set(failed_checks), "candidate rejection components differ")
            expected_reason = ("PAIRWISE_DIVERSITY_REJECTION" if pairwise > .98 + 1e-12
                               else "NUMERICAL_CONSTRAINT_FAILURE" if not planned_ok
                               else "FP32_CURVATURE_REJECTION" if not passed and "curvature" in failed_checks
                               else "FP32_PHYSICAL_REJECTION" if not passed else None)
            check(entry["reason"] == expected_reason and entry["accepted"] == (expected_reason is None),
                  "candidate rejection precedence differs")
            if key in baseline_keys:
                shared_count += 1
                check(_norm(trial - baseline_keys[key]) <= 1e-11 * max(1., radius), "shared baseline proposal vector differs")
            if entry["accepted"]:
                accepted.append(trial)
                accepted_q.append(q)
            candidate_index += 1
        if entry["accepted"]:
            expected_sham += 1
            expected_slice_index, expected_attempt = 0, 0
        else:
            expected_attempt += 1
            if expected_attempt == 32:
                expected_attempt = 0
                expected_slice_index += 1
                if expected_slice_index == len(allowed_slices):
                    check(entry is attempts[-1], "search continued beyond exhausted slices")
                    expected_slice_index = 0
    shams = np.asarray(get("shams"), dtype=np.float64)
    accepted_array = np.stack(accepted) if accepted else np.empty((0, 96))
    check(np.array_equal(shams, accepted_array), "final shams differ from accepted candidates")
    check(candidate_index == len(candidates) and len(accepted) == selection["n_shams"], "candidate completion counts differ")
    check(selection["rejection_counts"] == dict(Counter(e["reason"] for e in attempts if not e["accepted"])),
          "candidate rejection summary differs")
    check(selection["eligible"] == (len(accepted) == 8), "search eligibility differs from accepted count")
    if len(accepted) == 8:
        check(selection["reason"] is None and attempts[-1]["accepted"], "successful search has an invalid terminal record")
    elif selection["reason"] == "ROUNDED_CANDIDATE_BUDGET_EXHAUSTED":
        last = attempts[-1] if attempts else None
        check(last is not None and not last["accepted"] and last["sham_index"] == len(accepted)
              and last["slice_index"] == allowed_slices[-1] and last["attempt"] == 31,
              "budget exhaustion ended before the final allowed attempt")
    elif selection["reason"] == "NUMERICAL_CONSTRAINT_FAILURE":
        check(bool(attempts) and attempts[-1]["reason"] == "NUMERICAL_CONSTRAINT_FAILURE"
              and not attempts[-1]["accepted"], "numerical failure lacks a terminal failed trial")
    else:
        shared_failure = geometry.get("shared_geometry_failure_verified")
        check(not attempts and shared_failure is not None and selection["reason"] == shared_failure,
              "failed search has an unsupported or unverified shared termination")
    check(not any(e["reason"] == "NUMERICAL_CONSTRAINT_FAILURE" for e in attempts[:-1]),
          "search continued after an immediate numerical failure")
    return {"n_candidate_vectors": len(candidates), "n_accepted": len(accepted),
            "n_shared_baseline_proposals_checked": shared_count}


def audit(receipt: Path, raw: Path, out: Path | None = None, *, expected_seed=61066):
    """Audit a frozen receipt; the production CLI fixes expected_seed=61066."""
    receipt, raw = Path(receipt), Path(raw)
    document = json.loads(receipt.read_text())
    failures, results = [], []

    def require(condition, message):
        if not condition:
            failures.append(message)

    require(document["raw_sha256"] == _sha(raw), "raw archive digest differs")
    sidecar = receipt.with_suffix(receipt.suffix + ".sha256")
    require(sidecar.exists() and sidecar.read_text().split()[0] == _sha(receipt), "receipt digest sidecar differs")
    require(document["excluded_source_seed"] == expected_seed and document["excluded_model_seed"] == expected_seed,
            "excluded source/model seed differs")
    require(document["n_excluded_bases"] == 8 and document["n_goals_per_base"] == 8,
            "independent source/goal grouping differs")
    require(document["n_untrained_model_initializations"] == 1 and document["n_new_trained_model_roots"] == 0,
            "model evidence class differs")
    require(document["registered_min_admitted"] == 61, "registered admission gate differs")
    require(document["new_test_generated"] is False and document["trained_weights_loaded"] is False
            and document["edited_task_outcomes_scored"] is False
            and document["native_gate_used_in_proposal_search"] is False, "source-only/native-search boundary differs")
    require(document["paired_proposal_seed_namespace"] == "LGA10B", "paired proposal namespace differs")
    cases = document["cases"]
    require([(c["base"], c["goal"]) for c in cases] == [(b, g) for b in range(8) for g in GOALS],
            "64 paired cases are missing, duplicated or reordered")
    with np.load(raw, allow_pickle=False) as arrays:
        for name in arrays.files:
            require(bool(np.isfinite(arrays[name]).all()), "nonfinite raw array: " + name)
        x, y = arrays["source_inputs"], arrays["source_labels"]
        require(x.shape == (8, 81) and y.shape == (8, 81), "source dimensions differ")
        require(np.all((x >= 1) & (x <= 9)) and np.array_equal(y, np.roll(x, 1, axis=1)), "cyclic source law differs")
        require(hashlib.sha256(np.asarray(x, dtype="<i8").tobytes()).hexdigest() == document["source_inputs_sha256"],
                "source input digest differs")
        require(hashlib.sha256(np.asarray(y, dtype="<i8").tobytes()).hexdigest() == document["source_labels_sha256"],
                "source label digest differs")
        head, hp = arrays["head_weight"], arrays["head_projector"]
        require(np.array_equal(arrays["head_weight_fp32"].astype(np.float64), head), "head promotion differs")
        require(_norm(head @ hp) <= 1e-8 and _norm(hp @ hp - hp) <= 1e-8, "current-head kernel differs")
        for case in cases:
            first = len(failures)
            base, goal, prefix = case["base"], case["goal"], case["raw_prefix"]
            label = f"base {base} goal {goal}: "

            def check(condition, message):
                require(condition, label + message)

            get = lambda key: arrays[prefix + key]
            row = {"base": base, "goal": goal, "arms": {}}
            try:
                h64, h32 = arrays[f"base_{base:02d}__hrec_fp64"], arrays[f"base_{base:02d}__hrec_fp32"]
                check(h64.shape == h32.shape == (1, 81, 48) and h64.dtype == np.float64 and h32.dtype == np.float32,
                      "reached state shape or dtype differs")
                donor = np.asarray(get("donor_input"))
                expected_donor = x[base].copy()
                p = (goal - 1) % 81
                expected_donor[p] = int(expected_donor[p]) % 9 + 1
                check(np.array_equal(donor, expected_donor), "legal donor differs")
                metadata = case["donor_metadata"]
                check(metadata["original_goal_label"] == int(x[base, p])
                      and metadata["donor_goal_label"] == int(expected_donor[p])
                      and metadata["changed_targets"] == [goal], "donor source classes differ")
                if "reference" not in case:
                    check(bool(case["case_reason"]) and all(not case["arms"][arm][key]
                          for arm in ARMS for key in ("construction_eligible", "primary_eligible", "native_ready", "overall_ready")),
                          "missing geometry admitted a case")
                else:
                    reference = case["reference"]
                    check(reference["teacher_seed"] == expected_seed and reference["base_index"] == base
                          and reference["goal"] == goal, "primary proposal identity differs")
                    geometry = audit_geometry(case, get, check, head_projector=hp, head=head)
                    if "grid_exclusion_audit" in geometry:
                        row["grid_exclusion_audit"] = geometry["grid_exclusion_audit"]
                    if "shared_geometry_failure_verified" in geometry:
                        check(all(not case["arms"][arm][key] for arm in ARMS
                                  for key in ("construction_eligible", "primary_eligible", "native_ready", "overall_ready")),
                              "shared failed geometry admitted an endpoint")
                        row["shared_geometry_failure_verified"] = geometry["shared_geometry_failure_verified"]
                    check(case["hvp_counts"]["reference_construction"] == reference["construction_hvp_count"],
                          "original proposal HVP count differs")
                    if "target" in geometry and "dense_primary" in case:
                        hpair = h64[0, [p, goal]].reshape(96)
                        check(np.array_equal(hpair, get("hrec_pair")), "primary pair state differs")
                        dense, dense_raw = np.asarray(get("dense_hessian")), np.asarray(get("dense_hessian_raw"))
                        dense_report = case["dense_primary"]
                        skew, dnorm = _norm(dense_raw - dense_raw.T), _norm(dense_raw)
                        dlimit = 1e-8 * max(1., dnorm)
                        check(dense.shape == dense_raw.shape == (96, 96)
                              and np.array_equal(dense, .5 * (dense_raw + dense_raw.T)), "dense Hessian symmetrization differs")
                        check(dense_report["passed"] == (skew <= dlimit)
                              and _close(dense_report["skew_frobenius_norm"], skew)
                              and _close(dense_report["symmetry_limit"], dlimit, atol=1e-16), "dense symmetry gate differs")
                        check(dense_report["hvp_count"] == 96 and case["hvp_counts"]["dense_primary"] == 96,
                              "dense coordinate HVP count differs")
                        for arm_name in ARMS:
                            arm = case["arms"][arm_name]
                            aget = lambda name, arm_name=arm_name: get(arm_name + "__" + name)
                            if "selection" not in arm:
                                check(not arm["primary_eligible"] and not arm["native_ready"] and not arm["overall_ready"]
                                      and bool(arm["primary_reason"]), "uncaptured arm admitted")
                                continue
                            selection = arm["selection"]
                            check(arm["construction_eligible"] == bool(selection["eligible"]), "construction decision differs")
                            for key in ("selected", "z0", "q0", "row_basis", "k_projector", "complement_basis",
                                        "hz0", "hq0", "hv", "projected_hessian", "min_vectors", "max_vectors"):
                                if prefix + "reference__" + key in arrays:
                                    check(np.array_equal(aget(key), get("reference__" + key)), "paired geometry differs: " + key)
                            if "primary_validation_exception" in arm:
                                check(not arm["primary_eligible"] and not arm["overall_ready"],
                                      "incomplete primary validation admitted an arm")
                                primary = False
                                intended = np.concatenate((geometry["selected"][None], np.asarray(aget("shams"))), axis=0)
                                check(all(0 <= case["hvp_counts"][arm_name + "_final_" + suffix] <= len(intended)
                                          for suffix in ("fp64", "fp32")), "partial primary HVP count exceeds selected vectors")
                            else:
                                primary, intended = audit_primary(arm, aget, check, geometry=geometry,
                                                                  hpair=hpair, head=head, dense=dense,
                                                                  dense_valid=bool(dense_report["passed"]))
                                check(case["hvp_counts"][arm_name + "_final_fp64"] == len(intended)
                                      and case["hvp_counts"][arm_name + "_final_fp32"] == len(intended), "fresh primary HVP count differs")
                            nready = False
                            native_report = None
                            if "native" in arm:
                                check(len(intended) == 9 and arm["native"]["goal"] == goal
                                      and arm["native"]["donor_class"] == metadata["donor_goal_label"]
                                      and arm["native"]["original_class"] == metadata["original_goal_label"],
                                      "native case identity differs")
                                native_report = audit_native(arm["native"], lambda key: aget("native__" + key), check,
                                    target=geometry["target"], intended=intended, h32=h32, h64=h64,
                                    reference_j=geometry["jacobian"], head_projector=hp, head=head)
                                nready = native_report["native_ready"]
                            else:
                                check(len(intended) < 9 or bool(case["case_reason"])
                                      or "native_validation_exception" in arm,
                                      "complete arm lacks native gate")
                            check(arm["native_ready"] == nready and arm["overall_ready"] == bool(primary and nready),
                                  "native or overall arm decision differs")
                            row["arms"][arm_name] = {"primary_eligible": primary, "native_ready": nready,
                                                     "overall_ready": bool(primary and nready), "native": native_report}
                        new_arm = case["arms"]["roundtrip"]
                        if "selection" in new_arm and "minima" in geometry:
                            def candidate_get(key):
                                return get("baseline__shams") if key == "baseline_shams" else get("roundtrip__" + key)
                            row["candidate_audit"] = audit_candidates(case, candidate_get, check, geometry=geometry,
                                hpair=hpair, head=head, dense=dense, expected_seed=expected_seed)
                        elif "selection" in new_arm:
                            check(not new_arm["selection"]["eligible"]
                                  and not new_arm["selection"]["attempts"]
                                  and new_arm["selection"]["n_shams"] == 0,
                                  "incomplete shared geometry produced candidates")
                    else:
                        check(all(not case["arms"][arm][key] for arm in ARMS
                                  for key in ("construction_eligible", "primary_eligible", "native_ready", "overall_ready")),
                              "missing primary oracle admitted an endpoint")
                        check("shared_geometry_failure_verified" in geometry or bool(case["case_reason"]),
                              "missing primary oracle lacks an explicit failure")
            except (KeyError, ValueError, ArithmeticError, IndexError, np.linalg.LinAlgError) as error:
                check(False, f"incomplete or invalid case capture: {type(error).__name__}: {error}")
            row["audit_passed"] = len(failures) == first
            results.append(row)
    require(document["n_cases"] == len(cases) == 64, "total paired case count differs")
    for arm in ARMS:
        summary = document["arms"][arm]
        for key, count_key, status_key in (("primary_eligible", "n_primary_admitted", "primary_status"),
                                           ("native_ready", "n_native_ready", "native_status"),
                                           ("overall_ready", "n_overall_ready", "overall_status")):
            count = sum(bool(c["arms"][arm][key]) for c in cases)
            per_base = [sum(bool(c["arms"][arm][key]) for c in cases if c["base"] == b) for b in range(8)]
            require(summary[count_key] == count and summary["per_base"][key] == per_base,
                    arm + " grouped count differs: " + key)
            require(summary[status_key] == ("PASS" if count >= 61 else "FAIL"), arm + " threshold verdict differs: " + key)
        require(summary["n_constructed"] == sum(c["arms"][arm]["construction_eligible"] for c in cases),
                arm + " construction count differs")
        for ready_key, reason_key, summary_key in (("primary_eligible", "primary_reason", "primary_failure_reasons"),
                                                   ("native_ready", "native_reason", "native_failure_reasons")):
            require(summary[summary_key] == dict(Counter(c["arms"][arm][reason_key] for c in cases if not c["arms"][arm][ready_key])),
                    arm + " failure summary differs")
    for key in ("primary_eligible", "native_ready", "overall_ready"):
        differences = [int(c["arms"]["roundtrip"][key]) - int(c["arms"]["baseline"][key]) for c in cases]
        macro = [sum(d for d, c in zip(differences, cases) if c["base"] == b) / 8 for b in range(8)]
        paired = document["paired"][key]
        require(paired["case_differences"] == differences and paired["case_difference_count"] == sum(differences)
                and paired["per_base_fraction_differences"] == macro
                and paired["source_macro_fraction_difference"] == sum(macro) / 8, "paired descriptive endpoint differs: " + key)
    for key, count in document["hvp_counts"].items():
        require(count == sum(c["hvp_counts"][key] for c in cases), "aggregate primary HVP count differs: " + key)
    for key, count in document["native_hvp_counts"].items():
        require(count == sum(c["arms"][arm].get("native", {}).get("hvp_counts", {}).get(key, 0)
                            for c in cases for arm in ARMS), "aggregate native HVP count differs: " + key)
    result = {"schema": "LGA10C/independent-paired-numpy-audit-v1", "audit_status": "PASS" if not failures else "FAIL",
              "receipt_sha256": _sha(receipt), "raw_sha256": _sha(raw), "numpy_version": np.__version__,
              "model_matcher_or_torch_imported": False, "n_cases": len(results),
              "n_cases_audit_passed": sum(r["audit_passed"] for r in results), "failures": failures, "cases": results}
    if out is not None:
        Path(out).write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print("LGA10C_INDEPENDENT_PAIRED_AUDIT", result["audit_status"], result["n_cases_audit_passed"], len(results), flush=True)
    for failure in failures:
        print("AUDIT_FAILURE", failure, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.receipt, args.raw, args.out, expected_seed=61066)
    raise SystemExit(0 if result["audit_status"] == "PASS" else 1)




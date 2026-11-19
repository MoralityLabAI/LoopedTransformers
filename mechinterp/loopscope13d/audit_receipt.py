"""Independent NumPy audit of LGA-10D's two execution contracts.

Canonical per-arm views reuse the frozen LGA-10C geometry, candidate,
selection-space and native validators. This module adds shared-source and
promoted-witness checks; it imports no model, matcher, oracle or Torch.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from mechinterp.loopscope13c.audit_receipt import (
    _close, _norm, audit_candidates, audit_geometry, audit_native, audit_primary,
    physical_checks,
)
from mechinterp.loopscope13d.capture import open_capture


ARMS = ("reference_derivatives", "native_derivatives")
GOALS = (0, 10, 20, 30, 40, 50, 60, 70)
ENDPOINTS = ("selection_validated", "promoted_ready", "native_fp32_ready")
STEPS = np.array((2. ** -8, 2. ** -9, 2. ** -10))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _same(left, right, *, atol=1e-12, rtol=1e-10):
    left, right = np.asarray(left), np.asarray(right)
    return bool(left.shape == right.shape and np.isfinite(left).all() and np.isfinite(right).all()
                and np.allclose(left, right, atol=atol, rtol=rtol))


def _check_report(report, passed, errors, limits, check, label):
    check(report["passed"] == passed, label + " decision differs")
    for key, value in errors.items():
        check(_close(value, report["max_errors"][key]), label + " error differs: " + key)
        check(_close(limits[key], report["limits"][key], atol=1e-14), label + " limit differs: " + key)


def audit_common(case, get, check, *, h32, hp, expected_seed):
    """Rebuild the common native donor, radius, energy tangent and target."""
    goal, parent = case["goal"], case["common_parent"]
    pair = [(goal - 1) % 81, goal]
    donor = get("hdon_fp32")
    check(donor.dtype == np.float32 and donor.shape == h32.shape, "native donor state differs")
    natural = (donor.astype(np.float64) - h32.astype(np.float64))[0, pair].reshape(96)
    energy = h32[0, pair].astype(np.float64).reshape(96)
    P = np.zeros((96, 96))
    P[:48, :48] = P[48:, 48:] = hp
    gh = P @ energy
    active = _norm(gh) > max(1e-12, 1e-9 * _norm(energy))
    K = P - np.outer(gh, gh) / (gh @ gh) if active else P
    selected = get("common__selected")
    radius = _norm(natural)
    check(np.array_equal(get("common__natural_pair"), natural), "native source subtraction differs")
    check(np.array_equal(get("common__energy_gradient"), energy)
          and np.array_equal(get("common__application_pair64"), energy), "common application/energy state differs")
    check(_same(get("common__k_projector"), K), "common native K differs")
    check(_close(parent["natural_radius"], radius) and parent["pair_positions"] == pair,
          "common parent radius or positions differ")
    source = parent["native_source"]
    identity = {"teacher_seed": expected_seed, "base_index": case["base"], "goal": goal,
                "original_class": case["donor_metadata"]["original_goal_label"],
                "donor_class": case["donor_metadata"]["donor_goal_label"]}
    check(all(source[key] == value for key, value in identity.items()), "common source identity differs")
    check(source["subtraction_order"] == "hdon32.double() - h32.double()"
          and source["retained_auxiliary_parent_eligibility"] is True,
          "common parent construction contract differs")
    if parent["eligible"]:
        projected = K @ natural
        check(active and radius > 1e-10 and _norm(projected) > 1e-10
              and case["k_dimension"] == 75, "eligible common parent has degenerate geometry")
        check(_same(selected, radius * projected / _norm(projected)), "selected donor differs from common source")
        check(_close(case["radius"], radius), "selected donor dose differs")
        future, gf = get("common__future_gradient"), get("common__future_k_gradient")
        check(_same(gf, K @ future) and _norm(gf) > 1e-10, "common auxiliary gradient differs")
        check(_close(parent["selected_signed_derivative"], float(selected @ gf)),
              "common auxiliary derivative differs")
        for family in ("pair_random_k_", "pair_gradient_match_"):
            for index in range(8):
                value = get("common__parent_direction__" + family + str(index)).reshape(96)
                check(_norm(K @ value - value) <= 1e-7 * max(1., radius)
                      and abs(_norm(value) - radius) <= 1e-7 * max(1., radius),
                      "common auxiliary physical control differs")
                if family == "pair_gradient_match_":
                    target = float(selected @ gf)
                    check(abs(float(value @ gf) - target) <= 1e-7 * max(1., abs(target)),
                          "common auxiliary margin control differs")
        if case["common_target_hvp_count"] == 1:
            target = float(selected @ get("common__target_hvp_native"))
            check(_close(case["target_curvature_native"], target)
                  and case["common_target_hvp_attempt_count"] == 1, "common fresh target differs")
        else:
            target = None
            check(case["common_target_hvp_count"] == 0 and case["common_target_hvp_attempt_count"] in (0, 1)
                  and case["case_reason"] == "EXPLICIT_COMMON_CASE_EXCEPTION" and bool(case.get("exception")),
                  "eligible parent lacks common target or explicit failure")
    else:
        target = None
        check(bool(parent["reason"]) and np.array_equal(selected, np.zeros(96))
              and case["common_target_hvp_count"] == 0, "failed common parent was used")
    return {"selected": selected, "natural_pair": natural, "energy_gradient": energy,
            "k": get("common__k_projector"), "radius": radius, "target": target}


def audit_witness(witness, get, check, *, intended, h32, goal, target):
    """Recompute both Richardson scales from archived first gradients."""
    a = lambda key: np.asarray(get(key))
    for key in ("base_state64", "actual_edits64", "pair_gradients", "probe_edits", "probe_actual_deltas",
                "fresh_hvps", "central_hvp_estimates", "richardson_hvp_estimates"):
        check(a(key).dtype == np.float64, "promoted witness dtype differs: " + key)
    pair = h32[0, [(goal - 1) % 81, goal]].reshape(96)
    actual = (pair[None] + intended.astype(np.float32)).astype(np.float64) - pair.astype(np.float64)[None]
    check(intended.shape == (9, 96) and np.array_equal(a("intended_edits"), intended),
          "promoted intended set differs")
    check(np.array_equal(a("base_state64"), h32.astype(np.float64))
          and np.array_equal(a("actual_edits64"), actual), "promoted native application differs")
    norms = np.linalg.norm(actual, axis=1)
    check(bool(np.all(norms > 0)), "promoted zero physical direction")
    units = actual / norms[:, None]
    probes = units[:, None, None, :] * STEPS[None, :, None, None] * np.array([1., -1.])[None, None, :, None]
    realized = (pair.astype(np.float64)[None, None, None] + probes) - pair.astype(np.float64)[None, None, None]
    for key, expected in (("norms", norms), ("unit_directions", units), ("steps", STEPS),
                          ("probe_edits", probes), ("probe_actual_deltas", realized)):
        check(_same(a(key), expected), "promoted probe reconstruction differs: " + key)
    gradients, fresh = a("pair_gradients"), a("fresh_hvps")
    check(gradients.shape == (9, 3, 2, 96) and fresh.shape == (9, 96)
          and a("probe_margins").shape == (9, 3, 2), "promoted derivative capture shapes differ")
    central = (gradients[:, :, 0] - gradients[:, :, 1]) / (2 * STEPS[None, :, None])
    richardson = (4 * central[:, 1:] - central[:, :-1]) / 3
    q = np.einsum("ij,ij->i", actual, fresh)
    qr = norms[:, None] ** 2 * np.einsum("ik,ijk->ij", units, richardson)
    coarse, fine, scales = abs(qr[:, 0] - q), abs(qr[:, 1] - q), abs(qr[:, 0] - qr[:, 1])
    parity = qr[:, 1] - qr[0, 1]
    recomputed = {"central_hvp_estimates": central, "richardson_hvp_estimates": richardson,
                  "q_fresh": q, "q_richardson_coarse": qr[:, 0], "q_richardson_fine": qr[:, 1],
                  "coarse_errors": coarse, "fine_errors": fine, "scale_discrepancies": scales,
                  "fd_parity": parity}
    for key, expected in recomputed.items():
        check(_same(a(key), expected), "Richardson reconstruction differs: " + key)
    tau = 1e-8 * max(1., abs(target))
    oracle = tau / 10
    flags = {"coarse_vs_fresh_passed": coarse <= oracle, "fine_vs_fresh_passed": fine <= oracle,
             "coarse_vs_fine_passed": scales <= oracle, "fd_parity_passed": abs(parity) <= tau}
    flags["oracle_consistency_passed"] = flags["coarse_vs_fresh_passed"] & flags["fine_vs_fresh_passed"] & flags["coarse_vs_fine_passed"]
    check(witness["steps"] == list(STEPS) and witness["probe_sign_order"] == [1, -1]
          and witness["richardson_order"] == ["coarse", "fine"], "promoted finite-difference contract differs")
    check(_close(witness["q_tolerance"], tau, atol=1e-16)
          and _close(witness["oracle_tolerance"], oracle, atol=1e-17)
          and _close(witness["target_curvature"], target), "promoted target or precision budget differs")
    check(witness["n_vectors"] == 9 and witness["n_shams"] == 8 and witness["fresh_hvp_count"] == 9
          and witness["first_gradient_count"] == 54 and len(witness["per_vector"]) == 9,
          "promoted oracle measurement counts differ")
    check(witness["fresh_hvp_measured_inside_helper"] is True and witness["selection_feedback"] is False
          and witness["is_interval_certificate"] is False, "promoted witness scope differs")
    values = {"physical_norm": norms, "q_fresh": q, "q_richardson_coarse": qr[:, 0],
              "q_richardson_fine": qr[:, 1], "coarse_error": coarse, "fine_error": fine,
              "scale_discrepancy": scales, "fd_parity": parity}
    for index, entry in enumerate(witness["per_vector"]):
        check(entry["index"] == index, "promoted vector order differs")
        for key, value in values.items():
            check(_close(entry[key], float(value[index]), atol=1e-12), "promoted scalar differs: " + key)
        for key, value in flags.items():
            check(entry[key] == bool(value[index]), "promoted vector gate differs: " + key)
    for key, value in flags.items():
        check(witness[key] == bool(value.all()), "promoted witness gate differs: " + key)
    maxima = {"coarse_vs_fresh": max(coarse), "fine_vs_fresh": max(fine),
              "coarse_vs_fine": max(scales), "fd_parity": max(abs(parity))}
    for key, value in maxima.items():
        check(_close(witness["max_errors"][key], value, atol=1e-12), "promoted maximum differs: " + key)
    passed = bool(flags["oracle_consistency_passed"].all() and flags["fd_parity_passed"].all())
    check(witness["passed"] == passed, "promoted witness conjunction differs")
    return passed, actual, fresh


def audit_promoted(report, get, check, *, common, native_geometry, native_dense,
                   native_dense_valid, intended, h32, head, goal):
    witness, actual, fresh = audit_witness(report["witness"], get, check,
        intended=intended, h32=h32, goal=goal, target=common["target"])
    if native_geometry is None or native_dense is None:
        check(report["passed"] is False and report["reason"] == "NATIVE_GEOMETRY_UNAVAILABLE",
              "unavailable native geometry admitted promoted continuation")
        return False
    physical, errors, limits = physical_checks(actual, fresh, common["selected"], common["target"],
        common["k"], native_geometry["jacobian"], head, common["energy_gradient"],
        native_geometry["row_basis"], fp32=True)
    _check_report(report["physical"], physical, errors, limits, check, "promoted physical")
    difference = actual @ native_dense.T - fresh
    quadratic = float(np.max(abs(np.einsum("ij,ij->i", actual, difference))))
    vector = float(np.max(np.linalg.norm(difference, axis=1)))
    limit = 1e-10 * max(1., abs(common["target"]))
    dense_pass = bool(quadratic <= limit and native_dense_valid)
    dense = report["dense_vs_fresh"]
    check(_close(dense["max_quadratic_error"], quadratic) and _close(dense["max_vector_error"], vector)
          and _close(dense["quadratic_limit"], limit, atol=1e-18)
          and dense["coordinate_skew_passed"] == native_dense_valid and dense["passed"] == dense_pass,
          "promoted native dense/fresh gate differs")
    passed = bool(witness and physical and dense_pass)
    check(report["passed"] == passed, "promoted continuation conjunction differs")
    return passed


def _arm_geometry(case, arm_name, get, check, *, common, h32, h64, hp, head, expected_seed):
    arm = case["arms"][arm_name]
    if "geometry" not in arm:
        check("geometry_exception" in arm or bool(case["case_reason"]), "missing arm geometry lacks failure evidence")
        check(not any(arm[key] for key in ("construction_eligible", *ENDPOINTS)), "missing geometry admitted an arm")
        return None, None, False
    report = arm["geometry"]
    aget = lambda key: get(arm_name + "__geometry__" + key)
    expected_state = h64 if arm_name == ARMS[0] else h32.astype(np.float64)
    check(aget("derivative_state64").dtype == np.float64 and aget("application_state32").dtype == np.float32
          and np.array_equal(aget("derivative_state64"), expected_state)
          and np.array_equal(aget("application_state32"), h32), "arm derivative/application anchor differs")
    check(report["teacher_seed"] == expected_seed and report["base_index"] == case["base"]
          and report["goal"] == case["goal"] and report["common_native_source"] is True,
          "arm geometry identity differs")
    for raw_key, common_key in (("selected", "selected"), ("natural_pair", "natural_pair"),
                                 ("energy_gradient", "energy_gradient"), ("k_projector", "k")):
        check(aget(raw_key).dtype == np.float64 and np.array_equal(aget(raw_key), common[common_key]),
              "arm common geometry differs: " + raw_key)
    counts, attempts = report["hvp_counts"], report["hvp_attempt_counts"]
    check(set(counts) == set(attempts) and all(0 <= counts[key] <= attempts[key] for key in counts),
          "geometry completed/attempted HVP counts differ")
    check(set(counts) <= {"reference_construction", "dense_primary", "external_validation", "selection_fp64", "selection_fp32"}
          and counts["external_validation"] == attempts["external_validation"] == 0,
          "unregistered geometry HVP purpose")
    if report["reason"] == "EXPLICIT_GEOMETRY_EXCEPTION":
        for category in ("reference_construction", "dense_primary"):
            if attempts[category]:
                directions = aget("partial_hvp__" + category + "__directions")
                values = aget("partial_hvp__" + category + "__values")
                check(directions.shape == (attempts[category], 96) and values.shape == (counts[category], 96),
                      "partial geometry HVP trace counts differ")
    if "reference" not in arm:
        check(bool(report["reason"]) and not any(arm[key] for key in ("construction_eligible", *ENDPOINTS)),
              "incomplete reference geometry admitted an endpoint")
        return None, None, False
    canonical = {"base": case["base"], "goal": case["goal"], "reference": arm["reference"],
                 "k_dimension": report["k_dimension"], "n_full_j_dark": report["n_full_j_dark"],
                 "arms": {"roundtrip": arm}}
    def geometry_get(key):
        section = "reference__" if key.startswith("reference__") else "geometry__"
        return get(arm_name + "__" + (key if section == "reference__" else section + key))
    geometry = audit_geometry(canonical, geometry_get, check, head_projector=hp, head=head)
    check(counts["reference_construction"] == arm["reference"]["construction_hvp_count"],
          "spectral reference HVP count differs")
    for key in ("selected", "row_basis", "k_projector"):
        check(np.array_equal(get(arm_name + "__reference__" + key), aget(key)), "reference geometry differs: " + key)
    for key in ("future_jacobian", "restricted_jacobian", "row_basis"):
        check(aget(key).dtype == np.float64, "geometry derivative dtype differs: " + key)
    check(_same(aget("restricted_jacobian"), geometry["jacobian"] @ common["k"]),
          "restricted full Jacobian differs from common K")
    if "target" in geometry:
        check(counts["reference_construction"] == 2 + geometry["basis"].shape[1] == 66,
              "spectral action capture implies a different completed HVP count")
        check(_close(report["construction_target_curvature"], geometry["target"])
              and _close(report["construction_curvature_tolerance"], 1e-8 * max(1., abs(geometry["target"])), atol=1e-16),
              "arm construction target/tolerance metadata differs")
    dense, dense_valid = None, False
    if "dense_primary" in arm:
        dense, raw = aget("dense_hessian"), aget("dense_hessian_raw")
        skew, norm = _norm(raw - raw.T), _norm(raw)
        limit = 1e-8 * max(1., norm)
        dense_valid = bool(skew <= limit)
        dr = arm["dense_primary"]
        check(dense.dtype == raw.dtype == np.float64 and dense.shape == raw.shape == (96, 96)
              and np.array_equal(dense, .5 * (raw + raw.T)),
              "dense Hessian symmetrization differs")
        check(dr["passed"] == dense_valid and _close(dr["skew_frobenius_norm"], skew)
              and _close(dr["raw_frobenius_norm"], norm)
              and _close(dr["symmetry_limit"], limit, atol=1e-16), "dense symmetry gate differs")
        check(dr["hvp_count"] == counts["dense_primary"] == 96, "dense coordinate HVP count differs")
        if dense_valid and "basis" in geometry:
            for direction, action in ((geometry["z0"], geometry["hz0"]), (geometry["q0"], geometry["hq0"]),
                                      (geometry["basis"], geometry["hv"])):
                check(_norm(raw @ direction - action) <= 1e-8 * max(1., _norm(action)),
                      "dense Hessian differs from this arm's spectral actions")
    if "selection" in arm:
        check(arm["construction_eligible"] == arm["selection"]["eligible"]
              and arm["n_shams"] == arm["selection"]["n_shams"], "construction endpoint differs")
        for key in ("selected", "z0", "q0", "row_basis", "k_projector", "complement_basis",
                    "hz0", "hq0", "hv", "projected_hessian", "min_vectors", "max_vectors"):
            if key in ("min_vectors", "max_vectors") and "minima" not in geometry:
                continue
            check(np.array_equal(get(arm_name + "__selection__" + key), get(arm_name + "__reference__" + key)),
                  "within-arm selection geometry differs: " + key)
        if "minima" in geometry:
            def candidates(key):
                return get(arm_name + ("__reference__shams" if key == "baseline_shams" else "__selection__" + key))
            audit_candidates(canonical, candidates, check, geometry=geometry, hpair=common["energy_gradient"],
                             head=head, dense=dense, expected_seed=expected_seed)
        else:
            check(not arm["selection"]["eligible"] and not arm["selection"]["attempts"],
                  "missing spectral slices produced candidates")
    else:
        check(not any(arm[key] for key in ("construction_eligible", *ENDPOINTS)) and bool(report["reason"]),
              "missing selection admitted an endpoint")
    return geometry, dense, dense_valid


def _audit_case(case, arrays, check, *, x, hp, head, expected_seed):
    base, goal, prefix = case["base"], case["goal"], case["raw_prefix"]
    get = lambda key: arrays[prefix + key]
    h32, h64 = arrays[f"base_{base:02d}__hrec_fp32"], arrays[f"base_{base:02d}__hrec_fp64"]
    check(h32.shape == h64.shape == (1, 81, 48) and h32.dtype == np.float32 and h64.dtype == np.float64,
          "reached state shape or dtype differs")
    p = (goal - 1) % 81
    donor = x[base].copy()
    donor[p] = int(donor[p]) % 9 + 1
    check(np.array_equal(get("donor_input"), donor), "legal donor mutation differs")
    metadata = case["donor_metadata"]
    check(metadata["original_goal_label"] == int(x[base, p]) and metadata["donor_goal_label"] == int(donor[p])
          and metadata["changed_targets"] == [goal], "legal donor classes differ")
    if "common_parent" not in case:
        check(bool(case["case_reason"]) and all(not case["arms"][arm][key] for arm in ARMS
              for key in ("construction_eligible", *ENDPOINTS)), "missing common parent admitted a case")
        return
    common = audit_common(case, get, check, h32=h32, hp=hp, expected_seed=expected_seed)
    if not case["common_parent"]["eligible"] or common["target"] is None:
        check(all(not case["arms"][arm][key] for arm in ARMS for key in ("construction_eligible", *ENDPOINTS)),
              "failed common source admitted an endpoint")
        return
    geometries = {}
    for name in ARMS:
        try:
            geometries[name] = _arm_geometry(case, name, get, lambda ok, msg: check(ok, name + ": " + msg),
                common=common, h32=h32, h64=h64, hp=hp, head=head, expected_seed=expected_seed)
        except (KeyError, ValueError, ArithmeticError, IndexError, TypeError, np.linalg.LinAlgError) as error:
            check(False, name + " geometry capture invalid: " + str(error))
            geometries[name] = (None, None, False)
    ng, nd, nd_valid = geometries[ARMS[1]]
    if nd is not None and nd_valid:
        z = common["selected"]
        check(abs(float(z @ nd @ z) - common["target"]) <= 1e-10 * max(1., abs(common["target"])),
              "native dense Hessian differs from the independent common target")
    reference_j = get("reference_drift_jacobian") if prefix + "reference_drift_jacobian" in arrays else None
    if geometries[ARMS[0]][0] is not None:
        check(np.array_equal(reference_j, geometries[ARMS[0]][0]["jacobian"]), "reference-state drift Jacobian differs")
    for name in ARMS:
        arm, (geometry, dense, dense_valid) = case["arms"][name], geometries[name]
        aget = lambda key: get(name + "__" + key)
        selection, promoted, native = False, False, False
        if "selection" not in arm:
            continue
        intended = np.concatenate((common["selected"][None], aget("selection__shams")), axis=0)
        if "selection_validation" in arm:
            report = arm["selection_validation"]
            selection, measured = audit_primary(report, lambda key: aget("selection_validation__" + key), check,
                geometry=geometry, hpair=common["energy_gradient"], head=head, dense=dense, dense_valid=dense_valid)
            check(np.array_equal(measured, intended), "selection validation set differs")
            check(all(arm["geometry"]["hvp_counts"]["selection_" + suffix] == len(intended)
                      for suffix in ("fp64", "fp32")), "fresh selection HVP count differs")
        else:
            check("selection_exception" in arm, "selected set lacks selection validation or failure")
        complete = len(intended) == 9
        if "promoted" in arm:
            witness = arm["promoted"]["witness"]
            check(complete and witness["goal"] == goal and witness["original_class"] == metadata["original_goal_label"]
                  and witness["donor_class"] == metadata["donor_goal_label"], "promoted witness identity differs")
            measured = audit_promoted(arm["promoted"], lambda key: aget("promoted__" + key), check,
                common=common, native_geometry=ng, native_dense=nd, native_dense_valid=nd_valid,
                intended=intended, h32=h32, head=head, goal=goal)
            promoted = bool(arm["construction_eligible"] and complete and measured)
        else:
            check(not complete or "promoted_exception" in arm, "complete set lacks promoted endpoint")
        if "native" in arm:
            report = arm["native"]
            check(complete and report["goal"] == goal and report["donor_class"] == metadata["donor_goal_label"]
                  and report["original_class"] == metadata["original_goal_label"], "native endpoint identity differs")
            audited = audit_native(report, lambda key: aget("native__" + key), check,
                target=common["target"], intended=intended, h32=h32, h64=h64, reference_j=reference_j,
                head_projector=hp, head=head)
            native = audited["native_ready"]
            if ng is not None:
                check(_same(aget("native__jacobian64_same_state"), ng["jacobian"]),
                      "fresh native-state promoted Jacobian differs from common geometry")
        else:
            check(not complete or "native_exception" in arm, "complete set lacks native endpoint")
        for key, value in zip(ENDPOINTS, (selection, promoted, native)):
            check(arm[key] == value, name + " endpoint differs: " + key)
        for endpoint in ("promoted", "native"):
            if endpoint + "_exception" in arm:
                check(arm[endpoint + "_exception"].get("call_counts_complete") is False,
                      "raised helper hides incomplete call counts")
    if all("native" in case["arms"][name] for name in ARMS):
        for key in ("actual_edits_native_fp64", "probe32", "probe64", "hvp32", "hvp64_same_probe", "hvp64_actual_delta"):
            check(np.array_equal(get(ARMS[0] + "__native__" + key)[0], get(ARMS[1] + "__native__" + key)[0]),
                  "shared donor native measurement differs: " + key)
    if all("promoted" in case["arms"][name] for name in ARMS):
        for key in ("actual_edits64", "fresh_hvps", "pair_gradients", "probe_edits"):
            check(np.array_equal(get(ARMS[0] + "__promoted__" + key)[0], get(ARMS[1] + "__promoted__" + key)[0]),
                  "shared donor promoted measurement differs: " + key)


def _audit_summaries(document, check):
    cases = document["cases"]
    counts = Counter(common_target=sum(case["common_target_hvp_count"] for case in cases))
    gradients, incomplete = 0, []
    for name in ARMS:
        rows, summary = [case["arms"][name] for case in cases], document["arms"][name]
        check(summary["n_constructed"] == sum(row["construction_eligible"] for row in rows), "construction total differs")
        for key, count_key, reason_key, reasons_key in (
            ("selection_validated", "n_selection_validated", "selection_reason", "selection_failure_reasons"),
            ("promoted_ready", "n_promoted_ready", "promoted_reason", "promoted_failure_reasons"),
            ("native_fp32_ready", "n_native_fp32_ready", "native_fp32_reason", "native_fp32_failure_reasons")):
            total = sum(bool(row[key]) for row in rows)
            per_base = [sum(case["arms"][name][key] for case in cases if case["base"] == base) for base in range(8)]
            check(summary[count_key] == total and summary["per_base"][key] == per_base, "grouped endpoint count differs: " + key)
            check(summary[reasons_key] == dict(Counter(row[reason_key] for row in rows if not row[key])),
                  "failure reason total differs: " + key)
            if key != "selection_validated":
                check(summary[key.replace("_ready", "_status")] == ("PASS" if total >= 61 else "FAIL"),
                      "execution-contract threshold differs: " + key)
        for case, row in zip(cases, rows):
            for key, value in row.get("geometry", {}).get("hvp_counts", {}).items():
                counts[name + "__" + key] += value
            witness = row.get("promoted", {}).get("witness", {})
            counts[name + "__promoted_fresh"] += witness.get("fresh_hvp_count", 0)
            gradients += witness.get("first_gradient_count", 0)
            for key, value in row.get("native", {}).get("hvp_counts", {}).items():
                counts[name + "__native_" + key] += value
            for endpoint in ("promoted", "native"):
                if endpoint + "_exception" in row:
                    incomplete.append({"base": case["base"], "goal": case["goal"], "arm": name, "endpoint": endpoint})
    check(document["hvp_counts"] == dict(counts) and document["independent_first_gradient_count"] == gradients,
          "purpose-call totals differ")
    check(document["call_counts_complete"] == (not incomplete)
          and document["incomplete_call_count_endpoints"] == incomplete, "call-count completeness differs")
    check(document["reference_drift_jacobian_fallback_count"] == sum(case.get("reference_drift_jacobian_fallback_count", 0)
          for case in cases), "diagnostic Jacobian fallback count differs")
    ceiling, rows = document["donor_precision_ceiling"], []
    for case in cases:
        measured = [name for name in ARMS if "native" in case["arms"][name]]
        report = case["arms"][measured[0]]["native"] if measured else None
        error = report["arms"][0]["probe_resolution_error"] if report is not None else None
        tolerance = report["q_tolerance"] if report is not None else None
        rows.append({"base": case["base"], "goal": case["goal"], "measured_arms": measured,
                     "probe_resolution_error": error, "q_tolerance": tolerance,
                     "probe_resolution_passed": error <= tolerance if report is not None else None})
    measured = sum(bool(row["measured_arms"]) for row in rows)
    failed = sum(row["probe_resolution_passed"] is False for row in rows)
    check(ceiling["per_case"] == rows and ceiling["n_measured_cases"] == measured
          and ceiling["n_unmeasured_cases"] == len(cases) - measured
          and ceiling["n_donor_probe_resolution_failures"] == failed
          and ceiling["native_fp32_case_upper_bound"] == len(cases) - failed
          and ceiling["coverage"] == ("complete" if measured == len(cases) else "partial"),
          "donor-only precision ceiling or its measured coverage differs")
    for key in ENDPOINTS:
        pairs = [(bool(case["arms"][ARMS[0]][key]), bool(case["arms"][ARMS[1]][key])) for case in cases]
        differences = [int(right) - int(left) for left, right in pairs]
        macro = [sum(value for value, case in zip(differences, cases) if case["base"] == base) / 8 for base in range(8)]
        report, transitions = document["paired"][key], Counter(pairs)
        check(report["case_differences"] == differences and report["case_difference_count"] == sum(differences)
              and report["per_base_fraction_differences"] == macro
              and report["source_macro_fraction_difference"] == sum(macro) / 8, "paired descriptive contrast differs")
        for label, pair in (("both_pass", (True, True)), ("reference_only", (True, False)),
                             ("native_only", (False, True)), ("both_fail", (False, False))):
            check(report[label] == transitions[pair], "paired transition count differs")
        check(report["registered_role"] == ("primary" if key == "promoted_ready" else "secondary_descriptive"),
              "registered contrast role differs")


def audit(receipt: Path, raw: Path, out: Path | None = None, *, expected_seed=61067):
    receipt, raw = Path(receipt), Path(raw)
    document = json.loads(receipt.read_text())
    failures, results = [], []
    def check(condition, message):
        if not condition:
            failures.append(message)
    check(document["raw_sha256"] == _sha(raw), "raw archive digest differs")
    sidecar = receipt.with_suffix(receipt.suffix + ".sha256")
    check(sidecar.exists() and sidecar.read_text().split()[0] == _sha(receipt), "receipt digest sidecar differs")
    check(document["excluded_source_seed"] == document["excluded_model_seed"] == expected_seed,
          "excluded model/source seed differs")
    check(document["n_excluded_bases"] == document["n_goals_per_base"] == 8 and document["n_cases"] == 64,
          "source grouping differs")
    check(document["n_untrained_model_initializations"] == 1 and document["n_new_trained_model_roots"] == 0,
          "model evidence class differs")
    check(document["registered_min_admitted"] == 61 and document["common_native_parent"] is True,
          "registered common-source admission differs")
    check(all(document[key] is False for key in ("new_test_generated", "trained_weights_loaded",
          "edited_task_outcomes_scored", "trained_release_authorized", "precision_endpoints_used_in_proposal_search")),
          "source-only precision-search boundary differs")
    check(document["paired_proposal_seed_namespace"] == "LGA10B", "proposal namespace differs")
    cases = document["cases"]
    check([(case["base"], case["goal"]) for case in cases] == [(base, goal) for base in range(8) for goal in GOALS],
          "64 cases are missing, duplicated or reordered")
    try:
        with open_capture(raw, document["capture"]) as arrays:
            for key in arrays.files:
                check(bool(np.isfinite(arrays[key]).all()), "nonfinite captured array: " + key)
            x, y, head, hp = (arrays[key] for key in ("source_inputs", "source_labels", "head_weight", "head_projector"))
            check(x.shape == y.shape == (8, 81) and np.all((x >= 1) & (x <= 9))
                  and np.array_equal(y, np.roll(x, 1, axis=1)), "cyclic source law differs")
            for label, value in (("inputs", x), ("labels", y)):
                check(hashlib.sha256(np.asarray(value, dtype="<i8").tobytes()).hexdigest() == document["source_" + label + "_sha256"],
                      "source digest differs: " + label)
            check(np.array_equal(arrays["head_weight_fp32"].astype(np.float64), head)
                  and _norm(head @ hp) <= 1e-8 and _norm(hp @ hp - hp) <= 1e-8
                  and _norm(hp - hp.T) <= 1e-8, "common head promotion/projector differs")
            for case in cases:
                before = len(failures)
                label = f"base {case['base']} goal {case['goal']}: "
                local = lambda okay, message: check(okay, label + message)
                try:
                    _audit_case(case, arrays, local, x=x, hp=hp, head=head, expected_seed=expected_seed)
                except (KeyError, ValueError, ArithmeticError, IndexError, TypeError, np.linalg.LinAlgError) as error:
                    local(False, "invalid case capture: " + type(error).__name__ + ": " + str(error))
                results.append({"base": case["base"], "goal": case["goal"], "audit_passed": len(failures) == before})
    except (OSError, KeyError, ValueError, ArithmeticError) as error:
        check(False, "invalid numeric capture: " + str(error))
    try:
        _audit_summaries(document, check)
    except (KeyError, ValueError, TypeError) as error:
        check(False, "invalid grouped receipt: " + str(error))
    result = {"schema": "LGA10D/independent-paired-numpy-audit-v1", "audit_status": "FAIL" if failures else "PASS",
              "receipt_sha256": _sha(receipt), "raw_sha256": _sha(raw), "numpy_version": np.__version__,
              "model_matcher_or_torch_imported": False, "n_cases": len(results),
              "n_cases_audit_passed": sum(row["audit_passed"] for row in results), "failures": failures, "cases": results}
    if out is not None:
        Path(out).write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print("LGA10D_INDEPENDENT_PAIRED_AUDIT", result["audit_status"], result["n_cases_audit_passed"], len(results), flush=True)
    for failure in failures:
        print("AUDIT_FAILURE", failure, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    options = parser.parse_args()
    result = audit(options.receipt, options.raw, options.out)
    raise SystemExit(0 if result["audit_status"] == "PASS" else 1)

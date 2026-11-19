"""Frozen LGA-10D seed-61067 retrospective FP32 decomposition for LGA-10E.

Reads aliased NPY arrays. No model, matcher, Torch, source RNG, or new HVPs.
Every contribution is a signed telescoping diagnostic, not a causal repair.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np

SEED = 61067
ARMS = ("reference_derivatives", "native_derivatives")
GOALS = tuple(range(0, 80, 10))
RAW_SHA256 = "20f264a7fe0044ca2b142a5d1b70ffb27b6915a914bb91f96c30ccb5de65246d"
RECEIPT_SHA256 = "223d7a247e1a4e581839017e6f44b1aad11a4ae20004e5de9757616a17a9a44f"
SCHEMA = "LGA10E/retrospective-native-fp32-attribution-v1"
COLUMNS = (
    "base", "goal", "arm", "edit_index", "kind", "left_contraction",
    "hvp_arithmetic", "probe_substitution", "total", "parity_total",
    "q_native", "q32_probe", "q64_probe", "q64_actual",
    "application_rounding_l2", "probe_displacement_l2",
)


class ReplayIntegrityError(ValueError):
    """A source or mathematical receipt is missing, altered or nonfinite."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayIntegrityError(message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_array_equal(observed, expected, label: str, *, atol=1e-12) -> None:
    left, right = np.asarray(observed), np.asarray(expected)
    require(left.shape == right.shape, label + ": wrong shape")
    require(np.isfinite(left).all() and np.isfinite(right).all(),
            label + ": nonfinite")
    require(bool(np.allclose(left, right, rtol=0.0, atol=atol)),
            label + ": content mismatch")


def decompose(q_native, q32_probe, q64_probe, q64_actual) -> dict[str, np.ndarray]:
    """A signed and exact telescoping identity for nine donor/sham scalars."""
    scalars = tuple(np.asarray(q, dtype=np.float64) for q in
                    (q_native, q32_probe, q64_probe, q64_actual))
    for scalar in scalars:
        require(scalar.shape == (9,), "nine donor/sham scalar observations required")
        require(bool(np.isfinite(scalar).all()), "nonfinite curvature scalar")
    qn, q32, q64p, q64a = scalars
    pieces = {
        "left_contraction": qn - q32,
        "hvp_arithmetic": q32 - q64p,
        "probe_substitution": q64p - q64a,
        "total": qn - q64a,
    }
    reconstructed = (pieces["left_contraction"] +
                     pieces["hvp_arithmetic"] + pieces["probe_substitution"])
    assert_array_equal(pieces["total"], reconstructed, "signed telescoping identity",
                       atol=5e-17)
    return pieces


def paired_deltas(vector: np.ndarray) -> np.ndarray:
    return vector[1:] - vector[0]


def verify_metadata(document: dict) -> None:
    require(document.get("schema") ==
            "LGA10D/excluded-source-61067-common-native-parent-v1",
            "unexpected original LGA-10D schema")
    require(document.get("excluded_model_seed") == SEED and
            document.get("excluded_source_seed") == SEED,
            "wrong source or model seed")
    require(document.get("n_cases") == 64 and
            document.get("n_excluded_bases") == 8 and
            document.get("n_goals_per_base") == 8,
            "wrong original source inventory")
    require(document.get("n_untrained_model_initializations") == 1 and
            document.get("n_new_trained_model_roots") == 0 and
            document.get("trained_weights_loaded") is False and
            document.get("edited_task_outcomes_scored") is False,
            "retrospective evidence-class mismatch")
    require(document.get("reserved_science_seeds_unopened") ==
            [31059, 31064, 31065], "protected science seeds changed")
    require(document.get("raw_sha256") == RAW_SHA256,
            "raw capture is not the original archived 61067 source")
    require(document.get("capture", {}).get("schema") == "LGA10D/aliased-npz-v1",
            "unexpected content-addressed array format")
    require(len(document.get("cases", [])) == 64, "incomplete source cases")


def inspect_arm(case: dict, arm_name: str, get, base: int, goal: int):
    prefix = f"case_{base:02}_{goal:02}__{arm_name}__native__"
    scalar_names = ("q_native", "q32_same_probe", "q64_same_probe", "q64_actual")
    vector_names = ("intended_edits", "actual_edits_native_fp64", "probe32",
                    "probe64", "hvp32", "hvp64_same_probe",
                    "hvp64_actual_delta", "native_pair_base_fp32",
                    "application_rounding_residual", "direction_cast_residual",
                    "probe_resolution_errors", "paired_resolution_errors")
    arrays = {name: get(prefix + name) for name in (*scalar_names, *vector_names)}
    for name in scalar_names:
        require(arrays[name].shape == (9,) and arrays[name].dtype == np.float64,
                prefix + name + ": invalid scalar")
    for name in vector_names[:7]:
        want_dtype = np.float32 if name in ("probe32", "hvp32") else np.float64
        require(arrays[name].shape == (9, 96) and arrays[name].dtype == want_dtype,
                prefix + name + ": invalid vector")
    require(arrays["native_pair_base_fp32"].shape == (2, 48) and
            arrays["native_pair_base_fp32"].dtype == np.float32,
            prefix + "native_pair_base_fp32: invalid base")
    for name in ("application_rounding_residual", "direction_cast_residual"):
        require(arrays[name].shape == (9, 96) and arrays[name].dtype == np.float64,
                prefix + name + ": invalid residual")
    for name in ("probe_resolution_errors", "paired_resolution_errors"):
        require(arrays[name].shape == (9,) and arrays[name].dtype == np.float64,
                prefix + name + ": invalid resolution")

    a = arrays["actual_edits_native_fp64"]
    p = arrays["probe64"]
    intended = arrays["intended_edits"]
    base32 = arrays["native_pair_base_fp32"].reshape(96)
    expected_a = ((base32[None] + intended.astype(np.float32)).astype(np.float64)
                  - base32[None].astype(np.float64))
    assert_array_equal(a, expected_a, prefix + "FP32 edit application", atol=0.0)
    assert_array_equal(p, arrays["probe32"].astype(np.float64),
                       prefix + "probe exact promotion", atol=0.0)
    assert_array_equal(arrays["application_rounding_residual"], a - intended,
                       prefix + "application rounding receipt")
    assert_array_equal(arrays["direction_cast_residual"], a - p,
                       prefix + "probe displacement receipt")

    h32 = arrays["hvp32"].astype(np.float64)
    h64p = arrays["hvp64_same_probe"]
    h64a = arrays["hvp64_actual_delta"]
    reconstructed = {
        "q_native": np.einsum("ij,ij->i", a, h32),
        "q32_same_probe": np.einsum("ij,ij->i", p, h32),
        "q64_same_probe": np.einsum("ij,ij->i", p, h64p),
        "q64_actual": np.einsum("ij,ij->i", a, h64a),
    }
    for key, value in reconstructed.items():
        assert_array_equal(arrays[key], value, prefix + key + " HVP reconstruction",
                           atol=1e-12)
    pieces = decompose(*(arrays[x] for x in scalar_names))
    assert_array_equal(arrays["probe_resolution_errors"],
                       np.abs(pieces["hvp_arithmetic"]),
                       prefix + "stored probe resolution")
    assert_array_equal(arrays["paired_resolution_errors"],
                       np.abs(paired_deltas(pieces["total"])).reshape(-1)
                       if False else np.r_[0., np.abs(paired_deltas(pieces["total"]))],
                       prefix + "stored paired resolution")
    native = case["arms"][arm_name]["native"]
    tol = float(native["q_tolerance"])
    require(math.isfinite(tol) and tol > 0, "invalid native gate tolerance")
    require(abs(tol - 1e-8) < 1e-14, "unexpected LGA-10D numeric tolerance")
    require(bool(native["native_ready"]) ==
            bool(case["arms"][arm_name]["native_fp32_ready"]),
            "native endpoint receipt mismatch")
    rows = []
    rounding = np.linalg.norm(a - intended, axis=1)
    displacement = np.linalg.norm(a - p, axis=1)
    for i in range(9):
        rows.append({
            "base": base, "goal": goal, "arm": arm_name, "edit_index": i,
            "kind": "donor" if i == 0 else f"sham_{i}",
            **{key: float(val[i]) for key, val in pieces.items()},
            "parity_total": float(pieces["total"][i] - pieces["total"][0]),
            "q_native": float(arrays["q_native"][i]),
            "q32_probe": float(arrays["q32_same_probe"][i]),
            "q64_probe": float(arrays["q64_same_probe"][i]),
            "q64_actual": float(arrays["q64_actual"][i]),
            "application_rounding_l2": float(rounding[i]),
            "probe_displacement_l2": float(displacement[i]),
        })
    pair = paired_deltas(pieces["total"])
    arith_pair = paired_deltas(pieces["hvp_arithmetic"])
    without_arithmetic = pair - arith_pair
    metrics = {
        "base": base, "goal": goal, "arm": arm_name,
        "native_fp32_ready_original": bool(case["arms"][arm_name]["native_fp32_ready"]),
        "promoted_ready_original": bool(case["arms"][arm_name]["promoted_ready"]),
        "donor_probe_error": float(abs(pieces["hvp_arithmetic"][0])),
        "donor_probe_passed": bool(abs(pieces["hvp_arithmetic"][0]) <= tol),
        "max_abs_native_vs_promoted_sham_parity": float(np.max(abs(pair))),
        "max_abs_sham_hvp_arithmetic": float(np.max(abs(arith_pair))),
        "max_abs_sham_probe_substitution": float(np.max(abs(paired_deltas(pieces["probe_substitution"])))),
        "max_abs_sham_left_contraction": float(np.max(abs(paired_deltas(pieces["left_contraction"])))),
        "max_abs_sham_parity_without_arithmetic_proxy": float(np.max(abs(without_arithmetic))),
        "pair_parity_passed_original_numeric_only": bool(np.all(abs(pair) <= tol)),
        "pair_parity_passed_without_arithmetic_proxy_only": bool(np.all(abs(without_arithmetic) <= tol)),
        "max_application_rounding_l2": float(np.max(rounding)),
        "max_probe_displacement_l2": float(np.max(displacement)),
    }
    return metrics, rows


def replay(archive_dir: Path) -> tuple[dict, list[dict]]:
    folder = Path(archive_dir)
    raw = folder / "raw.npz"
    receipt = folder / "feasibility.json"
    require(raw.is_file() and receipt.is_file(), "original raw/receipt files are required")
    require(digest(raw) == RAW_SHA256, "original raw.npz SHA-256 mismatch")
    require(digest(receipt) == RECEIPT_SHA256, "original feasibility.json SHA-256 mismatch")
    document = json.loads(receipt.read_text(encoding="utf-8"))
    verify_metadata(document)
    aliases = document["capture"]["logical_to_physical"]
    require(isinstance(aliases, dict), "missing aliased capture map")
    metrics, rows = [], []
    with ZipFile(raw) as store:
        members = set(store.namelist())
        def get(name: str) -> np.ndarray:
            physical = aliases.get(name)
            require(isinstance(physical, str) and
                    re.fullmatch(r"a_[0-9a-f]{64}", physical) is not None,
                    "missing or malformed NPY alias: " + name)
            member = physical + ".npy"
            require(member in members, "missing archive member: " + member)
            array = np.load(io.BytesIO(store.read(member)), allow_pickle=False)
            require(np.isfinite(array).all(), "nonfinite archived array: " + name)
            return array

        for index, case in enumerate(document["cases"]):
            b, g = divmod(index, 8)
            g = GOALS[g]
            require(case["base"] == b and case["goal"] == g,
                    "source case order or independence units changed")
            for arm in ARMS:
                part, arm_rows = inspect_arm(case, arm, get, b, g)
                metrics.append(part)
                rows.extend(arm_rows)

    require(len(rows) == 64 * 2 * 9 and len(metrics) == 128,
            "incomplete paired edit inventory")
    by_arm = {}
    for arm in ARMS:
        cases = [v for v in metrics if v["arm"] == arm]
        by_arm[arm] = {
            "n_cases": len(cases),
            "n_original_native_fp32_ready": sum(c["native_fp32_ready_original"] for c in cases),
            "n_original_promoted_ready": sum(c["promoted_ready_original"] for c in cases),
            "n_donor_probe_resolution_failed": sum(not c["donor_probe_passed"] for c in cases),
            "n_sham_parity_numeric_only_pass": sum(c["pair_parity_passed_original_numeric_only"] for c in cases),
            "n_sham_parity_without_arithmetic_proxy_only_pass":
                sum(c["pair_parity_passed_without_arithmetic_proxy_only"] for c in cases),
            "per_base": [
                {"base": b, "n_original_native_fp32_ready":
                 sum(c["native_fp32_ready_original"] for c in cases if c["base"] == b),
                 "n_donor_probe_resolution_failed":
                 sum(not c["donor_probe_passed"] for c in cases if c["base"] == b)}
                for b in range(8)
            ],
        }
    require(by_arm["native_derivatives"]["n_original_native_fp32_ready"] == 1 and
            by_arm["reference_derivatives"]["n_original_native_fp32_ready"] == 1 and
            by_arm["native_derivatives"]["n_original_promoted_ready"] == 64 and
            by_arm["reference_derivatives"]["n_original_promoted_ready"] == 9,
            "original registered endpoint totals altered")
    require(all(v["n_donor_probe_resolution_failed"] == 21 for v in by_arm.values()),
            "shared donor resolution coverage changed")
    output = {
        "schema": SCHEMA,
        "evidence_class": "RETROSPECTIVE_EXPOSED_61067_NUMERICAL_DIAGNOSIS",
        "no_new_source_or_model": True,
        "execution_contract_changed": False,
        "fresh_hvp_calls": 0,
        "original_raw_sha256": RAW_SHA256,
        "original_receipt_sha256": RECEIPT_SHA256,
        "n_independent_bases": 8,
        "n_correlated_goals_per_base": 8,
        "n_edits_per_arm_case": 9,
        "donor_only_original_upper_bound": 43,
        "components": ["left_contraction", "hvp_arithmetic", "probe_substitution"],
        "component_claim": "signed_telescoping_not_independent_causal_effects",
        "counterfactual_claim": "signed_arithmetic_proxy_only_not_new_gate_evaluation",
        "arms": by_arm,
        "cases": metrics,
    }
    return output, rows


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--csv-out", required=True, type=Path)
    args = parser.parse_args(argv)
    result, rows = replay(args.archive_dir)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True,
                                         allow_nan=False) + "\n", encoding="utf-8")
    with args.csv_out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print("LGA10E_RETROSPECTIVE " + json.dumps({
        "schema": SCHEMA, "arm_summary": result["arms"],
        "n_rows": len(rows), "new_hvps": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()

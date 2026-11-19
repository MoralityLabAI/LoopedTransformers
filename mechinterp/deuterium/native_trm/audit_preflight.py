"""Independent standard-library-only arithmetic audit for the native preflight.

Historical model inference runs in preflight.py. This file never imports torch.
It independently rechecks model/source/protocol bytes, accuracy, dose and closure
receipts and deliberately has no entry point to prospective data generators.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SCHEMA = "DEUTERIUM-NATIVE-01/native-source-only-preflight-v1"
EXPECTED_BUNDLE = "d32f249385df2eabefcacbee15c9ab2cccc73f01452300f275a11f621106a266"
EXPECTED_META = "7bf32b627d7b95e756119cec221c33afd9a6c1cd44e6904ef9002791026bc07d"
ORIGINAL_BLOBS = {
    "mechinterp/loopscope04/gradient_flow.py": "ec1c601c321230b14f974bb76ba3610da6e9aa8f",
    "mechinterp/loopscope05/fit_and_freeze.py": "d933ce68cce6bc08670b8f5221d646ee11896629",
    "mechinterp/loopscope05/terminal_sae_test.py": "e9a9e77b306d1939e106267e8fd3cd1d41f8582d",
}
TEACHERS = ("103", "211", "307")
THRESHOLD = 1e-6
DOSE_TOL = 1e-4


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def blob(path: Path) -> str:
    raw = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\x00" + raw).hexdigest()


def finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def audit(preflight: Path, teacher_bundle: Path, teacher_meta: Path, out: Path) -> dict:
    r = json.loads(preflight.read_text())
    m = json.loads(teacher_meta.read_text())
    errors = []
    def require(ok, label):
        if not ok:
            errors.append(label)

    require(r.get("schema") == SCHEMA, "schema")
    require(r.get("status") == "PREFLIGHT_PASS", "source_preflight_status")
    require(r.get("evidence_class") == "HISTORICAL_SOURCE_ENGINEERING_ONLY", "evidence_class")
    require(r.get("new_calibration_generated") is False and
            r.get("new_heldout_generated") is False and
            r.get("new_heldout_labels_read") is False and
            r.get("new_score_comparison_completed") is False, "source_separation")
    require(r.get("new_calibration_seed_reserved") == 731055 and
            r.get("new_heldout_seed_reserved") == 731057, "reserved_roster")
    require(r.get("teacher_seeds") == [103, 211, 307] and
            set(r.get("models", {})) == set(TEACHERS), "teacher_roster")
    require(r.get("historical_source_seed") == 31043 and
            r.get("historical_source_n") == 256, "old_source_roster")
    require(r.get("historical_x_sha256") ==
            m.get("prior_test_used_only_for_reproduction", {}).get("x_sha256") and
            r.get("historical_y_sha256") ==
            m.get("prior_test_used_only_for_reproduction", {}).get("y_sha256"), "old_source_hashes")
    require(digest(teacher_bundle) == EXPECTED_BUNDLE and
            digest(teacher_meta) == EXPECTED_META, "original_artifact_hashes")
    require(r.get("original_teacher_bundle_sha256") == EXPECTED_BUNDLE and
            r.get("original_teacher_meta_sha256") == EXPECTED_META, "recorded_artifact_hashes")
    require(r.get("protocol_sha256") == digest(HERE / "PROTOCOL.md"), "protocol_sha")
    require(r.get("original_source_git_blobs") == ORIGINAL_BLOBS and
            all(blob(REPO / p) == sha for p, sha in ORIGINAL_BLOBS.items()),
            "original_source_blobs")
    require(r.get("original_canonical_tensor_check") == "PASS", "canonical_tensor_loader")
    require(r.get("independent_original_reference_logits") == "UNAVAILABLE",
            "reference_logits_limit")
    measured = {}
    for seed in TEACHERS:
        x = r.get("models", {}).get(seed, {})
        n = x.get("old_source_tokens")
        accuracy = x.get("clean_old_source_accuracy")
        historical = x.get("original_train_meta_old_source_accuracy")
        expected = m.get("teacher_receipts", {}).get(seed, {}).get("reproduced_accuracy")
        require(n == 256 * 81 and finite(accuracy) and finite(historical) and
                finite(expected) and accuracy >= 0.95 and
                abs(accuracy - historical) <= 1 / (256 * 81) + 1e-8 and
                abs(historical - expected) <= 1e-12, seed + ":clean_accuracy")
        require(x.get("weight_target") == "block.chan_mix.2.weight" and
                x.get("weight_shape") == [48, 128] and
                x.get("weight_edit_conceptual_rank") == 1 and
                x.get("all_four_recurrent_weight_uses_edited") is True,
                seed + ":target_and_rank")
        for key in ("full_manual_forward_max_logit_error",
                    "zero_hidden_edit_max_logit_error",
                    "weight_edit_native_manual_max_logit_error",
                    "weight_restore_max_logit_error",
                    "weight_restore_max_parameter_error",
                    "weight_edit_actual_relative_dose_error"):
            value = x.get(key)
            limit = DOSE_TOL if key.endswith("relative_dose_error") else THRESHOLD
            require(finite(value) and 0 <= value <= limit, seed + ":" + key)
        for key in ("hidden_edit_native_max_logit_change",
                    "weight_edit_native_max_logit_change",
                    "hidden_edit_intended_l2", "hidden_edit_actual_l2",
                    "weight_edit_intended_frobenius",
                    "weight_edit_actual_frobenius",
                    "weight_edit_max_matrix_delta_error"):
            require(finite(x.get(key)) and x[key] >= 0, seed + ":" + key)
        require(finite(x.get("hidden_edit_native_max_logit_change")) and
                x["hidden_edit_native_max_logit_change"] > 1e-8 and
                finite(x.get("weight_edit_native_max_logit_change")) and
                x["weight_edit_native_max_logit_change"] > 1e-8, seed + ":positive_effect")
        if finite(x.get("hidden_edit_actual_l2")) and finite(x.get("hidden_edit_intended_l2")):
            require(abs(x["hidden_edit_actual_l2"] - x["hidden_edit_intended_l2"]) /
                    max(x["hidden_edit_intended_l2"], 1e-20) <= DOSE_TOL,
                    seed + ":activation_dose")
        measured[seed] = {
            "historical_clean_accuracy": accuracy,
            "native_weight_change": x.get("weight_edit_native_max_logit_change"),
            "native_hidden_change": x.get("hidden_edit_native_max_logit_change"),
        }
    result = {
        "schema": "DEUTERIUM-NATIVE-01/stdlib-independent-preflight-audit-v1",
        "audit_status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "torch_imported": "torch" in sys.modules,
        "teacher_checks": measured,
        "original_artifact_sha256": {"bundle": digest(teacher_bundle),
                                     "meta": digest(teacher_meta)},
        "prospective_scientific_verdict": "NOT_RUN",
    }
    if result["torch_imported"]:
        result["audit_status"] = "FAIL"
        result["errors"].append("torch_imported")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    if result["audit_status"] != "PASS":
        raise RuntimeError("independent preflight audit failed: " + ", ".join(result["errors"]))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--teacher-bundle", type=Path, required=True)
    parser.add_argument("--teacher-meta", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.preflight, args.teacher_bundle, args.teacher_meta, args.out)
    print("DEUTERIUM_NATIVE_INDEPENDENT_PREFLIGHT", result["audit_status"], flush=True)


if __name__ == "__main__":
    main()

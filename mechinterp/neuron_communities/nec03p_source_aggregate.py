"""NEC-03P all-attempted 88-root recruitment ledger, independent source-only audit.

Pure stdlib. Missing/failed root jobs are counted in the 88-root intent-
to-treat denominator. Eligible root counts are *source-instrument
candidates*, not the causal-sham samples needed by a 95/95 tolerance
interval. No development evaluation or confirmation inputs are scored.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

ROOTS = tuple(range(8, 96))
SCHEMA = "NEC-03P/independent-source-recruitment-88-root-ledger/v1"


def read(path: Path):
    return json.loads(path.read_text())


def sha_file(path: Path):
    return sha256(path.read_bytes()).hexdigest()


def aggregate(root: Path, out: Path, lock: Path):
    expected = {f"nec03p-source-root-{index}" for index in ROOTS}
    unexpected = {
        p.name for p in root.glob("nec03p-source-root-*")
        if p.is_dir() and p.name not in expected
    }
    if unexpected:
        raise AssertionError("root cohort drift or accidental embargo access")
    lock_sha = sha_file(lock)
    records = []
    for index in ROOTS:
        path = root / f"nec03p-source-root-{index}"
        source_path = path / "nec03p_source_receipt.json"
        audit_path = path / "independent_nec03p_source_audit.json"
        if not source_path.is_file() or not audit_path.is_file():
            records.append({
                "world_id": f"dev5-{index:03d}", "world_index": index,
                "status": "missing_or_failed_neural_audit",
                "attempted_ITT": True, "paired_source_eligible": False,
                "arms": {},
            })
            continue
        source, audit = read(source_path), read(audit_path)
        if (source["schema"] != "NEC-03P/development-source-recruitment-native-reset-sae/v1"
            or audit["schema"] !=
               "NEC-03P/independent-development-source-recruitment-audit/v1"
            or source["world_id"] != f"dev5-{index:03d}"
            or audit["world_id"] != source["world_id"]
            or source["world_index"] != index
            or audit["world_index"] != index
            or source["world_sha256"] != audit["world_sha256"]
            or source["source_lock_sha256"] != lock_sha
            or audit["source_receipt_sha256"] != sha_file(source_path)
            or not audit["both_stage_checkpoint_bytes_and_training_source_labels_audited"]
            or not audit["both_arm_source_SAEs_and_geometry_independently_checked"]
            or not source["optimizer_reset_at900"]
            or source["exact_continuous_adamw_1800_claim"] is not False
            or not source["all_arms_in_intent_to_treat_denominator"]
            or (source["source_training_groups"], source["source_fit_groups"],
                source["source_rank_groups"], source["source_geometry_groups"]) != (80,40,40,8)
            or any(source[key] for key in (
                "new_development_evaluation_groups_scored", "reserved_context_groups_scored",
                "embargoed_worlds_scored", "four_input_confirmation_worlds_scored"))
            or any(audit[key] for key in (
                "new_development_evaluation_groups_scored", "reserved_context_groups_scored",
                "embargoed_worlds_scored", "four_input_confirmation_worlds_scored"))
            or source["scientific_gate"] != "REDESIGN_OR_PILOT_ONLY"
            or audit["scientific_gate"] != "REDESIGN_OR_PILOT_ONLY"
            or set(source["arms"]) != {"composed", "additive"}
            or set(audit["arms"]) != {"composed", "additive"}):
            raise AssertionError(f"forged or malformed recruited source root {index}")
        arms = {}
        for arm in ("composed", "additive"):
            from_train = source["arms"][arm]
            from_audit = audit["arms"][arm]
            native = from_train["native"]
            old = native["old900_source_score"]["correct"]
            new = native["reset1800_source_score"]["correct"]
            if (old != from_audit["old900_correct"]
                or new != from_audit["reset1800_correct"]
                or native["reset1800_model_sha256"] != from_audit["new_model_sha256"]
                or from_train["source_model_eligible"] != from_audit["source_eligible"]
                or from_audit["source_candidate_exists"] != (
                    from_train["source_candidate"] is not None)
                or not native["fresh_AdamW_moments_at900"]
                or native["total_native_updates"] != 1800
                or not (0 <= old <= 2560 and 0 <= new <= 2560)):
                raise AssertionError(f"source checkpoint, ITT, or eligibility mismatch: {index}/{arm}")
            sae = from_train["SAE_source"]
            if sae is None and from_audit.get("sae_mse") is not None:
                raise AssertionError("source-only SAE measurement without model")
            arms[arm] = {
                "old900_correct": old, "reset1800_correct": new,
                "old900_competent": old >= 2304,
                "reset1800_competent": new >= 2304,
                "source_candidate_exists": from_audit["source_candidate_exists"],
                "fresh_sae_present": sae is not None,
                "fresh_sae_mse": from_audit.get("sae_mse"),
                "source_null_feasible": from_audit["source_null_feasible"],
                "source_eligible": from_audit["source_eligible"],
            }
        eligible = all(arms[a]["source_eligible"] for a in ("composed", "additive"))
        if (source["both_arms_source_eligible"] != eligible
            or audit["source_root_eligible"] != eligible):
            raise AssertionError(f"source paired world inclusion misclassified: {index}")
        records.append({
            "world_id": source["world_id"], "world_index": index,
            "world_sha256": source["world_sha256"],
            "source_manifest_sha256": source["source_manifest_sha256"],
            "status": "independent_source_audit_passed",
            "source_receipt_sha256": sha_file(source_path),
            "independent_audit_sha256": sha_file(audit_path),
            "attempted_ITT": True,
            "paired_source_eligible": eligible, "arms": arms,
        })
    audited = [r for r in records if r["status"] == "independent_source_audit_passed"]
    allarms = [r["arms"][a] for r in audited for a in ("composed", "additive")]
    paired_eligible = sum(int(r["paired_source_eligible"]) for r in records)
    result = {
        "schema": SCHEMA,
        "frozen_recruitment_root_ids": [r["world_id"] for r in records],
        "registered_recruitment_root_count": 88,
        "all_registered_roots_retained_in_ITT": len(records) == 88,
        "independently_audited_roots": len(audited),
        "missing_or_failed_neural_audit_roots": 88 - len(audited),
        "audited_native_model_arms": len(allarms),
        "reset1800_native_source_competent_arms": sum(int(a["reset1800_competent"]) for a in allarms),
        "fresh_source_SAE_models": sum(int(a["fresh_sae_present"]) for a in allarms),
        "source_eligible_model_arms": sum(int(a["source_eligible"]) for a in allarms),
        "paired_source_eligible_roots": paired_eligible,
        "minimum_independent_source_eligible_roots_for_future_95_95_sham_tail": 59,
        "at_least_59_source_eligible_roots": paired_eligible >= 59,
        "all_recruitment_attempts_independently_audited": len(audited) == 88,
        "ready_for_separately_registered_causal_sham_calibration":
            paired_eligible >= 59 and len(audited) == 88,
        "actual_causal_sham_root_samples_observed": 0,
        "sham_95_95_tolerance_bound_estimated": False,
        "development_evaluation_groups_scored": 0,
        "reserved_context_groups_scored": 0,
        "embargoed_worlds_scored": 0,
        "four_input_confirmation_worlds_scored": 0,
        "scientific_gate": "REDESIGN_OR_PILOT_ONLY",
        "roots": records,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out/"nec03p_source_recruitment_ledger.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)+"\n")
    print("NEC03P_SOURCE_88_ROOT_ITT_LEDGER",len(audited),paired_eligible)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--lock",type=Path,default=Path(__file__).with_name("NEC03P_SOURCE_LOCK.json"))
    a=parser.parse_args(argv)
    aggregate(a.root,a.out,a.lock)


if __name__=="__main__":
    main()

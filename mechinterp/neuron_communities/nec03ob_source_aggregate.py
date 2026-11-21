"""NEC-03O-B prospective 4-root source-only ITT aggregate, stdlib only."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

SCHEMA = "NEC-03O-B/four-prospective-source-root-ITT-aggregate/v1"
ROOTS = (4, 5, 6, 7)


def sha_file(path: Path):
    return sha256(path.read_bytes()).hexdigest()


def aggregate(root: Path, out: Path):
    expected = {f"nec03ob-source-root-{i}" for i in ROOTS}
    paths = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("nec03ob-source-root-")]
    if {p.name for p in paths} != expected:
        raise AssertionError("prospective ITT requires exactly four unselected roots")
    rows, seen = [], set()
    for i in ROOTS:
        path = root / f"nec03ob-source-root-{i}"
        run = json.loads((path / "nec03ob_source_receipt.json").read_text())
        audit = json.loads((path / "independent_nec03ob_source_audit.json").read_text())
        if (run["world_id"] != f"dev5-{i:03d}" or run["world_index"] != i
            or audit["world_index"] != i or run["world_sha256"] != audit["world_sha256"]
            or audit["source_receipt_sha256"] != sha_file(path / "nec03ob_source_receipt.json")
            or not audit["both_stage_checkpoint_bytes_and_training_source_labels_audited"]
            or not audit["both_arm_source_SAEs_and_geometry_independently_checked"]
            or not run["all_arms_in_intent_to_treat_denominator"]
            or run["exact_continuous_adamw_1800_claim"] is not False
            or not run["optimizer_reset_at900"]
            or run["new_development_evaluation_groups_scored"] != 0
            or audit["new_development_evaluation_groups_scored"] != 0
            or audit["reserved_context_groups_scored"] != 0
            or audit["embargoed_worlds_scored"] != 0
            or audit["four_input_confirmation_worlds_scored"] != 0
            or run["scientific_gate"] != "REDESIGN_OR_PILOT_ONLY"
            or audit["scientific_gate"] != "REDESIGN_OR_PILOT_ONLY"
            or set(run["arms"]) != {"composed","additive"}
            or set(audit["arms"]) != {"composed","additive"}):
            raise AssertionError("prospective source-root evidence or seal violated")
        if run["world_sha256"] in seen:
            raise AssertionError("duplicate fresh development source world")
        seen.add(run["world_sha256"])
        paired = {}
        for arm in ("composed","additive"):
            model = run["arms"][arm]["native"]
            fit = run["arms"][arm]["SAE_source"]
            verified = audit["arms"][arm]
            if (verified["old900_correct"] != model["old900_source_score"]["correct"]
                or verified["reset1800_correct"] != model["reset1800_source_score"]["correct"]
                or verified["source_eligible"] != run["arms"][arm]["source_model_eligible"]
                or verified["source_candidate_exists"] != (run["arms"][arm]["source_candidate"] is not None)
                or model["fresh_AdamW_moments_at900"] is not True
                or model["total_native_updates"] != 1800):
                raise AssertionError("prospective model milestone or ITT feasibility incorrect")
            old, new = verified["old900_correct"], verified["reset1800_correct"]
            if not (0 <= old <= 2560 and 0 <= new <= 2560):
                raise AssertionError("invalid native source score")
            paired[arm] = {
                "old900_source_correct": old,
                "reset1800_source_correct": new,
                "old900_native_competent": old >= 2304,
                "reset1800_native_competent": new >= 2304,
                "new_reset1800_sae_present": fit is not None,
                "new_sae_mse": verified.get("sae_mse"),
                "new_sae_exact16_fraction": verified.get("sae_exact_active_fraction"),
                "source_angular_coverage": verified.get("directional_coverage"),
                "source_positive_dose_coverage": verified.get("positive_dose_fraction"),
                "source_median_dose_retained": verified.get("median_retained_dose"),
                "source_null_feasible": verified["source_null_feasible"],
                "source_eligible": verified["source_eligible"],
            }
        if audit["source_root_eligible"] != all(x["source_eligible"] for x in paired.values()):
            raise AssertionError("source paired-root ITT claim incorrect")
        rows.append({
            "world_id": run["world_id"], "world_sha256": run["world_sha256"],
            "source_manifest_sha256": run["source_manifest_sha256"],
            "both_arms_source_eligible": audit["source_root_eligible"],
            "arms": paired,
        })
    arms = [r["arms"][a] for r in rows for a in ("composed","additive")]
    outcome = {
        "schema": SCHEMA, "fresh_five_input_world_roots": 4,
        "fresh_native_model_arms": 8,
        "old900_native_competent_arms": sum(x["old900_native_competent"] for x in arms),
        "reset1800_native_competent_arms": sum(x["reset1800_native_competent"] for x in arms),
        "source_sae_refits_completed": sum(x["new_reset1800_sae_present"] for x in arms),
        "source_eligible_arms": sum(x["source_eligible"] for x in arms),
        "paired_worlds_both_source_eligible": sum(r["both_arms_source_eligible"] for r in rows),
        "new_development_evaluation_groups_scored": 0,
        "reserved_context_groups_scored": 0, "embargoed_worlds_scored": 0,
        "original_four_input_confirmation_worlds_scored": 0,
        "source_training_only": True, "scientific_gate": "REDESIGN_OR_PILOT_ONLY",
        "roots": rows,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "nec03ob_four_root_source_aggregate.json").write_text(
        json.dumps(outcome, sort_keys=True, indent=2) + "\n")
    print("NEC03OB_FOUR_FRESH_SOURCE_ITT_PASS", outcome["source_eligible_arms"], outcome["reset1800_native_competent_arms"])
    return outcome


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    aggregate(a.root,a.out)

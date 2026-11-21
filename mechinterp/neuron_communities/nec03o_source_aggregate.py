"""NEC-03O four-root pure-stdlib source-only ITT aggregation."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

SCHEMA = "NEC-03O/four-source-world-independent-refit-aggregate/v1"
ROOTS = range(4)


def sha_file(p: Path):
    return sha256(p.read_bytes()).hexdigest()


def aggregate(root: Path, out: Path):
    paths = sorted(root.glob("nec03o-refit-source-root-*"))
    expected = {f"nec03o-refit-source-root-{i}" for i in ROOTS}
    if {p.name for p in paths} != expected:
        raise AssertionError("ITT requires all four original development source roots")
    records, seen = [], set()
    for index in ROOTS:
        folder = root / f"nec03o-refit-source-root-{index}"
        r = json.loads((folder / "nec03o_source_refit_receipt.json").read_text())
        a = json.loads((folder / "independent_nec03o_source_audit.json").read_text())
        if (r["world_index"] != index or a["world_index"] != index
            or r["world_id"] != f"dev5-{index:03d}"
            or r["world_sha256"] != a["world_sha256"]
            or a["source_refit_receipt_sha256"] != sha_file(folder / "nec03o_source_refit_receipt.json")
            or not a["original_900_native_and_reset1800_model_provenance_verified"]
            or not a["fresh_1800_specific_source_SAE_and_rank_audited"]
            or not a["exhaustive_disjoint_active_sham_routes_reenumerated_for_available_controls"]
            or a["new_development_or_reserved_contexts_scored"] != 0
            or a["embargoed_or_confirmation_worlds_scored"] != 0
            or r["scientific_gate"] != "REDESIGN_OR_PILOT_ONLY"
            or a["scientific_gate"] != "REDESIGN_OR_PILOT_ONLY"
            or set(r["arms"]) != {"composed", "additive"}
            or set(a["arms"]) != {"composed", "additive"}):
            raise AssertionError("source-only audited root malformed")
        if r["world_sha256"] in seen:
            raise AssertionError("duplicate world in source aggregate")
        seen.add(r["world_sha256"])
        arms = {}
        for arm in ("composed", "additive"):
            entry, audit = r["arms"][arm], a["arms"][arm]
            native = entry["native_reset1800"]
            if (native["source_score"]["correct"] != audit["source_correct"]
                or native["source_score"]["total"] != 2560
                or audit["source_total"] != 2560
                or native["new_model_sha256"] != audit["new_model_sha256"]
                or entry["source_model_eligible"] != audit["source_model_eligible"]
                or audit["source_null_feasible"] not in (True, False)):
                raise AssertionError("model-level source evidence or ITT gate misreported")
            ck = folder / native["new_checkpoint_file"]
            if ck.exists() and sha_file(ck) != native["new_checkpoint_sha256"]:
                raise AssertionError("native checkpoint was mutated after original independent audit")
            sae = entry["SAE_source"]
            if sae:
                ck = folder / sae["source_dictionary_file"]
                if sha_file(ck) != sae["source_dictionary_sha256"]:
                    raise AssertionError("source SAE byte SHA256 corrupted")
            arms[arm] = {
                "native_source_correct_of_2560": audit["source_correct"],
                "native_competent": audit["source_correct"] >= 2304,
                "old900_sae_normalized_mse": audit["old_source_sae_normalized_mse"],
                "new1800_sae_normalized_mse": audit.get("new_source_sae_normalized_mse"),
                "new1800_sae_retrained": sae is not None,
                "source_null_feasible": audit["source_null_feasible"],
                "source_eligible": audit["source_model_eligible"],
            }
        if a["source_root_eligible"] != all(z["source_eligible"] for z in arms.values()):
            raise AssertionError("world eligibility inconsistent")
        records.append({"world_id": r["world_id"], "world_sha256": r["world_sha256"],
                        "source_manifest_sha256": r["source_manifest_sha256"],
                        "arms": arms, "both_arms_source_eligible": a["source_root_eligible"]})
    allarms = [r["arms"][arm] for r in records for arm in ("composed", "additive")]
    report = {
        "schema": SCHEMA, "world_roots": 4, "native_model_arms": 8,
        "sources_only": True, "source_root_ids": [r["world_id"] for r in records],
        "new_sae_models_refit": sum(int(a["new1800_sae_retrained"]) for a in allarms),
        "new_sae_models_absent": sum(int(not a["new1800_sae_retrained"]) for a in allarms),
        "native_source_competent_arms": sum(int(a["native_competent"]) for a in allarms),
        "new_source_feasible_arms": sum(int(a["source_eligible"]) for a in allarms),
        "worlds_with_both_arms_source_feasible": sum(int(r["both_arms_source_eligible"]) for r in records),
        "new_development_eval_groups_scored": 0,
        "reserved_groups_scored": 0, "embargoed_worlds_scored": 0,
        "four_input_confirmation_worlds_scored": 0,
        "scientific_gate": "REDESIGN_OR_PILOT_ONLY", "roots": records,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "nec03o_four_source_root_aggregate.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("NEC03O_FOUR_SOURCE_ROOT_ITT_AGGREGATE_PASS", report["new_source_feasible_arms"])
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    aggregate(a.root, a.out)

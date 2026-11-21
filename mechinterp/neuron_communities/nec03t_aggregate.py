"""NEC-03T independent eight-world full-affine SAE necessity ITT.

Pure stdlib; every registered root and both teacher arms retained.
Missing independent audits and originally source-ineligible 3/5 roots
have zero primary contrast, with full cohort completeness required.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path

SCHEMA="NEC-03T/eight-root-source-affine-necessity-ITT/v1"
ROOTS=tuple(range(8))
QUALIFIED=(0,1,2,4,6,7)
ZERO_ROOTS=(3,5)
MODES=(
    "clean","hubs","native_nonhub","full_affine",
    "contrib16_only","ablate_orig_source","ablate_orig_sham",
    "ablate_wide_source","ablate_wide_sham",
    "ablate_hash_source","ablate_hash_sham",
)


def sha(path:Path):
    return sha256(path.read_bytes()).hexdigest()


def signflip(values):
    nonzero=[float(v) for v in values if abs(float(v))>1e-12]
    if not nonzero:
        return 1.0,0
    obs=abs(sum(nonzero))
    extreme=sum(abs(sum(s*v for s,v in zip(signs,nonzero)))>=obs-1e-12
                for signs in product((-1.,1.),repeat=len(nonzero)))
    return extreme/(1<<len(nonzero)),len(nonzero)


def aggregate(root:Path,out:Path,lock:Path):
    source_lock=sha(lock)
    if any(p.name not in {f"nec03t-source-root-{i}" for i in ROOTS}
           for p in root.glob("nec03t-source-root-*") if p.is_dir()):
        raise AssertionError("new/embargoed root included in source experiment")
    entries=[]
    for i in ROOTS:
        base=root/f"nec03t-source-root-{i}"
        runner_path=base/"nec03t_source_necessity_report.json"
        audit_path=base/"independent_nec03t_source_necessity_audit.json"
        if not runner_path.exists() or not audit_path.exists():
            entries.append({
                "world_id":f"dev5-{i:03d}","world_index":i,
                "independent_audit_complete":False,
                "root_primary_necessity_DID":0.,
                "source_ITT_failure":i in ZERO_ROOTS,
                "status":"missing_or_failed_root_source_neural_audit",
            })
            continue
        run=json.loads(runner_path.read_text())
        audit=json.loads(audit_path.read_text())
        if (run["schema"]!="NEC-03T/source-only-full-affine-necessity-ablation/v1"
            or audit["schema"]!="NEC-03T/independent-source-affine-necessity-replay/v1"
            or run["world_id"]!=f"dev5-{i:03d}"
            or audit["world_id"]!=run["world_id"]
            or run["world_index"]!=i or audit["world_index"]!=i
            or run["world_sha256"]!=audit["world_sha256"]
            or audit["source_lock_sha256"]!=source_lock
            or run["source_lock_sha256"]!=source_lock
            or audit["original_report_sha256"]!=sha(runner_path)
            or run["original_paired_source_eligible"]!=(i in QUALIFIED)
            or run["original_ITT_source_failure"]!=(i in ZERO_ROOTS)
            or run["source_groups_scored"]!=list(range(24,32))
            or run["source_training_cases_per_arm"]!=256
            or audit["source_cases_per_arm"]!=256
            or not audit["all_original_source_groups_and_true_labels_verified"]
            or not audit["all_native_and_SAE_checkpoint_provenance_verified"]
            or not audit["full_affine_source_background_independently_reconstructed"]
            or not audit["all_disjoint_ranked_shams_and_hash_random_controls_recomputed"]
            or not audit["all_matched_physical_ablation_doses_and_invalid_cases_replayed"]
            or not audit["all_eleven_full_recurrent_outcomes_independently_recomputed"]
            or any(run[field]!=0 or audit[field]!=0 for field in (
                "new_development_eval_groups_scored","reserved_groups_scored",
                "newly_recruited_worlds_scored","embargoed_worlds_scored",
                "old_four_input_confirmation_worlds_scored"))
            or run["new_native_or_SAE_training_updates"]!=0
            or run["scientific_gate"]!="REDESIGN_OR_PILOT_ONLY"
            or audit["scientific_gate"]!="REDESIGN_OR_PILOT_ONLY"
            or set(run["arms"])!={"composed","additive"}
            or set(audit["arms"])!={"composed","additive"}):
            raise AssertionError(f"NEC03T root source or causal-audit identity failed {i}")
        arms={}
        for arm in ("composed","additive"):
            original=run["arms"][arm]
            verified=audit["arms"][arm]
            if (original["correct"]!=verified["correct"]
                or original["valid_angular_case_count"]!=verified["matched_angular_cases"]
                or original["valid_hash_case_count"]!=verified["matched_hash_cases"]
                or original["source_ranked_necessity_correct_count"]!=verified["necessity_correct_count"]
                or original["source_model_sha256"]!=verified["source_model_sha256"]
                or original["source_sae_checkpoint_sha256"]!=verified["source_sae_checkpoint_sha256"]
                or original["original_source_candidate_present"]!=verified["original_source_candidate_present"]
                or original["original_arm_source_eligible"]!=verified["original_arm_source_eligible"]
                or original["raw_sha256"]!=verified["raw_sha256"]
                or set(original["correct"])!=set(MODES)
                or any(not 0<=x<=256 for x in original["correct"].values())):
                raise AssertionError(f"NEC03T root {i} model arm {arm} independently changed")
            arms[arm]={
                "correct":verified["correct"],
                "necessity_correct_counts":verified["necessity_correct_count"],
                "valid_angular_case_count":verified["matched_angular_cases"],
                "valid_hash_case_count":verified["matched_hash_cases"],
                "original_source_eligible":verified["original_arm_source_eligible"],
            }
        primary=(arms["composed"]["necessity_correct_counts"]["orig"]-
                 arms["additive"]["necessity_correct_counts"]["orig"])/256.
        if (not math.isclose(primary,run["raw_primary_source_necessity_DID"],abs_tol=1e-12)
            or not math.isclose(primary if i in QUALIFIED else 0,
                run["primary_ITT_root_effect"],abs_tol=1e-12)
            or not math.isclose(primary if i in QUALIFIED else 0,
                audit["paired_root_primary_ITT_effect"],abs_tol=1e-12)):
            raise AssertionError("registered root-level necessity primary result changed")
        entries.append({
            "world_id":run["world_id"],"world_index":i,
            "world_sha256":run["world_sha256"],
            "independent_audit_complete":True,
            "source_ITT_failure":i in ZERO_ROOTS,
            "root_primary_necessity_DID":primary if i in QUALIFIED else 0.,
            "unselected_raw_primary_DID":primary,
            "source_report_sha256":sha(runner_path),
            "independent_audit_sha256":sha(audit_path),
            "arms":arms,
        })
    audited=[v for v in entries if v["independent_audit_complete"]]
    totals={
        arm:{mode:sum(e["arms"][arm]["correct"][mode] for e in audited)
             for mode in MODES}
        for arm in ("composed","additive")
    }
    effects=[r["root_primary_necessity_DID"] for r in entries]
    primary=sum(effects)/8
    p,nonzero=signflip(effects)
    result={
        "schema":SCHEMA,
        "root_statistical_unit":"one original independently seeded five-input Boolean world",
        "registered_root_count":8,
        "independently_audited_root_count":len(audited),
        "missing_or_failed_root_audits":8-len(audited),
        "all_original_roots_in_ITT":len(entries)==8,
        "two_paired_native_model_arms_per_root":True,
        "source_groups_scored":list(range(24,32)),
        "source_histories_per_model_arm":256,
        "original_source_qualified_roots":list(QUALIFIED),
        "original_source_ineligible_ITT_zero_roots":list(ZERO_ROOTS),
        "primary_root_necessity_effects":effects,
        "primary_macro_mean_necessity_effect":primary,
        "primary_comparison_family_count":1,
        "minimum_meaningful_effect_fraction":.02,
        "primary_exceeds_meaningful_floor":primary>.02,
        "exploratory_two_sided_exact_root_sign_flip_p":p,
        "nonzero_root_effects":nonzero,
        "source_correct_counts":totals,
        "signed_clean_minus_hubs_gap":{
            arm:totals[arm]["clean"]-totals[arm]["hubs"] for arm in totals},
        "source_training_only":True,
        "original_source_native_and_SAE_weights_unchanged":True,
        "new_development_eval_groups_scored":0,
        "reserved_groups_scored":0,
        "newly_recruited_worlds_scored":0,
        "embargoed_worlds_scored":0,
        "old_four_input_confirmation_worlds_scored":0,
        "causal_sham_95_95_tolerance_bound_estimated":False,
        "source_in_sample_is_population_confirmation":False,
        "all_independent_root_audits_pass":len(audited)==8,
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
        "roots":entries,
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"nec03t_eight_root_source_necessity_ITT.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("NEC03T_FROZEN_EIGHT_ROOT_NECESSITY_ITT",
          len(audited),round(primary,9),p)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--lock",type=Path,
                   default=Path(__file__).with_name("NEC03T_SOURCE_LOCK.json"))
    a=p.parse_args()
    aggregate(a.root,a.out,a.lock)

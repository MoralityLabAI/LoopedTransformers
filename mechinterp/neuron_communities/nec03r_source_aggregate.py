"""NEC-03R independent eight-root source-only clean-gap recovery aggregate.

All eight registered source roots and 16 model arms remain in denominator.
Uses only source-training correctness and separately replayed model/SAE
donor geometry. No new development, reserved or confirmation readout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from hashlib import sha256
from itertools import product

ROOTS=tuple(range(8))
SOURCE_ELIGIBLE=(0,1,2,4,6,7)
SCHEMA="NEC-03R/eight-world-source-only-recoverability-ITT/v1"
ARMS=("composed","additive")
MODES=(
    "clean","hubs","native_nonhub","sae_affine_nonhub",
    "sae16_contrib_nonhub","ranked_n","ranked_angle",
    "angle_sham","ranked_random","random_active",
)


def sha_file(p:Path)->str:
    return sha256(p.read_bytes()).hexdigest()


def exact_sign_flip(values):
    nonzero=[float(x) for x in values if abs(float(x))>1e-12]
    if not nonzero:
        return 1.0,0
    total=1<<len(nonzero)
    observed=abs(sum(nonzero))
    extreme=sum(abs(sum(a*b for a,b in zip(signs,nonzero)))>=observed-1e-12
                for signs in product((-1.,1.),repeat=len(nonzero)))
    return extreme/total,len(nonzero)


def aggregate(folder:Path,out:Path,lock:Path):
    lock_sha=sha_file(lock)
    expected={f"nec03r-source-root-{i}" for i in ROOTS}
    paths={p.name for p in folder.iterdir() if p.is_dir() and p.name.startswith("nec03r-source-root-")}
    if paths-expected:
        raise AssertionError("unregistered model source world in NEC03R aggregate")
    roots=[]
    complete=0
    for i in ROOTS:
        path=folder/f"nec03r-source-root-{i}"
        report_path=path/"nec03r_source_recovery_report.json"
        audit_path=path/"independent_nec03r_source_recovery_audit.json"
        if not report_path.exists() or not audit_path.exists():
            roots.append({"world_id":f"dev5-{i:03d}",
                          "world_index":i,"audit_complete":False,
                          "root_ITT_angle_DID":0.,"source_ITT_failure":i not in SOURCE_ELIGIBLE,
                          "reason":"missing_or_failed_source_neural_audit"})
            continue
        report=json.loads(report_path.read_text())
        audit=json.loads(audit_path.read_text())
        if (report["schema"]!="NEC-03R/source-only-sae-clean-gap-recovery/v1"
            or audit["schema"]!="NEC-03R/independent-source-recoverability-replay/v1"
            or report["world_index"]!=i or audit["world_index"]!=i
            or report["world_id"]!=f"dev5-{i:03d}"
            or report["world_id"]!=audit["world_id"]
            or report["world_sha256"]!=audit["world_sha256"]
            or audit["source_report_sha256"]!=sha_file(report_path)
            or report["source_lock_sha256"]!=lock_sha
            or not audit["all_eight_complete_source_groups_per_arm_independently_replayed"]
            or not audit["source_full_SAE_and_native_nonhub_reconstruction_verified"]
            or not audit["all_disjoint_angle_shams_and_hashed_random_controls_recomputed"]
            or not audit["all_recurrent_native_SAE_source_and_control_logits_replayed"]
            or not audit["all_original_training_source_truth_labels_recomputed"]
            or audit["arm_count"]!=2 or audit["source_cases_per_arm"]!=256
            or report["source_training_histories_per_arm"]!=256
            or report["source_groups_scored"]!=list(range(8,16))
            or report["models_or_SAEs_retrained"]!=0
            or any(report[key]!=0 for key in (
                "new_development_eval_contexts_scored","reserved_contexts_scored",
                "embargoed_worlds_scored","original_four_input_confirmation_worlds_scored"))
            or any(audit[key]!=0 for key in (
                "new_development_eval_contexts_scored","reserved_contexts_scored",
                "embargoed_worlds_scored","original_four_input_confirmation_worlds_scored"))
            or report["FSI_01_gate"]!="REDESIGN_OR_PILOT_ONLY"
            or audit["FSI_01_gate"]!="REDESIGN_OR_PILOT_ONLY"
            or set(report["arms"])!=set(ARMS)
            or set(audit["arms"])!=set(ARMS)):
            raise AssertionError(f"source or model causal recovery audit altered for root {i}")
        arms={}
        for arm in ARMS:
            a=report["arms"][arm]
            b=audit["arms"][arm]
            if (b["counts"]!=a["per_intervention_correct_counts"]
                or b["raw_sha256"]!=a["source_root_raw_sha256"]
                or b["native_model_sha256"]!=a["native_sha256"]
                or b["sae_sha256"]!=a["sae_checkpoint_sha256"]
                or b["original_90pct_native_and_SAE_source_eligible"]!=a["source_eligible_under_original_90pct_gate"]
                or a["total_source_training_cases"]!=256
                or a["source_context_groups_scored"]!=list(range(8,16))
                or set(b["counts"])!=set(MODES)):
                raise AssertionError(f"source arm score or checkpoint provenance changed {i}/{arm}")
            arms[arm]={
                "correctness":b["counts"],
                "clean_minus_hubs_correct_gap":b["clean_hubs_gap"],
                "recovered_fraction":b["fraction_if_gap_positive"],
                "verified_source_geometry_medians":b["verified_source_geometry_medians"],
                "source_candidate_present":b["source_candidate_present"],
                "original_source_eligible":b["original_90pct_native_and_SAE_source_eligible"],
            }
        raw_did=(arms["composed"]["correctness"]["ranked_angle"]-
                 arms["composed"]["correctness"]["angle_sham"]-
                 arms["additive"]["correctness"]["ranked_angle"]+
                 arms["additive"]["correctness"]["angle_sham"])/256.
        admitted=i in SOURCE_ELIGIBLE
        expected_did=raw_did if admitted else 0.
        if (abs(raw_did-report["raw_root_angle_sham_DID"])>1e-12
            or abs(expected_did-audit["root_ITT_angle_DID"])>1e-12
            or abs(expected_did-report["primary_ITT_angle_sham_DID"])>1e-12
            or report["original_paired_root_source_eligible"]!=admitted
            or audit["source_eligible_root_in_prior_study"]!=admitted):
            raise AssertionError("eight-root ITT effect altered in source-only aggregate")
        roots.append({
            "world_id":report["world_id"],"world_index":i,
            "world_sha256":report["world_sha256"],
            "audit_complete":True,
            "source_ITT_failure":not admitted,
            "root_ITT_angle_DID":expected_did,
            "raw_source_root_angle_DID":raw_did,
            "report_sha256":sha_file(report_path),
            "independent_audit_sha256":sha_file(audit_path),
            "arms":arms,
        })
        complete+=1
    known=[x for x in roots if x["audit_complete"]]
    totals={
        arm:{mode:sum(x["arms"][arm]["correctness"][mode] for x in known)
             for mode in MODES}
        for arm in ARMS
    }
    gaps={
        arm:totals[arm]["clean"]-totals[arm]["hubs"] for arm in ARMS
    }
    fractions={
        arm:{mode:( (totals[arm][mode]-totals[arm]["hubs"])/gaps[arm]
                    if gaps[arm]>0 else None)
             for mode in MODES if mode not in ("clean","hubs")}
        for arm in ARMS
    }
    values=[x["root_ITT_angle_DID"] for x in roots]
    p,k=exact_sign_flip(values)
    result={
        "schema":SCHEMA,
        "unit":"one independently seeded original five-input development world",
        "root_count_ITT":8,
        "independent_root_audits_complete":complete,
        "failed_or_missing_root_audits":8-complete,
        "all_eight_arms_pairs_retained":len(roots)==8,
        "arms_per_complete_root":2,
        "source_training_histories_per_model_arm":256,
        "source_context_groups":list(range(8,16)),
        "measured_modes":list(MODES),
        "source_eligible_ITT_worlds":list(SOURCE_ELIGIBLE),
        "source_ineligible_ITT_zero_worlds":[3,5],
        "primary_ITT_root_angle_source_vs_sham_effects":values,
        "primary_ITT_macro_mean_angle_source_vs_sham":sum(values)/8,
        "minimum_scientific_meaningful_effect":.02,
        "primary_effect_above_registered_scientific_floor":sum(values)/8>.02,
        "exploratory_exact_two_sided_root_signflip_p":p,
        "root_signflip_nonzero_effects":k,
        "correct_counts_across_original_training_source_roots":totals,
        "signed_clean_minus_hubs_correct_gap":gaps,
        "fraction_of_signed_clean_hubs_gap_recovered":fractions,
        "primary_angle_matched_control_scientifically_resolved":False,
        "independent_sham_95_95_tolerance_calibrated":False,
        "new_development_eval_contexts_scored":0,
        "reserved_contexts_scored":0,
        "embargoed_worlds_scored":0,
        "old_four_input_confirmation_worlds_scored":0,
        "all_source_audits_passed":complete==8,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
        "roots":roots,
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"nec03r_eight_world_source_recovery_aggregate.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("NEC03R_EIGHT_ROOT_SOURCE_RECOVERABILITY_ITT",complete,result["primary_ITT_macro_mean_angle_source_vs_sham"])
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--lock",type=Path,default=Path(__file__).with_name("NEC03R_SOURCE_LOCK.json"))
    a=p.parse_args()
    aggregate(a.root,a.out,a.lock)

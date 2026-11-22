"""NEC-03V independent eight-world, 16-arm source group necessity aggregate.

Pure stdlib; fixed original roots, paired 8-world primary ITT,
original paired source-admission zeros 3,5, missed jobs tracked.
Requires independent neural/SAE physical sham audit of every root.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path

SCHEMA="NEC-03V/eight-world-coherent-group-source-only-ITT/v1"
ROOTS=tuple(range(8))
ELIGIBLE=(0,1,2,4,6,7)
ZERO=(3,5)
ARMS=("composed","additive")
MODES=(
    "clean","hubs","native_nonhub","full_affine",
    "contrib16_only","coherent4_source","coherent4_sham",
    "coherent6_source","coherent6_sham","hash4_source","hash4_sham",
)


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def signflip(values):
    nonzero=[float(x) for x in values if abs(float(x))>1e-12]
    if not nonzero:return 1.,0
    threshold=abs(sum(nonzero))
    p=sum(abs(sum(v*s for v,s in zip(nonzero,signs)))>=threshold-1e-12
          for signs in product((-1.,1.),repeat=len(nonzero)))
    return p/(1<<len(nonzero)),len(nonzero)


def aggregate(root,out,lock):
    locksha=sha(lock)
    names={f"nec03v-source-root-{i}" for i in ROOTS}
    if any(p.name not in names for p in Path(root).glob("nec03v-source-root-*") if p.is_dir()):
        raise AssertionError("unregistered world in source coherent-group aggregate")
    records=[]
    complete=0
    for i in ROOTS:
        path=Path(root)/f"nec03v-source-root-{i}"
        rp=path/"nec03v_source_group_root_report.json"
        ap=path/"independent_nec03v_source_group_audit.json"
        if not rp.is_file() or not ap.is_file():
            records.append({"world_index":i,"world_id":f"dev5-{i:03d}",
                "independent_audit_complete":False,
                "original_paired_source_ITT_failure":i in ZERO,
                "primary_root_effect":0.,
                "reason":"missing_or_failed_independent_model_source_audit"})
            continue
        result=json.loads(rp.read_text())
        aud=json.loads(ap.read_text())
        if (result["schema"]!="NEC-03V/source-only-coherent-group-necessity/v1"
            or aud["schema"]!="NEC-03V/independent-coherent-group-physical-null-replay/v1"
            or result["world_index"]!=i or aud["world_index"]!=i
            or result["world_id"]!=f"dev5-{i:03d}"
            or result["world_id"]!=aud["world_id"]
            or result["world_sha256"]!=aud["world_sha256"]
            or result["source_lock_sha256"]!=locksha
            or aud["source_lock_sha256"]!=locksha
            or aud["original_root_report_sha256"]!=sha(rp)
            or result["source_groups_scored"]!=list(range(32,40))
            or result["source_cases_per_model_arm"]!=256
            or aud["source_training_cases_per_arm"]!=256
            or result["paired_qualified_in_original_source_roster"]!=(i in ELIGIBLE)
            or result["original_ITT_source_failure"]!=(i in ZERO)
            or aud["original_source_eligible_world"]!=(i in ELIGIBLE)
            or result["primary_comparison_family_size"]!=1
            or result["minimum_meaningful_primary"]!=.02
            or not all(aud.get(key) for key in (
                "all_original_source_inputs_and_Boolean_labels_independently_replayed",
                "frozen_model_and_SAE_sha256_verified",
                "all_coherent_group_source_ids_and_disjoint_shams_reconstructed",
                "all_original_sham_corners_and_physical_32d_output_doses_recomputed",
                "all_eleven_full_recurrent_neural_outcomes_independently_replayed",
                "all_source_case_dropout_and_hash_controls_verified",
                "both_original_teacher_arms_replayed"))
            or any(result[key]!=0 for key in (
                "new_native_or_SAE_training_updates","new_development_eval_groups_scored",
                "reserved_groups_scored","new_recruited_worlds_scored",
                "five_input_embargoed_worlds_scored",
                "old_four_input_confirmation_worlds_scored"))
            or any(aud[key]!=0 for key in (
                "development_eval_groups_scored","reserved_groups_scored",
                "new_recruited_worlds_scored","five_input_embargoed_worlds_scored",
                "old_four_input_confirmation_worlds_scored"))
            or result["FSI_01_gate"]!="REDESIGN_OR_PILOT_ONLY"
            or aud["FSI_01_gate"]!="REDESIGN_OR_PILOT_ONLY"
            or set(result["arms"])!=set(ARMS)
            or set(aud["arms"])!=set(ARMS)):
            raise AssertionError(f"source physical group/audit ancestry broken world {i}")
        arms={}
        for name in ARMS:
            row=result["arms"][name]
            check=aud["arms"][name]
            if (row["correct_counts"]!=check["correct_counts"]
                or row["matched_group_valid_cases"]!=check["valid_physical_null_cases"]
                or row["hash_valid_cases"]!=check["hash_valid_cases"]
                or row["four_source_ranked_necessity_correct_cases"]!=check["four_necessity_correct_case_difference"]
                or row["six_source_ranked_necessity_correct_cases"]!=check["six_necessity_correct_case_difference"]
                or row["source_model_sha256"]!=check["model_sha256"]
                or row["source_sae_checkpoint_sha256"]!=check["SAE_sha256"]
                or row["source_sae_rank_sha256"]!=check["source_rank_sha256"]
                or row["raw_sha256"]!=check["raw_sha256"]
                or row["original_arm_source_eligible"]!=check["original_source_eligible"]
                or row["original_source_candidate_present"]!=check["source_candidate_present"]
                or row["source_groups_scored"]!=list(range(32,40))
                or row["cases_per_arm"]!=256
                or set(row["correct_counts"])!=set(MODES)
                or set(row["matched_group_valid_cases"])!={"four","six"}
                or any(not 0<=n<=256 for n in row["correct_counts"].values())):
                raise AssertionError(f"NEC03V source model {i} teacher {name} score/checkpoint differs")
            if not math.isclose(
                row["within_arm_full_affine_median_margin_scale"],
                check["full_affine_baseline_margin_scale"],abs_tol=1e-6,rel_tol=1e-6):
                raise AssertionError("source margin scale changed independently")
            if not math.isclose(
                row["secondary_normalized_source_vs_sham_margin_mean"],
                check["secondary_normalized_sham_minus_source_margin_mean"],
                abs_tol=1e-6,rel_tol=1e-6):
                raise AssertionError("secondary normalized margin changed")
            arms[name]={
                "correct_counts":check["correct_counts"],
                "four_source_necessity_correct_cases":check["four_necessity_correct_case_difference"],
                "six_source_necessity_correct_cases":check["six_necessity_correct_case_difference"],
                "valid_physical_null_cases":check["valid_physical_null_cases"],
                "hash_valid_cases":check["hash_valid_cases"],
                "within_arm_full_affine_median_margin_scale":
                    check["full_affine_baseline_margin_scale"],
                "secondary_normalized_margin_effect":
                    check["secondary_normalized_sham_minus_source_margin_mean"],
                "source_eligible":check["original_source_eligible"],
            }
        did=(arms["composed"]["four_source_necessity_correct_cases"]-
             arms["additive"]["four_source_necessity_correct_cases"])/256
        if (abs(did-result["raw_root_primary_coherent4_necessity_DID"])>1e-12
            or abs((did if i in ELIGIBLE else 0.)-
                   result["registered_ITT_primary_root_DID"])>1e-12
            or abs((did if i in ELIGIBLE else 0.)-aud["primary_root_ITT_DID"])>1e-12):
            raise AssertionError("registered original-root coherent-group necessity effect changed")
        records.append({
            "world_index":i,"world_id":result["world_id"],
            "world_sha256":result["world_sha256"],
            "independent_audit_complete":True,
            "original_paired_source_ITT_failure":i in ZERO,
            "primary_root_effect":did if i in ELIGIBLE else 0.,
            "raw_unselected_primary_effect":did,
            "root_report_sha256":sha(rp),
            "independent_audit_sha256":sha(ap),
            "arms":arms,
        })
        complete+=1
    known=[v for v in records if v["independent_audit_complete"]]
    totals={arm:{
        mode:sum(v["arms"][arm]["correct_counts"][mode] for v in known)
        for mode in MODES}
        for arm in ARMS}
    gap={arm:totals[arm]["clean"]-totals[arm]["hubs"] for arm in ARMS}
    recovered={arm:{
        name:((correct-totals[arm]["hubs"])/gap[arm] if gap[arm]>0 else None)
        for name,correct in totals[arm].items() if name not in ("clean","hubs")}
        for arm in ARMS}
    source_root_values=[r["primary_root_effect"] for r in records]
    primary=sum(source_root_values)/8.
    p,n=signflip(source_root_values)
    eligible_known=[r for r in known if r["world_index"] in ELIGIBLE]
    qualified_positive=sum(r["primary_root_effect"]>0 for r in eligible_known)
    all_cover=bool(len(eligible_known)==6 and all(
        r["arms"][arm]["valid_physical_null_cases"]["four"]>=205
        for r in eligible_known for arm in ARMS))
    resolution=(complete==8 and primary>.02 and qualified_positive>=5 and all_cover)
    result={
        "schema":SCHEMA,
        "original_world_root_ITT_count":8,
        "independently_audited_root_count":complete,
        "missing_or_failed_root_audits":8-complete,
        "original_paired_source_eligible_roots":list(ELIGIBLE),
        "original_source_ineligible_zero_roots":list(ZERO),
        "native_teacher_arms_per_world":2,
        "source_training_histories_per_arm":256,
        "original_source_groups":list(range(32,40)),
        "original_native_or_SAE_optimizer_updates":0,
        "primary_coherent4_source_necessity_root_effects":source_root_values,
        "primary_eight_root_macro":primary,
        "primary_meaningful_floor":.02,
        "primary_exceeds_two_percentage_point_floor":primary>.02,
        "eligible_root_positive_count":qualified_positive,
        "eligible_root_positive_gate_at_least_five":qualified_positive>=5,
        "all_original_qualified_teacher_arms_valid_physical_null_coverage_80pct":all_cover,
        "source_only_mechanism_resolution_gate_passed":resolution,
        "exploratory_two_sided_exact_root_signflip_p":p,
        "nonzero_root_effect_count":n,
        "independently_verified_source_correct_totals":totals,
        "signed_clean_minus_hubs_correctness_gap":gap,
        "signed_source_gap_recovered_fraction":recovered,
        "secondary_normalized_margin_unit":"within teacher arm divided by max(1,median absolute full-affine signed true-label margin); exploratory only",
        "original_eight_root_population_confirmation":False,
        "all_eight_independent_root_neural_audits_complete":complete==8,
        "new_development_eval_groups_scored":0,
        "reserved_groups_scored":0,
        "new_recruited_worlds_scored":0,
        "five_input_embargoed_worlds_scored":0,
        "old_four_input_confirmation_worlds_scored":0,
        "independent_causal_sham_95_95_calibrated":False,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
        "roots":records,
    }
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/"nec03v_eight_root_source_group_ITT.json").write_text(
        json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("NEC03V_EIGHT_ROOT_SOURCE_COHERENT_GROUP_ITT",
          complete,primary,p,"resolution_gate",resolution)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--lock",type=Path,
        default=Path(__file__).with_name("NEC03V_SOURCE_LOCK.json"))
    a=p.parse_args()
    aggregate(a.root,a.out,a.lock)

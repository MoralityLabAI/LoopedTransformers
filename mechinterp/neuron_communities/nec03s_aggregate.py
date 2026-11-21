"""NEC-03S eight-root source-only offset x budget factorial ITT aggregate.

Pure standard library. Every original root stays in the attempted
denominator. Original paired-source-ineligible roots 3 and 5 are fixed
primary zeros, and missing independent audits are also recorded as
failures rather than removed from the root denominator.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path

ROOTS=tuple(range(8))
ELIGIBLE=(0,1,2,4,6,7)
ITT_ZERO=(3,5)
ARMS=("composed","additive")
BUDGETS=("orig","wide")
SCHEMA="NEC-03S/eight-root-source-offset-active-budget-ITT/v1"


def hash_file(path:Path)->str:
    return sha256(path.read_bytes()).hexdigest()


def root_sign_flip(values):
    present=[float(v) for v in values if abs(float(v))>1e-12]
    if not present:
        return 1.0,0
    observed=abs(sum(present))
    combinations=1<<len(present)
    count=sum(
        abs(sum(x*y for x,y in zip(signs,present)))>=observed-1e-12
        for signs in product((-1.,1.),repeat=len(present)))
    return count/combinations,len(present)


def aggregate(folder:Path,out:Path,lock:Path):
    pinned_lock=hash_file(lock)
    expected={f"nec03s-source-root-{i}" for i in ROOTS}
    unexpected={p.name for p in folder.glob("nec03s-source-root-*")
                if p.is_dir() and p.name not in expected}
    if unexpected:
        raise AssertionError("unregistered or heldout world result in NEC03S")
    roots=[]
    complete=0
    for i in ROOTS:
        directory=folder/f"nec03s-source-root-{i}"
        rpath=directory/"nec03s_source_factorial_report.json"
        apath=directory/"independent_nec03s_source_factorial_audit.json"
        if not rpath.is_file() or not apath.is_file():
            roots.append({
                "world_index":i,"world_id":f"dev5-{i:03d}",
                "status":"missing_or_failed_independent_root_audit",
                "source_training_contexts_scored":None,
                "registered_ITT_zero":i in ITT_ZERO,
                "primary_root_interaction":0.,
            })
            continue
        raw=json.loads(rpath.read_text())
        audited=json.loads(apath.read_text())
        if (raw["schema"]!="NEC-03S/source-only-offset-budget-four-way-physical-dose/v1"
            or audited["schema"]!="NEC-03S/independent-source-offset-budget-audit/v1"
            or raw["world_id"]!=f"dev5-{i:03d}"
            or raw["world_id"]!=audited["world_id"]
            or raw["world_index"]!=i or audited["world_index"]!=i
            or raw["world_sha256"]!=audited["world_sha256"]
            or raw["source_lock_sha256"]!=pinned_lock
            or audited["source_lock_sha256"]!=pinned_lock
            or audited["original_report_sha256"]!=hash_file(rpath)
            or raw["original_paired_source_eligible"]!=(i in ELIGIBLE)
            or raw["preregistered_source_ITT_failure"]!=(i in ITT_ZERO)
            or raw["source_groups_scored"]!=list(range(16,24))
            or raw["training_source_case_count_per_arm"]!=256
            or raw["primary_familywise_comparisons"]!=1
            or raw["minimum_meaningful_absolute_effect"]!=.02
            or not audited["all_original_source_groups_and_truth_labels_independently_replayed"]
            or not audited["original_source_native_and_sae_checkpoints_immutable"]
            or not audited["all_disjoint_minimax_angle_and_hash_random_shams_reenumerated"]
            or not audited["all_four_offset_control_doses_identically_anchored_and_nonamplifying"]
            or not audited["all_26_full_recurrent_logit_modes_independently_replayed"]
            or not audited["all_16_original_model_arms_in_fixed_eight_root_denominator"]
            or audited["source_cases_per_arm"]!=256
            or raw["native_and_sae_training_updates"]!=0
            or any(raw[k]!=0 or audited[k]!=0 for k in (
                "development_eval_groups_scored","reserved_groups_scored",
                "new_source_recruited_worlds_scored","embargoed_worlds_scored",
                "original_four_input_confirmation_worlds_scored"))
            or raw["scientific_gate"]!="REDESIGN_OR_PILOT_ONLY"
            or audited["scientific_gate"]!="REDESIGN_OR_PILOT_ONLY"
            or set(raw["arms"])!=set(ARMS)
            or set(audited["arms"])!=set(ARMS)):
            raise AssertionError(f"registered source-only cohort or independent proof broke at root {i}")
        arms={}
        for arm in ARMS:
            row=raw["arms"][arm]
            check=audited["arms"][arm]
            if (row["correctness"]!=check["correctness"]
                or row["available_common_dose_input_cases"]!=check["available_common_dose_input_cases"]
                or row["offset_interaction_correct_difference"]!=check["offset_interaction_correct_difference"]
                or row["raw_sha256"]!=check["raw_sha256"]
                or row["model_weights_sha256"]!=check["model_weights_sha256"]
                or row["sae_checkpoint_sha256"]!=check["sae_sha256"]
                or row["source_report_sha256"]!=check["source_report_sha256"]
                or row["original_arm_source_eligible"]!=check["original_source_eligible"]
                or row["original_source_candidate_present"]!=check["original_source_candidate_present"]
                or set(row["offset_interaction_correct_difference"])!=set(BUDGETS)
                or any(not (0<=v<=256) for v in row["correctness"].values())):
                raise AssertionError(f"independent arm source label or model checkpoint mismatch {i}/{arm}")
            arms[arm]={
                "correctness":check["correctness"],
                "valid_common_dose_inputs":check["available_common_dose_input_cases"],
                "offset_interaction_correct_difference":check["offset_interaction_correct_difference"],
                "original_source_eligible":check["original_source_eligible"],
                "original_source_candidate_present":check["original_source_candidate_present"],
            }
        primary_n=(arms["composed"]["offset_interaction_correct_difference"]["orig"]-
                   arms["additive"]["offset_interaction_correct_difference"]["orig"])/256.
        secondary_w=(arms["composed"]["offset_interaction_correct_difference"]["wide"]-
                     arms["additive"]["offset_interaction_correct_difference"]["wide"])/256.
        if (not math.isclose(primary_n,raw["raw_primary_original_budget_interaction_DID"],abs_tol=1e-12)
            or not math.isclose(primary_n,audited["raw_original_budget_interaction"],abs_tol=1e-12)
            or not math.isclose(primary_n if i in ELIGIBLE else 0.,
                raw["primary_ITT_root_interaction_DID"],abs_tol=1e-12)
            or not math.isclose(primary_n if i in ELIGIBLE else 0.,
                audited["primary_original_budget_ITT_root_interaction"],abs_tol=1e-12)):
            raise AssertionError("source factorial predeclared primary interaction changed")
        roots.append({
            "world_index":i,"world_id":raw["world_id"],
            "world_sha256":raw["world_sha256"],
            "status":"complete_independent_source_audit",
            "source_training_contexts_scored":list(range(16,24)),
            "registered_ITT_zero":i in ITT_ZERO,
            "primary_root_interaction":primary_n if i in ELIGIBLE else 0.,
            "unselected_raw_primary_interaction":primary_n,
            "wide_budget_secondary_root_interaction":secondary_w if i in ELIGIBLE else 0.,
            "raw_report_sha256":hash_file(rpath),
            "independent_audit_sha256":hash_file(apath),
            "arms":arms,
        })
        complete+=1
    measured=[v for v in roots if v["status"]=="complete_independent_source_audit"]
    totals={
        arm:{
            mode:sum(v["arms"][arm]["correctness"][mode] for v in measured)
            for mode in (
                "clean","hubs","native_nonhub","offset_only","full16_0","full16_1",
                "natural_orig_0","natural_orig_1","natural_wide_0","natural_wide_1",
                "orig_src0","orig_sham0","orig_src1","orig_sham1",
                "wide_src0","wide_sham0","wide_src1","wide_sham1",
                "orig_random_src0","orig_random_sham0",
                "orig_random_src1","orig_random_sham1",
                "wide_random_src0","wide_random_sham0",
                "wide_random_src1","wide_random_sham1"
            )
        } for arm in ARMS
    }
    gap={arm:totals[arm]["clean"]-totals[arm]["hubs"] for arm in ARMS}
    recovered={
        arm:{mode:((score-totals[arm]["hubs"])/gap[arm] if gap[arm]>0 else None)
             for mode,score in totals[arm].items() if mode not in ("clean","hubs")}
        for arm in ARMS
    }
    values=[r["primary_root_interaction"] for r in roots]
    p,k=root_sign_flip(values)
    avg=sum(values)/8
    result={
        "schema":SCHEMA,
        "registered_root_count":8,
        "independent_audited_root_count":complete,
        "failed_or_missing_source_audits":8-complete,
        "two_native_model_arms_per_world":True,
        "all_sixteen_arms_represented":len(measured)*2,
        "source_training_histories_per_arm":256,
        "frozen_original_source_group_indices":list(range(16,24)),
        "primary_original_budget_offset_interaction_root_effects":values,
        "primary_eight_root_macro_mean":avg,
        "primary_meaningful_effect_floor":.02,
        "primary_passes_meaningful_floor":avg>.02,
        "exploratory_exact_two_sided_root_sign_flip_p":p,
        "sign_flip_nonzero_root_effect_count":k,
        "prequalified_source_eligible_roots":list(ELIGIBLE),
        "registered_source_ineligible_ITT_zero_roots":list(ITT_ZERO),
        "secondary_wide_budget_root_effects":[x.get("wide_budget_secondary_root_interaction",0.) for x in roots],
        "correct_counts_on_original_training_source":totals,
        "clean_minus_hubs_correctness_gaps":gap,
        "descriptive_signed_gap_recovery_fractions":recovered,
        "new_reserved_contexts_scored":0,
        "new_development_eval_contexts_scored":0,
        "new_recruited_worlds_scored":0,
        "new_embargoed_worlds_scored":0,
        "old_four_input_confirmation_worlds_scored":0,
        "causal_sham_59_root_tolerance_bound_estimated":False,
        "source_only_pilot_is_confirmation":False,
        "all_eight_independent_audits_passed":complete==8,
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
        "roots":roots,
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"nec03s_eight_root_source_offset_budget_ITT.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("NEC03S_SOURCE_FACTORIAL_EIGHT_ROOT_ITT",complete,round(avg,9),p)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--lock",type=Path,default=Path(__file__).with_name("NEC03S_SOURCE_LOCK.json"))
    a=p.parse_args()
    aggregate(a.root,a.out,a.lock)

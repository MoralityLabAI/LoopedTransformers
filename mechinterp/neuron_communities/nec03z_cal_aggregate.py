"""NEC-03Z-CAL: eight-root pure-stdlib calibration and prospective test seal.

Registers 5 doses before group48..55 scoring, then applies strict
physical coverage, sham inertness, dose-variation and baseline headroom
gates across six ORIGINAL independent-source-eligible world roots.
Group56..63 stays sealed regardless of calibration outcome; an
independent frozen source-only evaluation protocol is required later.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

SCHEMA="NEC-03Z/eight-root-original-source-sham-dose-calibration/v1"
ROOTS=tuple(range(8))
ELIGIBLE=(0,1,2,4,6,7)
ITT_ZERO=(3,5)
TAGS=("1000","0500","0250","0125","00625")
DOSES=(1.,.5,.25,.125,.0625)
ARMS=("composed","additive")
MODES=("clean","hubs","native_nonhub","full_affine")+tuple(
    f"{side}_d{tag}" for tag in TAGS for side in ("source","sham")
)


def sha(p:Path):
    return sha256(p.read_bytes()).hexdigest()


def aggregate(root:Path,out:Path,lock:Path):
    config=json.loads(lock.read_text())
    lock_hash=sha(lock)
    if config["schema"]!="NEC-03Z/frozen-original-source-sham-inertness-CAL/v1":
        raise AssertionError("NEC03Z source calibration lock version altered")
    expected={f"nec03z-cal-root-{i}" for i in ROOTS}
    if any(p.name not in expected for p in root.glob("nec03z-cal-root-*") if p.is_dir()):
        raise AssertionError("unregistered or protected source world artifact found")
    roots=[]
    completed=0
    for i in ROOTS:
        path=root/f"nec03z-cal-root-{i}"
        rp=path/"nec03z_cal_source_root_report.json"
        ap=path/"independent_nec03z_cal_source_audit.json"
        if not rp.is_file() or not ap.is_file():
            roots.append({
                "world_index":i,"world_id":f"dev5-{i:03d}",
                "audit_complete":False,"source_ITT_zero":i in ITT_ZERO,
                "reason":"missing_or_failed_independent_calibration_model_audit"})
            continue
        report=json.loads(rp.read_text())
        audit=json.loads(ap.read_text())
        if (
            report["schema"]!="NEC-03Z/source-only-inertness-dose-CAL/v1"
            or audit["schema"]!="NEC-03Z/independent-archive-model-dose-calibration-replay/v1"
            or report["world_index"]!=i or audit["world_index"]!=i
            or report["world_id"]!=f"dev5-{i:03d}"
            or report["world_id"]!=audit["world_id"]
            or report["world_sha256"]!=audit["world_sha256"]
            or report["source_lock_sha256"]!=lock_hash
            or audit["source_lock_sha256"]!=lock_hash
            or audit["original_root_report_sha256"]!=sha(rp)
            or report["calibration_source_groups_scored"]!=list(range(48,56))
            or report["sealed_analysis_source_groups_scored"]!=[]
            or report["source_cases_per_native_teacher"]!=256
            or audit["source_cases_per_arm"]!=256
            or report["original_paired_source_eligible"]!=(i in ELIGIBLE)
            or audit["original_qualified_source_ITT"]!=(i in ELIGIBLE)
            or report["original_ITT_zero"]!=(i in ITT_ZERO)
            or report["new_native_SAE_training_updates"]!=0
            or report["scientific_gate"]!="REDESIGN_OR_PILOT_ONLY"
            or audit["scientific_gate"]!="REDESIGN_OR_PILOT_ONLY"
            or any(report[k]!=0 or audit[k]!=0 for k in (
                "development_eval_scored","reserved_scored",
                "recruited_worlds_scored","embargoed_worlds_scored",
                "old_four_input_confirmation_scored"))
            or audit["sealed_source_analysis_groups_scored"]!=0
            or not all(audit.get(k) for k in (
                "all_original_source_calibration_truths_independently_recomputed",
                "all_five_dose_fractions_and_original_solver_routes_independently_reconstructed",
                "all_original_native_SAE_source_ancestry_verified",
                "all_fourteen_complete_recurrent_modes_independently_replayed",
                "whole_trajectory_invalid_sham_source_inputs_return_to_full_affine",
                "all_original_source_correctness_and_sham_damage_counts_verified"))
            or set(report["arms"])!=set(ARMS)
            or set(audit["arms"])!=set(ARMS)
        ):
            raise AssertionError(f"NEC03Z original source calibration provenance changed root {i}")
        arms={}
        for arm in ARMS:
            s=report["arms"][arm]
            a=audit["arms"][arm]
            if (
                s["correct"]!=a["correct"]
                or s["dose_calibration"]!=a["dose_calibration"]
                or s["model_sha256"]!=a["model_sha256"]
                or s["sae_sha256"]!=a["sae_sha256"]
                or s["source_rank_sha256"]!=a["source_rank_sha256"]
                or s["raw_sha256"]!=a["raw_sha256"]
                or s["original_arm_source_eligible"]!=a["original_source_eligible"]
                or set(s["correct"])!=set(MODES)
                or set(s["dose_calibration"])!=set(TAGS)
                or s["source_groups"]!=list(range(48,56))
                or s["source_cases"]!=256
            ):
                raise AssertionError(f"NEC03Z independent dose audit mismatch {i}/{arm}")
            for tag,d in zip(TAGS,DOSES):
                v=s["dose_calibration"][tag]
                if (
                    v["dose_fraction"]!=d
                    or not 0<=v["available_original_source_cases"]<=256
                    or not 0<=v["baseline_correct_matched_cases"]<=v["available_original_source_cases"]
                    or v["sham_correct_to_wrong"]>v["baseline_correct_matched_cases"]
                    or v["sham_wrong_to_correct"]<0
                    or v["source_correct_to_wrong"]<0
                ):
                    raise AssertionError("dose candidate original source validity exceeded denominator")
            arms[arm]={
                "correct":s["correct"],
                "full_affine_baseline_accuracy":s["correct"]["full_affine"]/256,
                "dose_calibration":s["dose_calibration"],
                "original_source_eligible":s["original_arm_source_eligible"],
                "source_candidate_present":s["original_source_candidate_present"],
                "native_model_sha256":s["model_sha256"],
                "source_SAE_sha256":s["sae_sha256"],
                "original_source_rank_sha256":s["source_rank_sha256"],
                "raw_jsonl_sha256":s["raw_sha256"],
            }
        roots.append({
            "world_index":i,"world_id":report["world_id"],
            "world_sha256":report["world_sha256"],
            "audit_complete":True,"source_ITT_zero":i in ITT_ZERO,
            "report_sha256":sha(rp),"independent_audit_sha256":sha(ap),
            "arms":arms,
        })
        completed+=1

    complete=completed==8
    headroom={
        arm:sum(
            .60<=r["arms"][arm]["full_affine_baseline_accuracy"]<=.95
            for r in roots if r["audit_complete"] and r["world_index"] in ELIGIBLE
        ) for arm in ARMS
    }
    headroom_pass=complete and all(headroom[a]>=2 for a in ARMS)
    candidates=[]
    for tag,dose in zip(TAGS,DOSES):
        per_arm=[]
        for r in roots:
            if not r["audit_complete"] or r["world_index"] not in ELIGIBLE:
                continue
            for arm in ARMS:
                row=r["arms"][arm]["dose_calibration"][tag]
                fraction=row["sham_inertness_damage_fraction"]
                reasons=[]
                if row["available_original_source_cases"]<205:
                    reasons.append("matched_controls_under_205")
                if fraction is None or fraction>.02:
                    reasons.append("sham_correct_to_wrong_exceeds_2pct")
                relative=row["median_min_site_relative_dose"]
                if relative is None or relative<.05:
                    reasons.append("physical_variation_floor_under_005")
                per_arm.append({
                    "world_index":r["world_index"],"arm":arm,
                    "available_original_source_cases":row["available_original_source_cases"],
                    "baseline_correct_matched_cases":row["baseline_correct_matched_cases"],
                    "sham_correct_to_wrong":row["sham_correct_to_wrong"],
                    "sham_wrong_to_correct":row["sham_wrong_to_correct"],
                    "sham_damage_rate_given_correct_baseline":fraction,
                    "median_relative_removal_dose":relative,
                    "arm_gates_passed":not reasons,"failure_reasons":reasons,
                })
        all_arm_gates=complete and len(per_arm)==12 and all(
            row["arm_gates_passed"] for row in per_arm)
        allowed=all_arm_gates and headroom_pass
        candidates.append({
            "dose_tag":tag,"dose_fraction":dose,
            "all_six_qualified_root_both_arm_gates_pass":all_arm_gates,
            "headroom_gate_passed":headroom_pass,
            "calibration_candidate_passes_all_gates":allowed,
            "per_qualified_world_arm":per_arm,
        })
    qualifying=[row for row in candidates
                if row["calibration_candidate_passes_all_gates"]]
    picked=qualifying[0] if qualifying else None
    decision=(
        "CALIBRATED_SOURCE_ONLY_AWAIT_SEPARATE_FROZEN_EVALUATION_PROTOCOL"
        if picked else "STOP_SEAL_ANALYSIS_SOURCE_GROUPS_56_TO_63"
    )
    result={
        "schema":SCHEMA,
        "calibration_worlds":list(ROOTS),
        "original_paired_source_eligible_worlds":list(ELIGIBLE),
        "old_source_ITT_zeros":list(ITT_ZERO),
        "original_calibration_source_groups":list(range(48,56)),
        "sealed_next_analysis_source_groups":list(range(56,64)),
        "source_histories_per_arm":256,
        "independently_audited_original_source_roots":completed,
        "all_original_root_audits_completed":complete,
        "all_source_calibration_native_SAE_weights_unchanged":True,
        "source_doses_in_frozen_order":list(DOSES),
        "headroom_60_to_95pct_eligible_root_count_by_teacher":headroom,
        "headroom_minimum_roots_per_teacher":2,
        "baseline_headroom_gate_passed":headroom_pass,
        "maximum_sham_damage_rate_given_matched_baseline_correct":.02,
        "minimum_matched_cases_per_qualified_root_teacher":205,
        "minimum_median_relative_physical_output_dose":.05,
        "original_selected_continuous_sham_identity_frozen":True,
        "does_not_choose_dose_based_on_source_sham_necessity_effect":True,
        "candidate_dose_calibrations":candidates,
        "selected_dose_fraction":picked["dose_fraction"] if picked else None,
        "selected_dose_tag":picked["dose_tag"] if picked else None,
        "calibration_decision":decision,
        "source_analysis_test_groups_scored":0,
        "new_development_eval_groups_scored":0,
        "reserved_groups_scored":0,
        "new_recruited_worlds_scored":0,
        "five_input_embargoed_worlds_scored":0,
        "old_four_input_confirmation_worlds_scored":0,
        "new_model_SAE_optimizer_updates":0,
        "new_population_causal_sham_roots_scored":0,
        "independent_95_95_sham_tolerance_calibrated":False,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
        "roots":roots,
        "calibration_source_lock_sha256":lock_hash,
    }
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    (out/"nec03z_cal_original_source_gate.json").write_text(
        json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    with (out/"nec03z_cal_root_arm_dose_table.csv").open("w",newline="") as handle:
        writer=csv.writer(handle)
        writer.writerow([
            "world","arm","dose","full_affine_accuracy","matched",
            "baseline_correct_matched","sham_correct_to_wrong","sham_wrong_to_correct",
            "source_correct_to_wrong","sham_damage_rate","median_min_site_relative_dose",
        ])
        for r in roots:
            if not r["audit_complete"]:
                continue
            for arm in ARMS:
                for tag in TAGS:
                    d=r["arms"][arm]["dose_calibration"][tag]
                    writer.writerow([
                        r["world_index"],arm,d["dose_fraction"],
                        r["arms"][arm]["full_affine_baseline_accuracy"],
                        d["available_original_source_cases"],
                        d["baseline_correct_matched_cases"],d["sham_correct_to_wrong"],
                        d["sham_wrong_to_correct"],d["source_correct_to_wrong"],
                        d["sham_inertness_damage_fraction"],d["median_min_site_relative_dose"],
                    ])
    print("NEC03Z_FROZEN_SOURCE_CALIBRATION_GATE",
          completed,decision,"headroom",headroom,
          "candidate_pass",[(v["dose_fraction"],v["calibration_candidate_passes_all_gates"])
                            for v in candidates])
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--lock",type=Path,
                   default=Path(__file__).with_name("NEC03Z_CAL_LOCK.json"))
    args=p.parse_args()
    aggregate(args.root,args.out,args.lock)

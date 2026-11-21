"""NEC-03Q eight-root development-only causal ITT report, pure stdlib.

Six original source-admitted roots are independently replayed; roots 3
and 5 are pre-registered source-ineligible and contribute exactly zero.
Exact root sign-flip p is exploratory and conditional on symmetry; FSI-01
never confirms an effect from this selected eight-world development set.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path

ROOTS=tuple(range(8))
ADMITTED=(0,1,2,4,6,7)
ZERO_ROOTS=(3,5)
SCHEMA="NEC-03Q/eight-root-development-DID-ITT-aggregate/v1"


def sha_file(p:Path):
    return sha256(p.read_bytes()).hexdigest()


def exact_two_sided_root_sign_flip(values):
    """Exact conditional null over source-selected independent root signs."""
    active=[float(z) for z in values if abs(float(z))>1e-12]
    if not active:
        return 1.0,0
    obs=abs(sum(active))
    total=1<<len(active)
    extreme=sum(
        abs(sum(sign*z for sign,z in zip(s,active)))>=obs-1e-12
        for s in product((-1.,1.),repeat=len(active)))
    return extreme/total,len(active)


def aggregate(folder:Path,out:Path,execution_lock:Path):
    lock_sha=sha_file(execution_lock)
    records=[]
    for i in ROOTS:
        if i in ZERO_ROOTS:
            records.append({
                "world_id":f"dev5-{i:03d}","world_index":i,
                "source_eligible":False,
                "source_ITT_failure_reason":(
                    "additive source candidate absent" if i==3
                    else "composed native source competence below 90%"),
                "root_ITT_effect":0.0,
                "audit_status":"pre_frozen_ITT_zero",
                "source_and_sham_development_contexts_scored":0,
            })
            continue
        path=folder/f"nec03q-causal-root-{i}"
        source=path/"nec03q_development_causal_report.json"
        audit_path=path/"independent_nec03q_causal_audit.json"
        if not source.exists() or not audit_path.exists():
            records.append({
                "world_id":f"dev5-{i:03d}","world_index":i,
                "source_eligible":True,"root_ITT_effect":0.0,
                "audit_status":"missing_or_failed_root_audit_ITT_zero",
                "source_and_sham_development_contexts_scored":None,
            })
            continue
        report=json.loads(source.read_text())
        audit=json.loads(audit_path.read_text())
        if (report["schema"]!="NEC-03Q/five-input-development-causal-DID/v1"
            or audit["schema"]!="NEC-03Q/independent-development-causal-exhaustive-audit/v1"
            or report["world_id"]!=f"dev5-{i:03d}"
            or audit["world_id"]!=report["world_id"]
            or report["world_index"]!=i or audit["world_index"]!=i
            or report["world_sha256"]!=audit["world_sha256"]
            or report["frozen_execution_lock_sha256"]!=lock_sha
            or audit["frozen_execution_lock_sha256"]!=lock_sha
            or audit["reported_root_sha256"]!=sha_file(source)
            or not audit["all_dev_group_memberships_and_true_task_labels_independently_recomputed"]
            or not audit["all_32D_disjoint_active_controls_independently_exhaustively_reenumerated"]
            or not audit["all_paired_recurrent_logits_and_correctness_independently_replayed"]
            or not audit["source_checkpoints_and_SAEs_immutable"]
            or report["development_groups_evaluated"]!=24
            or report["focal_masks_per_group"]!=32
            or not report["source_eligible_in_original_source_only_study"]
            or report["primary_comparison_family_size"]!=1
            or report["minimum_meaningful_effect"]!=.02
            or report["reserved_contexts_scored"]!=0
            or audit["reserved_contexts_scored"]!=0
            or report["embargoed_worlds_scored"]!=0
            or audit["embargoed_worlds_scored"]!=0
            or report["original_four_input_confirmation_worlds_scored"]!=0
            or audit["original_four_input_confirmation_worlds_scored"]!=0
            or report["FSI_01_gate"]!="REDESIGN_OR_PILOT_ONLY"
            or audit["FSI_01_gate"]!="REDESIGN_OR_PILOT_ONLY"
            or set(report["arms"])!={"composed","additive"}
            or set(audit["arms"])!={"composed","additive"}):
            raise AssertionError(f"source-only ancestry or independently audited root {i} failed")
        signed={}
        for arm in ("composed","additive"):
            measured=report["arms"][arm]
            verified=audit["arms"][arm]
            if (measured["source_correct"]!=verified["source_correct"]
                or measured["sham_correct"]!=verified["sham_correct"]
                or measured["valid_input_controls"]!=verified["valid_input_controls"]
                or measured["raw_sha256"]!=verified["raw_sha256"]
                or measured["signed_source_minus_sham_correct"]!=verified["source_minus_sham_correct"]
                or measured["total_input_cases"]!=768
                or verified["case_count"]!=768):
                raise AssertionError("missing or divergent paired case-level developmental correctness")
            signed[arm]=verified["source_minus_sham_correct"]
        effect=(signed["composed"]-signed["additive"])/768
        if not math.isclose(effect,report["root_ITT_effect"],abs_tol=1e-12) or not math.isclose(effect,audit["root_ITT_effect"],abs_tol=1e-12):
            raise AssertionError("development root difference-in-differences diverged")
        records.append({
            "world_id":report["world_id"],"world_index":i,
            "world_sha256":report["world_sha256"],"source_eligible":True,
            "audit_status":"independent_source_and_sham_replay_pass",
            "development_context_groups_per_arm":24,
            "paired_cases_per_arm":768,
            "composed_source_minus_sham_correct":signed["composed"],
            "additive_source_minus_sham_correct":signed["additive"],
            "root_ITT_effect":effect,
            "report_sha256":sha_file(source),
            "audit_sha256":sha_file(audit_path),
        })
    effects=[r["root_ITT_effect"] for r in records]
    mean=sum(effects)/8
    p,nonzero=exact_two_sided_root_sign_flip(effects)
    audited=sum(r["audit_status"]=="independent_source_and_sham_replay_pass" for r in records)
    report={
        "schema":SCHEMA,
        "root_statistical_unit":"one original independently seeded five-input Boolean world",
        "registered_world_roots":8,
        "source_admitted_development_world_roots":6,
        "previous_source_ineligible_ITT_zero_worlds":2,
        "independently_replayed_causal_development_roots":audited,
        "unavailable_audit_roots_in_ITT":6-audited,
        "all_eight_worlds_retained":len(records)==8,
        "primary_root_effects":[r["root_ITT_effect"] for r in records],
        "primary_macro_mean_DID":mean,
        "primary_scientific_effect_floor":.02,
        "primary_comparisons":1,
        "exact_exploratory_two_sided_root_sign_flip_p":p,
        "root_sign_flip_nonzero_effects":nonzero,
        "sign_flip_exchangeability_is_experimental_assumption":True,
        "finite_sample_confirmed_causal_effect":False,
        "run_complete_and_independently_audited":audited==6,
        "source_model_or_sae_training_updates":0,
        "new_reserved_context_groups_scored":0,
        "new_embargoed_worlds_scored":0,
        "old_four_input_confirmation_worlds_scored":0,
        "root_count_for_inference":8,
        "matched_sham_calibration_95_95_estimated":False,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
        "roots":records,
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"nec03q_eight_root_development_ITT.json").write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("NEC03Q_EIGHT_ROOT_DEVELOPMENT_CAUSAL_ITT",audited,round(mean,7),round(p,7))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--execution-lock",type=Path,default=Path(__file__).with_name("NEC03Q_DEVELOPMENT_EXECUTION_LOCK.json"))
    a=p.parse_args()
    aggregate(a.root,a.out,a.execution_lock)

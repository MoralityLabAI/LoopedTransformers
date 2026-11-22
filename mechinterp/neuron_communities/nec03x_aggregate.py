"""NEC-03X pure-stdlib eight-root continuous physical-null ITT aggregate.

All eight original Boolean development worlds retained, roots3/5 fixed
primary zeros from historical source admission, two model arms/world,
exact original 256 source-training cases/arm. Only independently
replayed neural/solver source evidence counts toward complete result.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path

SCHEMA="NEC-03X/eight-original-source-world-continuous-physical-null-ITT/v1"
ROOTS=tuple(range(8))
ELIGIBLE=(0,1,2,4,6,7)
ZERO=(3,5)
ARMS=("composed","additive")
MODES=(
    "clean","hubs","native_nonhub","full_affine","contrib16_only",
    "continuous4_source","continuous4_sham",
    "discrete4_source","discrete4_sham",
)


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def root_signflip(values):
    nonzero=[float(v) for v in values if abs(v)>1e-12]
    if not nonzero:return 1.,0
    obs=abs(sum(nonzero))
    exceed=sum(abs(sum(v*s for v,s in zip(nonzero,signs)))>=obs-1e-12
               for signs in product((-1.,1.),repeat=len(nonzero)))
    return exceed/(2**len(nonzero)),len(nonzero)


def aggregate(root,out,lock):
    lock_sha=sha(lock)
    root=Path(root)
    expected={f"nec03x-source-root-{i}" for i in ROOTS}
    if any(p.name not in expected for p in root.glob("nec03x-source-root-*") if p.is_dir()):
        raise AssertionError("original source aggregate contains new or unregistered roots")
    roots=[]
    count=0
    for i in ROOTS:
        path=root/f"nec03x-source-root-{i}"
        rp=path/"nec03x_source_root_report.json"
        ap=path/"independent_nec03x_continuous_source_audit.json"
        if not rp.is_file() or not ap.is_file():
            roots.append({
                "world_index":i,"world_id":f"dev5-{i:03d}",
                "independent_audit_complete":False,
                "root_primary_ITT_DID":0.,"source_ITT_zero":i in ZERO,
                "reason":"missing_or_failed_original_source_neural_audit"})
            continue
        rec=json.loads(rp.read_text())
        audit=json.loads(ap.read_text())
        if (rec["schema"]!="NEC-03X/source-only-continuous-disjoint-physical-null/v1"
            or audit["schema"]!="NEC-03X/independent-box-solver-and-continuous-physical-sham-replay/v1"
            or rec["world_index"]!=i or audit["world_index"]!=i
            or rec["world_id"]!=f"dev5-{i:03d}"
            or rec["world_id"]!=audit["world_id"]
            or rec["world_sha256"]!=audit["world_sha256"]
            or rec["source_lock_sha256"]!=lock_sha
            or audit["source_lock_sha256"]!=lock_sha
            or audit["original_root_report_sha256"]!=sha(rp)
            or rec["source_training_groups_scored"]!=list(range(40,48))
            or rec["source_cases_per_model_arm"]!=256
            or audit["source_cases_per_arm"]!=256
            or rec["original_paired_source_eligible"]!=(i in ELIGIBLE)
            or audit["source_root_original_paired_eligible"]!=(i in ELIGIBLE)
            or rec["original_ITT_zero"]!=(i in ZERO)
            or rec["model_or_SAE_optimizer_updates"]!=0
            or any(rec[k]!=0 or audit[k]!=0 for k in (
                "new_development_eval_groups_scored","reserved_groups_scored",
                "new_recruited_worlds_scored","five_input_embargoed_worlds_scored",
                "old_four_input_confirmation_worlds_scored"))
            or rec["FSI_01_gate"]!="REDESIGN_OR_PILOT_ONLY"
            or audit["FSI_01_gate"]!="REDESIGN_OR_PILOT_ONLY"
            or not all(audit.get(k) for k in (
                "frozen_original_native_and_SAE_checkpoint_ancestry_verified",
                "all_495_disjoint_continuous_supports_and_bounded_32_step_solver_replayed",
                "all_source_coherent_quartets_and_control_quality_gates_recomputed",
                "original_secondary_discrete_null_independently_replayed",
                "original_32d_output_L2_doses_and_nonamplifying_gains_verified",
                "all_original_256_training_source_labels_per_teacher_recomputed",
                "all_nine_12_visit_neural_recurrent_modes_independently_replayed"))
            or set(rec["arms"])!=set(ARMS) or set(audit["arms"])!=set(ARMS)):
            raise AssertionError(f"NEC03X original root {i} continuous null or frozen outcomes changed")
        arms={}
        for arm in ARMS:
            original=rec["arms"][arm]
            verified=audit["arms"][arm]
            if (original["correct_counts"]!=verified["correct_counts"]
                or original["source_model_sha256"]!=verified["source_model_sha256"]
                or original["source_sae_sha256"]!=verified["SAE_checkpoint_sha256"]
                or original["source_rank_sha256"]!=verified["source_rank_sha256"]
                or original["raw_sha256"]!=verified["raw_sha256"]
                or original["source_candidate_present"]!=verified["source_candidate_present"]
                or original["original_source_eligible"]!=verified["original_source_eligible"]
                or original["matched_continuous_inputs"]!=verified["matched_continuous_inputs"]
                or original["matched_discrete_inputs"]!=verified["matched_discrete_inputs"]
                or original["continuous_sham_minus_source_correct"]!=verified["continuous_correct_case_difference"]
                or original["discrete_sham_minus_source_correct"]!=verified["discrete_correct_case_difference"]
                or set(original["correct_counts"])!=set(MODES)
                or any(not 0<=v<=256 for v in original["correct_counts"].values())
                or original["source_cases"]!=256
                or original["source_group_indices"]!=list(range(40,48))):
                raise AssertionError(f"NEC03X original root {i} {arm} incorrect or unverified")
            for k,v in (("secondary_within_model_margin_scale","margin_scale"),
                        ("secondary_normalized_continuous_sham_minus_source_margin",
                         "secondary_normalized_source_sham_margin")):
                if not math.isclose(original[k],verified[v],abs_tol=1e-6,rel_tol=1e-6):
                    raise AssertionError("source-only secondary normalized margin changed")
            arms[arm]={
                "correct_counts":verified["correct_counts"],
                "continuous_sham_minus_source_correct":
                    verified["continuous_correct_case_difference"],
                "discrete_sham_minus_source_correct":
                    verified["discrete_correct_case_difference"],
                "matched_continuous_inputs":verified["matched_continuous_inputs"],
                "matched_discrete_inputs":verified["matched_discrete_inputs"],
                "secondary_normalized_margin":verified["secondary_normalized_source_sham_margin"],
                "source_eligible":verified["original_source_eligible"],
                "source_candidate_present":verified["source_candidate_present"],
            }
        effect=(arms["composed"]["continuous_sham_minus_source_correct"]-
                arms["additive"]["continuous_sham_minus_source_correct"])/256
        if (abs(effect-rec["raw_continuous_primary_root_DID"])>1e-12
            or abs((effect if i in ELIGIBLE else 0.)-rec["primary_original_ITT_root_DID"])>1e-12
            or abs((effect if i in ELIGIBLE else 0.)-audit["primary_root_ITT_DID"])>1e-12):
            raise AssertionError("NEC03X original root primary ITT changed")
        roots.append({
            "world_id":rec["world_id"],"world_index":i,
            "world_sha256":rec["world_sha256"],
            "independent_audit_complete":True,
            "source_ITT_zero":i in ZERO,
            "root_primary_ITT_DID":effect if i in ELIGIBLE else 0.,
            "original_raw_effect":effect,
            "source_root_report_sha256":sha(rp),
            "independent_source_audit_sha256":sha(ap),
            "arms":arms,
        })
        count+=1
    known=[row for row in roots if row["independent_audit_complete"]]
    totals={arm:{m:sum(row["arms"][arm]["correct_counts"][m] for row in known)
                  for m in MODES}
            for arm in ARMS}
    values=[row["root_primary_ITT_DID"] for row in roots]
    primary=sum(values)/8
    p,n=root_signflip(values)
    qualified=[row for row in known if row["world_index"] in ELIGIBLE]
    positives=sum(row["root_primary_ITT_DID"]>0 for row in qualified)
    coverage=count==8 and all(
        row["arms"][arm]["matched_continuous_inputs"]>=205
        for row in qualified for arm in ARMS)
    # Even a passing source-only developmental gate is strictly a pilot.
    pilot=(count==8 and primary>.02 and positives>=5 and coverage and p<=.05)
    result={
        "schema":SCHEMA,
        "root_statistical_unit":"one independently seeded original development Boolean world",
        "original_eight_root_ITT_count":8,
        "independently_audited_source_root_count":count,
        "failed_or_missing_source_audits":8-count,
        "original_eligible_paired_source_roots":list(ELIGIBLE),
        "historical_source_ineligible_ITT_zeros":list(ZERO),
        "original_source_training_groups":list(range(40,48)),
        "original_source_cases_per_native_teacher_arm":256,
        "original_native_model_arms_per_world":2,
        "root_primary_continuous_disjoint_sham_effects":values,
        "one_primary_macro_correctness_effect":primary,
        "one_primary_scientifically_meaningful_effect_floor":.02,
        "one_primary_above_floor":primary>.02,
        "eligible_world_roots_positive":positives,
        "eligible_world_positive_gate":positives>=5,
        "original_qualified_arm_continuous_null_coverage_205_each":coverage,
        "exploratory_root_exact_two_sided_signflip_p":p,
        "root_signflip_nonzero_effect_count":n,
        "exploratory_root_signflip_gate_leq_005":p<=.05,
        "complete_source_only_pilot_resolution_gate":pilot,
        "all_eight_original_neural_source_audits_passed":count==8,
        "native_model_or_SAE_updates":0,
        "source_only_is_population_confirmation":False,
        "independent_causal_sham_95_95_tail_calibrated":False,
        "source_correct_counts_by_native_teacher":totals,
        "signed_clean_minus_hubs_gap":{
            arm:totals[arm]["clean"]-totals[arm]["hubs"] for arm in ARMS},
        "new_development_eval_groups_scored":0,
        "reserved_groups_scored":0,
        "new_recruited_worlds_scored":0,
        "five_input_embargoed_worlds_scored":0,
        "old_four_input_confirmation_worlds_scored":0,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
        "roots":roots,
    }
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    (out/"nec03x_eight_original_root_continuous_null_ITT.json").write_text(
        json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("NEC03X_EIGHT_ROOT_CONTINUOUS_SHAM_AUDITED_ITT",
          count,primary,p,positives,"all_qualified_arms_205",coverage,"pilot_gate",pilot)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--lock",type=Path,
                   default=Path(__file__).with_name("NEC03X_SOURCE_LOCK.json"))
    args=p.parse_args()
    aggregate(args.root,args.out,args.lock)

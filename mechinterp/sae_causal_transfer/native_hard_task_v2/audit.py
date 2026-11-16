"""LS-SAE08 independent stdlib-only source and full81 Sudoku numeric receipt replay.

Verifies original immutable calibration hash, deterministic first-admissible
source-grid roster, original physical directions/power search, all finite
ACT3/6/12 logits, exact Sudoku truth and 2000 bootstrap primary gates.
Does not re-execute pretrained native Torch forward model.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from mechinterp.sae_causal_transfer.native_hard_task_v2 import families as FAMILY
from mechinterp.sae_causal_transfer.native_task_transfer_v1 import audit as BASE_AUD

SCHEMA="LS-SAE08/model-blind-admitted-hard-native-causal-task-v1"
PROTOCOL="6b38c3dcf65dbc3a812e0f7c63b79dc14c6893fe"
CHECKPOINT="932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3"
SOURCE_NPZ="073b7771676e44eb934c50bdf1e7639ffeadbdff12998979cfe4c43e42b2da25"
SOURCE_MANIFEST="cb44b910f94210745b7a7df304e575d5a194ee8cd7bec51a88281fccc95bf33a"
FIT_ARRAY="83dacde8f13926fc41fc48cb17bcfba8c399538754efb7104e41e94bd93797ec"
RADII=("0.20","2.00","8.00")
HORIZONS=("ACT3","ACT6","ACT12")
ARMS=("state_power","logit_power","sae","random")


def close(a,b,atol=2e-5):
    return BASE_AUD.close(a,b,atol=atol)


def verify_case(case,source,index):
    goal=FAMILY.median_blank(source["x"])
    if (case["puzzle_index"]!=index or case["seed"]!=source["seed"]
        or case["visible_goal_position"]!=goal
        or case["source_calibration_locked"] is not True):
        raise ValueError("frozen source site/index differs from original visible Sudoku")
    ratio=case["radius_over_clean_source_norm"]
    original=float(case["original_source_zH_token_norm"])
    if not math.isfinite(original) or original<=0 or set(ratio)!=set(RADII):
        raise ValueError("native original hidden-source dose receipts missing")
    for dose in RADII:
        if not close(ratio[dose],float(dose)/original):
            raise ValueError("physical source dose is not fixed")
    meta=case["direction_metadata"]
    geometry=meta["geometry"]
    if set(geometry)!=set(ARMS) or set(case["source_direction_status"])!=set(ARMS):
        raise ValueError("source physical edit arms changed")
    selected={k:BASE_AUD.check_geometry(geometry[k],goal) for k in ARMS}
    for key in ARMS:
        if case["source_direction_status"][key]!=(selected[key] is not None):
            raise ValueError("source direction status differs from its physical vector")
    BASE_AUD.check_power(meta["state_power_transcript"],selected["state_power"])
    BASE_AUD.check_power(meta["logit_power_transcript"],selected["logit_power"])
    if selected["random"] is None:
        raise ValueError("registered RANDOM physical dose control missing")
    if set(case["native_81x9_logits"])!=set(HORIZONS) or set(case["task_score"])!=set(HORIZONS):
        raise ValueError("original 3/6/12 ACT horizon record incomplete")
    for horizon in HORIZONS:
        raw=case["native_81x9_logits"][horizon]
        BASE_AUD.check_logits(raw["clean"])
        scores=case["task_score"][horizon]
        if set(scores)!=set(ARMS):
            raise ValueError("finite native task control arms changed")
        for arm in ARMS:
            if selected[arm] is None and arm in raw:
                raise ValueError("ineligible source unit vector received a finite output")
            per_radius=scores[arm]
            if set(per_radius)!=set(RADII):
                raise ValueError("registered fixed physical source radii changed")
            for dose in RADII:
                edits=raw.get(arm,{}).get(dose)
                score=per_radius[dose]
                BASE_AUD.replay_pair(score,raw["clean"],edits,
                                     source["x"],source["target"],goal,
                                     float(dose),geometry[arm],horizon)
                if score is not None and horizon=="ACT3" and dose=="0.20":
                    if geometry[arm] is None:
                        raise ValueError("native logit JVP missing for source edit")
                    plus=edits["plus"];minus=edits["minus"]
                    sec=BASE_AUD.centered(BASE_AUD.difference(plus,minus))
                    sec=[[v/.4 for v in row] for row in sec]
                    jvp=geometry[arm]["centered_digit_logit_jvp_81x9"]
                    err=BASE_AUD.matrix_norm(BASE_AUD.difference(sec,jvp))
                    rel=err/(1+BASE_AUD.matrix_norm(jvp))
                    if (not close(err,score["centered_logit_secant_minus_jvp_fro"])
                        or not close(rel,score["centered_logit_secant_minus_jvp_relative"])):
                        raise ValueError("original finite native-logit JVP cross-check differs")


def summary(rows,family):
    effects=[];correct=[];events=0
    inexact=improved=0
    for row in rows:
        act6=row["task_score"]["ACT6"]
        a=act6["state_power"]["8.00"]
        b=act6["random"]["8.00"]
        if b is None:
            raise ValueError("source random equal-dose control unscored")
        clean=a["clean"] if a is not None else b["clean"]
        if clean["blank_correct"]<clean["blank_total"]:
            inexact+=1
            if a is not None and any(
                sign["blank_correct"]>clean["blank_correct"] and
                sign["clue_disagreements"]==0
                for sign in (a["plus"],a["minus"])):
                improved+=1
        Fp=a["symmetric_blank_prediction_flip_score"] if a is not None else 0
        Fr=b["symmetric_blank_prediction_flip_score"]
        d=float(Fp-Fr)
        effects.append(d)
        events+=int(d>=1.0)
        correct.append((a["symmetric_blank_correctness_gain"] if a is not None else 0)
                       -b["symmetric_blank_correctness_gain"])
    rng=random.Random(202627130+len(family))
    boot=[]
    for _ in range(2000):
        idx=[rng.randrange(len(effects)) for _ in effects]
        boot.append(sum(effects[j] for j in idx)/len(idx))
    boot.sort()
    ci=[boot[50],boot[1949]]
    return {
       "n_puzzles":len(rows),
       "power_minus_random_flip_pairs":effects,
       "power_flip_advantage_events_at_least_one":events,
       "mean_paired_blank_prediction_flip_effect":sum(effects)/len(effects),
       "puzzle_bootstrap95":ci,"bootstrap_draws":2000,
       "mean_paired_blank_correctness_gain":sum(correct)/len(correct),
       "clean_inexact_puzzles_ACT6":inexact,
       "power_best_sign_clue_consistent_improvement_on_inexact":improved,
       "registered_success":len(rows)==8 and events>=6 and ci[0]>0
    }


def audit(r):
    if (r["schema"]!=SCHEMA or r["preregistration_commit"]!=PROTOCOL
        or r["source_model_file_sha256"]!=CHECKPOINT
        or r["original_calibration_npz_file_sha256"]!=SOURCE_NPZ
        or r["original_calibration_manifest_sha256"]!=SOURCE_MANIFEST
        or r["original_calibration_fit_array_sha256"]!=FIT_ARRAY
        or r["source_candidate_budget_per_slot"]!=32
        or r["horizons"]!=[3,6,12] or r["physical_radii"]!=[.2,2.,8.]
        or r["direction_arm_order"]!=list(ARMS)
        or r["state_power_updates"]!=4 or r["logit_power_updates"]!=4
        or r["mode"] not in ("pilot","full")
        or r["full_protocol"]!=(r["mode"]=="full")):
        raise ValueError("source-registered LS-SAE08 protocol/source changed")
    original=FAMILY.full_manifest()
    if original!=r["all_source_candidate_roster_sha256"]:
        raise ValueError("model-blind first-admitted source roster differs")
    groups=("pilot_52","pilot_54") if r["mode"]=="pilot" else ("test_52","test_54")
    if set(r["families"])!=set(groups):
        raise ValueError("pilot/full source families changed")
    for family in groups:
        rows=FAMILY.roster(family)
        saved=r["families"][family]
        if saved["source_roster"]!=rows or len(saved["per_puzzle"])!=len(rows):
            raise ValueError("source candidate selection was changed after test")
        for i,(case,source) in enumerate(zip(saved["per_puzzle"],rows)):
            verify_case(case,source,i)
        expected=summary(saved["per_puzzle"],family.replace("pilot_","test_"))
        reported=saved["statistics"]
        for key,value in expected.items():
            if key not in reported:
                raise ValueError("original preregistered scientific count omitted")
            prior=reported[key]
            if isinstance(value,list):
                if len(value)!=len(prior) or any(not close(a,b) for a,b in zip(value,prior)):
                    raise ValueError("source-registered paired/bootstrap array differs: "+key)
            elif isinstance(value,float):
                if not close(value,prior):
                    raise ValueError("registered paired effect changed: "+key)
            elif prior!=value:
                raise ValueError("registered coverage or gate modified: "+key)
    overall=(r["full_protocol"] and
             all(r["families"][f]["statistics"]["registered_success"]
                 for f in ("test_52","test_54")))
    if r["registered_success_both_families"]!=overall:
        raise ValueError("registered two-family scientific decision altered")
    return {"schema":"LS-SAE08/independent-stdlib-hard-Sudoku-numeric-replay-v1",
            "preheldout_original_checkpoint_SAE_SHA_checked":True,
            "all_first_admissible_source_candidate_rosters_replayed":True,
            "all_ACT3_6_12_native_81x9_signed_logits_checked":True,
            "both_matched_JtJ_power_budget_numerical_transcripts_replayed":True,
            "original_27unit_Sudoku_correctness_and_error_recovery_replayed":True,
            "all_original_2000_cluster_bootstrap_and_gates_replayed":True,
            "full_protocol":r["full_protocol"],
            "tested_unique_puzzles":sum(len(x["per_puzzle"]) for x in r["families"].values()),
            "both_families_scientific_gate":overall,
            "limitation":"Independent source-exact and saved-native-numeric replay; no second pretrained model execution or formal interval certification."}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    original=args.input.read_bytes()
    rec=audit(json.loads(original))
    rec["original_raw_sha256"]=hashlib.sha256(original).hexdigest()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(rec,indent=2,sort_keys=True)+"\n")
    print(json.dumps(rec,indent=2,sort_keys=True))


if __name__=="__main__":
    main()

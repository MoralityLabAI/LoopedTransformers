"""NEC-04B pure-stdlib six-ROOT native Looper headroom aggregate.

Only original pilot calibration roots0..5, two frozen teacher arms and
continuous optimizer checkpoints at 120/240/480/900 training steps.
Entire six-root set and every prior independently checked model outcome
are required. Never infer world-population generalization from the
calibration-owned validation masks. No SAE or causal sham outcome.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

SCHEMA="NEC-04B/six-calibration-native-headroom-budget-audit/v1"
ROOTS=tuple(range(6))
ARMS=("composed","additive")
BUDGETS=(120,240,480,900)
BANDS={
    "moderate":(0.60,0.75),
    "intermediate":(0.75,0.90),
    "high":(0.90,0.98),
}
SOURCE=2560
VALIDATION=768


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def band_for_accuracy(x):
    if not 0<=x<=1:
        raise AssertionError("native validation accuracy outside unit interval")
    for name,(lower,upper) in BANDS.items():
        if lower<=x<upper or (name=="high" and x==upper):
            return name
    return "outside_headroom"


def aggregate(root,pretraining,out):
    root=Path(root)
    pretraining=Path(pretraining)
    lock=pretraining/"nec04b_pretraining_oracles.json"
    locked=json.loads(lock.read_text())
    if (locked["root_indices"]!=list(ROOTS)
        or locked["native_arms"]!=list(ARMS)
        or locked["training_step_budgets"]!=list(BUDGETS)
        or len(locked["roots"])!=6
        or locked["model_optimizer_steps_before_preflight"]!=0):
        raise AssertionError("pretraining world/teacher source gate invalid")
    found={p.name for p in root.glob("nec04b-pilot-root-*") if p.is_dir()}
    if found!={f"nec04b-pilot-root-{i}" for i in ROOTS}:
        raise AssertionError("registered six-root native training artifacts incomplete/unexpected")
    roots=[]
    for index in ROOTS:
        folder=root/f"nec04b-pilot-root-{index}"
        src_file=folder/"nec04b_native_headroom_root.json"
        audit_file=folder/"nec04b_independent_native_headroom_audit.json"
        if not (src_file.is_file() and audit_file.is_file()):
            raise AssertionError("independent pilot checkpoint audit missing")
        rec=json.loads(src_file.read_text())
        aud=json.loads(audit_file.read_text())
        ref=locked["roots"][index]
        if (rec["schema"]!="NEC-04B/native-headroom-six-calibration-root-pilot/v1"
            or aud["schema"]!="NEC-04B/independent-frozen-native-checkpoint-headroom-replay/v1"
            or rec["root_index"]!=index or aud["root_index"]!=index
            or rec["world_id"]!=ref["world_id"]
            or rec["world_id"]!=aud["world_id"]
            or rec["world_sha256"]!=ref["world_sha256"]
            or aud["world_sha256"]!=ref["world_sha256"]
            or rec["source_lock_sha256"]!=aud["source_lock_sha256"]
            or rec["source_lock_sha256"]!=locked["source_lock_sha256"]
            or rec["pretraining_teacher_oracle_sha256"]!=sha(lock)
            or aud["original_pretraining_oracle_ledger_sha256"]!=sha(lock)
            or rec["original_source_validation_file_sha256"]!=ref["pilot_file_sha256"]
            or rec["source_train_rows"]!=SOURCE
            or rec["owned_calibration_world_validation_rows"]!=VALIDATION
            or rec["budget_steps"]!=list(BUDGETS)
            or rec["native_model_training_updates_per_arm"]!=900
            or rec["native_SAE_training_updates"]!=0
            or rec["all_checkpoints_within_each_arm_sealed_before_validation"] is not True
            or aud["original_root_score_receipt_sha256"]!=sha(src_file)
            or not all(aud[k] for k in (
                "all_six_oracle_roots_scoped",
                "two_original_teacher_oracles_independently_recomputed",
                "all_8_frozen_checkpoints_native_model_SHA256_verified",
                "all_8_optimizer_state_step_counts_verified",
                "all_8_source_and_validation_full_logits_independently_replayed"
            ))
            or aud["training_trajectory_independently_reoptimized"] is not False
            or any(rec[k]!=0 for k in (
                "native_model_or_SAE_validation_on_new_world_roots",
                "new_development_or_confirmation_world_specs_materialized",
                "original_calibration_root_reserved_contexts_scored",
                "old_NEC03P_or_old_confirmation_worlds_scored"
            ))
            or any(aud[k]!=0 for k in (
                "original_reserved_contexts_scored",
                "new_development_root_specs_instantiated",
                "new_confirmation_root_specs_instantiated",
                "old_NEC03P_worlds_scored"
            ))
            or rec["FSI_01_gate"]!="REDESIGN_OR_PILOT_ONLY"
            or aud["FSI_01_gate"]!="REDESIGN_OR_PILOT_ONLY"
            or set(rec["arms"])!=set(ARMS)
            or set(aud["arms"])!=set(ARMS)):
            raise AssertionError("immutable six root/teacher pilot roster or validation results changed")
        byarm={}
        for arm in ARMS:
            original=rec["arms"][arm]
            replay=aud["arms"][arm]
            manifest=original["checkpoint_seal_manifest"]
            if (len(original["source_and_validation_results"])!=4
                or len(manifest["checkpoints"])!=4
                or len(replay)!=4
                or manifest["all_four_checkpoints_sealed_before_validation"] is not True
                or manifest["optimizer_resets"]!=0
                or manifest["updates_total"]!=900):
                raise AssertionError("original checkpoint budget/optimizer continuity changed")
            armrow=[]
            for j,budget in enumerate(BUDGETS):
                val=original["source_and_validation_results"][j]
                checked=replay[j]
                old_train=val["source_context_training_metrics"]
                old_eval=val["owned_calibration_world_validation_metrics"]
                if (val["step"]!=budget or checked["step"]!=budget
                    or old_train["correct"]!=checked["source_train"]["correct"]
                    or old_eval["correct"]!=checked["owned_validation"]["correct"]
                    or val["checkpoint_sha256"]!=checked["checkpoint_sha256"]
                    or val["model_sha256"]!=checked["native_weights_sha256"]
                    or old_train["cases"]!=SOURCE or
                    old_eval["cases"]!=VALIDATION
                    or abs(old_train["accuracy"]-checked["source_train"]["accuracy"])>1e-12
                    or abs(old_eval["accuracy"]-checked["owned_validation"]["accuracy"])>1e-12):
                    raise AssertionError("independent validation root/budget score mismatch")
                accuracy=old_eval["correct"]/VALIDATION
                armrow.append({
                    "step":budget,
                    "source_training_correct":old_train["correct"],
                    "source_training_accuracy":old_train["accuracy"],
                    "owned_validation_correct":old_eval["correct"],
                    "owned_validation_accuracy":accuracy,
                    "owned_validation_balanced_accuracy":old_eval["balanced_accuracy"],
                    "training_minus_validation_accuracy":old_train["accuracy"]-accuracy,
                    "validation_headroom_band":band_for_accuracy(accuracy),
                    "checkpoint_sha256":checked["checkpoint_sha256"],
                    "native_model_sha256":checked["native_weights_sha256"],
                })
            byarm[arm]=armrow
        roots.append({
            "world_id":rec["world_id"],"root_index":index,
            "world_sha256":rec["world_sha256"],
            "tier":rec["structural_difficulty_tier"],
            "original_program_sha256":rec["original_composed_program_sha256"],
            "original_additive_oracle_sha256":rec["original_additive_oracle_sha256"],
            "original_checkpoint_score_receipt_sha256":sha(src_file),
            "independent_checkpoint_replay_sha256":sha(audit_file),
            "arms":byarm,
        })
    snapshots=[]
    for j,step in enumerate(BUDGETS):
        tiers={}
        arms={}
        for arm in ARMS:
            histogram={band:0 for band in (*BANDS,"outside_headroom")}
            accuracies=[]
            for r in roots:
                result=r["arms"][arm][j]
                histogram[result["validation_headroom_band"]]+=1
                accuracies.append(result["owned_validation_accuracy"])
            arms[arm]={
                "root_count":6,
                "headroom_band_root_counts":histogram,
                "mean_root_macro_validation_accuracy":sum(accuracies)/6,
                "root_accuracies_in_frozen_order":accuracies,
                "required_two_independent_roots_in_each_headroom_band":
                    all(histogram[band]>=2 for band in BANDS),
            }
        for tier in ("easy","medium","hard"):
            tiers[tier]={
                "root_indices":[r["root_index"] for r in roots if r["tier"]==tier],
                "validation_by_teacher":{
                    arm:[r["arms"][arm][j]["owned_validation_accuracy"]
                         for r in roots if r["tier"]==tier]
                    for arm in ARMS
                }
            }
        snapshots.append({
            "steps":step,"arms":arms,
            "tier_root_accuracy":tiers,
            "both_teacher_headroom_family_gate_passes":all(
                arms[a]["required_two_independent_roots_in_each_headroom_band"]
                for a in ARMS),
        })
    qualifying=[x for x in snapshots if x["both_teacher_headroom_family_gate_passes"]]
    selected=qualifying[0] if qualifying else None
    result={
        "schema":SCHEMA,
        "original_NEC04A_root_commitment_sha256":locked["original_registry_sha256"],
        "original_NEC04A_source_sha256":locked["original_calibration_source_sha256"],
        "original_pretraining_oracle_ledger_sha256":sha(lock),
        "pretraining_source_lock_sha256":locked["source_lock_sha256"],
        "native_model_architecture":"PEF07-Looper 32D 12 recurrent steps",
        "original_calibration_root_indices":list(ROOTS),
        "original_independent_root_statistical_units":6,
        "original_calibration_source_contexts_per_root":80,
        "owned_calibration_world_validation_contexts_per_root":24,
        "full_focal_counterfactual_masks_per_context":32,
        "fixed_four_native_checkpoint_budgets":list(BUDGETS),
        "all_six_roots_have_two_independently_scored_teacher_arms":True,
        "all_48_native_checkpoint_states_sha_audited":True,
        "all_48_training_and_owned_validation_metrics_replayed":True,
        "two_teacher_families":list(ARMS),
        "headroom_ranges":{
            "moderate":[.60,.75],"intermediate":[.75,.90],"high":[.90,.98],
        },
        "headroom_min_independent_roots_each_band_per_family":2,
        "checkpoint_budget_snapshots":snapshots,
        "headroom_gate_passed_any_budget":selected is not None,
        "selected_training_budget_for_full_24_root_calibration":
            selected["steps"] if selected else None,
        "decision":(
            "HEADROOM_PILOT_PASS_AWAIT_SEPARATE_FROZEN_SAE_AND_EXPANSION"
            if selected else
            "STOP_NATIVE_HEADROOM_SELECTION_CALIBRATION_ONLY"
        ),
        "source_SAE_fitted":False,
        "source_specific_sham_scored":False,
        "new_native_optimizer_updates_for_each_pilot_root_teacher":900,
        "original_reserved_contexts_scored":0,
        "new_development_world_root_specs_materialized":0,
        "new_confirmation_world_root_specs_materialized":0,
        "old_NEC03P_recruited_worlds_scored":0,
        "old_NEC03_embargoed_worlds_scored":0,
        "population_causal_sham_roots_scored":0,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
        "roots":roots,
    }
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    (out/"nec04b_six_root_headroom_aggregate.json").write_text(
        json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("NEC04B_SIX_ROOT_NATIVE_HEADROOM_INDEPENDENTLY_REPLAYED",
          "budget_pass",result["selected_training_budget_for_full_24_root_calibration"],
          "gate",result["decision"])
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--roots",type=Path,required=True)
    p.add_argument("--pretraining",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    aggregate(a.roots,a.pretraining,a.out)


if __name__=="__main__":
    main()

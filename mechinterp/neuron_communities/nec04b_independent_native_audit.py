"""NEC-04B independent replay audit: native checkpoints and headroom.

Regenerates both original target teachers independently and scores every
sealed native Looper state on all 2560 source and 768 calibration-world
owned validation examples for six frozen roots. Verifies each original
checkpoint/optimizer-state SHA and 120/240/480/900 step counters.

This is checkpoint/outcome replay, NOT a second optimization trajectory
and NOT independent heldout-world causal validation.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch

from mechinterp.neuron_communities import nec04a_boolean_gym as gym
from mechinterp.neuron_communities.nec04a_independent_oracle import oracle
from mechinterp.neuron_communities.nec04b_oracles import (
    PILOT,ARMS,BUDGETS,SOURCE,VALIDATION,verify_lock,file_sha,
)
from mechinterp.neuron_communities.nec04b_independent_oracles import independently_linear_spec
from research.pef07.pef07 import Looper,model_digest,as_tensor

SCHEMA_AUDIT="NEC-04B/independent-frozen-native-checkpoint-headroom-replay/v1"
SCHEMA_SCORE="NEC-04B/native-headroom-six-calibration-root-pilot/v1"
CHUNK=128


def own_seed(index,arm):
    raw=sha256(b"NEC04B/native-training-seed/v1|"+
        gym.seed_for_identity("calibration",index)+arm.encode()).digest()
    return 1+int.from_bytes(raw[:8],"big")%(2**31-2)


@torch.no_grad()
def replay_logits(model,bits):
    model.eval()
    values=[]
    for k in range(0,len(bits),CHUNK):
        values.extend(model.logits(bits[k:k+CHUNK]).detach().cpu().tolist())
    if len(values)!=len(bits):raise AssertionError("checkpoint audit returned incomplete logits")
    return values


def evaluate_independent(logits,truth,prior):
    if len(logits)!=len(truth) or set(prior)!={
        "correct","cases","accuracy","correct_true_positives",
        "correct_true_negatives","truth_positive_cases","balanced_accuracy",
        "class_predictions","native_logits",
    }:
        raise AssertionError("source/native headroom scored outcome receipt incomplete")
    expected_correct=0
    tp=tn=0
    positive=sum(truth)
    if not 0<positive<len(truth):
        raise AssertionError("original source/owned-validation teacher class degenerate")
    for i,(logits2,y) in enumerate(zip(logits,truth)):
        expected=prior["native_logits"][i]
        if len(expected)!=2 or max(abs(float(a)-float(b)) for a,b in zip(logits2,expected))>3e-4:
            raise AssertionError("independent full native 12-step logits differ from archived scores")
        predicted=1 if logits2[1]>logits2[0] else 0
        if predicted!=prior["class_predictions"][i]:
            raise AssertionError("independent checkpoint prediction differs")
        expected_correct+=int(predicted==y)
        tp+=int(predicted==y==1)
        tn+=int(predicted==y==0)
    accuracy=expected_correct/len(truth)
    balanced=.5*(tp/positive+tn/(len(truth)-positive))
    if (expected_correct!=prior["correct"] or
        len(truth)!=prior["cases"] or
        tp!=prior["correct_true_positives"] or
        tn!=prior["correct_true_negatives"] or
        positive!=prior["truth_positive_cases"] or
        abs(accuracy-prior["accuracy"])>1e-12 or
        abs(balanced-prior["balanced_accuracy"])>1e-12):
        raise AssertionError("independent source/validation model correctness altered")
    return {
        "cases":len(truth),"correct":expected_correct,
        "accuracy":accuracy,"balanced_accuracy":balanced,
        "truth_positive":positive,
    }


def validate(root_index,oracles:Path,output:Path):
    if root_index not in PILOT:
        raise PermissionError("NEC04B audit permits only precommitted calibration roots 0..5")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    _,lock_hash=verify_lock()
    ledger_file=oracles/"nec04b_pretraining_oracles.json"
    ledger=json.loads(ledger_file.read_text())
    ref=ledger["roots"][root_index]
    input_file=oracles/ref["pilot_file"]
    if (ledger["root_indices"]!=list(PILOT) or
        ledger["source_lock_sha256"]!=lock_hash or
        file_sha(input_file)!=ref["pilot_file_sha256"]):
        raise AssertionError("pilot independently verified teacher source evidence altered")
    raw=[json.loads(v) for v in input_file.read_text().splitlines()]
    if len(raw)!=SOURCE+VALIDATION:
        raise AssertionError("root original training/validation context count changed")
    world,_=gym.calibration_world(root_index)
    if world["world_sha256"]!=ref["world_sha256"]:
        raise AssertionError("original NEC04A world identity changed")
    weights,threshold,pos,truth_sha,linear_truth=independently_linear_spec(root_index,world)
    if (ref["additive_spec"]["signed_weights_12"]!=weights
        or ref["additive_spec"]["threshold_twice"]!=threshold
        or ref["additive_spec"]["truth_table_4096_sha256"]!=truth_sha):
        raise AssertionError("independent original additive teacher malformed")
    teacher={}
    for arm in ARMS:
        labels=[]
        for item in raw:
            bits=item["history_bits12"]
            if arm=="composed":
                truth=int(oracle(world["canonical_program"],bits))
            else:
                idx=sum(int(b)<<j for j,b in enumerate(bits))
                truth=int((linear_truth>>idx)&1)
            if item[f"{arm}_label"]!=truth:
                raise AssertionError("independently recomputed Boolean task target disagrees")
            labels.append(truth)
        teacher[arm]=labels
    report_file=output/"nec04b_native_headroom_root.json"
    report=json.loads(report_file.read_text())
    if (report["schema"]!=SCHEMA_SCORE
        or report["root_index"]!=root_index
        or report["world_id"]!=world["world_id"]
        or report["world_sha256"]!=world["world_sha256"]
        or report["source_lock_sha256"]!=lock_hash
        or report["original_source_validation_file_sha256"]!=file_sha(input_file)
        or report["pretraining_teacher_oracle_sha256"]!=file_sha(ledger_file)
        or report["budget_steps"]!=list(BUDGETS)
        or report["source_train_rows"]!=SOURCE
        or report["owned_calibration_world_validation_rows"]!=VALIDATION
        or report["native_model_training_updates_per_arm"]!=900
        or report["native_SAE_training_updates"]!=0
        or report["all_checkpoints_within_each_arm_sealed_before_validation"] is not True
        or any(report[k]!=0 for k in (
            "native_model_or_SAE_validation_on_new_world_roots",
            "new_development_or_confirmation_world_specs_materialized",
            "original_calibration_root_reserved_contexts_scored",
            "old_NEC03P_or_old_confirmation_worlds_scored"))
        or report["FSI_01_gate"]!="REDESIGN_OR_PILOT_ONLY"
        or set(report["arms"])!=set(ARMS)):
        raise AssertionError("registered original six-root native calibration pilot changed")
    x=as_tensor([row["history_bits12"] for row in raw])
    result_arms={}
    for arm in ARMS:
        a=report["arms"][arm]
        manifest_file=output/f"nec04b.root{root_index}.{arm}.checkpoints.json"
        if file_sha(manifest_file)!=a["checkpoint_seal_manifest_sha256"]:
            raise AssertionError("four-checkpoint first-seal manifest SHA changed")
        manifest=json.loads(manifest_file.read_text())
        if (manifest!=a["checkpoint_seal_manifest"] or
            manifest["source_lock_sha256"]!=lock_hash or
            manifest["root_index"]!=root_index or
            manifest["arm"]!=arm or
            manifest["all_four_checkpoints_sealed_before_validation"] is not True or
            manifest["optimizer_resets"]!=0 or
            manifest["updates_total"]!=900 or
            len(manifest["checkpoints"])!=len(BUDGETS) or
            len(a["source_and_validation_results"])!=len(BUDGETS)):
            raise AssertionError("training-budget checkpoint seals or native optimizer continuity missing")
        seed=own_seed(root_index,arm)
        if (manifest["world_sha256"]!=world["world_sha256"] or
            report["root_seed_model_arms"][arm]!=seed):
            raise AssertionError("root original weight seed or source identity changed")
        outcomes=[]
        for k,step in enumerate(BUDGETS):
            rec=manifest["checkpoints"][k]
            prior=a["source_and_validation_results"][k]
            f=output/f"nec04b.root{root_index}.{arm}.step{step}.pt"
            if (rec["file"]!=f.name or rec["step"]!=step or
                file_sha(f)!=rec["checkpoint_sha256"] or
                rec["checkpoint_sha256"]!=prior["checkpoint_sha256"] or
                prior["step"]!=step):
                raise AssertionError("optimizer checkpoint byte SHA changed")
            payload=torch.load(f,map_location="cpu",weights_only=True)
            if (payload["schema"]!=SCHEMA_SCORE or
                payload["world_sha256"]!=world["world_sha256"] or
                payload["root_index"]!=root_index or
                payload["native_arm"]!=arm or
                payload["model_seed"]!=seed or
                payload["pilot_source_case_sha256"]!=file_sha(input_file) or
                payload["source_lock_sha256"]!=lock_hash or
                payload["step"]!=step or payload["optimizer_updates"]!=step or
                payload["batch_size"]!=64 or
                payload["optimizer"]!="AdamW" or payload["lr"]!=.003 or
                payload["weight_decay"]!=0 or
                payload["gradient_clip"]!=1.0 or
                payload["native_weights_sha256"]!=rec["model_sha256"]):
                raise AssertionError("original native optimizer step or root checkpoint provenance changed")
            optimizer_steps=[]
            for item in payload["optimizer_state"]["state"].values():
                if "step" in item:
                    optimizer_steps.append(int(item["step"].item()))
            if not optimizer_steps or any(count!=step for count in optimizer_steps):
                raise AssertionError("optimizer state did not carry continuous registered updates")
            native=Looper().eval()
            native.load_state_dict(payload["model_state"],strict=True)
            if (model_digest(native)!=rec["model_sha256"]
                or rec["model_sha256"]!=prior["model_sha256"]):
                raise AssertionError("native weight SHA not stable across independent replay")
            train_logit=replay_logits(native,x[:SOURCE])
            val_logit=replay_logits(native,x[SOURCE:])
            train_res=evaluate_independent(
                train_logit,teacher[arm][:SOURCE],
                prior["source_context_training_metrics"])
            val_res=evaluate_independent(
                val_logit,teacher[arm][SOURCE:],
                prior["owned_calibration_world_validation_metrics"])
            outcomes.append({
                "step":step,"checkpoint_sha256":file_sha(f),
                "native_weights_sha256":model_digest(native),
                "source_train":train_res,"owned_validation":val_res,
                "oracle_teacher":"root-committed composed AST" if arm=="composed" else "root-committed independent signed linear threshold",
                "optimizer_updates_verified":step,
            })
            print("NEC04B_INDEPENDENT_NATIVE_CHECKPOINT_AND_HEADROOM_REPLAY_PASS",
                world["world_id"],arm,step,val_res["correct"],"of",VALIDATION,
                flush=True)
        result_arms[arm]=outcomes
    artifact={
        "schema":SCHEMA_AUDIT,
        "source_lock_sha256":lock_hash,
        "root_index":root_index,
        "world_id":world["world_id"],
        "world_sha256":world["world_sha256"],
        "original_root_score_receipt_sha256":file_sha(report_file),
        "original_pretraining_oracle_ledger_sha256":file_sha(ledger_file),
        "all_six_oracle_roots_scoped":True,
        "two_original_teacher_oracles_independently_recomputed":True,
        "all_8_frozen_checkpoints_native_model_SHA256_verified":True,
        "all_8_optimizer_state_step_counts_verified":True,
        "all_8_source_and_validation_full_logits_independently_replayed":True,
        "training_trajectory_independently_reoptimized":False,
        "native_SAE_models_trained":0,
        "source_samples_per_arm":SOURCE,
        "calibration_owned_validation_samples_per_arm":VALIDATION,
        "original_reserved_contexts_scored":0,
        "new_development_root_specs_instantiated":0,
        "new_confirmation_root_specs_instantiated":0,
        "old_NEC03P_worlds_scored":0,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
        "arms":result_arms,
    }
    (output/"nec04b_independent_native_headroom_audit.json").write_text(
        json.dumps(artifact,sort_keys=True,indent=2,allow_nan=False)+"\n")
    return artifact


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root-index",type=int,required=True)
    p.add_argument("--oracles",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    validate(a.root_index,a.oracles,a.out)


if __name__=="__main__":
    main()

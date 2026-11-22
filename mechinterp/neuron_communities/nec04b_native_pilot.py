"""NEC-04B: fixed six-calibration-root native Looper headroom pilot.

Only 6 NEC04A calibration-world roots (0..5). Train on original 80
source-training context groups per root. Seal 120/240/480/900-step
checkpoints for composed and independent linear-threshold additive arms
before any 24-context owned-calibration-world validation is scored.

One continuous AdamW optimizer per arm (no reset between checkpoints).
No SAE training, causal sham, root recruitment or source-selected
mechanistic intervention in this headroom-only pilot.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import platform

import torch
import torch.nn.functional as F

from mechinterp.neuron_communities.nec04b_oracles import (
    PILOT,ARMS,BUDGETS,SOURCE,VALIDATION,
    file_sha,verify_lock,root_checkpoint_seed,
)
from research.pef07.pef07 import Looper,model_digest,as_tensor

SCHEMA="NEC-04B/native-headroom-six-calibration-root-pilot/v1"
BATCH=64
LR=0.003
CLIP=1.0
WEIGHT_DECAY=0.0
CHUNK=128


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def checked_inputs(root_index,source):
    ledger_path=source/"nec04b_pretraining_oracles.json"
    ledger=json.loads(ledger_path.read_text())
    _,lock_sha=verify_lock()
    if (ledger["source_lock_sha256"]!=lock_sha
        or ledger["root_indices"]!=list(PILOT)
        or ledger["native_arms"]!=list(ARMS)
        or ledger["training_step_budgets"]!=list(BUDGETS)):
        raise AssertionError("pre-scoring original root and model pilot lock changed")
    ref=ledger["roots"][root_index]
    f=source/ref["pilot_file"]
    if (f.name!=f"nec04b_pilot_root_{root_index}.jsonl" or
        ref["root_index"]!=root_index or
        file_sha(f)!=ref["pilot_file_sha256"]):
        raise AssertionError("original independent teacher oracle source case SHA changed")
    data=rows(f)
    if (len(data)!=SOURCE+VALIDATION
        or len([x for x in data if x["split"]=="source_train"])!=SOURCE
        or len([x for x in data if x["split"]=="calibration_world_validation"])!=VALIDATION
        or len({tuple(x["history_bits12"]) for x in data})!=SOURCE+VALIDATION):
        raise AssertionError("source-only complete split or unique 12bit contexts broken")
    for k,x in enumerate(data):
        split="source_train" if k<SOURCE else "calibration_world_validation"
        if x["world_id"]!=ref["world_id"] or x["split"]!=split:
            raise AssertionError("original calibration root or split changed")
    return ledger,ref,data


def score_logits(model,x):
    model.eval()
    out=[]
    with torch.no_grad():
        for i in range(0,len(x),CHUNK):
            batch=model.logits(x[i:i+CHUNK])
            out.extend(batch.detach().cpu().tolist())
    if len(out)!=len(x):
        raise AssertionError("checkpoint headroom evaluation partial output")
    return out


def score_predictions(logits,target):
    pred=[int(v[1]>v[0]) for v in logits]
    total=len(target)
    correct=sum(a==int(b) for a,b in zip(pred,target))
    tp=sum(int(a==b==1) for a,b in zip(pred,target))
    tn=sum(int(a==b==0) for a,b in zip(pred,target))
    positive=sum(target)
    negative=total-positive
    return {
        "correct":correct,"cases":total,
        "accuracy":correct/total,
        "correct_true_positives":tp,"correct_true_negatives":tn,
        "truth_positive_cases":positive,
        "balanced_accuracy":(
            .5*(tp/positive+tn/negative) if positive and negative else None
        ),
        "class_predictions":pred,
        "native_logits":logits,
    }


def train_arm(root_index,arm,ref,data,out,lock_sha):
    if arm not in ARMS:
        raise ValueError("unknown native calibration model arm")
    seed=root_checkpoint_seed(root_index,arm)
    if seed!=ref["native_seeds"][arm]:
        raise AssertionError("registered root native initial weight seed changed")
    torch.manual_seed(seed)
    model=Looper().train()
    initial_sha=model_digest(model)
    optimizer=torch.optim.AdamW(
        model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY
    )
    gen=torch.Generator(device="cpu").manual_seed(seed^0x5DEECE66D)
    inputs=as_tensor([x["history_bits12"] for x in data])
    train_x=inputs[:SOURCE]
    train_y=torch.tensor([int(x[f"{arm}_label"]) for x in data[:SOURCE]],dtype=torch.long)
    val_x=inputs[SOURCE:]
    if (train_x.shape!=(SOURCE,12) or val_x.shape!=(VALIDATION,12)
        or not 0<int(train_y.sum())<SOURCE):
        raise AssertionError("source/validation teacher labels or sizes changed")
    checkpoint=[]
    first_loss=None
    last_loss=None
    for step in range(1,max(BUDGETS)+1):
        take=torch.randint(SOURCE,(BATCH,),generator=gen)
        bx=train_x[take]
        by=train_y[take]
        optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model.logits(bx),by)
        if not torch.isfinite(loss):
            raise AssertionError("nonfinite frozen native source loss")
        if first_loss is None:first_loss=float(loss.detach())
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),CLIP)
        if not torch.isfinite(grad):
            raise AssertionError("nonfinite gradient norm")
        optimizer.step()
        last_loss=float(loss.detach())
        if step in BUDGETS:
            model_sha=model_digest(model)
            payload={
                "schema":SCHEMA,
                "source_lock_sha256":lock_sha,
                "root_index":root_index,"world_id":ref["world_id"],
                "world_sha256":ref["world_sha256"],
                "native_arm":arm,"model_seed":seed,
                "pilot_source_case_sha256":ref["pilot_file_sha256"],
                "step":step,"batch_size":BATCH,
                "optimizer":"AdamW","lr":LR,
                "weight_decay":WEIGHT_DECAY,
                "gradient_clip":CLIP,
                "optimizer_updates":step,
                "model_state":model.state_dict(),
                "optimizer_state":optimizer.state_dict(),
                "native_weights_sha256":model_sha,
                "native_initial_weights_sha256":initial_sha,
            }
            checkpoint_path=out/f"nec04b.root{root_index}.{arm}.step{step}.pt"
            torch.save(payload,checkpoint_path)
            checkpoint.append({
                "step":step,"file":checkpoint_path.name,
                "checkpoint_sha256":file_sha(checkpoint_path),
                "model_sha256":model_sha,
                "initial_weights_sha256":initial_sha,
                "source_loss_initial":first_loss,
                "source_loss_last_update":last_loss,
            })
            print("NEC04B_CAL_NATIVE_CHECKPOINT_SEALED",
                  ref["world_id"],arm,step,checkpoint[-1]["checkpoint_sha256"],
                  flush=True)
    if len(checkpoint)!=4:
        raise AssertionError("one of four native budgets not frozen")
    # All four native checkpoints are sealed before looking at any
    # calibration-world validation labels or measuring headroom.
    sealed_manifest={
        "schema":SCHEMA,"root_index":root_index,
        "world_id":ref["world_id"],"world_sha256":ref["world_sha256"],
        "arm":arm,"source_lock_sha256":lock_sha,
        "original_program_sha256":ref["program_sha256"],
        "original_additive_target_sha256":ref["additive_spec"]["target_spec_sha256"],
        "root_source_input_sha256":ref["pilot_file_sha256"],
        "all_four_checkpoints_sealed_before_validation":True,
        "optimizer_resets":0,"updates_total":max(BUDGETS),
        "checkpoints":checkpoint,
    }
    path=out/f"nec04b.root{root_index}.{arm}.checkpoints.json"
    path.write_text(json.dumps(sealed_manifest,sort_keys=True,indent=2)+"\n")
    # Validation truths are first read after the entire 900-step arm
    # and all four fixed checkpoint hashes have been sealed.
    val_y=torch.tensor([int(x[f"{arm}_label"]) for x in data[SOURCE:]],dtype=torch.long)
    if not 0<int(val_y.sum())<VALIDATION:
        raise AssertionError("validation oracle labels degenerate")
    observations=[]
    for rec in checkpoint:
        frozen_path=out/rec["file"]
        if file_sha(frozen_path)!=rec["checkpoint_sha256"]:
            raise AssertionError("sealed root checkpoint modified before validation")
        payload=torch.load(frozen_path,map_location="cpu",weights_only=True)
        frozen=Looper().eval()
        frozen.load_state_dict(payload["model_state"],strict=True)
        if model_digest(frozen)!=rec["model_sha256"]:
            raise AssertionError("sealed original native model weights mismatch")
        train_logits=score_logits(frozen,train_x)
        val_logits=score_logits(frozen,val_x)
        train=score_predictions(train_logits,train_y.tolist())
        validation=score_predictions(val_logits,val_y.tolist())
        observations.append({
            "step":rec["step"],"checkpoint_sha256":rec["checkpoint_sha256"],
            "model_sha256":rec["model_sha256"],
            "source_context_training_metrics":train,
            "owned_calibration_world_validation_metrics":validation,
        })
        print("NEC04B_REAL_NATIVE_HEADROOM_SCORED",
              ref["world_id"],arm,rec["step"],
              "source",train["correct"],"of",SOURCE,
              "validation",validation["correct"],"of",VALIDATION,
              flush=True)
    return sealed_manifest,observations


def run(root_index,source,out):
    if root_index not in PILOT:
        raise PermissionError("only new NEC04A calibration roots0..5")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    lock,lock_sha=verify_lock()
    ledger,ref,data=checked_inputs(root_index,source)
    out.mkdir(parents=True,exist_ok=True)
    result={
        "schema":SCHEMA,
        "root_index":root_index,
        "world_id":ref["world_id"],
        "world_sha256":ref["world_sha256"],
        "structural_difficulty_tier":ref["tier"],
        "original_composed_program_sha256":ref["program_sha256"],
        "original_additive_oracle_sha256":ref["additive_spec"]["target_spec_sha256"],
        "source_lock_sha256":lock_sha,
        "pretraining_teacher_oracle_sha256":file_sha(
            source/"nec04b_pretraining_oracles.json"
        ),
        "original_source_validation_file_sha256":ref["pilot_file_sha256"],
        "original_NEC04A_source_corpus_sha256":ledger["original_calibration_source_sha256"],
        "native_architecture":"PEF07-Looper 32D, 12 recurrent steps",
        "torch_version":str(torch.__version__),
        "torch_cpu_capability":torch.backends.cpu.get_cpu_capability(),
        "numpy_external_random_seeds_used":False,
        "pythonhashseed":os.environ.get("PYTHONHASHSEED"),
        "processor":platform.processor(),
        "source_train_rows":SOURCE,
        "owned_calibration_world_validation_rows":VALIDATION,
        "budget_steps":list(BUDGETS),
        "all_checkpoints_within_each_arm_sealed_before_validation":True,
        "root_seed_model_arms":{a:ref["native_seeds"][a] for a in ARMS},
        "native_model_training_updates_per_arm":900,
        "native_SAE_training_updates":0,
        "native_model_or_SAE_validation_on_new_world_roots":0,
        "new_development_or_confirmation_world_specs_materialized":0,
        "original_calibration_root_reserved_contexts_scored":0,
        "old_NEC03P_or_old_confirmation_worlds_scored":0,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
        "arms":{},
    }
    for arm in ARMS:
        manifest,observations=train_arm(root_index,arm,ref,data,out,lock_sha)
        result["arms"][arm]={
            "checkpoint_seal_manifest_sha256":file_sha(
                out/f"nec04b.root{root_index}.{arm}.checkpoints.json"),
            "checkpoint_seal_manifest":manifest,
            "source_and_validation_results":observations,
        }
    raw=(json.dumps(result,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
    (out/"nec04b_native_headroom_root.json").write_bytes(raw)
    print("NEC04B_NATIVE_SIX_ROOT_PILOT_RESULT_SAVED",root_index,
          sha256(raw).hexdigest(),flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root-index",type=int,required=True)
    p.add_argument("--oracles",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    run(a.root_index,a.oracles,a.out)


if __name__=="__main__":
    main()

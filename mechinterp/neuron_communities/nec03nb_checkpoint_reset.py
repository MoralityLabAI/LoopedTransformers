"""NEC-03N-B: fixed checkpoint-anchored source-only AdamW reset calibration.

This experimental condition starts from the EXACT frozen NEC-03M
900-step native checkpoint, resets AdamW optimizer moments to zero, and
performs 900 additional source-training updates using the registered
per-teacher minibatch index generator advanced past its original 900
batches. It is NOT the continuous-AdamW 1800-step NEC-03N condition,
whose strict replay failed on the original heterogeneous CPU runners.

All four already-opened dev5-000..003 world roots and both original
composed/additive models are retained. Training and scoring access only
their original 80 source groups × 32 focal-bit variants (2560 histories).
No SAE retrain, causal intervention, development-evaluation, reserved,
five-input embargoed or original four-input confirmation readout.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from mechinterp.neuron_communities.nec03l_five_input_worlds import (
    worlds,registration,context_catalog,labels,encoded,
)
from mechinterp.neuron_communities.nec03g_worlds import registry as old_registry
from research.pef07.pef07 import Looper,model_digest,as_tensor

SCHEMA="NEC-03N-B/frozen900-checkpoint-reset-AdamW-900-source/v1"
ROOTS=(0,1,2,3)
SOURCE_STATES=2560
OLD_STEPS=900
ADDITIONAL_STEPS=900
TOTAL_STEPS=1800
BATCH=128
LR=.003
CLIP=1.
WEIGHT_DECAY=0.
CLEAN_SOURCE_FLOOR=.90


def sha_file(path:Path) -> str:
    h=sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def batch_digest(indices):
    t=indices.detach().cpu().contiguous()
    return sha256(bytes(t.reshape(-1).view(torch.uint8).tolist())).hexdigest()


def verify_lock() -> dict:
    root=Path(__file__).parent
    path=root/"NEC03NB_RESET_LOCK.json"
    cfg=json.loads(path.read_text())
    if (cfg.get("schema")!="NEC-03N-B/frozen-source900-reset-optimizer-lock/v1"
        or cfg.get("original_NEC03M_run_id")!=44894980345
        or cfg.get("failed_exact_continuation_run_id")!=44895917012
        or cfg.get("original_five_input_registry_sha256")!=registration()["registry_sha256"]
        or cfg.get("original_four_input_confirmation_spec_sha256")!=
             old_registry()["confirmation_full_spec_sha256"]
        or cfg.get("source_world_indices")!=list(ROOTS)
        or cfg.get("source_training_histories")!=SOURCE_STATES
        or cfg.get("original_checkpoint_training_steps")!=OLD_STEPS
        or cfg.get("new_steps_with_reset_AdamW")!=ADDITIONAL_STEPS
        or cfg.get("total_model_updates")!=TOTAL_STEPS
        or cfg.get("batch_size")!=BATCH
        or cfg.get("native_learning_rate")!=LR
        or cfg.get("original_source_head_competence_floor")!=CLEAN_SOURCE_FLOOR
        or cfg.get("new_sae_models_trained")!=0
        or cfg.get("development_eval_contexts_scored")!=0
        or cfg.get("reserved_contexts_scored")!=0
        or cfg.get("embargoed_five_input_worlds_scored")!=0
        or cfg.get("original_four_input_confirmation_worlds_scored")!=0):
        raise AssertionError("NEC03N-B fixed source-reset experiment changed after registration")
    for name,expected in cfg["git_blob_sha1"].items():
        contents=(root/name).read_bytes()
        gitsha=__import__("hashlib").sha1(
            b"blob "+str(len(contents)).encode()+b"\x00"+contents).hexdigest()
        if gitsha!=expected:
            raise AssertionError("registered reset-AdamW source changed: "+name)
    cfg["lock_sha256"]=sha_file(path)
    return cfg


@torch.no_grad()
def source_confusion(model,source,world,*,arm):
    if len(source)!=SOURCE_STATES or arm not in ("composed","additive"):
        raise ValueError("only 80 source training groups in two known arms")
    predictions=[]
    truth_counts=[[0,0],[0,0]]
    for i in range(0,len(source),128):
        bits=as_tensor(source[i:i+128])
        ground=labels(bits,world,control=arm=="additive")
        pred=model.logits(bits).argmax(1)
        for y,z in zip(ground.tolist(),pred.tolist()):
            truth_counts[y][z]+=1
            predictions.append(int(z))
    correct=truth_counts[0][0]+truth_counts[1][1]
    total=sum(map(sum,truth_counts))
    if total!=SOURCE_STATES:
        raise AssertionError("model source input coverage incomplete")
    return {
        "correct":correct,"total":SOURCE_STATES,
        "accuracy":correct/SOURCE_STATES,
        "confusion_truth_rows_pred_columns":truth_counts,
        "source_prediction_labels":predictions,
        "per_class_recall":[truth_counts[i][i]/max(sum(truth_counts[i]),1)
                            for i in (0,1)],
        "training_source_only":True,
    }


def train_checkpoint_anchored(original:Path,world,source,arm,out:Path):
    old=json.loads((original/"source_pilot_receipt.json").read_text())
    prior=old["arms"][arm]["native"]
    ckpt=original/prior["checkpoint_file"]
    if sha_file(ckpt)!=prior["checkpoint_sha256"]:
        raise AssertionError("source-trained old native checkpoint bytes altered")
    if (prior["source_training_steps"]!=OLD_STEPS
        or prior["training_source_histories"]!=SOURCE_STATES):
        raise AssertionError("prior checkpoint not source900 under registered world")
    seed=world["native_model_seed"]+(100000 if arm=="additive" else 0)
    if prior["training_seed"]!=seed:
        raise AssertionError("original world model seed mismatch")
    payload=torch.load(ckpt,map_location="cpu",weights_only=True)
    if (payload["world_sha256"]!=world["world_sha256"]
        or payload["arm"]!=arm or payload["steps"]!=OLD_STEPS
        or payload["seed"]!=seed
        or payload["model_weights_sha256"]!=prior["final_model_weights_sha256"]):
        raise AssertionError("checkpoint payload not original immutable five-bit model")
    torch.manual_seed(seed)
    model=Looper()
    model.load_state_dict(payload["model"],strict=True)
    before=model_digest(model)
    if before!=prior["final_model_weights_sha256"]:
        raise AssertionError("loaded native model differs from frozen source900 weights")
    model.eval()
    old_score=source_confusion(model,source,world,arm=arm)
    if (old_score["correct"]!=prior["source_correct"]
        or abs(old_score["accuracy"]-prior["source_accuracy"])>1e-10):
        raise AssertionError("original source900 class behavior not reproducible")
    model.train()
    x=as_tensor(source)
    y=labels(x,world,control=arm=="additive")
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    if opt.state:
        raise AssertionError("fresh AdamW reset state unexpectedly nonempty")
    rng=torch.Generator().manual_seed(seed*7919+59)
    for _ in range(OLD_STEPS):
        torch.randint(len(x),(BATCH,),generator=rng)
    first_indices=None
    last_indices=None
    first_loss=last_loss=None
    for _ in range(ADDITIONAL_STEPS):
        idx=torch.randint(len(x),(BATCH,),generator=rng)
        if first_indices is None: first_indices=batch_digest(idx)
        last_indices=batch_digest(idx)
        opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model.logits(x[idx]),y[idx])
        if not torch.isfinite(loss):
            raise AssertionError("nonfinite checkpoint-anchored native loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),CLIP)
        opt.step()
        if first_loss is None:first_loss=float(loss.detach())
        last_loss=float(loss.detach())
    model.eval()
    trained_sha=model_digest(model)
    checkpoint_file=f"{arm}.warm_reset_native1800.pt"
    path=out/checkpoint_file
    torch.save({
        "schema":SCHEMA,
        "world_id":world["world_id"],
        "world_sha256":world["world_sha256"],
        "arm":arm,"model_seed":seed,
        "prior_frozen900_model_sha256":before,
        "prior_frozen900_checkpoint_file_sha256":prior["checkpoint_sha256"],
        "new_reset_AdamW_steps":ADDITIONAL_STEPS,
        "original_learning_updates_before_warmstart":OLD_STEPS,
        "total_model_updates":TOTAL_STEPS,
        "checkpoint_source_only":True,
        "model_weights_sha256":trained_sha,
        "model":model.state_dict()
    },path)
    new_score=source_confusion(model,source,world,arm=arm)
    return {
        "arm":arm,"seed":seed,
        "old_native_model_sha256":before,
        "old_frozen900_checkpoint_sha256":prior["checkpoint_sha256"],
        "old_source_score":old_score,
        "original_optimizer_state_reconstructed":False,
        "reset_optimizer_state_at_checkpoint":True,
        "optimizer":"fresh AdamW(lr=.003,wd=0) state at old900 + 900 source batches",
        "source_batch_stream_offset":OLD_STEPS,
        "source_index_generator_seed":seed*7919+59,
        "first_new_batch_index_sha256":first_indices,
        "last_new_batch_index_sha256":last_indices,
        "new_optimizer_update_count":ADDITIONAL_STEPS,
        "total_updates":TOTAL_STEPS,
        "first_warmstart_batch_loss":first_loss,
        "last_warmstart_batch_loss":last_loss,
        "new_native_model_sha256":trained_sha,
        "new_checkpoint_file":checkpoint_file,
        "new_checkpoint_sha256":sha_file(path),
        "new_source_score":new_score,
        "old_competence":old_score["accuracy"]>=CLEAN_SOURCE_FLOOR,
        "new_competence":new_score["accuracy"]>=CLEAN_SOURCE_FLOOR,
        "original_source_score_only":True,
    }


def run(index:int,old_root:Path,out:Path):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    lock=verify_lock()
    if index not in ROOTS:
        raise PermissionError("frozen900 reset optimizer only on previously trained dev5-000..003")
    w=worlds()[index]
    source,manifest=context_catalog(w)
    old=json.loads((old_root/"source_pilot_receipt.json").read_text())
    if (old["world_sha256"]!=w["world_sha256"]
        or old["source_manifest_sha256"]!=manifest["source_sha256"]
        or old["five_input_registry_sha256"]!=registration()["registry_sha256"]
        or old["original_four_input_confirmation_spec_sha256"]!=
            old_registry()["confirmation_full_spec_sha256"]
        or old["development_evaluation_context_groups_scored"]!=0
        or old["reserved_context_groups_scored"]!=0
        or old["new_embargoed_worlds_scored"]!=0
        or old["original_confirmation_worlds_scored"]!=0
        or json.loads((old_root/"world_spec.json").read_text())!=w
        or json.loads((old_root/"source_split_manifest.json").read_text())!=manifest):
        raise AssertionError("original five-input model/source cohort no longer frozen")
    out.mkdir(parents=True,exist_ok=True)
    both={arm:train_checkpoint_anchored(old_root,w,source["train"],arm,out)
          for arm in ("composed","additive")}
    report={
        "schema":SCHEMA,
        "world_id":w["world_id"],"world_index":index,
        "world_sha256":w["world_sha256"],
        "five_input_world_registry_sha256":registration()["registry_sha256"],
        "original_four_input_confirmation_spec_sha256":
            old_registry()["confirmation_full_spec_sha256"],
        "original_NEC03M_run":44894980345,
        "original_900_receipt_sha256":sha_file(old_root/"source_pilot_receipt.json"),
        "registered_source_reset_lock_sha256":lock["lock_sha256"],
        "original_exact_continuation_1800_failure_run":44895917012,
        "old_optimizer_state_reset_at900":True,
        "exact_continuous_AdamW_1800_claim":False,
        "source_histories":SOURCE_STATES,
        "old_native_steps":OLD_STEPS,
        "additional_AdamW_reset_steps":ADDITIONAL_STEPS,
        "no_five_input_new_worlds_scored":True,
        "new_sae_models_fit":0,
        "new_interventions_scored":0,
        "new_five_input_development_evaluation_context_groups_scored":0,
        "reserved_context_groups_scored":0,
        "five_input_embargoed_worlds_scored":0,
        "original_four_input_confirmation_worlds_scored":0,
        "original_NEC03M_competence_and_source_AE_unchanged":True,
        "source_only_in_sample_accuracy":True,
        "arms":both,
        "both_arms_source_competent":all(
            item["new_competence"] for item in both.values()),
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    (out/"checkpoint_reset_source_report.json").write_bytes(
        json.dumps(report,sort_keys=True,separators=(",",":")).encode()+b"\n")
    print("NEC03NB_SOURCE_RESET_CHECKPOINT_CALIBRATION_WRITTEN")
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--world-index",type=int,required=True)
    parser.add_argument("--old-root",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    run(args.world_index,args.old_root,args.out)

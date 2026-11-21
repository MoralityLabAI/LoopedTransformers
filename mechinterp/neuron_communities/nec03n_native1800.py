"""NEC-03N: exact original 900-step replay into fixed 1800-step source training.

This is source-only five-input model competence calibration on the four
ALREADY OPENED NEC-03M native training-source worlds dev5-000..003.

Recreate every original AdamW parameter update (including sampling stream
and optimizer moments) from seed. At step900 require an EXACT model-weight
digest and full 2560-source-prediction agreement with the original frozen
900-step checkpoint/receipts, then continue to exactly 1800 steps.

No model/SAE source-fit or development-evaluation, embargoed, reserved or
old four-input confirmation scores are generated.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from mechinterp.neuron_communities.nec03l_five_input_worlds import (
    worlds,registration,context_catalog,labels,encoded,additive_control_truth,
)
from mechinterp.neuron_communities.nec03g_worlds import registry as old_registry
from research.pef07.pef07 import Looper,model_digest,as_tensor

SCHEMA="NEC-03N/source-only-fixed1800-continuous-optimizer-v1"
INDICES=(0,1,2,3)
OLD_STEPS=900
NEW_STEPS=1800
BATCH=128
LEARNING_RATE=.003
WEIGHT_DECAY=0.
CLIP=1.
SOURCE_HISTORY_COUNT=2560
COMPETENCE_FLOOR=.90


def file_sha(p:Path):
    h=sha256()
    with p.open("rb") as f:
        for data in iter(lambda:f.read(1<<20),b""):
            h.update(data)
    return h.hexdigest()


def verify_training_lock():
    """Exact immutable source, threshold, code and old-world commitments."""
    path=Path(__file__).with_name("NEC03N_SOURCE_LOCK.json")
    data=json.loads(path.read_text())
    if (data.get("schema")!="NEC-03N/fixed1800-source-replay-lock/v1"
        or data.get("world_indices")!=list(INDICES)
        or data.get("source_registration_sha256")!=registration()["registry_sha256"]
        or data.get("old_four_input_confirmation_sha256")!=old_registry()["confirmation_full_spec_sha256"]
        or data.get("old_nec03m_run_id")!=44894980345
        or data.get("milestone_replay_steps")!=OLD_STEPS
        or data.get("full_training_steps")!=NEW_STEPS
        or data.get("batch_size")!=BATCH
        or data.get("learning_rate")!=LEARNING_RATE
        or data.get("weight_decay")!=WEIGHT_DECAY
        or data.get("clip_gradient_l2")!=CLIP
        or data.get("source_history_count")!=SOURCE_HISTORY_COUNT
        or data.get("clean_in_sample_competence_floor")!=COMPETENCE_FLOOR
        or data.get("new_development_evaluation_groups_scored")!=0
        or data.get("reserved_groups_scored")!=0
        or data.get("embargoed_worlds_scored")!=0
        or data.get("old_confirmation_worlds_scored")!=0):
        raise AssertionError("source-only 1800-step learning protocol was not frozen")
    base=Path(__file__).parent
    for filename,expected in data["git_blob_sha1"].items():
        blob=(base/filename).read_bytes()
        identity=__import__("hashlib").sha1(
            b"blob "+str(len(blob)).encode()+b"\x00"+blob).hexdigest()
        if identity!=expected:
            raise AssertionError("source training implementation changed after preregistration")
    data["lock_sha256"]=file_sha(path)
    return data


@torch.no_grad()
def source_confusion(model,source,world,*,arm):
    """Full in-sample source score, true classes only used for reporting."""
    if len(source)!=SOURCE_HISTORY_COUNT:
        raise ValueError("source accuracy must cover all 2,560 training histories")
    correct=0
    confusion=[[0,0],[0,0]]
    labels_count=[0,0]
    for start in range(0,len(source),128):
        bits=as_tensor(source[start:start+128])
        target=labels(bits,world,control=arm=="additive")
        guess=model.logits(bits).argmax(1)
        correct+=int((target==guess).sum())
        for truth,pred in zip(target.tolist(),guess.tolist()):
            confusion[truth][pred]+=1
            labels_count[truth]+=1
    if sum(sum(x) for x in confusion)!=SOURCE_HISTORY_COUNT:
        raise AssertionError("source labels/guesses incomplete")
    return {
        "total":SOURCE_HISTORY_COUNT,
        "correct":correct,
        "accuracy":correct/SOURCE_HISTORY_COUNT,
        "confusion_truth_rows_pred_columns":confusion,
        "truth_class_counts":labels_count,
        "class_recall":[
            confusion[c][c]/labels_count[c] if labels_count[c] else None
            for c in (0,1)],
        "source_training_accuracy_only":True,
    }


def replay_one(original:Path,world,catalog,arm,out:Path):
    if world["cohort"]!="development" or arm not in ("composed","additive"):
        raise PermissionError("invalid source teacher/embargoed cohort")
    old=json.loads((original/"source_pilot_receipt.json").read_text())
    prior=old["arms"][arm]["native"]
    original_checkpoint=original/prior["checkpoint_file"]
    if file_sha(original_checkpoint)!=prior["checkpoint_sha256"]:
        raise AssertionError("prior 900-step source model checkpoint bytes altered")
    if prior["source_training_steps"]!=OLD_STEPS:
        raise AssertionError("wrong prior 900-step training recipe")
    if prior["training_source_histories"]!=SOURCE_HISTORY_COUNT:
        raise AssertionError("old native model was trained on other source data")
    seed=world["native_model_seed"]+(100000 if arm=="additive" else 0)
    if prior["training_seed"]!=seed:
        raise AssertionError("old teacher seed is not the registry's fixed seed")
    torch.manual_seed(seed)
    model=Looper().train()
    start_digest=model_digest(model)
    if start_digest!=prior["initial_model_weights_sha256"]:
        raise AssertionError("freshly seeded original Looper cannot be replayed")
    x=as_tensor(catalog["train"])
    target=labels(x,world,control=(arm=="additive"))
    generator=torch.Generator().manual_seed(seed*7919+59)
    opt=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
    first=milestone_loss=last=None
    at900=None
    for step in range(1,NEW_STEPS+1):
        idx=torch.randint(len(x),(BATCH,),generator=generator)
        opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model.logits(x[idx]),target[idx])
        if not torch.isfinite(loss):
            raise AssertionError("nonfinite fixed native source training")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),CLIP)
        opt.step()
        if first is None: first=float(loss.detach())
        if step==OLD_STEPS:
            milestone_loss=float(loss.detach())
            if abs(milestone_loss-prior["final_train_loss"])>1e-6:
                raise AssertionError("milestone loss differs from historical fixed source training")
            hash900=model_digest(model)
            if hash900!=prior["final_model_weights_sha256"]:
                raise AssertionError("900-step replay diverged from immutable NEC03M checkpoint")
            original_payload=torch.load(original_checkpoint,map_location="cpu",weights_only=True)
            if original_payload["model_weights_sha256"]!=hash900:
                raise AssertionError("old 900-step model payload contradicts original receipt")
            for key,tensor in model.state_dict().items():
                if not torch.equal(tensor.detach(),original_payload["model"][key]):
                    raise AssertionError("same 900-step source model has different tensor bytes")
            model.eval()
            at900=source_confusion(model,catalog["train"],world,arm=arm)
            if (at900["correct"]!=prior["source_correct"]
                or at900["total"]!=prior["source_total"]):
                raise AssertionError("at900 re-training source scores differ from old receipt")
            model.train()
            # Evaluate the source in eval mode only; Adam optimizer state is
            # unchanged and will continue through the full 1800 steps.
        last=float(loss.detach())
    model.eval()
    identity=model_digest(model)
    out.mkdir(parents=True,exist_ok=True)
    filename=f"{arm}.native1800.pt"
    payload={
        "world_id":world["world_id"],"world_sha256":world["world_sha256"],
        "teacher":arm,"seed":seed,"steps":NEW_STEPS,
        "intermediate_900_weight_sha256":prior["final_model_weights_sha256"],
        "model_weights_sha256":identity,"model":model.state_dict()}
    torch.save(payload,out/filename)
    score=source_confusion(model,catalog["train"],world,arm=arm)
    return {
        "arm":arm,"native_model_seed":seed,
        "source_training_steps":NEW_STEPS,"source_training_histories":SOURCE_HISTORY_COUNT,
        "original_initial_model_sha256":start_digest,
        "original_900_checkpoint_sha256":prior["checkpoint_sha256"],
        "old_900_model_sha256":prior["final_model_weights_sha256"],
        "replayed_900_model_sha256":hash900,
        "all_900_checkpoint_tensors_matched_exactly":True,
        "old_900_source_score":at900,
        "new_1800_source_score":score,
        "new_1800_model_sha256":identity,
        "new_checkpoint_file":filename,"new_checkpoint_sha256":file_sha(out/filename),
        "source_native_optimizer_steps":NEW_STEPS,
        "same_AdamW_state_continued_through_1800":True,
        "independent_source_model_reinitialized_from_original_seed":True,
        "first_training_batch_loss":first,
        "training_loss_step900":milestone_loss,
        "training_loss_step1800":last,
        "old_900_source_competent":at900["accuracy"]>=COMPETENCE_FLOOR,
        "new_1800_source_competent":score["accuracy"]>=COMPETENCE_FLOOR,
        "teacher_truth_table":world["truth_table_32"] if arm=="composed" else
              additive_control_truth(world["truth_table_32"]),
    }


def run(index:int,old_root:Path,out:Path):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    lock=verify_training_lock()
    if index not in INDICES:
        raise PermissionError("1800-step calibration only on already-trained dev5-000..003")
    world=worlds()[index]
    catalog,source=context_catalog(world)
    old=json.loads((old_root/"source_pilot_receipt.json").read_text())
    if (old["world_sha256"]!=world["world_sha256"]
        or old["five_input_registry_sha256"]!=registration()["registry_sha256"]
        or old["source_manifest_sha256"]!=source["source_sha256"]
        or old["development_evaluation_context_groups_scored"]!=0
        or old["new_embargoed_worlds_scored"]!=0
        or old["original_confirmation_worlds_scored"]!=0):
        raise AssertionError("original native 900-step source world identity moved")
    if json.loads((old_root/"world_spec.json").read_text())!=world:
        raise AssertionError("source native teacher spec differs")
    if json.loads((old_root/"source_split_manifest.json").read_text())!=source:
        raise AssertionError("source split assignment changed")
    arms={}
    for arm in ("composed","additive"):
        arms[arm]=replay_one(old_root,world,catalog,arm,out)
    report={
        "schema":SCHEMA,
        "world_id":world["world_id"],"world_index":index,
        "world_sha256":world["world_sha256"],
        "old_900_run_id":44894980345,
        "old_900_receipt_sha256":file_sha(old_root/"source_pilot_receipt.json"),
        "new_1800_source_lock_sha256":lock["lock_sha256"],
        "new_five_input_registry_sha256":registration()["registry_sha256"],
        "old_original_four_input_confirmation_sha256":
            old_registry()["confirmation_full_spec_sha256"],
        "all_prior_900_source_weights_replayed_exactly":True,
        "in_sample_source_history_count":SOURCE_HISTORY_COUNT,
        "native_1800_optimizer_steps_per_arm":NEW_STEPS,
        "source_training_only":True,
        "new_dev_evaluation_groups_scored":0,
        "reserved_groups_scored":0,
        "embargoed_worlds_scored":0,
        "old_confirmation_worlds_scored":0,
        "new_source_AE_models_fitted":0,
        "new_causal_interventions_scored":0,
        "arms":arms,
        "original_900_pilot_outcomes_unchanged":True,
        "scientific_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"native1800_source_calibration.json").write_bytes(encoded(report))
    print("NEC03N_FROZEN_FIVE_INPUT_1800_SOURCE_TRAINING_WRITTEN")
    return report


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--world-index",type=int,required=True)
    p.add_argument("--old-root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args(argv)
    run(args.world_index,args.old_root,args.out)


if __name__=="__main__":
    main()

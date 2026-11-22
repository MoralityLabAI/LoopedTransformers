"""NEC-04F three-arm 32D Looper with shared intermediate XOR readout.

Exact original root12..17 NEC04A composed teacher. Primary matched pair:
one shared 32D->2 auxiliary head supervised at four focal-arrival visits
by either exact XOR prefixes or source-only hash-balanced shuffled controls.
Vanilla baseline carries same aux architecture with zero auxiliary loss.
All arms use identical original source batches, initialization and 900
continuous AdamW updates, with model+optimizer SHA-sealed before validation.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from mechinterp.neuron_communities.nec04f_oracles import (
    ROOTS, ARMS, STAGES, STEPS, SOURCE, VALIDATION, SCHEMA as INPUT_SCHEMA,
    code_identity, file_sha, lock, seed,
)
from research.pef07.pef07 import Looper, as_tensor, model_digest

SCHEMA = "NEC-04F/three-arm-recurrent-auxiliary-XOR-native-900steps/v1"
BATCH = 64
LR = .003
CLIP = 1.
ALPHA = .5
MAX_STEPS = 900
SAMPLER_XOR = 0x5DEECE66D
CHUNK = 128


class AuxLooper(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = Looper()
        self.aux_head = nn.Linear(32, 2)

    def forward(self, bits, visits: tuple[int, int, int, int]):
        if bits.ndim != 2 or bits.shape[1] != 12 or len(visits) != 4:
            raise ValueError("requires exact original 12-bit histories and four stage visits")
        if tuple(sorted(visits)) != visits or len(set(visits)) != 4:
            raise ValueError("original focal arrival visits not strictly increasing")
        memory = self.core.memory_init[None,:].expand(len(bits), -1)
        auxiliary = []
        for j in range(12):
            memory = self.core.step(memory, self.core.bit_embedding(bits[:,j]))
            if j+1 in visits:
                auxiliary.append(self.aux_head(memory))
        if len(auxiliary) != 4:
            raise AssertionError("one of four registered focal arrival visits missing")
        return self.core.head(memory), torch.stack(auxiliary, dim=1)


def load_original(index: int, folder: Path):
    if index not in ROOTS:
        raise PermissionError("NEC04F native pilot restricted to original calibration12..17")
    _,lock_sha = lock()
    ledger_file = folder/"nec04f_pretraining_oracles.json"
    ledger = json.loads(ledger_file.read_text())
    if (
        ledger["schema"] != INPUT_SCHEMA
        or ledger["lock_sha256"] != lock_sha
        or ledger["root_indices"] != list(ROOTS)
        or ledger["training_arms"] != list(ARMS)
        or ledger["checkpoint_steps"] != list(STEPS)
        or ledger["stage_k"] != list(STAGES)
        or ledger["optimizer_updates_before_preflight"] != 0
        or ledger["source_code_sha256_before_training"] != code_identity()
        or (os.environ.get("GITHUB_SHA") and
            ledger["preflight_source_git_sha"] != os.environ["GITHUB_SHA"])
        or ledger["FSI_01_gate"] != "REDESIGN_OR_PILOT_ONLY"
    ):
        raise AssertionError("original source/preregistered code/aux training policy changed")
    root = ledger["roots"][index-ROOTS[0]]
    source = folder/root["rows_file"]
    if (
        root["index"] != index
        or root["matched_seed"] != seed(index)
        or source.name != f"nec04f_root_{index}.jsonl"
        or file_sha(source) != root["rows_sha256"]
    ):
        raise AssertionError("registered original world root/source SHA mismatched")
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    if (
        len(rows) != SOURCE+VALIDATION
        or len({tuple(r["history_bits12"]) for r in rows}) != SOURCE+VALIDATION
        or any(r["split"]!="source_train" or r["root_index"]!=index for r in rows[:SOURCE])
        or any(r["split"]!="calibration_world_validation" or r["root_index"]!=index
               or "shuffled_aux" in r for r in rows[SOURCE:])
        or root["source_cases"]!=SOURCE or root["owned_validation_cases"]!=VALIDATION
    ):
        raise AssertionError("original source-vs-owned-validation split incomplete")
    return ledger,root,rows, file_sha(ledger_file)


def logits(model: AuxLooper, x: torch.Tensor, visits):
    model.eval()
    final,aux=[],[]
    with torch.no_grad():
        for n in range(0,len(x),CHUNK):
            a,b=model(x[n:n+CHUNK],visits)
            final.extend(a.detach().cpu().tolist())
            aux.extend(b.detach().cpu().tolist())
    if len(final)!=len(x) or len(aux)!=len(x):
        raise AssertionError("native root final or intermediate logits missing")
    return final,aux


def metrics(logits, truth):
    if len(logits)!=len(truth) or not truth:
        raise AssertionError("original native score incomplete")
    guesses=[int(row[1] > row[0]) for row in logits]
    pos=sum(truth)
    true_pos=sum(int(x==y==1) for x,y in zip(guesses,truth))
    true_neg=sum(int(x==y==0) for x,y in zip(guesses,truth))
    return {
        "cases":len(truth),
        "correct":true_pos+true_neg,
        "accuracy":(true_pos+true_neg)/len(truth),
        "balanced_accuracy":(
            .5*(true_pos/pos+true_neg/(len(truth)-pos))
            if 0<pos<len(truth) else None
        ),
        "true_positive_cases":pos,
        "correct_true_positives":true_pos,
        "correct_true_negatives":true_neg,
        "predicted_positive_cases":sum(guesses),
        "original_logits":logits,
        "class_predictions":guesses,
    }


def branch_metrics(logits, labels, rows):
    outputs={}
    for selector,name in ((1,"parity"),(0,"alternate")):
        where=[i for i,row in enumerate(rows) if row["mux_selector"]==selector]
        if not where:
            raise AssertionError("missing original calibration world MUX branch")
        outputs[name]=metrics([logits[i] for i in where],
                              [labels[i] for i in where])
    return outputs


def train(index: int, arm: str, root: dict, rows: list, out: Path,
          lock_sha: str):
    if arm not in ARMS:
        raise ValueError("unregistered native auxiliary task arm")
    start_seed=seed(index)
    torch.manual_seed(start_seed)
    model=AuxLooper().train()
    original_initial_sha=model_digest(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=0.)
    generator=torch.Generator(device="cpu").manual_seed(start_seed^SAMPLER_XOR)
    stage_visits=tuple(root["stage_visits"][str(k)] for k in STAGES)
    inputs=as_tensor([r["history_bits12"] for r in rows[:SOURCE]])
    original=torch.tensor([r["composed_label"] for r in rows[:SOURCE]],
                          dtype=torch.long)
    exact=torch.tensor([[r["xor_prefix"][str(k)] for k in STAGES]
                       for r in rows[:SOURCE]],dtype=torch.long)
    shuffled=torch.tensor([[r["shuffled_aux"][str(k)] for k in STAGES]
                          for r in rows[:SOURCE]],dtype=torch.long)
    if (inputs.shape!=(SOURCE,12) or original.shape!=(SOURCE,)
        or exact.shape!=(SOURCE,4) or shuffled.shape!=(SOURCE,4)):
        raise AssertionError("original source-only training label shape mismatch")
    label = shuffled if arm=="aux_shuffled" else exact
    loss_coefficient=0. if arm=="vanilla" else ALPHA
    first=None
    checkpoints=[]
    sampler_receipt=sha256()
    for step in range(1,MAX_STEPS+1):
        take=torch.randint(SOURCE,(BATCH,),generator=generator)
        sampler_receipt.update(take.numpy().tobytes())
        final,intermediate=model(inputs[take],stage_visits)
        full_loss=F.cross_entropy(final,original[take])
        aux_loss=sum(F.cross_entropy(intermediate[:,i],label[take,i])
                     for i in range(4))/4
        objective=full_loss+loss_coefficient*aux_loss
        if not bool(torch.isfinite(objective)):
            raise AssertionError("nonfinite original matched native XOR training loss")
        if first is None:
            first=float(objective.detach())
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),CLIP)
        if not bool(torch.isfinite(grad)):
            raise AssertionError("nonfinite matched native task/auxiliary gradient")
        optimizer.step()
        if step in STEPS:
            model_sha=model_digest(model)
            blob={
                "schema":SCHEMA, "root_index":index,"world_id":root["world_id"],
                "world_sha256":root["world_sha256"],
                "source_training_sha256":root["rows_sha256"],
                "original_composed_program_sha256":root["program_sha256"],
                "source_lock_sha256":lock_sha,
                "native_arm":arm,"native_seed":start_seed,
                "native_initial_weight_sha256":original_initial_sha,
                "model_weight_sha256":model_sha,
                "stage_counts":list(STAGES),"stage_visits":list(stage_visits),
                "step":step,"optimizer_updates":step,
                "optimizer":"AdamW","lr":LR,"weight_decay":0.,
                "gradient_clip_norm":CLIP,"batch_size":BATCH,
                "auxiliary_weight":loss_coefficient,
                "auxiliary_stage_loss_weighting":"equal four stages",
                "source_sampler_prefix_sha256":sampler_receipt.hexdigest(),
                "model_state":model.state_dict(),
                "optimizer_state":optimizer.state_dict(),
            }
            chk=out/f"nec04f.root{index}.{arm}.step{step}.pt"
            torch.save(blob,chk)
            checkpoints.append({
                "step":step,"file":chk.name,"sha256":file_sha(chk),
                "model_sha256":model_sha,
                "initial_weights_sha256":original_initial_sha,
                "sampler_prefix_sha256":sampler_receipt.hexdigest(),
                "first_objective":first,"last_objective":float(objective.detach()),
            })
            print("NEC04F_ORIGINAL_NATIVE_AUX_CHECKPOINT_SEALED",
                  index,arm,step,checkpoints[-1]["sha256"],flush=True)
    if len(checkpoints)!=len(STEPS):
        raise AssertionError("fixed original optimizer checkpoint schedule incomplete")
    manifest={
        "schema":SCHEMA,"root_index":index,"world_id":root["world_id"],
        "arm":arm,"source_lock_sha256":lock_sha,
        "source_training_sha256":root["rows_sha256"],
        "native_initial_weight_sha256":original_initial_sha,
        "same_original_minibatch_sampler_seed":start_seed^SAMPLER_XOR,
        "optimizer_resets":0,"optimizer_updates_total":MAX_STEPS,
        "all_native_model_and_optimizer_checkpoints_sealed_before_validation":True,
        "checkpoints":checkpoints,
    }
    (out/f"nec04f.root{index}.{arm}.checkpoints.json").write_text(
        json.dumps(manifest,sort_keys=True,indent=2)+"\n"
    )
    return manifest


def run_root(index:int, pretraining:Path, out:Path):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg,lock_sha=lock()
    ledger,ref,rows,pretraining_sha=load_original(index,pretraining)
    out.mkdir(parents=True,exist_ok=True)
    manifests={arm:train(index,arm,ref,rows,out,lock_sha) for arm in ARMS}
    initials={manifests[arm]["native_initial_weight_sha256"] for arm in ARMS}
    sampler_hashes={tuple(cp["sampler_prefix_sha256"]
                          for cp in manifests[arm]["checkpoints"]) for arm in ARMS}
    if len(initials)!=1 or len(sampler_hashes)!=1:
        raise AssertionError("matched model initial weights or source minibatches differ between arms")

    # ALL 3 arms and BOTH checkpoints are sealed before consulting the
    # original owned-calibration WORLD validation labels.
    x=as_tensor([r["history_bits12"] for r in rows])
    all_truth=[r["composed_label"] for r in rows]
    all_xor=[[r["xor_prefix"][str(k)] for k in STAGES] for r in rows]
    stages=tuple(ref["stage_visits"][str(k)] for k in STAGES)
    evidence={}
    for arm,manifest in manifests.items():
        results=[]
        for cp in manifest["checkpoints"]:
            chk=out/cp["file"]
            if file_sha(chk)!=cp["sha256"]:
                raise AssertionError("native checkpoint changed before validation scoring")
            saved=torch.load(chk,map_location="cpu",weights_only=True)
            model=AuxLooper().eval()
            model.load_state_dict(saved["model_state"],strict=True)
            if model_digest(model)!=cp["model_sha256"]:
                raise AssertionError("original native model weights changed before scoring")
            input_logits,aux_logits=logits(model,x,stages)
            split_scores={}
            for split,start,end in (
                ("source_train",0,SOURCE),
                ("owned_calibration_validation",SOURCE,SOURCE+VALIDATION),
            ):
                raw_input=input_logits[start:end]
                target=all_truth[start:end]
                row_slice=rows[start:end]
                full=metrics(raw_input,target)
                by_branch=branch_metrics(raw_input,target,row_slice)
                by_aux={
                    str(k):metrics(
                        [row[ordinal] for row in aux_logits[start:end]],
                        [row[ordinal] for row in all_xor[start:end]]
                    )
                    for ordinal,k in enumerate(STAGES)
                }
                split_scores[split]={
                    "native_final":full,
                    "native_final_by_mux_branch":by_branch,
                    "intermediate_true_prefix":by_aux,
                }
            results.append({
                "step":cp["step"],
                "original_checkpoint_sha256":cp["sha256"],
                "original_model_sha256":cp["model_sha256"],
                "source_and_owned_validation_scores":split_scores,
                "all_final_native_logits":input_logits,
                "all_intermediate_auxiliary_logits":aux_logits,
            })
        evidence[arm]={
            "checkpoint_seal_manifest_sha256":
                file_sha(out/f"nec04f.root{index}.{arm}.checkpoints.json"),
            "checkpoint_seal_manifest":manifest,
            "observations":results,
        }
    output={
        "schema":SCHEMA,
        "root_index":index,"world_id":ref["world_id"],
        "world_sha256":ref["world_sha256"],
        "difficulty_tier":ref["difficulty_tier"],
        "original_program_sha256":ref["program_sha256"],
        "original_source_validation_sha256":ref["rows_sha256"],
        "source_lock_sha256":lock_sha,
        "pretraining_oracle_ledger_sha256":pretraining_sha,
        "training_git_commit_sha":ledger["preflight_source_git_sha"],
        "source_cases":SOURCE,"owned_calibration_validation_cases":VALIDATION,
        "native_initial_weights_shared_all_arms":True,
        "source_batch_order_matched_all_arms":True,
        "optimizer_updates_per_arm":MAX_STEPS,
        "model_checkpoints_sealed_before_validation":True,
        "new_source_SAE_training_updates":0,
        "original_calibration_reserved_contexts_scored":0,
        "development_world_roots_materialized":0,
        "confirmation_world_roots_materialized":0,
        "old_nec03p_roots_scored":0,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
        "arms":evidence,
    }
    (out/"nec04f_native_root_result.json").write_text(
        json.dumps(output,sort_keys=True,indent=2,allow_nan=False)+"\n"
    )
    print("NEC04F_MATCHED_THREE_ARM_ROOT_NATIVE_RESULTS_SAVED",index,flush=True)
    return output


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root-index",required=True,type=int)
    p.add_argument("--pretraining",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    run_root(a.root_index,a.pretraining,a.out)


if __name__=="__main__":
    main()

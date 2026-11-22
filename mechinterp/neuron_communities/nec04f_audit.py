"""NEC-04F independent original-model checkpoint and source/validation replay.

Verifies all six native states per world (3 arms × 240/900 continuous
optimizer steps), full weight+optimizer SHA, root-specific matched source
sampler, independently regenerated composed/auxiliary truth and raw model
final+four intermediate logits on 2,560 source and 768 owned-calibration
examples. Independently scored, no second optimizer training.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

import torch

from mechinterp.neuron_communities import nec04a_boolean_gym as gym
from mechinterp.neuron_communities.nec04a_independent_oracle import oracle
from mechinterp.neuron_communities.nec04f_oracles import (
    ROOTS, ARMS, STAGES, STEPS, SOURCE, VALIDATION,
    code_identity,file_sha,lock,seed,
)
from mechinterp.neuron_communities.nec04f_independent_oracles import (
    independently_prefix,independently_shuffled_labels,
)
from mechinterp.neuron_communities.nec04f_native import (
    SCHEMA as NATIVE_SCHEMA, AuxLooper, logits as replay_logits,
)
from research.pef07.pef07 import as_tensor,model_digest

AUDIT_SCHEMA = "NEC-04F/independent-three-arm-native-optimizer-aux-logit-replay/v1"
SAMPLER_XOR = 0x5DEECE66D
TOL = 3e-4


def require(cond:bool,message:str):
    if not cond:
        raise AssertionError(message)


def independently_score(logits, truth, prior):
    require(len(logits)==len(truth)==prior["cases"],"original native scored-case denominator missing")
    pred=[]
    correct=tp=tn=0
    positive=sum(truth)
    assert all(x in (0,1) for x in truth)
    for i,(l,y) in enumerate(zip(logits,truth)):
        original = prior["original_logits"][i]
        require(len(l)==len(original)==2
                and max(abs(float(x)-float(z)) for x,z in zip(l,original))<=TOL,
                "source/owned-native logits changed at independent replay")
        p=int(l[1]>l[0])
        require(p==prior["class_predictions"][i],
                "original final or auxiliary class prediction differs")
        pred.append(p)
        tp+=int(p==y==1)
        tn+=int(p==y==0)
    correct=tp+tn
    require(
        correct==prior["correct"]
        and positive==prior["true_positive_cases"]
        and tp==prior["correct_true_positives"]
        and tn==prior["correct_true_negatives"]
        and sum(pred)==prior["predicted_positive_cases"]
        and abs(correct/len(truth)-prior["accuracy"])<1e-12,
        "root original clean final/intermediate correctness altered",
    )
    balanced=(.5*(tp/positive+tn/(len(truth)-positive))
              if 0<positive<len(truth) else None)
    if balanced is None:
        require(prior["balanced_accuracy"] is None,
                "original degenerate target class scores mutated")
    else:
        require(prior["balanced_accuracy"] is not None
                and abs(balanced-prior["balanced_accuracy"])<1e-12,
                "independent balanced accuracy differs")
    return {"cases":len(truth),"correct":correct,
            "accuracy":correct/len(truth),
            "balanced_accuracy":balanced,
            "target_positive_count":positive,
            "predicted_positive_count":sum(pred)}


def branch_indices(rows,selector):
    return [i for i,r in enumerate(rows) if r["mux_selector"]==selector]


def independent_native_sampler(seed_value, checkpoint_steps):
    gen=torch.Generator(device="cpu").manual_seed(seed_value^SAMPLER_XOR)
    h=sha256()
    outputs={}
    for t in range(1,max(checkpoint_steps)+1):
        batch=torch.randint(SOURCE,(64,),generator=gen)
        h.update(batch.numpy().tobytes())
        if t in checkpoint_steps:
            outputs[t]=h.hexdigest()
    return outputs


def audit(index:int, pretraining:Path, output:Path):
    require(index in ROOTS,"independent native auditor only original calibration12..17")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    _,lock_sha=lock()
    source_file=pretraining/"nec04f_pretraining_oracles.json"
    ledger=json.loads(source_file.read_text())
    ref=ledger["roots"][index-ROOTS[0]]
    rows_file=pretraining/ref["rows_file"]
    require(ledger["root_indices"]==list(ROOTS)
            and ledger["training_arms"]==list(ARMS)
            and ledger["checkpoint_steps"]==list(STEPS)
            and ledger["lock_sha256"]==lock_sha
            and ledger["source_code_sha256_before_training"]==code_identity()
            and ref["index"]==index
            and ref["matched_seed"]==seed(index)
            and file_sha(rows_file)==ref["rows_sha256"],
            "original matched code, source/teacher SHA or root identity changed")
    if os.environ.get("GITHUB_SHA"):
        require(ledger["preflight_source_git_sha"]==os.environ["GITHUB_SHA"],
                "independent replay on unregistered source revision")
    world,_=gym.calibration_world(index)
    require(world["world_sha256"]==ref["world_sha256"]
            and world["program_sha256"]==ref["program_sha256"],
            "previously untrained world program changed")
    rows=[json.loads(row) for row in rows_file.read_text().splitlines()]
    require(len(rows)==SOURCE+VALIDATION,
            "original source + calibration owned validation cases incomplete")
    truth=[]
    aux_true=[]
    for j,item in enumerate(rows):
        bits=item["history_bits12"]
        exact=int(oracle(world["canonical_program"],bits))
        independently=[independently_prefix(bits,world["focal_positions"],k)
                       for k in STAGES]
        require(item["composed_label"]==exact
                and [item["xor_prefix"][str(k)] for k in STAGES]==independently
                and item["mux_selector"]==bits[world["context_positions"][0]],
                "independent circuit or prefix auxiliary task label mismatch")
        if j<SOURCE:
            group,mask=divmod(j,32)
            assert all(item["shuffled_aux"][str(k)]==
                       independently_shuffled_labels(index,group,k)[mask]
                       for k in STAGES)
        else:
            require("shuffled_aux" not in item,
                    "shuffled source-only control labels leaked to validation")
        truth.append(exact)
        aux_true.append(independently)
    score_file=output/"nec04f_native_root_result.json"
    score=json.loads(score_file.read_text())
    require(
        score["schema"]==NATIVE_SCHEMA
        and score["root_index"]==index
        and score["world_id"]==world["world_id"]
        and score["world_sha256"]==ref["world_sha256"]
        and score["original_program_sha256"]==ref["program_sha256"]
        and score["source_lock_sha256"]==lock_sha
        and score["pretraining_oracle_ledger_sha256"]==file_sha(source_file)
        and score["original_source_validation_sha256"]==file_sha(rows_file)
        and score["training_git_commit_sha"]==ledger["preflight_source_git_sha"]
        and score["native_initial_weights_shared_all_arms"] is True
        and score["source_batch_order_matched_all_arms"] is True
        and score["optimizer_updates_per_arm"]==900
        and score["model_checkpoints_sealed_before_validation"] is True
        and score["new_source_SAE_training_updates"]==0
        and score["original_calibration_reserved_contexts_scored"]==0
        and score["development_world_roots_materialized"]==0
        and score["confirmation_world_roots_materialized"]==0
        and score["old_nec03p_roots_scored"]==0
        and score["FSI_01_gate"]=="REDESIGN_OR_PILOT_ONLY"
        and set(score["arms"])==set(ARMS),
        "original source-owned evaluation or paired model experiment mutated",
    )
    stage_visits=tuple(ref["stage_visits"][str(k)] for k in STAGES)
    expected_sampling=independent_native_sampler(seed(index),STEPS)
    x=as_tensor([r["history_bits12"] for r in rows])
    audits={}
    initial_hashes=set()
    for arm in ARMS:
        saved_arm=score["arms"][arm]
        manifest_file=output/f"nec04f.root{index}.{arm}.checkpoints.json"
        require(file_sha(manifest_file)==saved_arm["checkpoint_seal_manifest_sha256"],
                "model+optimizer checkpoint manifest byte SHA changed")
        manifest=json.loads(manifest_file.read_text())
        require(
            manifest==saved_arm["checkpoint_seal_manifest"]
            and manifest["schema"]==NATIVE_SCHEMA
            and manifest["arm"]==arm
            and manifest["root_index"]==index
            and manifest["source_lock_sha256"]==lock_sha
            and manifest["source_training_sha256"]==ref["rows_sha256"]
            and manifest["optimizer_resets"]==0
            and manifest["optimizer_updates_total"]==900
            and manifest["all_native_model_and_optimizer_checkpoints_sealed_before_validation"] is True
            and manifest["same_original_minibatch_sampler_seed"]==seed(index)^SAMPLER_XOR
            and len(manifest["checkpoints"])==len(STEPS)
            and len(saved_arm["observations"])==len(STEPS),
            "original complete continuous native optimizer checkpoint history invalid",
        )
        initial_hashes.add(manifest["native_initial_weight_sha256"])
        records=[]
        for ordinal,step in enumerate(STEPS):
            cp=manifest["checkpoints"][ordinal]
            obs=saved_arm["observations"][ordinal]
            chk=output/cp["file"]
            require(cp["file"]==f"nec04f.root{index}.{arm}.step{step}.pt"
                    and cp["step"]==step and obs["step"]==step
                    and file_sha(chk)==cp["sha256"]==obs["original_checkpoint_sha256"]
                    and cp["sampler_prefix_sha256"]==expected_sampling[step]
                    and cp["initial_weights_sha256"]==manifest["native_initial_weight_sha256"]
                    and cp["model_sha256"]==obs["original_model_sha256"],
                    "native model/optimizer SHA or matched minibatch sequence differs")
            checkpoint=torch.load(chk,map_location="cpu",weights_only=True)
            require(
                checkpoint["schema"]==NATIVE_SCHEMA
                and checkpoint["root_index"]==index
                and checkpoint["native_arm"]==arm
                and checkpoint["native_seed"]==seed(index)
                and checkpoint["world_sha256"]==ref["world_sha256"]
                and checkpoint["source_training_sha256"]==ref["rows_sha256"]
                and checkpoint["source_lock_sha256"]==lock_sha
                and checkpoint["step"]==checkpoint["optimizer_updates"]==step
                and checkpoint["stage_visits"]==list(stage_visits)
                and checkpoint["stage_counts"]==list(STAGES)
                and checkpoint["optimizer"]=="AdamW"
                and checkpoint["lr"]==.003 and checkpoint["weight_decay"]==0.
                and checkpoint["gradient_clip_norm"]==1.
                and checkpoint["batch_size"]==64
                and checkpoint["auxiliary_weight"]==(0. if arm=="vanilla" else .5)
                and checkpoint["source_sampler_prefix_sha256"]==expected_sampling[step],
                "original equal-budget native task/aux model checkpoint contract changed",
            )
            counters=[int(item["step"].item()) for item in
                      checkpoint["optimizer_state"]["state"].values()
                      if "step" in item]
            require(bool(counters) and all(n==step for n in counters),
                    "native AdamW state counters fail uninterrupted step history")
            model=AuxLooper().eval()
            model.load_state_dict(checkpoint["model_state"],strict=True)
            require(model_digest(model)==cp["model_sha256"]
                    ==checkpoint["model_weight_sha256"],
                    "native model or shared auxiliary head weight SHA mismatch")
            check_logits,check_aux=replay_logits(model,x,stage_visits)
            require(len(check_logits)==len(rows) and len(check_aux)==len(rows)
                    and len(obs["all_final_native_logits"])==len(rows)
                    and len(obs["all_intermediate_auxiliary_logits"])==len(rows),
                    "original source and validation complete native outputs missing")
            for case,(test,past,aux,aux_past) in enumerate(zip(
                check_logits,obs["all_final_native_logits"],check_aux,
                obs["all_intermediate_auxiliary_logits"]
            )):
                require(max(abs(u-v) for u,v in zip(test,past))<=TOL
                        and len(aux)==len(aux_past)==4
                        and max(abs(u-v) for row,p_row in zip(aux,aux_past)
                                for u,v in zip(row,p_row))<=TOL,
                        "independently replayed native 12-step or intermediate auxiliary logits changed")
            observations={}
            for split,start,end in (
                ("source_train",0,SOURCE),
                ("owned_calibration_validation",SOURCE,SOURCE+VALIDATION),
            ):
                snapshot=obs["source_and_owned_validation_scores"][split]
                original_truth=truth[start:end]
                original_slice=rows[start:end]
                final=check_logits[start:end]
                measured=independently_score(final,original_truth,snapshot["native_final"])
                branches={}
                for selector,name in ((1,"parity"),(0,"alternate")):
                    pos=branch_indices(original_slice,selector)
                    require(bool(pos),"missing native MUX branch")
                    branches[name]=independently_score(
                        [final[j] for j in pos],
                        [original_truth[j] for j in pos],
                        snapshot["native_final_by_mux_branch"][name],
                    )
                aux_measure={}
                for stage_idx,k in enumerate(STAGES):
                    aux_measure[str(k)]=independently_score(
                        [item[stage_idx] for item in check_aux[start:end]],
                        [item[stage_idx] for item in aux_true[start:end]],
                        snapshot["intermediate_true_prefix"][str(k)],
                    )
                observations[split]={
                    "original_composed_native":measured,
                    "native_by_original_MUX_branch":branches,
                    "auxiliary_true_intermediate_XOR":aux_measure,
                }
            records.append({
                "step":step,
                "original_checkpoint_sha256":cp["sha256"],
                "model_weight_sha256":cp["model_sha256"],
                "continuous_AdamW_counters_verified":len(counters),
                "original_matched_minibatch_sequence_sha256":expected_sampling[step],
                "all_source_owned_native_12step_and_aux_logits_replayed":True,
                "independent_metrics":observations,
            })
            print("NEC04F_INDEPENDENT_NATIVE_12STEP_FOUR_AUX_CHECKPOINT_PASS",
                  index,arm,step,observations["owned_calibration_validation"]
                  ["native_by_original_MUX_branch"]["parity"]["correct"],flush=True)
        audits[arm]=records
    require(len(initial_hashes)==1,
            "three original arms did not share exact model + aux head initial weights")
    output_audit={
        "schema":AUDIT_SCHEMA,
        "root_index":index,
        "world_id":world["world_id"],
        "world_sha256":world["world_sha256"],
        "original_pretraining_ledger_sha256":file_sha(source_file),
        "original_root_scored_receipt_sha256":file_sha(score_file),
        "source_lock_sha256":lock_sha,
        "original_source_and_validation_sha256":file_sha(rows_file),
        "all_six_native_model_optimizer_checkpoints_byte_sha256_verified":True,
        "all_six_native_full_and_four_auxiliary_logits_independently_replayed":True,
        "all_six_continuous_optimizer_state_counters_verified":True,
        "all_three_arm_native_initial_weights_identical":True,
        "all_three_arm_source_batch_order_sha_equal":True,
        "optimizer_trajectories_independently_retrained":False,
        "native_checkpoints_replayed":len(ARMS)*len(STEPS),
        "original_reserved_contexts_scored":0,
        "new_development_world_roots_generated":0,
        "new_confirmation_world_roots_generated":0,
        "old_nec03p_causal_sham_worlds_scored":0,
        "native_SAE_updates":0,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
        "arms":audits,
    }
    (output/"nec04f_independent_native_audit.json").write_text(
        json.dumps(output_audit,sort_keys=True,indent=2,allow_nan=False)+"\n"
    )
    return output_audit


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root-index",required=True,type=int)
    p.add_argument("--pretraining",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    audit(a.root_index,a.pretraining,a.out)


if __name__=="__main__":
    main()

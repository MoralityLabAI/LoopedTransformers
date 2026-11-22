"""NEC-04B independent scalar audit of pretraining target labels and split.

Pure stdlib and the precommitted NEC04A root registry. Recomputes a
fixed additive Boolean linear-threshold teacher independently of the
pilot task maker and checks all 6*(2560+768) teacher label pairs before
any Looper native training or validation-budget selection.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from mechinterp.neuron_communities import nec04a_boolean_gym as gym
from mechinterp.neuron_communities.nec04a_independent_oracle import oracle
from mechinterp.neuron_communities.nec04b_oracles import (
    SCHEMA,PILOT,ARMS,BUDGETS,SOURCE,VALIDATION,EXPECTED_CORPUS_SHA,
    EXPECTED_REGISTRY,verify_lock,file_sha,
)

SCHEMA_AUDIT="NEC-04B/independent-pretraining-teacher-label-audit/v1"


def byte_hash(obj):
    return sha256(json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def independently_linear_spec(index,world):
    secret=sha256(b"NEC04B/linear-threshold-additive-oracle/v1|"+
                  gym.seed_for_identity("calibration",index)).digest()
    focal=set(world["focal_positions"])
    vals=[]
    for k in range(12):
        blob=sha256(secret+b"/weight/"+bytes([k])).digest()
        mags=[1,2,3] if k in focal else [1,2]
        vals.append(mags[blob[0]%len(mags)]*(-1 if blob[1]%2 else 1))
    scores=[]
    for history in range(4096):
        bits=[(history>>k)&1 for k in range(12)]
        scores.append(sum(w*(2*b-1) for w,b in zip(vals,bits)))
    candidates=[]
    for cut in range(min(scores)-1,max(scores)+1):
        ones=sum(v>cut for v in scores)
        candidates.append((abs(ones-2048),abs(cut),cut,ones))
    _,_,cut,count=min(candidates)
    threshold_twice=2*cut+1
    truth=0
    for i,score in enumerate(scores):
        if 2*score>threshold_twice:truth|=1<<i
    return vals,threshold_twice,count,sha256(truth.to_bytes(512,"little")).hexdigest(),truth


def audit(folder:Path,out:Path):
    cfg,lock_sha=verify_lock()
    ledger_path=folder/"nec04b_pretraining_oracles.json"
    ledger=json.loads(ledger_path.read_text())
    if (ledger["schema"]!=SCHEMA or ledger["source_lock_sha256"]!=lock_sha
        or ledger["original_calibration_source_sha256"]!=EXPECTED_CORPUS_SHA
        or ledger["original_registry_sha256"]!=EXPECTED_REGISTRY
        or ledger["root_indices"]!=list(PILOT)
        or ledger["native_arms"]!=list(ARMS)
        or ledger["training_step_budgets"]!=list(BUDGETS)
        or ledger["calibration_world_source_histories_per_arm"]!=SOURCE
        or ledger["calibration_world_owned_validation_histories_per_arm"]!=VALIDATION
        or ledger["model_optimizer_steps_before_preflight"]!=0
        or ledger["original_calibration_reserved_context_groups_scored"]!=0
        or ledger["development_world_specs_instantiated"]!=0
        or ledger["confirmation_world_specs_instantiated"]!=0
        or ledger["newly_recruited_world_causal_sham_scored"]!=0
        or ledger["FSI_01_gate"]!="REDESIGN_OR_PILOT_ONLY"
        or len(ledger["roots"])!=len(PILOT)):
        raise AssertionError("frozen source/owned-validation pilot teacher roster differs")
    results=[]
    for index,ref in enumerate(ledger["roots"]):
        world,_=gym.calibration_world(index)
        if (ref["root_index"]!=index or
            ref["world_id"]!=world["world_id"] or
            ref["world_sha256"]!=world["world_sha256"] or
            ref["program_sha256"]!=world["program_sha256"] or
            ref["original_truth_4096_sha256"]!=world["truth_table_4096_sha256"] or
            ref["pilot_file"]!=f"nec04b_pilot_root_{index}.jsonl" or
            ref["tier"]!=world["difficulty_tier"] or
            ref["source_rows"]!=SOURCE or ref["validation_rows"]!=VALIDATION):
            raise AssertionError("original precommitted world root changed")
        f=folder/ref["pilot_file"]
        if file_sha(f)!=ref["pilot_file_sha256"]:
            raise AssertionError("pilot teacher source/validation input blob changed")
        raw=f.read_text().splitlines()
        if len(raw)!=SOURCE+VALIDATION:
            raise AssertionError("pilot original source/validation contexts truncated")
        weights,threshold,count,truthhash,truth=independently_linear_spec(index,world)
        add=ref["additive_spec"]
        if (add["root_index"]!=index or add["world_id"]!=world["world_id"] or
            add["signed_weights_12"]!=weights or add["threshold_twice"]!=threshold
            or add["positive_count_full4096"]!=count
            or add["truth_table_4096_sha256"]!=truthhash or
            add["target_spec_sha256"]!=byte_hash(
                {k:v for k,v in add.items() if k!="target_spec_sha256"}
            ) or not .35<=add["class_balance_full4096"]<=.65):
            raise AssertionError("independent original additive linear oracle SHA mismatch")
        source_contexts,parts=gym.contexts_for_world(index)
        validation_contexts=parts["development"]
        if (len(source_contexts)!=80 or len(validation_contexts)!=24 or
            len(parts["reserved"])!=24 or
            set(source_contexts)&set(validation_contexts) or
            set(parts["reserved"])&set(validation_contexts) or
            set(source_contexts)&set(parts["reserved"])):
            raise AssertionError("source contexts leaked to owned calibration-world validation")
        source_ids=set()
        validation_ids=set()
        for k,line in enumerate(raw):
            row=json.loads(line)
            expected_source=k<SOURCE
            group=k//32
            mask=k%32
            split="source_train" if expected_source else "calibration_world_validation"
            if not expected_source:
                group=80+(k-SOURCE)//32
            if (row["world_id"]!=world["world_id"] or row["root_index"]!=index
                or row["split"]!=split or row["context_group"]!=group
                or row["focal_mask"]!=mask):
                raise AssertionError("pilot source-validation block ownership changed")
            context=(source_contexts[group] if expected_source
                     else validation_contexts[group-80])
            original=[0]*12
            for bit,pos in enumerate(world["focal_positions"]):
                original[pos]=(mask>>bit)&1
            for bit,pos in enumerate(world["context_positions"]):
                original[pos]=context[bit]
            bits=row["history_bits12"]
            if bits!=original:
                raise AssertionError("original source/validation history differs from independent construction")
            composed=int(oracle(world["canonical_program"],bits))
            idx=sum(v<<j for j,v in enumerate(bits))
            linear=(truth>>idx)&1
            if (row["composed_label"]!=composed
                or row["additive_label"]!=linear):
                raise AssertionError("independent composed/additive oracle label disagree")
            marker=tuple(bits)
            if expected_source:
                if marker in source_ids:raise AssertionError("duplicated source mask history")
                source_ids.add(marker)
            else:
                if marker in validation_ids:raise AssertionError("duplicated validation mask history")
                validation_ids.add(marker)
        if (len(source_ids)!=SOURCE or len(validation_ids)!=VALIDATION
            or source_ids&validation_ids):
            raise AssertionError("repeated or leaked original root data")
        results.append({
            "root_index":index,"world_id":world["world_id"],
            "world_sha256":world["world_sha256"],
            "source_rows":SOURCE,"validation_rows":VALIDATION,
            "additive_truth_table_sha256":truthhash,
            "source_validation_sha256":file_sha(f),
            "native_teacher_arms":["composed","additive"],
        })
    result={
        "schema":SCHEMA_AUDIT,
        "original_pretraining_ledger_sha256":file_sha(ledger_path),
        "source_lock_sha256":lock_sha,
        "original_24_calibration_root_registry_sha256":EXPECTED_REGISTRY,
        "pilot_roots_verified":len(results),
        "source_label_rows_verified":SOURCE*len(results)*2,
        "owned_validation_label_rows_verified":VALIDATION*len(results)*2,
        "independent_scalar_composed_oracle":True,
        "independent_scalar_linear_additive_oracle":True,
        "whole_32_mask_context_ownership_verified":True,
        "original_reserved_context_labels_scored":0,
        "new_development_root_programs_generated":0,
        "new_confirmation_root_programs_generated":0,
        "old_recruited_or_confirmation_roots_scored":0,
        "native_training_steps_before_oracle_audit":0,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
        "roots":results,
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"nec04b_independent_teacher_oracle_audit.json").write_text(
        json.dumps(result,sort_keys=True,indent=2)+"\n"
    )
    print("NEC04B_PRETRAINING_INDEPENDENT_SOURCE_VALIDATION_ORACLES_PASS",
          len(results),result["owned_validation_label_rows_verified"])
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--folder",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    audit(a.folder,a.out)


if __name__=="__main__":
    main()

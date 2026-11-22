"""NEC-04B: frozen six-calibration-root native task-oracle materialization.

Pure stdlib: reconstruct old NEC04A calibration-world SOURCE labels from
the SHA-pinned corpus, separately materialize exactly 24 owned validation
contexts in calibration WORLDS 0..5, and construct a deterministic additive
linear-threshold Boolean teacher before any native checkpoint training.

The 24 NEC04A development and 16 confirmation WORLD roots remain
uninstantiated; the 24 reserved contexts per calibration world stay sealed.
"""
from __future__ import annotations

import argparse
from hashlib import sha256,sha1
import json
from pathlib import Path

from mechinterp.neuron_communities import nec04a_boolean_gym as gym
from mechinterp.neuron_communities.nec04a_independent_oracle import oracle

SCHEMA="NEC-04B/frozen-calibration-pilot-teacher-oracles/v1"
LOCK_NAME="NEC04B_PILOT_LOCK.json"
PILOT=tuple(range(6))
ARMS=("composed","additive")
SOURCE=80*32
VALIDATION=24*32
BUDGETS=(120,240,480,900)
EXPECTED_CORPUS_SHA="82c80643d3c99871913f5228f1a1cde1eec4484787c1784872b508349793e9e0"
EXPECTED_REGISTRY="88bf58d32fdf15a1270986c238afb3def1a7e6371d3eee64c4edd264c9cf041e"
WEIGHT_DOMAIN=b"NEC04B/linear-threshold-additive-oracle/v1|"


def json_bytes(obj):
    return (json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def file_sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def verify_lock():
    p=Path(__file__).with_name(LOCK_NAME)
    v=json.loads(p.read_text())
    exact={
        "schema":"NEC-04B/pretrained-calibration-oracle-and-Looper-pilot-registration/v1",
        "original_64_root_registry_sha256":EXPECTED_REGISTRY,
        "original_source_calibration_corpus_sha256":EXPECTED_CORPUS_SHA,
        "pilot_calibration_root_indices":list(PILOT),
        "pilot_model_teacher_arms":list(ARMS),
        "train_context_groups":80,
        "validation_owned_context_groups":24,
        "reserved_context_groups_scored":0,
        "native_training_steps":list(BUDGETS),
        "additive_truth_rule":"signed deterministic integer linear 12-bit score, root fixed nonzero weights and half-integer threshold chosen closest global balanced 50pct class frequency, tie smaller threshold",
        "model_architecture":"research.pef07.pef07.Looper 32D twelve-step recurrent 2-logit head",
        "native_optimizer":"AdamW lr=0.003 weight_decay=0.0 grad_clip_norm=1.0 batch64, no reset at checkpoints",
        "native_batch_size":64,
        "native_steps_max":900,
        "native_model_training_seeds":"1 + SHA256 domain-root-seed-arm first8 big endian modulo (2^31-2)",
        "source_context_training_only":True,
        "validation_scored_only_after_four_per_arm_checkpoints_sealed":True,
        "development_world_specs_materialized":0,
        "confirmation_world_specs_materialized":0,
        "calibration_reserved_contexts_scored":0,
        "old_NEC03Z_source_group_56_63_scored":0,
        "old_NEC03P_recruited_worlds_scored":0,
        "old_32_five_input_embargoed_worlds_scored":0,
        "old_64_four_input_confirmation_worlds_scored":0,
        "source_SAE_training_started":False,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    if any(v.get(k)!=val for k,val in exact.items()):
        raise AssertionError("NEC04B original world roots, teacher or native pilot changed")
    repo=Path(__file__).resolve().parents[2]
    for name,expected in v["git_blob_sha1"].items():
        b=(repo/name).read_bytes()
        actual=sha1(b"blob "+str(len(b)).encode()+b"\x00"+b).hexdigest()
        if actual!=expected:
            raise AssertionError("NEC04B registered source code changed "+name)
    return v,file_sha(p)


def _rng_bytes(index):
    if type(index)!=int or index not in PILOT:
        raise PermissionError("NEC04B only original six committed calibration roots")
    return sha256(WEIGHT_DOMAIN+gym.seed_for_identity("calibration",index)).digest()


def additive_weights(index):
    s=_rng_bytes(index)
    world,_=gym.calibration_world(index)
    focal=set(world["focal_positions"])
    context=set(world["context_positions"])
    weights=[]
    for i in range(12):
        b=sha256(s+b"/weight/"+bytes([i])).digest()
        magnitudes=(1,2,3) if i in focal else (1,2)
        value=magnitudes[b[0]%len(magnitudes)]
        if b[1]&1:
            value=-value
        weights.append(value)
    assert all(v!=0 for v in weights)
    assert set(focal)|set(context)==set(range(12))
    return weights


def linear_score(bits,weights):
    if (len(bits)!=12 or len(weights)!=12 or
        any(type(v)!=int or v not in (0,1) for v in bits)):
        raise ValueError("requires exact 12-bit original history")
    return sum(w*(2*v-1) for w,v in zip(weights,bits))


def additive_spec(index):
    weight=additive_weights(index)
    scores=[]
    for x in range(4096):
        bits=[(x>>j)&1 for j in range(12)]
        scores.append(linear_score(bits,weight))
    choices=[]
    for cut in range(min(scores)-1,max(scores)+1):
        pos=sum(score>cut for score in scores)
        choices.append((abs(pos-2048),abs(cut),cut,pos))
    _,_,cut,pos=min(choices)
    if not .35<=pos/4096<=.65:
        raise AssertionError("frozen root affine teacher not sufficiently balanced")
    threshold_twice=2*cut+1
    truth=sum((int(2*score>threshold_twice)<<k for k,score in enumerate(scores)))
    world,_=gym.calibration_world(index)
    original=gym.metrics(truth,world["focal_positions"],world["context_positions"])
    if (len(original["essential_focal_bits"])!=5 or
        len(original["essential_context_bits"])<3):
        raise AssertionError("fixed additive oracle lacks original focal/context dependence")
    record={
        "schema":SCHEMA,
        "world_id":world["world_id"],
        "root_index":index,
        "source_world_sha256":world["world_sha256"],
        "target_type":"signed_linear_threshold",
        "signed_weights_12":weight,
        "score_encoding":"sum(weight[i]*(2*history[i]-1))",
        "positive_if":"2*score>threshold_twice",
        "threshold_twice":threshold_twice,
        "positive_count_full4096":pos,
        "class_balance_full4096":pos/4096,
        "essential_focal_bits":original["essential_focal_bits"],
        "essential_context_bits":original["essential_context_bits"],
        "truth_table_4096_sha256":sha256(truth.to_bytes(512,"little")).hexdigest(),
    }
    record["target_spec_sha256"]=gym.digest(record)
    return record,truth


def additive_label(bits,spec):
    if spec["schema"]!=SCHEMA or spec["target_type"]!="signed_linear_threshold":
        raise AssertionError("unknown root-specific additive oracle type")
    return int(2*linear_score(bits,spec["signed_weights_12"])>spec["threshold_twice"])


def root_checkpoint_seed(index,arm):
    _rng_bytes(index)
    if arm not in ARMS:
        raise ValueError("unregistered source native model arm")
    raw=sha256(b"NEC04B/native-training-seed/v1|"+
               gym.seed_for_identity("calibration",index)+
               arm.encode()).digest()
    return 1+int.from_bytes(raw[:8],"big")%(2**31-2)


def rows_for_owned_validation(index,world,addspec):
    # ONLY 24 owned developmental-context blocks of CALIBRATION world root.
    # Never materializes a new developmental/confirmation WORLD program.
    _,parts=gym.contexts_for_world(index)
    original_context=list(parts["development"])
    if len(original_context)!=24:
        raise AssertionError("original calibration-world validation owned contexts changed")
    f=world["focal_positions"]
    c=world["context_positions"]
    for group,bits in enumerate(original_context):
        for mask in range(32):
            history=[0]*12
            for k,pos in enumerate(f):
                history[pos]=(mask>>k)&1
            for k,pos in enumerate(c):
                history[pos]=bits[k]
            yield {
                "world_id":world["world_id"],"root_index":index,
                "split":"calibration_world_validation",
                "context_group":80+group,"focal_mask":mask,
                "history_bits12":history,
                "composed_label":int(oracle(world["canonical_program"],history)),
                "additive_label":additive_label(history,addspec),
            }


def materialize_original_pilot(original:Path,out:Path):
    cfg,lock_hash=verify_lock()
    base=original/"nec04a-calibration"
    reg=json.loads((base/"nec04a_world_root_commitments.json").read_text())
    manifest=json.loads((base/"nec04a_calibration_source_manifest.json").read_text())
    allworlds=[json.loads(x) for x in
               (base/"nec04a_calibration_worlds.jsonl").read_text().splitlines()]
    corpus_file=base/"nec04a_calibration_source_train.jsonl"
    if (reg["registration_sha256"]!=EXPECTED_REGISTRY
        or manifest["source_corpus_sha256"]!=EXPECTED_CORPUS_SHA
        or file_sha(corpus_file)!=EXPECTED_CORPUS_SHA
        or file_sha(base/"nec04a_calibration_worlds.jsonl")!=manifest["calibration_world_specs_sha256"]
        or len(allworlds)!=24):
        raise AssertionError("original NEC04A oracle/source corpus ancestry mismatch")
    data=[json.loads(row) for row in corpus_file.read_text().splitlines()]
    if len(data)!=61440:
        raise AssertionError("original calibration source corpus truncated")
    out.mkdir(parents=True,exist_ok=True)
    roots=[]
    for index in PILOT:
        world=allworlds[index]
        expected_world,truth=gym.calibration_world(index)
        if world!=expected_world or world["root_index"]!=index:
            raise AssertionError("original calibration-world program altered")
        addspec,addtruth=additive_spec(index)
        if world["world_sha256"]!=addspec["source_world_sha256"]:
            raise AssertionError("root additive target bound to wrong source model world")
        source_rows=[]
        for row in data[index*SOURCE:(index+1)*SOURCE]:
            if (row["world_id"]!=world["world_id"] or
                row["world_sha256"]!=world["world_sha256"] or
                row["split"]!="source_train" or
                row["cohort"]!="calibration"
                or not 0<=row["source_context_group"]<80):
                raise AssertionError("original source row ownership changed")
            bits=row["history_bits12"]
            i=sum(b<<k for k,b in enumerate(bits))
            if int((truth>>i)&1)!=row["label"]:
                raise AssertionError("original composed source label mismatch")
            source_rows.append({
                "world_id":world["world_id"],"root_index":index,
                "split":"source_train",
                "context_group":row["source_context_group"],
                "focal_mask":row["focal_mask"],
                "history_bits12":bits,
                "composed_label":row["label"],
                "additive_label":int((addtruth>>i)&1),
            })
        if len(source_rows)!=SOURCE:
            raise AssertionError("original source case count changed")
        val=list(rows_for_owned_validation(index,world,addspec))
        if len(val)!=VALIDATION:
            raise AssertionError("original calibration world validation group incomplete")
        source_keys={tuple(row["history_bits12"]) for row in source_rows}
        val_keys={tuple(row["history_bits12"]) for row in val}
        if (len(source_keys)!=SOURCE or len(val_keys)!=VALIDATION or
            source_keys&val_keys):
            raise AssertionError("calibration-world source/validation context leakage")
        path=out/f"nec04b_pilot_root_{index}.jsonl"
        blob=b"".join(json_bytes(x) for x in source_rows+val)
        path.write_bytes(blob)
        roots.append({
            "root_index":index,"world_id":world["world_id"],
            "tier":world["difficulty_tier"],
            "world_sha256":world["world_sha256"],
            "program_sha256":world["program_sha256"],
            "original_truth_4096_sha256":world["truth_table_4096_sha256"],
            "additive_spec":addspec,
            "source_rows":SOURCE,"validation_rows":VALIDATION,
            "pilot_file":path.name,"pilot_file_sha256":sha256(blob).hexdigest(),
            "native_seeds":{arm:root_checkpoint_seed(index,arm) for arm in ARMS},
        })
    ledger={
        "schema":SCHEMA,
        "source_lock_sha256":lock_hash,
        "original_calibration_source_sha256":EXPECTED_CORPUS_SHA,
        "original_registry_sha256":EXPECTED_REGISTRY,
        "root_indices":list(PILOT),
        "native_arms":list(ARMS),
        "training_step_budgets":list(BUDGETS),
        "calibration_world_source_histories_per_arm":SOURCE,
        "calibration_world_owned_validation_histories_per_arm":VALIDATION,
        "original_calibration_reserved_context_groups_scored":0,
        "development_world_specs_instantiated":0,
        "confirmation_world_specs_instantiated":0,
        "model_optimizer_steps_before_preflight":0,
        "newly_recruited_world_causal_sham_scored":0,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
        "roots":roots,
    }
    (out/"nec04b_pretraining_oracles.json").write_bytes(json_bytes(ledger))
    print("NEC04B_PRETRAINING_ORACLES_PINNED",
          len(roots),len(roots)*(SOURCE+VALIDATION),
          sha256(json_bytes(ledger)).hexdigest())
    return ledger


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    materialize_original_pilot(a.original,a.out)


if __name__=="__main__":
    main()

"""NEC-03L: new FIVE-input Boolean-composition source-world registry.

Structurally separate from the old 4-input PEF/NEC03G world family.
Ninety-six prospective DEVELOPMENT-only roots, thirty-two additional
execution-embargoed roots. No training, scoring, heldout or confirmation
run exists here. The 64 NEC03G confirmation-world commitments are pinned
unchanged. Every 12-bit history appears once and only once per world.

5 focal bits -> 32 counterfactuals per context; the 7 remaining bits
yield 128 context groups: 80 source parameter train, 24 development
evaluation, 24 reserved. Source training is split 40/40 SAE fit/ranking.
"""
from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
from itertools import combinations, product
import json
from pathlib import Path

SCHEMA="NEC-03L/5-input-bool-program-source-v1"
SALT=b"nec03l-five-bit-programs-20261121-v1"
FIVE_INPUTS=5
TOTAL_BITS=12
FOCAL_MASKS=1<<FIVE_INPUTS
SOURCE_TRAIN_CONTEXTS=80
SOURCE_FIT_CONTEXTS=40
SOURCE_RANK_CONTEXTS=40
DEV_EVAL_CONTEXTS=24
RESERVED_CONTEXTS=24
DEVELOPMENT_ROOT_COUNT=96
EMBARGOED_ROOT_COUNT=32
TOTAL_ROOTS=DEVELOPMENT_ROOT_COUNT+EMBARGOED_ROOT_COUNT
TRAIN_STEPS=900
SAE_STEPS=480
ORIGINAL_REGISTRY_SHA256="8816304a7db020bd0ad016acca7d9195dfd4633e3316a17bd8e4b8e912fd25b4"
ORIGINAL_CONFIRMATION_SHA256="670ea6e1bb4fe86ab9691d0af4800d5623fe64e489ca7555561a80c9739f1f39"
OPS=("and","or","xor","nand","nor","xnor")
STRUCTURES=("paired-then-fifth","left-triple-plus-right-pair","right-triple-plus-left-single")


def encoded(obj):
    return (json.dumps(obj,sort_keys=True,separators=(",",":"))+"\n").encode()


def digest(obj):
    return sha256(encoded(obj)).hexdigest()


def op(a:int,b:int,name:str):
    if name=="and": return a&b
    if name=="or": return a|b
    if name=="xor": return a^b
    if name=="nand": return 1-(a&b)
    if name=="nor": return 1-(a|b)
    if name=="xnor": return 1-(a^b)
    raise ValueError("unsupported five-input Boolean gate")


def program_truth(structure:str,gates,negation:int):
    """32-bit little-endian truth table for an explicit five-input program."""
    if (structure not in STRUCTURES or len(gates)!=4
        or any(g not in OPS for g in gates) or negation not in range(32)):
        raise ValueError("invalid five-input grammar/program")
    table=0
    a0,a1,a2,a3=gates
    for mask in range(32):
        a,b,c,d,e=[((mask>>j)&1)^((negation>>j)&1) for j in range(5)]
        if structure=="paired-then-fifth":
            value=op(op(op(a,b,a0),op(c,d,a1),a2),e,a3)
        elif structure=="left-triple-plus-right-pair":
            value=op(op(op(a,b,a0),c,a1),op(d,e,a2),a3)
        else:
            value=op(a,op(op(b,c,a0),op(d,e,a1),a2),a3)
        table |=int(value)<<mask
    return table


def essential(table:int):
    if not 0<=table<(1<<32):
        raise ValueError("need a 32-bit truth function")
    return all(any(((table>>mask)&1)!=((table>>(mask^(1<<bit)))&1)
                   for mask in range(32)) for bit in range(5))


@lru_cache(maxsize=1)
def _canonical_teacher_programs():
    unique={}
    for structure,g1,g2,g3,g4,invert in product(
            STRUCTURES,OPS,OPS,OPS,OPS,range(32)):
        truth=program_truth(structure,(g1,g2,g3,g4),invert)
        if (not 10<=truth.bit_count()<=22 or not essential(truth)
            or truth==additive_control_truth(truth)):
            continue
        data={"structure":structure,"gates":[g1,g2,g3,g4],
              "literal_inversion_mask":invert,
              "truth_table_32":truth}
        if truth not in unique or encoded(data)<encoded(unique[truth]):
            unique[truth]=data
    ordered=list(unique.values())
    ordered.sort(key=lambda row:sha256(
        SALT+b"|5-input-teacher|"+
        str(row["truth_table_32"]).encode()).digest())
    if len(ordered)<TOTAL_ROOTS:
        raise AssertionError("five-input balanced grammar yields fewer than 128 distinct essential teachers")
    return tuple(ordered[:TOTAL_ROOTS])


@lru_cache(maxsize=1)
def _focal_positions():
    positions=[p for p in combinations(range(TOTAL_BITS),FIVE_INPUTS)
               if p[0]<=3 and p[-1]>=8 and sum(x<=6 for x in p)>=3
               and sum(x>=5 for x in p)>=2]
    positions.sort(key=lambda p:sha256(
        SALT+b"|5-input-focal|"+bytes(p)).digest())
    if len(positions)<TOTAL_ROOTS:
        raise AssertionError("insufficient five-input chronological focal arrangements")
    return tuple(positions[:TOTAL_ROOTS])


def worlds():
    programs=_canonical_teacher_programs()
    positions=_focal_positions()
    result=[]
    for i in range(TOTAL_ROOTS):
        cohort="development" if i<DEVELOPMENT_ROOT_COUNT else "embargoed"
        idx=i if cohort=="development" else i-DEVELOPMENT_ROOT_COUNT
        world_id=f"{cohort[:3]}5-{idx:03d}"
        seed=1+int.from_bytes(sha256(
            SALT+b"|native-checkpoint|"+world_id.encode()).digest()[:4],
            "big")%2100000000
        data={
            "schema":SCHEMA,"world_id":world_id,"world_index":i,
            "cohort":cohort,
            "program":dict(programs[i]),
            "focal_positions":list(positions[i]),
            "truth_table_32":programs[i]["truth_table_32"],
            "context_salt":f"nec03l-seven-bit-context:{world_id}",
            "native_model_seed":seed,
            "native_training_steps":TRAIN_STEPS,
            "source_sae_steps":SAE_STEPS,
            "additive_control_weights":[1,2,4,8,16],
            "source_training_context_groups":SOURCE_TRAIN_CONTEXTS,
            "sae_fit_context_groups":SOURCE_FIT_CONTEXTS,
            "source_feature_rank_groups":SOURCE_RANK_CONTEXTS,
            "development_context_groups":DEV_EVAL_CONTEXTS,
            "reserved_context_groups":RESERVED_CONTEXTS,
            "focal_flips_per_context":FOCAL_MASKS}
        data["world_sha256"]=digest(data)
        result.append(data)
    if (len({w["truth_table_32"] for w in result})!=TOTAL_ROOTS
        or len({tuple(w["focal_positions"]) for w in result})!=TOTAL_ROOTS
        or len({w["native_model_seed"] for w in result})!=TOTAL_ROOTS):
        raise AssertionError("five-input source-world independence constraint failed")
    return result


def additive_control_truth(truth:int):
    n=truth.bit_count()
    if not 0<n<32:
        raise ValueError("degenerate five-input teacher")
    # Unique binary weighted scores ensure exact class-prior matching.
    return sum(1<<mask for mask in range(32-n,32))


def labels(bits,world,*,control=False):
    import torch
    if (bits.ndim!=2 or bits.shape[1]!=TOTAL_BITS
        or len(world["focal_positions"])!=FIVE_INPUTS):
        raise ValueError("need 12-bit sequence with five focal positions")
    index=sum((bits[:,pos].long()<<i
               for i,pos in enumerate(world["focal_positions"])))
    truth=additive_control_truth(world["truth_table_32"]) if control else world["truth_table_32"]
    return ((torch.full_like(index,truth)>>index)&1).long()


def context_catalog(world):
    if world["cohort"]!="development":
        raise PermissionError("embargoed five-input world cannot be allocated to source/evaluation")
    pos=list(world["focal_positions"])
    others=[p for p in range(TOTAL_BITS) if p not in pos]
    if len(others)!=7 or len(set(pos))!=5:
        raise AssertionError("invalid five-input focal support")
    contexts=[tuple((mask>>i)&1 for i in range(7))
              for mask in range(1<<7)]
    salt=world["context_salt"].encode()
    contexts.sort(key=lambda v:(sha256(salt+bytes(v)).digest(),bytes(v)))
    partitions={"train":SOURCE_TRAIN_CONTEXTS,
                "development":DEV_EVAL_CONTEXTS,
                "reserved":RESERVED_CONTEXTS}
    out={key:[] for key in partitions}
    owners={}
    offset=0
    for name,n in partitions.items():
        for context in contexts[offset:offset+n]:
            label="".join(map(str,context))
            owners[label]=name
            for focal in range(FOCAL_MASKS):
                bits=[0]*TOTAL_BITS
                for j,p in enumerate(others):
                    bits[p]=context[j]
                for j,p in enumerate(pos):
                    bits[p]=(focal>>j)&1
                out[name].append(tuple(bits))
        offset+=n
    if (offset!=128 or
        len(set.union(*(set(rows) for rows in out.values())))!=4096):
        raise AssertionError("five-input task contexts overlap across split")
    manifest={
        "schema":SCHEMA,
        "world_id":world["world_id"],
        "world_sha256":world["world_sha256"],
        "context_positions":others,
        "focal_positions":pos,
        "contexts_ordered":["".join(map(str,x)) for x in contexts],
        "partition_counts":partitions,
        "context_group_assignment":owners,
        "source_fit_context_count":SOURCE_FIT_CONTEXTS,
        "source_rank_context_count":SOURCE_RANK_CONTEXTS,
        "whole_focal_counterfactual_block_stays_with_context":True,
        "development_and_reserved_contexts_not_used_for_source_selection":True}
    manifest["source_sha256"]=digest(manifest)
    return out,manifest


def registration():
    all_worlds=worlds()
    development=all_worlds[:DEVELOPMENT_ROOT_COUNT]
    embargoed=all_worlds[DEVELOPMENT_ROOT_COUNT:]
    out={
        "schema":SCHEMA,
        "new_independent_five_input_function_family":True,
        "old_NECO3G_worlds_untouched":True,
        "original_four_input_world_registry_sha256":ORIGINAL_REGISTRY_SHA256,
        "original_64_confirmation_world_spec_sha256":ORIGINAL_CONFIRMATION_SHA256,
        "five_input_development_worlds":DEVELOPMENT_ROOT_COUNT,
        "five_input_embargoed_worlds":EMBARGOED_ROOT_COUNT,
        "new_five_input_development_world_specs":development,
        "new_five_input_embargoed_world_commitments":[
            {"world_id":w["world_id"],"world_sha256":w["world_sha256"]}
            for w in embargoed],
        "complete_embargoed_world_spec_sha256":digest(embargoed),
        "complete_development_world_spec_sha256":digest(development),
        "embargoed_models_trained":0,"embargoed_contexts_scored":0,
        "original_four_input_confirmation_models_trained":0,
        "original_four_input_confirmation_contexts_scored":0,
        "planned_max_development_null_roots":DEVELOPMENT_ROOT_COUNT,
        "planned_five_input_sham_tail_calibration":"95/95 tolerance bound needs at least 59 independent eligible roots",
        "first_step":"registration and split tests only; no new model training"
    }
    out["registry_sha256"]=digest(out)
    return out


def write_registration(path:Path):
    data=registration()
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(encoded(data))
    return data


def main(argv=None):
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args(argv)
    r=write_registration(args.out)
    print(json.dumps({
        "five_input_development_roots":r["five_input_development_worlds"],
        "five_input_embargoed_roots_by_hash_only":r["five_input_embargoed_worlds"],
        "registry_sha256":r["registry_sha256"],
        "embargoed_models_trained":0,
        "original_confirmation_worlds_scored":0},sort_keys=True))


if __name__=="__main__":
    main()

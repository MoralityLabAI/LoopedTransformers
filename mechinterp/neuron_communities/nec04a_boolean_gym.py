"""NEC-04A: deterministic 12-bit, five-focal-input Boolean circuit gym.

Registration publishes seed commitments for 24 calibration, 24
development and 16 confirmation world roots BEFORE generating any
programs. Only calibration root specs and source-training contexts are
instantiable by this module. No Torch, native/SAE model or benchmark
scoring. Source labels are arithmetic truth of executable circuits.

An embargo is a CODE-PATH and experimental protocol boundary, not
cryptographic secrecy: the construction salt is public and deterministic.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path

SCHEMA="NEC-04A/seeded-five-focal-boolean-gym/v1"
SALT=b"NEC04A/seeded-boolean-gym/20261122/v1"
SPLITS={"calibration":24,"development":24,"confirmation":16}
TIER_GATES={"easy":11,"medium":15,"hard":19}
TIER_MAX_DEPTH={"easy":8,"medium":12,"hard":16}
TIERS=("easy","medium","hard")
TOTAL_BITS=12
FOCAL_BITS=5
CONTEXT_BITS=7
FOCAL_MASKS=1<<FOCAL_BITS
CONTEXT_GROUPS=1<<CONTEXT_BITS
SOURCE_GROUPS=80
DEVELOPMENT_GROUPS=24
RESERVED_GROUPS=24
MAX_PROGRAM_ATTEMPTS=64
BALANCE_BAND=(.30,.70)
MIN_ESSENTIAL_CONTEXT_BITS=3
MIN_BALANCED_FOCAL_GROUPS=64
PARTITIONS=("source","development","reserved")
MASK_ALL=(1<<(1<<TOTAL_BITS))-1


def packed(obj):
    return json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False).encode()


def digest(obj):
    return sha256(packed(obj)).hexdigest()


class HashRng:
    def __init__(self,seed:bytes):
        self.seed=seed
        self.counter=0

    def below(self,n:int)->int:
        if not isinstance(n,int) or n<=0:
            raise ValueError("positive random range required")
        raw=sha256(self.seed+self.counter.to_bytes(8,"big")).digest()
        self.counter+=1
        return int.from_bytes(raw,"big")%n

    def choose(self,items):
        if not items:raise ValueError("cannot choose from empty sequence")
        return items[self.below(len(items))]

    def shuffle(self,items):
        values=list(items)
        for j in range(len(values)-1,0,-1):
            k=self.below(j+1)
            values[j],values[k]=values[k],values[j]
        return values


def root_id(cohort,index):
    if cohort not in SPLITS or type(index)!=int or index not in range(SPLITS[cohort]):
        raise PermissionError("root outside precommitted NEC04A world split")
    prefix={"calibration":"cal","development":"dev","confirmation":"con"}[cohort]
    return f"nec04a-{prefix}-{index:03d}"


def seed_for_identity(cohort,index):
    return sha256(SALT+b"/root/"+root_id(cohort,index).encode()).digest()


def seed_commitment(cohort,index):
    seed=seed_for_identity(cohort,index)
    return sha256(b"NEC04A/seed-commitment/v1|"+seed).hexdigest()


@lru_cache(maxsize=1)
def focal_layouts():
    positions=list(combinations(range(TOTAL_BITS),FOCAL_BITS))
    positions.sort(key=lambda p:(sha256(SALT+b"/layout/"+bytes(p)).digest(),p))
    assert len(positions)==792
    return tuple(positions)


def global_root_number(cohort,index):
    return sum(SPLITS[s] for s in SPLITS if list(SPLITS).index(s)<list(SPLITS).index(cohort))+index


def root_tier(cohort,index):
    root_id(cohort,index)
    return TIERS[index%3]


def commitments():
    entries=[]
    for cohort,n in SPLITS.items():
        for index in range(n):
            id_=root_id(cohort,index)
            entries.append({
                "root_id":id_,"cohort":cohort,"index":index,
                "difficulty_tier":root_tier(cohort,index),
                "seed_commitment_sha256":seed_commitment(cohort,index),
                "original_focal_layout_commitment_sha256":sha256(
                    b"NEC04A/layout-commitment|"+
                    bytes(focal_layouts()[global_root_number(cohort,index)])
                ).hexdigest(),
                "program_materialized":False,
                "model_training_started":False,
                "outcome_scored":False,
            })
    result={
        "schema":SCHEMA,
        "public_deterministic_seed_commitments":True,
        "embargo_is_enforced_by_generator_code_path":True,
        "cryptographically_hidden_world_specifications":False,
        "previous_NEC03_world_roots_reused":False,
        "split_counts":dict(SPLITS),
        "difficulty_tiers":list(TIERS),
        "cohorts_registered_before_program_generation":True,
        "registration_worlds":entries,
        "development_programs_generated":0,
        "confirmation_programs_generated":0,
        "development_models_trained":0,
        "confirmation_models_trained":0,
        "evaluation_outcomes_scored":0,
        "old_NEC03P_recruited_models_scored":0,
        "old_embargoed_or_confirmation_roots_scored":0,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    result["registration_sha256"]=digest(result)
    return result


def depth(ast):
    if ast[0]=="VAR":return 0
    return 1+max(depth(x) for x in ast[1:] if isinstance(x,list))


def gates(ast):
    if ast[0]=="VAR":return 0
    return 1+sum(gates(x) for x in ast[1:] if isinstance(x,list))


def leaves(ast):
    if ast[0]=="VAR":return {ast[1]}
    result=set()
    for x in ast[1:]:
        result|=leaves(x)
    return result


def make_program(focal,context,tier,seed:bytes):
    if len(focal)!=5 or len(context)!=7 or set(focal)&set(context):
        raise ValueError("needs five disjoint focal and seven context positions")
    if tier not in TIERS:raise ValueError("invalid original tier")
    rng=HashRng(seed)
    f=[["VAR",int(v)] for v in focal]
    c=[["VAR",int(v)] for v in context]
    # Half the contexts take an explicitly balanced five-focal parity
    # branch. That branch guarantees global essentiality of every focal.
    anchor=f[0]
    for x in f[1:]:
        anchor=["XOR",anchor,x]
    parity_branch=["XOR",["XOR",anchor,c[1]],c[2]]
    # Exactly four/ eight/ twelve extra gates; every block includes the
    # four declared primitive types, with deterministic shuffled order.
    branch=f[4]
    steps={"easy":1,"medium":2,"hard":3}[tier]
    operations=rng.shuffle(["XOR","AND","NOT","MUX"]*steps)
    for op in operations:
        if op=="NOT":
            branch=["NOT",branch]
        elif op=="MUX":
            # Context0 is reserved for outer selector; never use task
            # labels or case correctness to choose these operations.
            cond=rng.choose(c[1:])
            alternate=rng.choose(f+c[1:])
            branch=["MUX",cond,branch,alternate]
        else:
            alternate=rng.choose(f+c[1:])
            branch=[op,branch,alternate]
    ast=["MUX",c[0],parity_branch,branch]
    if gates(ast)!=TIER_GATES[tier] or depth(ast)>TIER_MAX_DEPTH[tier]:
        raise AssertionError("frozen tier gate-depth budget violated")
    return ast


@lru_cache(maxsize=1)
def bit_masks():
    result=[]
    for bit in range(TOTAL_BITS):
        value=0
        for history in range(1<<TOTAL_BITS):
            if history&(1<<bit):value|=(1<<history)
        result.append(value)
    return tuple(result)


def evaluate_bit_parallel(ast,variables=None):
    if variables is None:variables=bit_masks()
    name=ast[0]
    if name=="VAR":
        return variables[ast[1]]
    if name=="NOT":
        return MASK_ALL ^ evaluate_bit_parallel(ast[1],variables)
    if name=="AND":
        return (evaluate_bit_parallel(ast[1],variables)&
                evaluate_bit_parallel(ast[2],variables))
    if name=="XOR":
        return (evaluate_bit_parallel(ast[1],variables)^
                evaluate_bit_parallel(ast[2],variables))
    if name=="MUX":
        cond=evaluate_bit_parallel(ast[1],variables)
        yes=evaluate_bit_parallel(ast[2],variables)
        no=evaluate_bit_parallel(ast[3],variables)
        return ((cond&yes)|((MASK_ALL^cond)&no))&MASK_ALL
    raise ValueError("unknown NEC04A circuit operation")


def all_essential_inputs(truth:int):
    influenced=[]
    for bit in range(TOTAL_BITS):
        stride=1<<bit
        changed=False
        for mask in range(1<<TOTAL_BITS):
            if not mask&stride:
                if ((truth>>mask)&1)!=((truth>>(mask|stride))&1):
                    changed=True
                    break
        if changed:influenced.append(bit)
    return influenced


def algebraic_degree(truth:int):
    # Möbius transform of the full 12-variable truth table, reporting
    # Boolean algebraic normal-form degree, independent of grammar depth.
    coeff=bytearray((truth>>i)&1 for i in range(1<<TOTAL_BITS))
    for bit in range(TOTAL_BITS):
        jump=1<<bit
        for mask in range(1<<TOTAL_BITS):
            if mask&jump:
                coeff[mask]^=coeff[mask^jump]
    return max((i.bit_count() for i,v in enumerate(coeff) if v),default=0)


def focal_truth_for_context(truth,focal,context,value:int):
    tbl=0
    for mask in range(FOCAL_MASKS):
        full=0
        for k,position in enumerate(focal):
            full|=((mask>>k)&1)<<position
        for k,position in enumerate(context):
            full|=((value>>k)&1)<<position
        tbl|=((truth>>full)&1)<<mask
    return tbl


def metrics(truth,focal,context):
    count=truth.bit_count()
    influence=all_essential_inputs(truth)
    balanced=0
    for value in range(CONTEXT_GROUPS):
        n=focal_truth_for_context(truth,focal,context,value).bit_count()
        balanced+=int(8<=n<=24)
    return {
        "positive_labels_total_4096":count,
        "class_one_rate":count/(1<<TOTAL_BITS),
        "essential_focal_bits":sorted(set(focal)&set(influence)),
        "essential_context_bits":sorted(set(context)&set(influence)),
        "balanced_focal_context_groups":balanced,
        "algebraic_normal_form_degree_12bits":algebraic_degree(truth),
    }


def accepted(metric):
    return (
        BALANCE_BAND[0]<=metric["class_one_rate"]<=BALANCE_BAND[1]
        and len(metric["essential_focal_bits"])==FOCAL_BITS
        and len(metric["essential_context_bits"])>=MIN_ESSENTIAL_CONTEXT_BITS
        and metric["balanced_focal_context_groups"]>=MIN_BALANCED_FOCAL_GROUPS
    )


def calibration_world(index:int):
    cohort="calibration"
    id_=root_id(cohort,index)
    focal=focal_layouts()[global_root_number(cohort,index)]
    context=tuple(bit for bit in range(TOTAL_BITS) if bit not in focal)
    tier=root_tier(cohort,index)
    seed=seed_for_identity(cohort,index)
    for attempt in range(MAX_PROGRAM_ATTEMPTS):
        candidate_seed=sha256(seed+b"/candidate/"+attempt.to_bytes(4,"big")).digest()
        prog=make_program(focal,context,tier,candidate_seed)
        truth=evaluate_bit_parallel(prog)
        meta=metrics(truth,focal,context)
        if not accepted(meta):
            continue
        packed_truth=truth.to_bytes(1<<(TOTAL_BITS-3),"little")
        record={
            "schema":SCHEMA,"world_id":id_,"cohort":cohort,
            "root_index":index,"difficulty_tier":tier,
            "root_seed_commitment_sha256":seed_commitment(cohort,index),
            "focal_positions":list(focal),
            "context_positions":list(context),
            "source_code_bits":TOTAL_BITS,
            "canonical_program":prog,
            "program_gates":gates(prog),"program_depth":depth(prog),
            "candidate_attempt_index":attempt,
            "fixed_max_program_attempts":MAX_PROGRAM_ATTEMPTS,
            "program_sha256":digest(prog),
            "truth_table_4096_sha256":sha256(packed_truth).hexdigest(),
            "program_metrics":meta,
            "native_models_trained":0,
            "source_SAEs_fitted":0,
            "model_baseline_accuracy_measured":False,
            "measured_model_headroom_band":None,
        }
        record["world_sha256"]=digest(record)
        return record,truth
    raise AssertionError("calibration world failed frozen model-independent grammar gates")


def contexts_for_world(index):
    # Only calibration roots enter this function: no dev/confirmation
    # programs, truth tables, or source/evaluation records are generated.
    world_id_=root_id("calibration",index)
    seed=seed_for_identity("calibration",index)
    contexts=[tuple((bits>>k)&1 for k in range(CONTEXT_BITS))
              for bits in range(CONTEXT_GROUPS)]
    contexts.sort(key=lambda row:(sha256(seed+b"/context/"+bytes(row)).digest(),row))
    return contexts[:SOURCE_GROUPS],{
        "source":tuple(contexts[:SOURCE_GROUPS]),
        "development":tuple(contexts[SOURCE_GROUPS:SOURCE_GROUPS+DEVELOPMENT_GROUPS]),
        "reserved":tuple(contexts[SOURCE_GROUPS+DEVELOPMENT_GROUPS:]),
        "world_id":world_id_,
    }


def source_rows(index,world,truth):
    if world["cohort"]!="calibration" or world["root_index"]!=index:
        raise PermissionError("NEC04A source case output is calibration-only")
    source,parts=contexts_for_world(index)
    focal=world["focal_positions"]
    context=world["context_positions"]
    for group,context_bits in enumerate(source):
        focal_table=focal_truth_for_context(
            truth,focal,context,
            sum(bit<<j for j,bit in enumerate(context_bits))
        )
        for mask in range(FOCAL_MASKS):
            history=[0]*TOTAL_BITS
            for j,position in enumerate(focal):
                history[position]=(mask>>j)&1
            for j,position in enumerate(context):
                history[position]=context_bits[j]
            yield {
                "schema":SCHEMA,
                "world_id":world["world_id"],
                "world_sha256":world["world_sha256"],
                "difficulty_tier":world["difficulty_tier"],
                "cohort":"calibration","split":"source_train",
                "source_context_group":group,
                "focal_mask":mask,
                "history_bits12":history,
                "label":(focal_table>>mask)&1,
            }


def materialize_calibration(out:Path):
    out.mkdir(parents=True,exist_ok=True)
    registration=commitments()
    roots=[]
    corpus_path=out/"nec04a_calibration_source_train.jsonl"
    with corpus_path.open("wb") as f:
        for index in range(SPLITS["calibration"]):
            world,truth=calibration_world(index)
            roots.append(world)
            for row in source_rows(index,world,truth):
                f.write(packed(row)+b"\n")
    if len({w["truth_table_4096_sha256"] for w in roots})!=len(roots):
        raise AssertionError("two original calibration programs share full truth tables")
    if len({w["program_sha256"] for w in roots})!=len(roots):
        raise AssertionError("same calibration program selected twice")
    root_path=out/"nec04a_calibration_worlds.jsonl"
    root_path.write_bytes(b"".join(packed(w)+b"\n" for w in roots))
    registration_path=out/"nec04a_world_root_commitments.json"
    registration_path.write_text(json.dumps(registration,sort_keys=True,indent=2)+"\n")
    manifest={
        "schema":SCHEMA,
        "phase":"calibration-source-corpus-only",
        "seed_registration_sha256":registration["registration_sha256"],
        "root_count":len(roots),
        "source_context_groups_per_root":SOURCE_GROUPS,
        "focal_masks_per_context":FOCAL_MASKS,
        "source_case_rows":len(roots)*SOURCE_GROUPS*FOCAL_MASKS,
        "program_tier_counts":{t:sum(w["difficulty_tier"]==t for w in roots)
                               for t in TIERS},
        "root_program_sha256":[w["program_sha256"] for w in roots],
        "root_truth_table_sha256":[w["truth_table_4096_sha256"] for w in roots],
        "source_corpus_sha256":sha256(corpus_path.read_bytes()).hexdigest(),
        "calibration_world_specs_sha256":sha256(root_path.read_bytes()).hexdigest(),
        "development_world_programs_generated":0,
        "confirmation_world_programs_generated":0,
        "development_or_confirmation_labels_generated":0,
        "model_forwards":0,
        "model_training_steps":0,
        "source_SAE_training_steps":0,
        "previous_NEC03P_worlds_scored":0,
        "old_embargoed_or_four_input_confirmation_scored":0,
        "real_model_headroom_evaluated":False,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    (out/"nec04a_calibration_source_manifest.json").write_text(
        json.dumps(manifest,sort_keys=True,indent=2)+"\n")
    print("NEC04A_CALIBRATION_SOURCE_CORPUS_GENERATED",
          len(roots),manifest["source_case_rows"],
          "sha256",manifest["source_corpus_sha256"])
    return manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--phase",choices=("commitments","calibration"),required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if args.phase=="commitments":
        c=commitments()
        (args.out/"nec04a_world_root_commitments.json").write_text(
            json.dumps(c,sort_keys=True,indent=2)+"\n")
        print("NEC04A_PRECOMMITTED_ROOTS",c["registration_sha256"])
    else:
        materialize_calibration(args.out)


if __name__=="__main__":
    main()

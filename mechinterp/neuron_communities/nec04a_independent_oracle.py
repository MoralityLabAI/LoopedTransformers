"""NEC-04A independent Boolean AST oracle and calibration-corpus auditor.

The generator uses whole-4096-input bit-parallel integer circuits.
This module independently evaluates scalar Boolean gate semantics,
verifies the published pre-program seed commitments and all original
source-training examples, without any Torch/model or noncalibration
world construction.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha1,sha256
import json
from pathlib import Path

from mechinterp.neuron_communities import nec04a_boolean_gym as gym

SCHEMA="NEC-04A/independent-calibration-source-oracle-audit/v1"
LOCK_NAME="NEC04A_REGISTRATION_LOCK.json"
EXPECTED_ROOT_REGISTRY="88bf58d32fdf15a1270986c238afb3def1a7e6371d3eee64c4edd264c9cf041e"
EXPECTED_SEED_HASH_ROOT="53f201c94b20699e81f9175fe95d86567dfd1925e2357ae519170c70207f152e"
EXPECTED_LAYOUT_HASH_ROOT="e1c69cd56acd5fb8cc62ae688e927743a533b4ef218eec1fdf76664d789b6e83"


def verify_lock():
    lock_path=Path(__file__).with_name(LOCK_NAME)
    lock=json.loads(lock_path.read_text())
    expected={
        "schema":"NEC-04A/root-commitment-before-calibration-materialization/v1",
        "old_NEC03Z_sealed_analysis_groups":[56,57,58,59,60,61,62,63],
        "new_root_counts":{"calibration":24,"development":24,"confirmation":16},
        "seed_registry_sha256":EXPECTED_ROOT_REGISTRY,
        "seed_commitments_aggregated_sha256":EXPECTED_SEED_HASH_ROOT,
        "layout_commitments_aggregated_sha256":EXPECTED_LAYOUT_HASH_ROOT,
        "new_calibration_root_count":24,
        "full_history_bits":12,
        "focal_bits":5,
        "source_context_groups_per_calibration_root":80,
        "source_cases_per_root":2560,
        "calibration_corpus_cases":61440,
        "label_balance_band":[.30,.70],
        "essential_focal_bits":5,
        "essential_context_bits_minimum":3,
        "minimum_balanced_focal_context_groups":64,
        "max_program_attempts":64,
        "tier_gates":{"easy":11,"medium":15,"hard":19},
        "tier_max_depth":{"easy":8,"medium":12,"hard":16},
        "native_or_SAE_training_updates":0,
        "development_root_specs_instantiated":0,
        "confirmation_root_specs_instantiated":0,
        "development_or_confirmation_labels_scored":0,
        "original_NEC03P_recruited_roots_scored":0,
        "old_NEC03_embargo_or_confirmation_scored":0,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    if any(lock.get(k)!=v for k,v in expected.items()):
        raise AssertionError("NEC04A pre-materialization root/seed/difficulty contract changed")
    repo=Path(__file__).resolve().parents[2]
    for path,pin in lock["git_blob_sha1"].items():
        raw=(repo/path).read_bytes()
        git_sha=sha1(b"blob "+str(len(raw)).encode()+b"\x00"+raw).hexdigest()
        if git_sha!=pin:
            raise AssertionError("NEC04A frozen source identity changed "+path)
    return lock,sha256(lock_path.read_bytes()).hexdigest()


def oracle(ast,history):
    """Scalar reference interpreter, independent of bit-parallel evaluator."""
    if not isinstance(ast,list) or not ast:
        raise AssertionError("AST grammar malformed")
    op=ast[0]
    if op=="VAR":
        if len(ast)!=2 or type(ast[1])!=int or not 0<=ast[1]<12:
            raise AssertionError("variable index not a valid native history bit")
        return bool(history[ast[1]])
    if op=="NOT":
        if len(ast)!=2:raise AssertionError("NOT must be unary")
        return not oracle(ast[1],history)
    if op=="AND":
        if len(ast)!=3:raise AssertionError("AND must be binary")
        # Evaluate both clauses to audit every child, without shortcut.
        a=oracle(ast[1],history)
        b=oracle(ast[2],history)
        return a and b
    if op=="XOR":
        if len(ast)!=3:raise AssertionError("XOR must be binary")
        a=oracle(ast[1],history)
        b=oracle(ast[2],history)
        return a!=b
    if op=="MUX":
        if len(ast)!=4:raise AssertionError("MUX must have three children")
        condition=oracle(ast[1],history)
        yes=oracle(ast[2],history)
        no=oracle(ast[3],history)
        return yes if condition else no
    raise AssertionError("unregistered grammar gate "+str(op))


def histories_in_full_context(focal,context,bits):
    for mask in range(32):
        history=[0]*12
        for j,pos in enumerate(focal):
            history[pos]=(mask>>j)&1
        for j,pos in enumerate(context):
            history[pos]=bits[j]
        yield tuple(history)


def audit(folder:Path,out:Path):
    cfg,lock_sha=verify_lock()
    manifest=json.loads((folder/"nec04a_calibration_source_manifest.json").read_text())
    registry=json.loads((folder/"nec04a_world_root_commitments.json").read_text())
    worlds_path=folder/"nec04a_calibration_worlds.jsonl"
    data_path=folder/"nec04a_calibration_source_train.jsonl"
    worlds=[json.loads(line) for line in worlds_path.read_text().splitlines()]
    rows=[json.loads(line) for line in data_path.read_text().splitlines()]
    if set(p.name for p in folder.iterdir())!={
        "nec04a_calibration_source_manifest.json",
        "nec04a_world_root_commitments.json",
        "nec04a_calibration_worlds.jsonl",
        "nec04a_calibration_source_train.jsonl",
    }:
        raise AssertionError("unexpected program/model/scoring artifact in calibration-only directory")
    if (registry!=gym.commitments()
        or registry["registration_sha256"]!=EXPECTED_ROOT_REGISTRY
        or len(registry["registration_worlds"])!=64
        or registry["development_programs_generated"]!=0
        or registry["confirmation_programs_generated"]!=0
        or registry["old_embargoed_or_confirmation_roots_scored"]!=0
        or manifest["seed_registration_sha256"]!=EXPECTED_ROOT_REGISTRY
        or manifest["source_corpus_sha256"]!=sha256(data_path.read_bytes()).hexdigest()
        or manifest["calibration_world_specs_sha256"]!=sha256(worlds_path.read_bytes()).hexdigest()
        or manifest["source_case_rows"]!=61440 or len(rows)!=61440
        or manifest["root_count"]!=24 or len(worlds)!=24
        or manifest["real_model_headroom_evaluated"] is not False
        or any(manifest[k]!=0 for k in (
            "development_world_programs_generated",
            "confirmation_world_programs_generated",
            "development_or_confirmation_labels_generated",
            "model_forwards","model_training_steps",
            "source_SAE_training_steps","previous_NEC03P_worlds_scored",
            "old_embargoed_or_four_input_confirmation_scored"))
        or manifest["FSI_01_gate"]!="REDESIGN_OR_PILOT_ONLY"):
        raise AssertionError("calibration corpus source manifest changed/holdout opened")

    commitments=registry["registration_worlds"]
    seed_aggregate=sha256("".join(e["seed_commitment_sha256"] for e in commitments).encode()).hexdigest()
    layout_aggregate=sha256("".join(e["original_focal_layout_commitment_sha256"] for e in commitments).encode()).hexdigest()
    if seed_aggregate!=EXPECTED_SEED_HASH_ROOT or layout_aggregate!=EXPECTED_LAYOUT_HASH_ROOT:
        raise AssertionError("original 64-root seeds/layouts diverged from precommitted digests")

    done=[];fingerprints=set()
    for index,world in enumerate(worlds):
        root=f"nec04a-cal-{index:03d}"
        pre=commitments[index]
        if (world["schema"]!=gym.SCHEMA or world["world_id"]!=root
            or world["cohort"]!="calibration" or world["root_index"]!=index
            or pre["root_id"]!=root or pre["cohort"]!="calibration"
            or world["difficulty_tier"]!=pre["difficulty_tier"]
            or world["root_seed_commitment_sha256"]!=pre["seed_commitment_sha256"]
            or world["program_gates"]!=gym.TIER_GATES[pre["difficulty_tier"]]
            or world["program_depth"]>gym.TIER_MAX_DEPTH[pre["difficulty_tier"]]
            or not 0<=world["candidate_attempt_index"]<64
            or any(world[k]!=0 for k in ("native_models_trained","source_SAEs_fitted"))
            or world["model_baseline_accuracy_measured"] is not False
            or world["measured_model_headroom_band"] is not None):
            raise AssertionError("materialized calibration world roster or grammar changed")
        tmp=dict(world); sha=tmp.pop("world_sha256")
        if gym.digest(tmp)!=sha:
            raise AssertionError("world original canonical SHA mismatch")
        program=world["canonical_program"]
        if world["program_sha256"]!=gym.digest(program):
            raise AssertionError("calibration root AST changed")
        if world["program_sha256"] in fingerprints:
            raise AssertionError("duplicate root program selected")
        fingerprints.add(world["program_sha256"])
        focal=world["focal_positions"]
        context=world["context_positions"]
        if (len(focal)!=5 or len(context)!=7 or
            sorted(focal+context)!=list(range(12))
            or tuple(focal)!=gym.focal_layouts()[index]):
            raise AssertionError("world original distinct focal positions changed")
        # Recompute all 4096 input truth labels through an independent
        # scalar interpreter. The generator used a separate bit-parallel
        # 4096-bit integer evaluator; agreement is an actual oracle check.
        calculated=0
        for history_index in range(1<<12):
            bits=tuple((history_index>>j)&1 for j in range(12))
            calculated|=int(oracle(program,bits))<<history_index
        oracle_hash=sha256(calculated.to_bytes(512,"little")).hexdigest()
        if oracle_hash!=world["truth_table_4096_sha256"]:
            raise AssertionError("independent 4096-case oracle disagrees with generated world")
        if not gym.accepted(gym.metrics(calculated,focal,context)):
            raise AssertionError("calibration program violates preregistered truth gates")
        source_ctx,partitions=gym.contexts_for_world(index)
        if (len(source_ctx)!=80 or len(partitions["development"])!=24
            or len(partitions["reserved"])!=24
            or set(source_ctx)&set(partitions["development"])
            or set(source_ctx)&set(partitions["reserved"])
            or set(partitions["development"])&set(partitions["reserved"])):
            raise AssertionError("whole seven-bit source contexts not disjoint from held-out contexts")
        offset=index*2560
        seen=set()
        for group,ctx_bits in enumerate(source_ctx):
            for mask,expected_history in enumerate(
                histories_in_full_context(focal,context,ctx_bits)
            ):
                row=rows[offset+group*32+mask]
                if (row["schema"]!=gym.SCHEMA
                    or row["cohort"]!="calibration"
                    or row["split"]!="source_train"
                    or row["world_id"]!=root
                    or row["world_sha256"]!=sha
                    or row["difficulty_tier"]!=world["difficulty_tier"]
                    or row["source_context_group"]!=group
                    or row["focal_mask"]!=mask
                    or tuple(row["history_bits12"])!=expected_history):
                    raise AssertionError("source context/complete focal block order or roots changed")
                truth_label=int(oracle(program,expected_history))
                if type(row["label"])!=int or row["label"]!=truth_label:
                    raise AssertionError("independent source label differs from original Boolean program")
                if expected_history in seen:
                    raise AssertionError("repeated original source context/mask history")
                seen.add(expected_history)
        if len(seen)!=2560:
            raise AssertionError("incomplete unique source-training context/focal grid")
        done.append({
            "world_id":root,"tier":world["difficulty_tier"],
            "source_rows":len(seen),
            "world_sha256":sha,
            "source_truth_sha256":oracle_hash,
            "program_gate_count":world["program_gates"],
            "program_depth":world["program_depth"],
            "positive_label_fraction":calculated.bit_count()/4096,
            "ANF_degree":world["program_metrics"]["algebraic_normal_form_degree_12bits"],
            "essential_context_bit_count":len(world["program_metrics"]["essential_context_bits"]),
            "balanced_focal_context_groups":world["program_metrics"]["balanced_focal_context_groups"],
        })
    if len(set(x["source_truth_sha256"] for x in done))!=24:
        raise AssertionError("two calibration roots share full 12-bit truth tables")
    if [x["world_id"] for x in done]!=[f"nec04a-cal-{i:03d}" for i in range(24)]:
        raise AssertionError("original world statistical units/sequence changed")
    result={
        "schema":SCHEMA,
        "registry_sha256":EXPECTED_ROOT_REGISTRY,
        "source_lock_sha256":lock_sha,
        "all_64_world_root_seed_commitments_pre_registered":True,
        "original_development_root_programs_generated":0,
        "original_confirmation_root_programs_generated":0,
        "calibration_roots_independently_verified":24,
        "independent_full_4096_case_truth_table_per_calibration_root":True,
        "independent_source_case_label_rows_verified":61440,
        "separate_original_context_source_groups":80,
        "all_32_focal_masks_per_context_verified":True,
        "all_original_five_focal_inputs_essential":True,
        "all_root_source_histories_unique":True,
        "all_original_source_contexts_disjoint_from_development_reserved":True,
        "original_model_or_SAE_forwards_or_train_steps":0,
        "measured_native_clean_or_full_affine_accuracy":False,
        "semantic_functional_inertness_calibrated":False,
        "population_causal_sham_claim":False,
        "old_NEC03P_recruited_roots_scored":0,
        "old_embargoed_or_confirmation_roots_scored":0,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
        "calibration_worlds":done,
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"nec04a_independent_oracle_audit.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("NEC04A_24_ROOT_SCALAR_ORACLE_AND_61440_SOURCE_CASES_PASS",
          len(done),result["registry_sha256"])
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--folder",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    audit(a.folder,a.out)


if __name__=="__main__":
    main()

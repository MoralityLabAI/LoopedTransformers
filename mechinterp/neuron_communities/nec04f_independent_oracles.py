"""NEC-04F independent scalar verification of original roots and matched aux targets.

Independent from primary source-row/shuffle construction: rebuild every
original source and owned validation 12-bit history, evaluate immutable
Boolean AST and each XOR prefix, reconstruct hash-ranked 16/16
intermediate source-label controls, enforce split and SHA receipts.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

from mechinterp.neuron_communities import nec04a_boolean_gym as gym
from mechinterp.neuron_communities.nec04a_independent_oracle import oracle
from mechinterp.neuron_communities.nec04f_oracles import (
    ROOTS, ARMS, STAGES, STEPS, SOURCE, VALIDATION, SCHEMA,
    ORIGINAL_SHA, REGISTRY_SHA, code_identity, file_sha, lock, seed,
)

SCHEMA_AUDIT = "NEC-04F/independent-preoptimizer-original-auxiliary-label-audit/v1"


def independently_shuffled_labels(index, group, stage):
    assert index in ROOTS and 0 <= group < 80 and stage in STAGES
    prefix = b"NEC04F/balanced-shuffled-source-aux-label/v1|"
    prefix += gym.seed_for_identity("calibration", index)
    ranking = []
    for mask in range(32):
        digest = sha256(
            prefix + int(group).to_bytes(2,"big") + bytes((stage,mask))
        ).digest()
        ranking.append((digest,mask))
    ranking.sort()
    positives = {mask for _,mask in ranking[:16]}
    return [int(mask in positives) for mask in range(32)]


def independently_prefix(bits, focal, count):
    indices = sorted(focal)
    assert len(indices) == 5 and count in STAGES
    result = 0
    for i in range(count):
        result = result ^ int(bits[indices[i]])
    return result


def audit(pretraining: Path, out: Path):
    cfg, lock_sha = lock()
    f = pretraining/"nec04f_pretraining_oracles.json"
    ledger = json.loads(f.read_text())
    if (
        ledger["schema"] != SCHEMA
        or ledger["lock_sha256"] != lock_sha
        or ledger["original_source_corpus_sha256"] != ORIGINAL_SHA
        or ledger["original_registry_sha256"] != REGISTRY_SHA
        or ledger["source_code_sha256_before_training"] != code_identity()
        or ledger["root_indices"] != list(ROOTS)
        or ledger["training_arms"] != list(ARMS)
        or ledger["stage_k"] != list(STAGES)
        or ledger["checkpoint_steps"] != list(STEPS)
        or ledger["original_source_rows_per_root"] != SOURCE
        or ledger["original_owned_validation_rows_per_root"] != VALIDATION
        or ledger["optimizer_updates_before_preflight"] != 0
        or len(ledger["roots"]) != len(ROOTS)
    ):
        raise AssertionError("original calibration root and pretraining source/auxiliary roster changed")
    if os.environ.get("GITHUB_SHA") and ledger["preflight_source_git_sha"] != os.environ["GITHUB_SHA"]:
        raise AssertionError("original pretraining source code commit not frozen")
    for prop in (
        "source_SAE_updates","reserved_contexts_scored",
        "new_development_world_roots_materialized",
        "new_confirmation_world_roots_materialized","old_NEC03P_worlds_scored",
    ):
        if ledger[prop] != 0:
            raise AssertionError("sealed original root or causal cohort opened in preflight")
    all_roots = []
    for ordinal,index in enumerate(ROOTS):
        original, truth = gym.calibration_world(index)
        ref = ledger["roots"][ordinal]
        if (
            ref["index"] != index
            or ref["world_id"] != original["world_id"]
            or ref["world_sha256"] != original["world_sha256"]
            or ref["program_sha256"] != original["program_sha256"]
            or ref["difficulty_tier"] != original["difficulty_tier"]
            or ref["truth_4096_sha256"] != original["truth_table_4096_sha256"]
            or ref["stage_visits"] != {
                str(k):sorted(original["focal_positions"])[k-1]+1 for k in STAGES
            }
            or ref["matched_seed"] != seed(index)
        ):
            raise AssertionError("original independent six-root world/program/seed identity altered")
        input_file = pretraining/ref["rows_file"]
        if (
            input_file.name != f"nec04f_root_{index}.jsonl"
            or file_sha(input_file) != ref["rows_sha256"]
        ):
            raise AssertionError("pretraining scalar oracle rows SHA differs")
        rows = [json.loads(line) for line in input_file.read_text().splitlines()]
        src,parts = gym.contexts_for_world(index)
        valid = parts["development"]
        reserve = parts["reserved"]
        if (
            len(rows) != SOURCE + VALIDATION or len(src) != 80
            or len(valid) != 24 or len(reserve) != 24
            or set(src) & set(valid) or set(src) & set(reserve)
            or set(valid) & set(reserve)
        ):
            raise AssertionError("old complete source, owned and reserved context allocation invalid")
        focal = sorted(original["focal_positions"])
        source_hashes = set()
        validation_hashes = set()
        for line_idx,item in enumerate(rows):
            source = line_idx < SOURCE
            group = line_idx//32 if source else (line_idx-SOURCE)//32
            mask = line_idx % 32
            context = src[group] if source else valid[group]
            expected_history = [0]*12
            for bit,position in enumerate(original["focal_positions"]):
                expected_history[position] = (mask>>bit)&1
            for bit,position in enumerate(original["context_positions"]):
                expected_history[position] = int(context[bit])
            index_bits = sum(int(bit)<<j for j,bit in enumerate(expected_history))
            original_truth = (truth>>index_bits)&1
            separate_oracle = int(oracle(original["canonical_program"],expected_history))
            if (
                item["root_index"] != index
                or item["world_id"] != original["world_id"]
                or item["split"] != (
                    "source_train" if source else "calibration_world_validation"
                )
                or item["context_group"] != (group if source else 80+group)
                or item["focal_mask"] != mask
                or item["history_bits12"] != expected_history
                or item["composed_label"] != original_truth
                or item["composed_label"] != separate_oracle
                or item["mux_selector"] != expected_history[original["context_positions"][0]]
                or item["xor_prefix"] != {
                    str(k):independently_prefix(expected_history,focal,k) for k in STAGES
                }
            ):
                raise AssertionError("old circuit/oracle/XOR prefix/source ownership changed")
            marker = tuple(expected_history)
            panel = source_hashes if source else validation_hashes
            if marker in panel:
                raise AssertionError("old 12-bit training/evaluation input repeated")
            panel.add(marker)
            if source:
                control = {
                    str(k):independently_shuffled_labels(index,group,k)[mask]
                    for k in STAGES
                }
                if item["shuffled_aux"] != control:
                    raise AssertionError("source-only fixed 16/16 balanced auxiliary label SHA changed")
            elif "shuffled_aux" in item:
                raise AssertionError("shuffled source-only control labels leaked to owned validation")
        if (len(source_hashes) != SOURCE or len(validation_hashes) != VALIDATION
            or source_hashes & validation_hashes):
            raise AssertionError("source vs owned calibration validation histories overlap")
        for group in range(80):
            group_rows = rows[group*32:(group+1)*32]
            for k in STAGES:
                assert sum(item["xor_prefix"][str(k)] for item in group_rows) == 16
                assert sum(item["shuffled_aux"][str(k)] for item in group_rows) == 16
        for group in range(24):
            group_rows = rows[SOURCE+group*32:SOURCE+(group+1)*32]
            for k in STAGES:
                assert sum(item["xor_prefix"][str(k)] for item in group_rows) == 16
        if (
            ref["source_cases"] != SOURCE or ref["owned_validation_cases"] != VALIDATION
            or ref["source_parity_cases"] != sum(x["mux_selector"] for x in rows[:SOURCE])
            or ref["validation_parity_cases"] != sum(x["mux_selector"] for x in rows[SOURCE:])
        ):
            raise AssertionError("source/owned score denominator, original MUX selection changed")
        all_roots.append({
            "root_index":index,
            "world_id":original["world_id"],
            "source_and_validation_sha256":file_sha(input_file),
            "source_cases":SOURCE,
            "owned_validation_cases":VALIDATION,
            "true_source_prefix_labels_verified":SOURCE * len(STAGES),
            "balanced_source_control_labels_verified":SOURCE * len(STAGES),
            "true_owned_prefix_labels_verified":VALIDATION * len(STAGES),
        })
    result={
        "schema":SCHEMA_AUDIT,
        "root_count":len(all_roots),
        "root_indices":list(ROOTS),
        "original_pretraining_ledger_sha256":file_sha(f),
        "original_lock_sha256":lock_sha,
        "git_source_hashes_before_training":code_identity(),
        "original_source_composed_oracle_labels_checked":SOURCE * len(ROOTS),
        "original_owned_validation_composed_oracle_labels_checked":VALIDATION * len(ROOTS),
        "original_true_source_prefix_labels_checked":SOURCE*len(ROOTS)*len(STAGES),
        "original_balanced_source_aux_control_labels_checked":SOURCE*len(ROOTS)*len(STAGES),
        "original_owned_true_prefix_labels_checked":VALIDATION*len(ROOTS)*len(STAGES),
        "source_context_32mask_and_16_16_control_balance_verified":True,
        "source_training_updates_at_preflight":0,
        "source_SAE_updates":0,
        "reserved_contexts_scored":0,
        "development_worlds_instantiated":0,
        "confirmation_worlds_instantiated":0,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
        "roots":all_roots,
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"nec04f_independent_pretraining_audit.json").write_text(
        json.dumps(result,sort_keys=True,indent=2)+"\n"
    )
    print("NEC04F_INDEPENDENT_SIX_WORLD_PRETRAINING_ORACLES_PASS",
          result["original_true_source_prefix_labels_checked"],
          result["original_balanced_source_aux_control_labels_checked"],flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--pretraining",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    audit(a.pretraining,a.out)


if __name__=="__main__":
    main()

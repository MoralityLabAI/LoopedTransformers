"""NEC-04C independent scalar audit of root6..11 teacher labels and context ownership."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from mechinterp.neuron_communities import nec04a_boolean_gym as gym
from mechinterp.neuron_communities.nec04a_independent_oracle import oracle
from mechinterp.neuron_communities.nec04c_pretraining import (
    ROOTS, ARMS, CHECKPOINTS, SOURCE, VALIDATION, SCHEMA,
    EXPECTED_CORPUS, EXPECTED_REGISTRY, digest, verify_lock, code_identity,
    seed_for_root,
)


def audit(folder: Path, out: Path):
    lock, sha = verify_lock()
    ledger_file = folder / "nec04c_pretraining_oracles.json"
    ledger = json.loads(ledger_file.read_text())
    if (
        ledger["schema"] != SCHEMA
        or ledger["lock_sha256"] != sha
        or ledger["original_source_registry_sha256"] != EXPECTED_REGISTRY
        or ledger["original_calibration_source_corpus_sha256"] != EXPECTED_CORPUS
        or ledger["root_indices"] != list(ROOTS)
        or ledger["training_arms"] != list(ARMS)
        or ledger["checkpoint_steps"] != list(CHECKPOINTS)
        or ledger["source_cases_per_root"] != SOURCE
        or ledger["owned_validation_cases_per_root"] != VALIDATION
        or ledger["code_sha256_before_training"] != code_identity()
        or ledger["optimizer_steps_before_preflight"] != 0
        or len(ledger["roots"]) != len(ROOTS)
    ):
        raise AssertionError("independent preflight: source, code, roster or optimizer status changed")
    github_sha = os.environ.get("GITHUB_SHA")
    if github_sha and ledger["preflight_source_commit_sha"] != github_sha:
        raise AssertionError("native training source commit differs from preflight commit")
    audit_roots = []
    for ordinal, index in enumerate(ROOTS):
        root = ledger["roots"][ordinal]
        world, _ = gym.calibration_world(index)
        if (
            root["root_index"] != index
            or root["world_id"] != world["world_id"]
            or root["world_sha256"] != world["world_sha256"]
            or root["program_sha256"] != world["program_sha256"]
            or root["original_truth_4096_sha256"] != world["truth_table_4096_sha256"]
            or root["matched_native_seed"] != seed_for_root(index)
        ):
            raise AssertionError("independent preflight: world and model seed mismatch")
        src = folder / root["training_file"]
        if (
            src.name != f"nec04c_training_root_{index}.jsonl"
            or digest(src) != root["training_file_sha256"]
        ):
            raise AssertionError("independent preflight: altered source/oracle bytes")
        rows = [json.loads(line) for line in src.read_text().splitlines()]
        if len(rows) != SOURCE + VALIDATION:
            raise AssertionError("independent preflight: wrong original source split size")
        source_context, groups = gym.contexts_for_world(index)
        own_val = groups["development"]
        if (
            len(source_context) != 80 or len(own_val) != 24
            or len(groups["reserved"]) != 24
            or set(source_context) & set(own_val)
            or set(source_context) & set(groups["reserved"])
            or set(own_val) & set(groups["reserved"])
        ):
            raise AssertionError("independent preflight: reserved context leakage")
        selector = world["context_positions"][0]
        if world["canonical_program"][:2] != ["MUX", ["VAR", selector]]:
            raise AssertionError("independent preflight: changed parity teacher")
        originals = set()
        positives = {"source": 0, "validation": 0}
        for position, row in enumerate(rows):
            source = position < SOURCE
            group_index = position // 32 if source else (position - SOURCE) // 32
            complete_mask = position % 32
            context = source_context[group_index] if source else own_val[group_index]
            bits = [0] * 12
            for k, bitpos in enumerate(world["focal_positions"]):
                bits[bitpos] = (complete_mask >> k) & 1
            for k, bitpos in enumerate(world["context_positions"]):
                bits[bitpos] = context[k]
            if (
                row["world_id"] != world["world_id"]
                or row["root_index"] != index
                or row["split"] != (
                    "source_train" if source else "calibration_world_validation"
                )
                or row["context_group"] != (group_index if source else 80 + group_index)
                or row["focal_mask"] != complete_mask
                or row["history_bits12"] != bits
                or row["mux_selector"] != bits[selector]
                or row["composed_label"] != int(oracle(world["canonical_program"], bits))
            ):
                raise AssertionError("independent preflight: scalar oracle/context row failure")
            if tuple(bits) in originals:
                raise AssertionError("independent preflight: repeated input history")
            originals.add(tuple(bits))
            positives["source" if source else "validation"] += int(row["mux_selector"])
        if (
            len(originals) != SOURCE + VALIDATION
            or positives["source"] != root["source_selector_one_cases"]
            or positives["validation"] != root["validation_selector_one_cases"]
            or not 0 < positives["source"] < SOURCE
            or not 0 < positives["validation"] < VALIDATION
        ):
            raise AssertionError("independent preflight: incomplete paired parity selector coverage")
        audit_roots.append({
            "root_index": index,
            "world_id": root["world_id"],
            "source_rows": SOURCE,
            "owned_validation_rows": VALIDATION,
            "original_source_validation_sha256": digest(src),
            "selector_one_source": positives["source"],
            "selector_one_owned_validation": positives["validation"],
        })
    for field in (
        "original_reserved_contexts_scored",
        "new_development_world_specs_materialized",
        "new_confirmation_world_specs_materialized",
        "old_NEC03P_worlds_scored",
        "native_SAE_training_updates",
    ):
        if ledger[field] != 0:
            raise AssertionError("independent preflight: protected new root cohort opened")
    if ledger["FSI_01_gate"] != "REDESIGN_OR_PILOT_ONLY":
        raise AssertionError("independent preflight: source-only status promoted")
    report = {
        "schema": "NEC-04C/independent-pretraining-six-root-oracle-audit/v1",
        "pretraining_ledger_sha256": digest(ledger_file),
        "lock_sha256": sha,
        "source_code_sha256": code_identity(),
        "root_indices": list(ROOTS),
        "root_count": 6,
        "source_labels_checked": SOURCE * 6,
        "owned_validation_labels_checked": VALIDATION * 6,
        "original_reserved_contexts_scored": 0,
        "development_world_roots_generated": 0,
        "confirmation_world_roots_generated": 0,
        "model_optimizer_updates_before_preflight": 0,
        "FSI_01_gate": "REDESIGN_OR_PILOT_ONLY",
        "roots": audit_roots,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "nec04c_independent_pretraining_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print("NEC04C_SIX_ROOT_SCALAR_ORACLE_AND_CONTEXT_SPLITS_AUDITED", flush=True)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folder", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    audit(args.folder, args.out)


if __name__ == "__main__":
    main()

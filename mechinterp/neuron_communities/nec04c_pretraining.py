"""NEC-04C: materialize fixed calibration root6..11 source/owned-validation teacher data.

Only source-training histories from the original SHA-pinned NEC04A artifact,
plus 24 originally owned validation contexts of each *calibration* world.
Protected developmental WORLD roots, confirmation roots and reserved contexts
remain uninstantiated. Runs and hashes all task/sampler inputs before training.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

from mechinterp.neuron_communities import nec04a_boolean_gym as gym
from mechinterp.neuron_communities.nec04a_independent_oracle import oracle

ROOTS = tuple(range(6, 12))
ARMS = ("vanilla", "parity_curriculum")
CHECKPOINTS = (240, 900)
SOURCE = 2560
VALIDATION = 768
SCHEMA = "NEC-04C/pretraining-six-original-calibration-world-roots/v1"
LOCK_NAME = "NEC04C_TRAIN_LOCK.json"
EXPECTED_CORPUS = "82c80643d3c99871913f5228f1a1cde1eec4484787c1784872b508349793e9e0"
EXPECTED_REGISTRY = "88bf58d32fdf15a1270986c238afb3def1a7e6371d3eee64c4edd264c9cf041e"
CODE_FILES = (
    "mechinterp/neuron_communities/nec04c_pretraining.py",
    "mechinterp/neuron_communities/nec04c_independent_pretraining.py",
    "mechinterp/neuron_communities/nec04c_native_curriculum.py",
    "mechinterp/neuron_communities/nec04c_independent_native_audit.py",
    "mechinterp/neuron_communities/nec04c_curriculum_aggregate.py",
    "mechinterp/neuron_communities/test_nec04c_curriculum.py",
    ".github/workflows/nec04c-curriculum.yml",
)


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def json_bytes(obj: dict) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def verify_lock() -> tuple[dict, str]:
    f = Path(__file__).with_name(LOCK_NAME)
    obj = json.loads(f.read_text())
    expected = {
        "schema": "NEC-04C/preregistered-parity-curriculum-six-untouched-calibration-worlds/v1",
        "issue": 296,
        "original_registry_sha256": EXPECTED_REGISTRY,
        "original_NEC04A_source_corpus_sha256": EXPECTED_CORPUS,
        "training_world_root_indices": list(ROOTS),
        "arms": list(ARMS),
        "training_steps_per_arm": 900,
        "checkpoint_steps": list(CHECKPOINTS),
        "curriculum_stage1_steps": 240,
        "curriculum_stage2_steps": 660,
        "curriculum_sampler_generator_seed_xor": 0x5DEECE66D,
        "batch_size": 64,
        "lr": 0.003,
        "weight_decay": 0,
        "grad_clip_norm": 1,
        "parity_macro_improvement_minimum": 0.10,
        "parity_positive_roots_minimum": 4,
        "alternate_macro_retention_minimum": -0.05,
        "source_context_groups": 80,
        "owned_validation_context_groups": 24,
        "owned_reserved_context_groups_unopened": 24,
        "original_development_world_root_specs_materialized": 0,
        "original_confirmation_world_root_specs_materialized": 0,
        "source_SAE_training": False,
        "causal_sham_scoring": False,
        "same_initial_weights_within_root": True,
        "single_pilot_budget_selection": False,
        "FSI_01_gate": "REDESIGN_OR_PILOT_ONLY",
    }
    if any(obj.get(key) != value for key, value in expected.items()):
        raise AssertionError("NEC04C frozen cohort, optimizer, training-arm or science gate changed")
    return obj, digest(f)


def code_identity() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {path: digest(root / path) for path in CODE_FILES}


def seed_for_root(index: int) -> int:
    if type(index) is not int or index not in ROOTS:
        raise PermissionError("NEC04C native training restricted to calibration world roots6..11")
    blob = sha256(
        b"NEC04B/native-training-seed/v1|"
        + gym.seed_for_identity("calibration", index)
        + b"composed"
    ).digest()
    return 1 + int.from_bytes(blob[:8], "big") % (2**31 - 2)


def _owned_val_rows(index: int, world: dict):
    _, contexts = gym.contexts_for_world(index)
    if len(contexts["development"]) != 24 or len(contexts["reserved"]) != 24:
        raise AssertionError("calibration-owned context allocation changed")
    focal, context = world["focal_positions"], world["context_positions"]
    for group, assigned in enumerate(contexts["development"]):
        for mask in range(32):
            bits = [0] * 12
            for k, pos in enumerate(focal):
                bits[pos] = (mask >> k) & 1
            for k, pos in enumerate(context):
                bits[pos] = assigned[k]
            yield {
                "root_index": index,
                "world_id": world["world_id"],
                "split": "calibration_world_validation",
                "context_group": 80 + group,
                "focal_mask": mask,
                "history_bits12": bits,
                "composed_label": int(oracle(world["canonical_program"], bits)),
                "mux_selector": bits[world["context_positions"][0]],
            }


def materialize(original: Path, out: Path):
    lock, lock_sha = verify_lock()
    base = original / "nec04a-calibration"
    reg = json.loads((base / "nec04a_world_root_commitments.json").read_text())
    manifest = json.loads((base / "nec04a_calibration_source_manifest.json").read_text())
    worlds_file = base / "nec04a_calibration_worlds.jsonl"
    source_file = base / "nec04a_calibration_source_train.jsonl"
    if (
        reg["registration_sha256"] != EXPECTED_REGISTRY
        or manifest["source_corpus_sha256"] != EXPECTED_CORPUS
        or digest(source_file) != EXPECTED_CORPUS
        or digest(worlds_file) != manifest["calibration_world_specs_sha256"]
    ):
        raise AssertionError("precommitted NEC04A source/seed/root hashes changed")
    worlds = [json.loads(line) for line in worlds_file.read_text().splitlines()]
    corpus = [json.loads(line) for line in source_file.read_text().splitlines()]
    if len(worlds) != 24 or len(corpus) != 61440:
        raise AssertionError("original calibration-only 24-world source artifact truncated")
    out.mkdir(parents=True, exist_ok=True)
    roots = []
    for index in ROOTS:
        world = worlds[index]
        generated, truth = gym.calibration_world(index)
        if world != generated or world["root_index"] != index:
            raise AssertionError("original NEC04A calibration world root mismatch")
        selector = world["context_positions"][0]
        if world["canonical_program"][:2] != ["MUX", ["VAR", selector]]:
            raise AssertionError("original five-bit parity selector changed")
        source_rows = []
        for row in corpus[index * SOURCE : (index + 1) * SOURCE]:
            if (
                row["world_id"] != world["world_id"]
                or row["world_sha256"] != world["world_sha256"]
                or row["split"] != "source_train"
                or row["cohort"] != "calibration"
                or not 0 <= row["source_context_group"] < 80
            ):
                raise AssertionError("source root/context ownership mismatch")
            bits = row["history_bits12"]
            key = sum(int(bit) << k for k, bit in enumerate(bits))
            if row["label"] != (truth >> key) & 1:
                raise AssertionError("original symbolic source label changed")
            source_rows.append({
                "root_index": index,
                "world_id": world["world_id"],
                "split": "source_train",
                "context_group": row["source_context_group"],
                "focal_mask": row["focal_mask"],
                "history_bits12": bits,
                "composed_label": row["label"],
                "mux_selector": bits[selector],
            })
        validation_rows = list(_owned_val_rows(index, world))
        all_rows = source_rows + validation_rows
        seen = {tuple(row["history_bits12"]) for row in all_rows}
        if (
            len(source_rows) != SOURCE or len(validation_rows) != VALIDATION
            or len(seen) != SOURCE + VALIDATION
            or any(sum(x["mux_selector"] == value for x in source_rows) == 0 for value in (0, 1))
            or any(sum(x["mux_selector"] == value for x in validation_rows) == 0 for value in (0, 1))
        ):
            raise AssertionError("calibration-only source/owned-validation group split malformed")
        output = out / f"nec04c_training_root_{index}.jsonl"
        blob = b"".join(json_bytes(row) for row in all_rows)
        output.write_bytes(blob)
        roots.append({
            "root_index": index,
            "world_id": world["world_id"],
            "tier": world["difficulty_tier"],
            "world_sha256": world["world_sha256"],
            "program_sha256": world["program_sha256"],
            "original_truth_4096_sha256": world["truth_table_4096_sha256"],
            "source_cases": SOURCE,
            "owned_validation_cases": VALIDATION,
            "source_selector_one_cases": sum(row["mux_selector"] for row in source_rows),
            "validation_selector_one_cases": sum(row["mux_selector"] for row in validation_rows),
            "training_file": output.name,
            "training_file_sha256": sha256(blob).hexdigest(),
            "matched_native_seed": seed_for_root(index),
        })
    ledger = {
        "schema": SCHEMA,
        "lock_sha256": lock_sha,
        "preflight_source_commit_sha": os.environ.get("GITHUB_SHA", "LOCAL_UNPINNED"),
        "code_sha256_before_training": code_identity(),
        "original_source_registry_sha256": EXPECTED_REGISTRY,
        "original_calibration_source_corpus_sha256": EXPECTED_CORPUS,
        "original_NEC04A_source_artifact_id": lock["original_source_artifact_id"],
        "root_indices": list(ROOTS),
        "training_arms": list(ARMS),
        "checkpoint_steps": list(CHECKPOINTS),
        "optimizer_steps_before_preflight": 0,
        "source_cases_per_root": SOURCE,
        "owned_validation_cases_per_root": VALIDATION,
        "original_reserved_contexts_scored": 0,
        "new_development_world_specs_materialized": 0,
        "new_confirmation_world_specs_materialized": 0,
        "old_NEC03P_worlds_scored": 0,
        "native_SAE_training_updates": 0,
        "FSI_01_gate": "REDESIGN_OR_PILOT_ONLY",
        "roots": roots,
    }
    (out / "nec04c_pretraining_oracles.json").write_bytes(json_bytes(ledger))
    print("NEC04C_FROZEN_NEW_SIX_ROOT_PRETRAINING_ORACLES", len(roots), flush=True)
    return ledger


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--original", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    materialize(a.original, a.out)


if __name__ == "__main__":
    main()

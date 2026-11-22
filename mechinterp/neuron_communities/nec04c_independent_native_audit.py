"""NEC-04C: independent replay of two paired native training arms on each root.

Checks checkpoint file hashes, optimizer state update counters and exact native
12-step logits on all source and originally owned calibration validation cases.
Separately evaluates the frozen MUX-selected parity and alternate branches.
This is independent checkpoint replay, not independent optimizer re-training.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from mechinterp.neuron_communities import nec04a_boolean_gym as gym
from mechinterp.neuron_communities.nec04a_independent_oracle import oracle
from mechinterp.neuron_communities.nec04c_pretraining import (
    ROOTS, ARMS, CHECKPOINTS, SOURCE, VALIDATION,
    digest, verify_lock, code_identity, seed_for_root,
)
from mechinterp.neuron_communities.nec04c_archived_parity_audit import branch_metrics
from mechinterp.neuron_communities.nec04b_independent_native_audit import (
    replay_logits, evaluate_independent,
)
from mechinterp.neuron_communities.nec04c_native_curriculum import SCHEMA
from research.pef07.pef07 import Looper, model_digest, as_tensor

AUDIT_SCHEMA = "NEC-04C/independent-two-arm-native-checkpoint-replay/v1"


def conditional(rows: list[dict], predictions: list[int], truth: list[int]):
    branch = {}
    for selector, name in ((1, "parity"), (0, "alternate")):
        labels = [y for row, y in zip(rows, truth) if row["mux_selector"] == selector]
        preds = [p for row, p in zip(rows, predictions) if row["mux_selector"] == selector]
        branch[name] = branch_metrics(labels, preds)
    if (
        branch["parity"]["correct"] + branch["alternate"]["correct"]
        != sum(p == y for p, y in zip(predictions, truth))
        or branch["parity"]["cases"] + branch["alternate"]["cases"] != len(rows)
    ):
        raise AssertionError("independent MUX branch audit did not reconstruct total")
    return branch


def audit(index: int, pretraining: Path, output: Path):
    if index not in ROOTS:
        raise PermissionError("independent native replay restricted to calibration roots6..11")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    _, lock_sha = verify_lock()
    ledger_file = pretraining / "nec04c_pretraining_oracles.json"
    ledger = json.loads(ledger_file.read_text())
    ref = ledger["roots"][index - ROOTS[0]]
    root_file = pretraining / ref["training_file"]
    if (
        ledger["root_indices"] != list(ROOTS)
        or ledger["training_arms"] != list(ARMS)
        or ledger["checkpoint_steps"] != list(CHECKPOINTS)
        or ledger["lock_sha256"] != lock_sha
        or ledger["code_sha256_before_training"] != code_identity()
        or ref["root_index"] != index
        or ref["matched_native_seed"] != seed_for_root(index)
        or digest(root_file) != ref["training_file_sha256"]
    ):
        raise AssertionError("independent native replay source/preflight/seed changed")
    if os.environ.get("GITHUB_SHA") and ledger["preflight_source_commit_sha"] != os.environ["GITHUB_SHA"]:
        raise AssertionError("independent native replay running from different preregistered code")
    rows = [json.loads(line) for line in root_file.read_text().splitlines()]
    world, _ = gym.calibration_world(index)
    if len(rows) != SOURCE + VALIDATION or world["world_sha256"] != ref["world_sha256"]:
        raise AssertionError("original calibration root changed")
    truth = []
    for k, row in enumerate(rows):
        bits = row["history_bits12"]
        original_label = int(oracle(world["canonical_program"], bits))
        if (
            row["root_index"] != index
            or row["world_id"] != world["world_id"]
            or row["composed_label"] != original_label
            or row["mux_selector"] != bits[world["context_positions"][0]]
            or row["split"] != ("source_train" if k < SOURCE else "calibration_world_validation")
        ):
            raise AssertionError("independent original NEC04A training/validation oracle differs")
        truth.append(original_label)
    score_path = output / "nec04c_native_root_result.json"
    result = json.loads(score_path.read_text())
    if (
        result["schema"] != SCHEMA
        or result["root_index"] != index
        or result["world_sha256"] != world["world_sha256"]
        or result["source_lock_sha256"] != lock_sha
        or result["pretraining_ledger_sha256"] != digest(ledger_file)
        or result["original_source_validation_file_sha256"] != digest(root_file)
        or result["native_initial_weights_shared_between_arms"] is not True
        or result["both_arms_all_four_checkpoints_sealed_before_validation"] is not True
        or result["source_only_training_updates_per_arm"] != 900
        or result["native_SAE_training_updates"] != 0
        or any(result[field] != 0 for field in (
            "original_reserved_contexts_scored",
            "development_world_roots_materialized",
            "confirmation_world_roots_materialized",
            "old_NEC03P_worlds_scored",
        ))
        or result["FSI_01_gate"] != "REDESIGN_OR_PILOT_ONLY"
        or set(result["arms"]) != set(ARMS)
    ):
        raise AssertionError("independent original paired neural protocol differs")
    x = as_tensor([row["history_bits12"] for row in rows])
    auditor_arms = {}
    for arm in ARMS:
        arm_result = result["arms"][arm]
        f = output / f"nec04c.root{index}.{arm}.checkpoints.json"
        if digest(f) != arm_result["sealed_checkpoint_manifest_sha256"]:
            raise AssertionError("independent paired checkpoint seal manifest modified")
        manifest = json.loads(f.read_text())
        if (
            manifest != arm_result["checkpoint_manifest"]
            or manifest["arm"] != arm or manifest["root_index"] != index
            or manifest["continuous_AdamW_optimizer"] is not True
            or manifest["checkpoint_files_frozen_before_any_owned_validation_scoring"] is not True
            or manifest["native_initial_model_sha256"] != result["native_initial_weights_sha256"]
            or manifest["total_updates"] != 900
            or manifest["optimizer_resets"] != 0
            or len(manifest["checkpoints"]) != len(CHECKPOINTS)
            or len(arm_result["scored_checkpoints"]) != len(CHECKPOINTS)
        ):
            raise AssertionError("paired optimizer continuity or initial equality broken")
        observations = []
        for j, step in enumerate(CHECKPOINTS):
            record = manifest["checkpoints"][j]
            scored = arm_result["scored_checkpoints"][j]
            path = output / record["file"]
            if (
                record["step"] != step
                or scored["step"] != step
                or record["file"] != f"nec04c.root{index}.{arm}.step{step}.pt"
                or digest(path) != record["checkpoint_sha256"]
                or record["checkpoint_sha256"] != scored["checkpoint_sha256"]
                or record["native_model_sha256"] != scored["native_model_sha256"]
            ):
                raise AssertionError("historical frozen 240/900 native checkpoint SHA mismatch")
            frozen = torch.load(path, map_location="cpu", weights_only=True)
            if (
                frozen["schema"] != SCHEMA or frozen["root_index"] != index
                or frozen["training_arm"] != arm
                or frozen["world_sha256"] != world["world_sha256"]
                or frozen["source_lock_sha256"] != lock_sha
                or frozen["source_validation_sha256"] != digest(root_file)
                or frozen["native_seed"] != seed_for_root(index)
                or frozen["native_initial_model_sha256"] != result["native_initial_weights_sha256"]
                or frozen["native_model_sha256"] != record["native_model_sha256"]
                or frozen["step"] != step or frozen["optimizer_updates"] != step
                or frozen["optimizer"] != "AdamW"
                or frozen["lr"] != .003 or frozen["weight_decay"] != 0
                or frozen["batch_size"] != 64 or frozen["gradient_clip_norm"] != 1.
                or frozen["sampler_stage1_updates"] != (240 if arm == "parity_curriculum" else 0)
                or frozen["sampler_stage2_updates"] != (660 if arm == "parity_curriculum" else 0)
            ):
                raise AssertionError("original paired model/optimizer sampler contract mutated")
            counters = [
                int(v["step"].item())
                for v in frozen["optimizer_state"]["state"].values()
                if "step" in v
            ]
            if not counters or any(count != step for count in counters):
                raise AssertionError("continuous AdamW state update counters do not match budget")
            native = Looper().eval()
            native.load_state_dict(frozen["model_state"], strict=True)
            if model_digest(native) != record["native_model_sha256"]:
                raise AssertionError("independently replayed native Looper weight hash mismatch")
            source_logits = replay_logits(native, x[:SOURCE])
            validation_logits = replay_logits(native, x[SOURCE:])
            old_source = scored["source_context_training_metrics"]
            old_validation = scored["owned_calibration_world_validation_metrics"]
            src = evaluate_independent(source_logits, truth[:SOURCE], old_source)
            val = evaluate_independent(validation_logits, truth[SOURCE:], old_validation)
            observations.append({
                "step": step,
                "checkpoint_sha256": record["checkpoint_sha256"],
                "native_model_sha256": record["native_model_sha256"],
                "source": src,
                "owned_validation": val,
                "source_branches": conditional(
                    rows[:SOURCE], old_source["class_predictions"], truth[:SOURCE]
                ),
                "owned_validation_branches": conditional(
                    rows[SOURCE:], old_validation["class_predictions"], truth[SOURCE:]
                ),
            })
            print("NEC04C_INDEPENDENT_SOURCE_AND_VALIDATION_NATIVE_REPLAY_PASS",
                  index, arm, step, val["correct"], flush=True)
        auditor_arms[arm] = observations
    report = {
        "schema": AUDIT_SCHEMA,
        "root_index": index,
        "world_id": world["world_id"],
        "source_lock_sha256": lock_sha,
        "pretraining_teacher_ledger_sha256": digest(ledger_file),
        "original_root_scoring_receipt_sha256": digest(score_path),
        "original_source_validation_file_sha256": digest(root_file),
        "same_native_initial_weights_across_treatment_arms_verified": True,
        "four_original_checkpoints_byte_SHA256_verified": True,
        "four_original_optimizer_step_counters_verified": True,
        "all_four_native_source_validation_logits_independently_replayed": True,
        "training_trajectory_independently_reoptimized": False,
        "native_SAE_updates": 0,
        "reserved_contexts_scored": 0,
        "development_world_roots_generated": 0,
        "confirmation_world_roots_generated": 0,
        "FSI_01_gate": "REDESIGN_OR_PILOT_ONLY",
        "arms": auditor_arms,
    }
    (output / "nec04c_independent_native_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root-index", required=True, type=int)
    p.add_argument("--pretraining", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    audit(args.root_index, args.pretraining, args.out)


if __name__ == "__main__":
    main()

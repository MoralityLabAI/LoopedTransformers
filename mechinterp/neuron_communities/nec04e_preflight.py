"""NEC-04E pre-inference provenance and fixed probe registration.

Read original NEC04C root artifacts, verify all 24 native checkpoint bytes,
previous independent logits and original six-world source/validation inputs
with the NEC04D archive auditor. Seal code hashes before any new h1..h12
inference or source-only ridge fitting. No native training.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

from mechinterp.neuron_communities import nec04a_boolean_gym as gym
from mechinterp.neuron_communities.nec04c_pretraining import (
    ROOTS, ARMS, CHECKPOINTS, SOURCE, VALIDATION,
    code_identity as original_code_identity,
    seed_for_root, verify_lock as verify_nec04c_lock,
)
from mechinterp.neuron_communities.nec04d_parity_walsh_audit import audit as old_audit

SCHEMA = "NEC-04E/preinference-original-six-root-native-checkpoint-receipts/v1"
LOCK = "NEC04E_PROBE_LOCK.json"
PINNED_PARENT_SHA = "b6e7a9a2e2b530e738967aa15d37c265643e98b4"
CODE_FILES = (
    "mechinterp/neuron_communities/NEC04E_PROBE_LOCK.json",
    "mechinterp/neuron_communities/NEC04E_PROTOCOL.md",
    "mechinterp/neuron_communities/nec04e_preflight.py",
    "mechinterp/neuron_communities/nec04e_hidden_probe.py",
    "mechinterp/neuron_communities/nec04e_independent_probe_audit.py",
    "mechinterp/neuron_communities/nec04e_aggregate.py",
    "mechinterp/neuron_communities/test_nec04e_hidden_probe.py",
    ".github/workflows/nec04e-hidden-probe.yml",
    "research/pef07/pef07.py",
    "mechinterp/neuron_communities/nec04d_parity_walsh_audit.py",
    "mechinterp/neuron_communities/nec04c_pretraining.py",
)


def sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def require(condition: bool, reason: str):
    if not condition:
        raise AssertionError(reason)


def lock():
    path = Path(__file__).with_name(LOCK)
    spec = json.loads(path.read_text())
    fixed = {
        "schema": "NEC-04E/read-only-source-fit-loopwise-hidden-probe-commit-lock/v1",
        "issue": 302,
        "parent_nec04c_workflow_run": 44993859839,
        "parent_nec04c_code_commit": PINNED_PARENT_SHA,
        "pilot_calibration_world_root_indices": list(ROOTS),
        "existing_native_arms": list(ARMS),
        "existing_optimizer_steps": list(CHECKPOINTS),
        "sequential_recurrent_visits": 12,
        "original_source_examples_per_root": SOURCE,
        "original_owned_validation_examples_per_root": VALIDATION,
        "original_reserved_context_groups_scored": 0,
        "new_development_WORLD_roots_instantiated": 0,
        "new_confirmation_WORLD_roots_instantiated": 0,
        "old_NEC03P_recruited_worlds_scored": 0,
        "target_tasks": [
            "prefix1", "prefix2", "prefix_current", "future_unseen",
            "full_parity_teacher",
        ],
        "probe_representations": [
            "trained_hidden", "random_initial_hidden", "visible_raw_bits",
            "source_shuffled_trained_hidden",
        ],
        "source_fit_only": True,
        "ridge_penalty": 1.0,
        "ridge_source_std_min": 0.0001,
        "ridge_intercept_penalty": 0,
        "prediction_cutoff": 0.5,
        "original_native_final_logit_max_abs_error": .0003,
        "original_native_full_memory_replay_max_abs_error": .0003,
        "hidden_embedding_size": 32,
        "new_task_model_optimizer_updates": 0,
        "source_SAE_training_updates": 0,
        "old_causal_sham_outcomes_scored": 0,
        "decision_threshold_minimum_owned_val_accuracy": .70,
        "decision_threshold_margin_vs_best_random_and_raw": .10,
        "decision_minimum_positive_roots": 4,
        "FSI_01_gate": "REDESIGN_OR_PILOT_ONLY",
    }
    require(all(spec.get(k) == value for k, value in fixed.items()),
            "registered prior cohorts, probes, controls, label sources or gates changed")
    require(
        spec["parent_nec04c_pretraining_artifact_id"] == 14742886436
        and spec["parent_nec04c_root_artifact_ids"] == {
            str(k): v for k, v in zip(ROOTS, (
                14742124547, 14743106877, 14743221555,
                14742897690, 14743391255, 14742099591,
            ))
        },
        "prior root artifacts changed",
    )
    return spec, sha(path)


def code_identity():
    root = Path(__file__).resolve().parents[2]
    return {name: sha(root / name) for name in CODE_FILES}


def preflight(old_roots: Path, old_pretraining: Path, out: Path):
    spec, lock_sha = lock()
    original_lock, original_lock_sha = verify_nec04c_lock()
    require(original_lock["training_world_root_indices"] == list(ROOTS),
            "older training-world roster modified")
    old_ledger_file = old_pretraining / "nec04c_pretraining_oracles.json"
    old_ledger = json.loads(old_ledger_file.read_text())
    require(
        old_ledger["code_sha256_before_training"] == original_code_identity()
        and old_ledger["preflight_source_commit_sha"] == PINNED_PARENT_SHA
        and old_ledger["root_indices"] == list(ROOTS)
        and old_ledger["training_arms"] == list(ARMS)
        and old_ledger["checkpoint_steps"] == list(CHECKPOINTS)
        and old_ledger["lock_sha256"] == original_lock_sha
        and old_ledger["original_source_registry_sha256"]
        == spec["original_NEC04A_seed_registry_sha256"]
        and old_ledger["original_calibration_source_corpus_sha256"]
        == spec["original_NEC04A_corpus_sha256"]
        and old_ledger["optimizer_steps_before_preflight"] == 0
        and old_ledger["FSI_01_gate"] == "REDESIGN_OR_PILOT_ONLY",
        "historical training seed/data/source program or code SHA changed",
    )
    out.mkdir(parents=True, exist_ok=True)
    # Reuse the already independent, pure-stdlib original score/label audit.
    # This verifies each old 12-bit history, scalar teacher label, complete
    # context ownership, all 24 old model checkpoint bytes and saved logits.
    historical = old_audit(old_roots, old_pretraining, out)
    require(
        historical["old_native_checkpoint_byte_SHA_verified"] == 24
        and historical["original_independent_native_replay_receipts_checked"] == 6
        and historical["original_reserved_contexts_scored"] == 0
        and historical["development_world_roots_instantiated"] == 0
        and historical["confirmation_world_roots_instantiated"] == 0
        and historical["new_model_inferences"] == 0,
        "older native checkpoint or protected cohort evidence incomplete",
    )
    roots = {}
    for index in ROOTS:
        ref = old_ledger["roots"][index - ROOTS[0]]
        folder = old_roots / f"nec04c-root-{index}"
        score_file = folder / "nec04c_native_root_result.json"
        replay_file = folder / "nec04c_independent_native_audit.json"
        native = json.loads(score_file.read_text())
        replay = json.loads(replay_file.read_text())
        require(
            native["root_index"] == index and replay["root_index"] == index
            and native["original_source_validation_file_sha256"]
            == ref["training_file_sha256"]
            and native["native_initial_weights_shared_between_arms"] is True
            and replay["same_native_initial_weights_across_treatment_arms_verified"]
            and all(replay[k] for k in (
                "four_original_checkpoints_byte_SHA256_verified",
                "four_original_optimizer_step_counters_verified",
                "all_four_native_source_validation_logits_independently_replayed",
            ))
            and ref["matched_native_seed"] == seed_for_root(index),
            "historical native pair failed preinference provenance",
        )
        checkpoints = {}
        for arm in ARMS:
            manifest = native["arms"][arm]["checkpoint_manifest"]
            checkpoints[arm] = {}
            for k, step in enumerate(CHECKPOINTS):
                record = manifest["checkpoints"][k]
                file = folder / record["file"]
                require(
                    record["step"] == step
                    and sha(file) == record["checkpoint_sha256"]
                    and record["checkpoint_sha256"]
                    == replay["arms"][arm][k]["checkpoint_sha256"],
                    "historical optimizer checkpoint SHA altered before hidden probe",
                )
                checkpoints[arm][str(step)] = {
                    "checkpoint_file": file.name,
                    "checkpoint_sha256": record["checkpoint_sha256"],
                    "model_sha256": record["native_model_sha256"],
                }
        roots[str(index)] = {
            "world_id": ref["world_id"],
            "world_sha256": ref["world_sha256"],
            "source_validation_sha256": ref["training_file_sha256"],
            "original_score_receipt_sha256": sha(score_file),
            "original_independent_replay_sha256": sha(replay_file),
            "initial_weights_sha256": native["native_initial_weights_sha256"],
            "matched_model_seed": seed_for_root(index),
            "original_model_checkpoints": checkpoints,
        }
    commit = os.environ.get("GITHUB_SHA", "LOCAL_UNPINNED")
    report = {
        "schema": SCHEMA,
        "preinference_lock_sha256": lock_sha,
        "preinference_source_commit_sha": commit,
        "original_nec04c_pretraining_sha256": sha(old_ledger_file),
        "original_nec04d_complete_archive_audit_sha256":
            sha(out / "nec04d_original_parity_walsh_audit.json"),
        "new_probe_code_sha256_before_new_hidden_inference": code_identity(),
        "original_NEC04A_seed_registry_sha256":
            spec["original_NEC04A_seed_registry_sha256"],
        "root_indices": list(ROOTS),
        "native_arms": list(ARMS),
        "original_native_checkpoint_budgets": list(CHECKPOINTS),
        "all_24_historical_native_checkpoint_byte_shas_verified": True,
        "six_prior_native_full_logit_audits_reverified": True,
        "source_only_probe_fit": True,
        "new_hidden_inference_executed_at_preflight": False,
        "new_task_model_training_updates": 0,
        "new_source_SAE_updates": 0,
        "original_reserved_contexts_scored": 0,
        "new_development_world_roots_generated": 0,
        "new_confirmation_world_roots_generated": 0,
        "FSI_01_gate": "REDESIGN_OR_PILOT_ONLY",
        "roots": roots,
    }
    (out / "nec04e_preinference_lock.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    print("NEC04E_ORIGINAL_24_CHECKPOINTS_AND_SOURCE_PROBE_PREFLIGHT_PASS",
          len(ROOTS), len(ROOTS)*len(ARMS)*len(CHECKPOINTS), flush=True)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--old-roots", required=True, type=Path)
    p.add_argument("--pretraining", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    preflight(args.old_roots, args.pretraining, args.out)


if __name__ == "__main__":
    main()

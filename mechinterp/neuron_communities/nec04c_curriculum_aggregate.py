"""NEC-04C: fixed six-world paired parity curriculum decision, pure stdlib.

The statistical unit is one precommitted calibration world. All six independent
world roots, both matched training arms, both checkpoint budgets and each prior
native checkpoint audit must be present. No development/confirmation world is
scored and no SAE or causal claim can be promoted by this script.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import sqrt
from pathlib import Path

ROOTS = tuple(range(6, 12))
ARMS = ("vanilla", "parity_curriculum")
BUDGETS = (240, 900)
SOURCE = 2560
VALIDATION = 768
SCHEMA = "NEC-04C/six-calibration-world-paired-parity-curriculum-aggregate/v1"
PRIMARY_FLOOR = 0.10
MIN_POSITIVE_ROOTS = 4
ALT_RETENTION_FLOOR = -0.05


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def macro_difference(rows: list[dict], field: str) -> float:
    return sum(row[field] for row in rows) / len(rows)


def aggregate(root_dir: Path, pretraining: Path, out: Path):
    original_ledger = pretraining / "nec04c_pretraining_oracles.json"
    ledger = json.loads(original_ledger.read_text())
    if (
        ledger["root_indices"] != list(ROOTS)
        or ledger["training_arms"] != list(ARMS)
        or ledger["checkpoint_steps"] != list(BUDGETS)
        or ledger["optimizer_steps_before_preflight"] != 0
        or len(ledger["roots"]) != len(ROOTS)
    ):
        raise AssertionError("six-root preregistered native curriculum pretraining invalid")
    expected = {f"nec04c-root-{index}" for index in ROOTS}
    found = {p.name for p in root_dir.iterdir() if p.is_dir()}
    if found != expected:
        raise AssertionError("all six preregistered native root artifacts are required")
    roots = []
    for ordinal, index in enumerate(ROOTS):
        folder = root_dir / f"nec04c-root-{index}"
        scored_file = folder / "nec04c_native_root_result.json"
        replay_file = folder / "nec04c_independent_native_audit.json"
        original = json.loads(scored_file.read_text())
        replay = json.loads(replay_file.read_text())
        ref = ledger["roots"][ordinal]
        if (
            original["schema"] != "NEC-04C/native-matched-parity-curriculum-pilot/v1"
            or replay["schema"] != "NEC-04C/independent-two-arm-native-checkpoint-replay/v1"
            or original["root_index"] != index or replay["root_index"] != index
            or original["world_id"] != ref["world_id"]
            or replay["world_id"] != ref["world_id"]
            or original["world_sha256"] != ref["world_sha256"]
            or original["source_lock_sha256"] != ledger["lock_sha256"]
            or replay["source_lock_sha256"] != ledger["lock_sha256"]
            or original["pretraining_ledger_sha256"] != digest(original_ledger)
            or replay["pretraining_teacher_ledger_sha256"] != digest(original_ledger)
            or replay["original_root_scoring_receipt_sha256"] != digest(scored_file)
            or original["original_source_validation_file_sha256"]
            != ref["training_file_sha256"]
            or replay["original_source_validation_file_sha256"]
            != ref["training_file_sha256"]
            or original["native_initial_weights_shared_between_arms"] is not True
            or replay["same_native_initial_weights_across_treatment_arms_verified"] is not True
            or replay["four_original_checkpoints_byte_SHA256_verified"] is not True
            or replay["four_original_optimizer_step_counters_verified"] is not True
            or replay["all_four_native_source_validation_logits_independently_replayed"] is not True
            or replay["training_trajectory_independently_reoptimized"] is not False
            or original["source_only_training_updates_per_arm"] != 900
            or original["source_train_rows"] != SOURCE
            or original["owned_validation_rows"] != VALIDATION
            or original["native_SAE_training_updates"] != 0
            or replay["native_SAE_updates"] != 0
            or original["FSI_01_gate"] != "REDESIGN_OR_PILOT_ONLY"
            or replay["FSI_01_gate"] != "REDESIGN_OR_PILOT_ONLY"
            or any(original[f] != 0 for f in (
                "original_reserved_contexts_scored",
                "development_world_roots_materialized",
                "confirmation_world_roots_materialized",
                "old_NEC03P_worlds_scored",
            ))
            or any(replay[f] != 0 for f in (
                "reserved_contexts_scored",
                "development_world_roots_generated",
                "confirmation_world_roots_generated",
            ))
            or set(original["arms"]) != set(ARMS)
            or set(replay["arms"]) != set(ARMS)
        ):
            raise AssertionError("original fixed six-root calibration model evidence incomplete")
        per_arm = {}
        for arm in ARMS:
            scored = original["arms"][arm]["scored_checkpoints"]
            audited = replay["arms"][arm]
            manifest = original["arms"][arm]["checkpoint_manifest"]
            if (
                len(scored) != 2 or len(audited) != 2
                or len(manifest["checkpoints"]) != 2
                or manifest["total_updates"] != 900
                or manifest["optimizer_resets"] != 0
                or manifest["native_initial_model_sha256"]
                != original["native_initial_weights_sha256"]
            ):
                raise AssertionError("native two-arm optimizer checkpoint schedule incomplete")
            checkpoints = []
            for j, step in enumerate(BUDGETS):
                observed = audited[j]
                native = scored[j]
                if (
                    observed["step"] != step or native["step"] != step
                    or observed["checkpoint_sha256"] != native["checkpoint_sha256"]
                    or observed["native_model_sha256"] != native["native_model_sha256"]
                    or observed["source"]["correct"]
                    != native["source_context_training_metrics"]["correct"]
                    or observed["owned_validation"]["correct"]
                    != native["owned_calibration_world_validation_metrics"]["correct"]
                    or observed["source"]["cases"] != SOURCE
                    or observed["owned_validation"]["cases"] != VALIDATION
                ):
                    raise AssertionError("independent native root training/validation replay differs")
                branches = observed["owned_validation_branches"]
                if (
                    branches["parity"]["cases"] + branches["alternate"]["cases"] != VALIDATION
                    or branches["parity"]["correct"] + branches["alternate"]["correct"]
                    != observed["owned_validation"]["correct"]
                    or abs(branches["parity"]["majority_baseline_accuracy"] - 0.5) > 1e-12
                ):
                    raise AssertionError("original five-bit parity branch accuracy or class prior broken")
                checkpoints.append({
                    "step": step,
                    "source_accuracy": observed["source"]["accuracy"],
                    "owned_validation_accuracy": observed["owned_validation"]["accuracy"],
                    "parity": branches["parity"],
                    "alternate": branches["alternate"],
                    "checkpoint_sha256": observed["checkpoint_sha256"],
                    "model_sha256": observed["native_model_sha256"],
                })
            per_arm[arm] = checkpoints
        rows = []
        for j, step in enumerate(BUDGETS):
            vanilla = per_arm["vanilla"][j]
            curriculum = per_arm["parity_curriculum"][j]
            if (
                vanilla["parity"]["cases"] != curriculum["parity"]["cases"]
                or vanilla["alternate"]["cases"] != curriculum["alternate"]["cases"]
            ):
                raise AssertionError("paired native teachers changed original validation denominators")
            rows.append({
                "step": step,
                "vanilla_whole_accuracy": vanilla["owned_validation_accuracy"],
                "curriculum_whole_accuracy": curriculum["owned_validation_accuracy"],
                "vanilla_parity_accuracy": vanilla["parity"]["accuracy"],
                "curriculum_parity_accuracy": curriculum["parity"]["accuracy"],
                "parity_delta": (
                    curriculum["parity"]["accuracy"] - vanilla["parity"]["accuracy"]
                ),
                "vanilla_alternate_accuracy": vanilla["alternate"]["accuracy"],
                "curriculum_alternate_accuracy": curriculum["alternate"]["accuracy"],
                "alternate_retention_delta": (
                    curriculum["alternate"]["accuracy"] - vanilla["alternate"]["accuracy"]
                ),
                "full_task_delta": (
                    curriculum["owned_validation_accuracy"] - vanilla["owned_validation_accuracy"]
                ),
                "owned_validation_parity_cases": vanilla["parity"]["cases"],
                "owned_validation_alternate_cases": vanilla["alternate"]["cases"],
            })
        roots.append({
            "root_index": index,
            "world_id": ref["world_id"],
            "tier": ref["tier"],
            "original_world_sha256": ref["world_sha256"],
            "original_source_validation_sha256": ref["training_file_sha256"],
            "native_initial_weights_sha256": original["native_initial_weights_sha256"],
            "original_scored_result_sha256": digest(scored_file),
            "original_independent_native_replay_sha256": digest(replay_file),
            "arms": per_arm,
            "paired_budget_rows": rows,
        })
    snapshots = []
    for j, budget in enumerate(BUDGETS):
        comparisons = [root["paired_budget_rows"][j] for root in roots]
        parity = [row["parity_delta"] for row in comparisons]
        mean_parity = sum(parity) / len(parity)
        variance = sum((x - mean_parity) ** 2 for x in parity) / (len(parity) - 1)
        positive = sum(x > 0 for x in parity)
        alternate = macro_difference(comparisons, "alternate_retention_delta")
        passes = (
            budget == 900
            and mean_parity >= PRIMARY_FLOOR
            and positive >= MIN_POSITIVE_ROOTS
            and alternate >= ALT_RETENTION_FLOOR
        )
        snapshots.append({
            "step": budget,
            "root_count": 6,
            "parity_root_macro_curriculum_minus_vanilla": mean_parity,
            "parity_root_sd_descriptive": sqrt(variance),
            "parity_root_standard_error_descriptive": sqrt(variance / 6),
            "positive_parity_root_deltas": positive,
            "alternate_root_macro_curriculum_minus_vanilla": alternate,
            "full_task_root_macro_curriculum_minus_vanilla": macro_difference(
                comparisons, "full_task_delta"
            ),
            "per_root_parity_deltas": parity,
            "per_root_alternate_retention_deltas": [
                row["alternate_retention_delta"] for row in comparisons
            ],
            "predeclared_minimum_parity_delta": PRIMARY_FLOOR,
            "predeclared_minimum_positive_roots": MIN_POSITIVE_ROOTS,
            "predeclared_alternate_retention_floor": ALT_RETENTION_FLOOR,
            "pilot_gate_passes": passes,
        })
    decision = (
        "CALIBRATION_PARITY_CURRICULUM_PILOT_PASS_NEEDS_NEW_NATIVE_HEADROOM_AND_SAE_PROTOCOL"
        if snapshots[-1]["pilot_gate_passes"]
        else "STOP_PARITY_CURRICULUM_CALIBRATION_PILOT"
    )
    result = {
        "schema": SCHEMA,
        "pretraining_ledger_sha256": digest(original_ledger),
        "root_indices": list(ROOTS),
        "experimental_unit": "one newly untouched NEC04A calibration world root",
        "all_six_original_worlds_paired_and_independently_audited": True,
        "all_24_native_checkpoints_hash_and_logit_audited": True,
        "checkpoint_budgets": list(BUDGETS),
        "source_SAE_fitted": False,
        "causal_sham_or_neuron_necessity_scored": False,
        "reserved_contexts_scored": 0,
        "development_world_roots_scored": 0,
        "confirmation_world_roots_scored": 0,
        "old_NEC03P_worlds_scored": 0,
        "FSI_01_gate": "REDESIGN_OR_PILOT_ONLY",
        "snapshots": snapshots,
        "roots": roots,
        "decision": decision,
        "selected_native_headroom_budget": None,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "nec04c_six_root_paired_curriculum_aggregate.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        "NEC04C_PAIRED_SIX_WORLD_CALIBRATION_RESULTS",
        "parity_delta", snapshots[-1]["parity_root_macro_curriculum_minus_vanilla"],
        "positive_roots", snapshots[-1]["positive_parity_root_deltas"],
        "alternate_retention", snapshots[-1]["alternate_root_macro_curriculum_minus_vanilla"],
        "decision", decision,
        flush=True,
    )
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--roots", type=Path, required=True)
    p.add_argument("--pretraining", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    aggregate(args.roots, args.pretraining, args.out)


if __name__ == "__main__":
    main()

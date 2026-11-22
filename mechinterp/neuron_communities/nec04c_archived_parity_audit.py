"""NEC-04C: read-only archived NEC-04B branch-level diagnostic.

Consumes only the already independently audited NEC-04B calibration worlds 0..5.
Checks all 48 historical model/checkpoint blobs, teacher inputs, scored logits,
and independent replay receipts before computing source and owned-validation
branch accuracy and class priors. Performs no model inference or training.
No new NEC04A world roots, reserved contexts, or NEC03P data are opened.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path

from mechinterp.neuron_communities import nec04a_boolean_gym as gym
from mechinterp.neuron_communities.nec04a_independent_oracle import (
    oracle,
    verify_lock as verify_nec04a_lock,
)

ROOTS = tuple(range(6))
ARMS = ("composed", "additive")
BUDGETS = (120, 240, 480, 900)
SOURCE = 2560
VALIDATION = 768
SCHEMA = "NEC-04C/read-only-NEC04B-archived-branch-audit/v1"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def branch_metrics(truth: list[int], predictions: list[int]) -> dict:
    """Compute correctness and balanced accuracy without reweighting case arms."""
    require(len(truth) == len(predictions) and bool(truth), "empty/mismatched branch")
    require(set(truth) <= {0, 1} and set(predictions) <= {0, 1}, "nonbinary labels")
    positive = sum(truth)
    negative = len(truth) - positive
    tp = sum(y == p == 1 for y, p in zip(truth, predictions))
    tn = sum(y == p == 0 for y, p in zip(truth, predictions))
    return {
        "cases": len(truth),
        "correct": tp + tn,
        "accuracy": (tp + tn) / len(truth),
        "truth_positives": positive,
        "truth_negatives": negative,
        "majority_baseline_accuracy": max(positive, negative) / len(truth),
        "predicted_positives": sum(predictions),
        "balanced_accuracy": (
            (tp / positive + tn / negative) / 2
            if positive > 0 and negative > 0 else None
        ),
    }


def validate_rows(folder: Path, reference: dict, index: int) -> tuple[list[dict], int]:
    world, _ = gym.calibration_world(index)
    require(reference["root_index"] == index, "root index changed")
    require(reference["world_id"] == world["world_id"], "world root mismatch")
    require(reference["world_sha256"] == world["world_sha256"], "world SHA mismatch")
    require(reference["program_sha256"] == world["program_sha256"], "program changed")
    f = folder / reference["pilot_file"]
    require(
        f.name == f"nec04b_pilot_root_{index}.jsonl"
        and digest(f) == reference["pilot_file_sha256"],
        "archived pretraining source/validation file changed",
    )
    lines = f.read_text().splitlines()
    require(len(lines) == SOURCE + VALIDATION, "old root data incomplete")
    records = [json.loads(line) for line in lines]
    source_contexts, parts = gym.contexts_for_world(index)
    development_contexts = parts["development"]
    require(
        len(source_contexts) == 80 and len(development_contexts) == 24
        and len(parts["reserved"]) == 24,
        "original source/owned-context split changed",
    )
    require(
        not set(source_contexts) & set(development_contexts)
        and not set(source_contexts) & set(parts["reserved"])
        and not set(development_contexts) & set(parts["reserved"]),
        "source, owned-validation, and reserved contexts overlap",
    )
    ast = world["canonical_program"]
    require(
        ast[0] == "MUX" and ast[1] == ["VAR", world["context_positions"][0]],
        "parity selector differs from original source-root program",
    )
    seen = set()
    linear = reference["additive_spec"]
    weights = linear["signed_weights_12"]
    cut = linear["threshold_twice"]
    for k, row in enumerate(records):
        source = k < SOURCE
        context_idx = k // 32 if source else (k - SOURCE) // 32
        original_context = (
            source_contexts[context_idx] if source else development_contexts[context_idx]
        )
        mask = k % 32
        reconstructed = [0] * 12
        for pos, bit in enumerate(world["focal_positions"]):
            reconstructed[bit] = (mask >> pos) & 1
        for pos, bit in enumerate(world["context_positions"]):
            reconstructed[bit] = original_context[pos]
        require(
            row["root_index"] == index and row["world_id"] == world["world_id"]
            and row["split"] == (
                "source_train" if source else "calibration_world_validation"
            )
            and row["context_group"] == (context_idx if source else 80 + context_idx)
            and row["focal_mask"] == mask
            and row["history_bits12"] == reconstructed,
            "original 32-mask source/owned-validation root ownership changed",
        )
        require(tuple(reconstructed) not in seen, "duplicate old source context")
        seen.add(tuple(reconstructed))
        composed = int(oracle(ast, reconstructed))
        additive = int(
            2 * sum(w * (2 * x - 1) for w, x in zip(weights, reconstructed)) > cut
        )
        require(
            row["composed_label"] == composed and row["additive_label"] == additive,
            "old source-oracle target labels changed",
        )
    require(len(seen) == SOURCE + VALIDATION, "original 12-bit histories incomplete")
    return records, world["context_positions"][0]


def audit_split(metrics: dict, rows: list[dict], arm: str, selector: int) -> dict:
    labels = [row[f"{arm}_label"] for row in rows]
    logits = metrics["native_logits"]
    preds = metrics["class_predictions"]
    require(
        len(labels) == len(logits) == len(preds) == metrics["cases"],
        "original saved model prediction coverage incomplete",
    )
    recomputed = []
    groups = {"parity": ([], []), "alternate": ([], [])}
    for row, target, scores, prediction in zip(rows, labels, logits, preds):
        require(
            len(scores) == 2 and all(isfinite(float(value)) for value in scores),
            "nonfinite old native model logits",
        )
        pred = int(scores[1] > scores[0])
        require(pred == prediction, "saved checkpoint logits/predictions mismatch")
        recomputed.append(pred)
        key = "parity" if row["history_bits12"][selector] else "alternate"
        groups[key][0].append(int(target))
        groups[key][1].append(pred)
    whole = branch_metrics(labels, recomputed)
    require(
        whole["correct"] == metrics["correct"]
        and whole["cases"] == metrics["cases"]
        and abs(whole["accuracy"] - metrics["accuracy"]) < 1e-12,
        "original checkpoint correctness receipt changed",
    )
    require(
        whole["truth_positives"] == metrics["truth_positive_cases"]
        and whole["balanced_accuracy"] is not None
        and abs(whole["balanced_accuracy"] - metrics["balanced_accuracy"]) < 1e-12,
        "original checkpoint balanced accuracy changed",
    )
    p = branch_metrics(*groups["parity"])
    a = branch_metrics(*groups["alternate"])
    require(p["cases"] + a["cases"] == len(rows), "branch partition lost examples")
    require(
        p["correct"] + a["correct"] == metrics["correct"],
        "conditional branch metrics fail to reconstruct original headroom",
    )
    return {"whole": whole, "parity": p, "alternate": a}


def audit(old_roots: Path, pretraining: Path, out: Path) -> dict:
    verify_nec04a_lock()
    ledger_file = pretraining / "nec04b_pretraining_oracles.json"
    ledger = json.loads(ledger_file.read_text())
    require(
        ledger["root_indices"] == list(ROOTS) and ledger["native_arms"] == list(ARMS)
        and ledger["training_step_budgets"] == list(BUDGETS)
        and ledger["model_optimizer_steps_before_preflight"] == 0
        and len(ledger["roots"]) == len(ROOTS),
        "previously registered six-root, two-teacher, four-budget roster changed",
    )
    require(
        ledger["original_calibration_reserved_context_groups_scored"] == 0
        and ledger["development_world_specs_instantiated"] == 0
        and ledger["confirmation_world_specs_instantiated"] == 0,
        "protected old cohort metadata altered",
    )
    require(
        {p.name for p in old_roots.iterdir() if p.is_dir()}
        == {f"nec04b-pilot-root-{i}" for i in ROOTS},
        "incomplete or additional historical root artifacts",
    )
    inspected = []
    for index in ROOTS:
        ref = ledger["roots"][index]
        rows, selector = validate_rows(pretraining, ref, index)
        folder = old_roots / f"nec04b-pilot-root-{index}"
        original_file = folder / "nec04b_native_headroom_root.json"
        replay_file = folder / "nec04b_independent_native_headroom_audit.json"
        original = json.loads(original_file.read_text())
        replay = json.loads(replay_file.read_text())
        require(
            original["schema"] == "NEC-04B/native-headroom-six-calibration-root-pilot/v1"
            and replay["schema"] == "NEC-04B/independent-frozen-native-checkpoint-headroom-replay/v1"
            and original["root_index"] == index and replay["root_index"] == index
            and original["world_sha256"] == ref["world_sha256"]
            and replay["world_sha256"] == ref["world_sha256"]
            and original["pretraining_teacher_oracle_sha256"] == digest(ledger_file)
            and replay["original_pretraining_oracle_ledger_sha256"] == digest(ledger_file)
            and replay["original_root_score_receipt_sha256"] == digest(original_file),
            "original independent checkpoint audit no longer matches original results",
        )
        for field in (
            "all_8_frozen_checkpoints_native_model_SHA256_verified",
            "all_8_optimizer_state_step_counts_verified",
            "all_8_source_and_validation_full_logits_independently_replayed",
        ):
            require(replay[field] is True, f"independent historical audit missing {field}")
        require(
            replay["training_trajectory_independently_reoptimized"] is False
            and original["native_SAE_training_updates"] == 0,
            "historical model/SAE scope changed",
        )
        require(
            original["original_calibration_root_reserved_contexts_scored"] == 0
            and original["new_development_or_confirmation_world_specs_materialized"] == 0
            and replay["original_reserved_contexts_scored"] == 0,
            "protected historical model outcomes opened",
        )
        arms = {}
        for arm in ARMS:
            original_arm = original["arms"][arm]
            audited_arm = replay["arms"][arm]
            manifest_file = folder / f"nec04b.root{index}.{arm}.checkpoints.json"
            require(
                digest(manifest_file) == original_arm["checkpoint_seal_manifest_sha256"],
                "frozen checkpoint seal manifest changed",
            )
            manifest = json.loads(manifest_file.read_text())
            require(
                manifest == original_arm["checkpoint_seal_manifest"]
                and manifest["all_four_checkpoints_sealed_before_validation"] is True
                and manifest["optimizer_resets"] == 0
                and manifest["updates_total"] == 900
                and len(manifest["checkpoints"]) == len(BUDGETS)
                and len(audited_arm) == len(BUDGETS),
                "historical optimizer trajectory seal inconsistent",
            )
            outcomes = []
            for j, step in enumerate(BUDGETS):
                item = original_arm["source_and_validation_results"][j]
                rec = manifest["checkpoints"][j]
                audited = audited_arm[j]
                checkpoint_file = folder / rec["file"]
                require(
                    rec["step"] == step and item["step"] == step
                    and audited["step"] == step
                    and checkpoint_file.name == f"nec04b.root{index}.{arm}.step{step}.pt"
                    and digest(checkpoint_file) == rec["checkpoint_sha256"]
                    and rec["checkpoint_sha256"] == audited["checkpoint_sha256"]
                    and rec["checkpoint_sha256"] == item["checkpoint_sha256"]
                    and rec["model_sha256"] == item["model_sha256"]
                    and rec["model_sha256"] == audited["native_weights_sha256"],
                    "old checkpoint byte seal or independent replay altered",
                )
                source = audit_split(
                    item["source_context_training_metrics"], rows[:SOURCE], arm, selector
                )
                validation = audit_split(
                    item["owned_calibration_world_validation_metrics"], rows[SOURCE:],
                    arm, selector,
                )
                require(
                    source["whole"]["correct"] == audited["source_train"]["correct"]
                    and validation["whole"]["correct"] == audited["owned_validation"]["correct"],
                    "original audit and independently recomputed correctness disagree",
                )
                outcomes.append({
                    "step": step,
                    "checkpoint_sha256": rec["checkpoint_sha256"],
                    "source": source,
                    "owned_validation": validation,
                })
            arms[arm] = outcomes
        inspected.append({
            "root_index": index,
            "world_id": ref["world_id"],
            "root_source_validation_sha256": ref["pilot_file_sha256"],
            "archived_root_receipt_sha256": digest(original_file),
            "archived_independent_replay_sha256": digest(replay_file),
            "arms": arms,
        })
    macro = {}
    for arm in ARMS:
        snapshots = []
        for j, step in enumerate(BUDGETS):
            branch = {}
            for name in ("whole", "parity", "alternate"):
                values = [
                    root["arms"][arm][j]["owned_validation"][name]["accuracy"]
                    for root in inspected
                ]
                correct = sum(
                    root["arms"][arm][j]["owned_validation"][name]["correct"]
                    for root in inspected
                )
                cases = sum(
                    root["arms"][arm][j]["owned_validation"][name]["cases"]
                    for root in inspected
                )
                branch[name] = {
                    "root_macro_accuracy": sum(values) / len(values),
                    "pooled_correct": correct,
                    "pooled_cases": cases,
                    "pooled_accuracy_descriptive_only": correct / cases,
                }
            snapshots.append({"step": step, "validation": branch})
        macro[arm] = snapshots
    result = {
        "schema": SCHEMA,
        "evidence_class": "retrospective-original-six-calibration-world-roots-only",
        "original_pretraining_teacher_ledger_sha256": digest(ledger_file),
        "root_indices": list(ROOTS),
        "arms": list(ARMS),
        "budgets": list(BUDGETS),
        "historical_checkpoints_sha_verified": len(ROOTS) * len(ARMS) * len(BUDGETS),
        "prior_independent_native_checkpoint_replay_receipts_checked": 6,
        "new_model_optimizer_updates": 0,
        "new_source_SAE_updates": 0,
        "new_root_outcomes_scored": 0,
        "reserved_contexts_scored": 0,
        "development_world_root_programs_materialized": 0,
        "confirmation_world_root_programs_materialized": 0,
        "FSI_01_gate": "REDESIGN_OR_PILOT_ONLY",
        "root_results": inspected,
        "root_macro_and_pooled_descriptives": macro,
    }
    out.mkdir(parents=True, exist_ok=True)
    target = out / "nec04c_archived_parity_audit.json"
    target.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(
        "NEC04C_ARCHIVE_ONLY_BRANCH_AUDIT_PASS",
        result["historical_checkpoints_sha_verified"],
        macro["composed"][-1]["validation"]["parity"]["pooled_correct"],
        macro["composed"][-1]["validation"]["parity"]["pooled_cases"],
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-roots", type=Path, required=True)
    parser.add_argument("--pretraining", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit(args.old_roots, args.pretraining, args.out)


if __name__ == "__main__":
    main()

"""NEC-04D: pure-stdlib fifth-order Walsh audit of historical NEC-04C logits.

Read-only, no model inference, no new training. Exactly six previously
scored NEC04A CALIBRATION world roots 6..11, two original treatment arms,
two original checkpoints, original source/owned-validation contexts only.
Rechecks previous independent checkpoint audit and checkpoint byte SHAs.
No original reserved contexts or separate development/confirmation WORLD roots.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import isfinite, sqrt
from pathlib import Path

from mechinterp.neuron_communities import nec04a_boolean_gym as gym
from mechinterp.neuron_communities.nec04a_independent_oracle import oracle
from mechinterp.neuron_communities.nec04c_pretraining import (
    ROOTS, ARMS, CHECKPOINTS, SOURCE, VALIDATION,
    EXPECTED_REGISTRY, EXPECTED_CORPUS, verify_lock, code_identity,
    seed_for_root,
)

SCHEMA = "NEC-04D/historical-NEC04C-five-focal-Walsh-output-audit/v1"
SCORING_SCHEMA = "NEC-04C/native-matched-parity-curriculum-pilot/v1"
REPLAY_SCHEMA = "NEC-04C/independent-two-arm-native-checkpoint-replay/v1"
PRETRAINING_SCHEMA = "NEC-04C/pretraining-six-original-calibration-world-roots/v1"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def walsh_coefficients(margins: list[float]) -> list[float]:
    """Coefficient subset s is 1/32 * sum_f m(f)*(-1)^popcount(f&s)."""
    require(len(margins) == 32, "Walsh block must have exactly 32 original focal masks")
    require(all(isfinite(float(x)) for x in margins), "nonfinite original native logit")
    vals = [float(x) for x in margins]
    return [
        sum(value * (-1 if (mask & subset).bit_count() % 2 else 1)
            for mask, value in enumerate(vals)) / 32
        for subset in range(32)
    ]


def context_spectrum(rows: list[dict], logits: list[list[float]]) -> dict:
    require(len(rows) == len(logits) == 32, "partial original context focal block")
    require([row["focal_mask"] for row in rows] == list(range(32)),
            "original focal mask ordering changed")
    require(len({row["context_group"] for row in rows}) == 1,
            "focal masks from different original context groups")
    require(all(row["mux_selector"] == 1 for row in rows),
            "fifth-order parity analysis requires original MUX selector=1")
    truth = [int(row["composed_label"]) for row in rows]
    require(sum(truth) == 16, "original complete five-bit parity teacher unbalanced")
    require(all(len(logit) == 2 and all(isfinite(float(v)) for v in logit)
                for logit in logits), "invalid original native two-class logits")
    margins = [float(z[1]) - float(z[0]) for z in logits]
    predictions = [int(z[1] > z[0]) for z in logits]
    coeff = walsh_coefficients(margins)
    mean = coeff[0]
    energy = sum(value * value for value in coeff[1:])
    rms = sqrt(sum(value * value for value in margins) / 32)
    aligned = sum(value * (2 * y - 1) for value, y in zip(margins, truth)) / 32
    # Parity is a 5-focal XOR with a frozen per-context sign from the
    # original source program's context bits. Its degree-5 Walsh
    # amplitude must match the oracle-aligned coefficient in magnitude.
    require(abs(abs(aligned) - abs(coeff[31])) <= 1e-6 * max(1., abs(aligned)),
            "source target has changed from five-way Boolean parity")
    # Orthogonal Walsh Parseval decomposition: mean-centered margin variance
    # across the original complete 32-mask context equals nonconstant energy.
    variance = sum((value - mean) ** 2 for value in margins) / 32
    require(abs(variance - energy) <= 1e-5 * max(1., variance),
            "Walsh spectrum failed original focal-mask Parseval identity")
    counts1 = sum(predictions)
    return {
        "context_group": rows[0]["context_group"],
        "cases": 32,
        "correct": sum(prediction == target for prediction, target
                       in zip(predictions, truth)),
        "truth_positive": 16,
        "predicted_positive": counts1,
        "predicted_majority_fraction": max(counts1, 32 - counts1) / 32,
        "logit_margin_mean": mean,
        "logit_margin_rms": rms,
        "logit_margin_focal_std": sqrt(energy),
        "oracle_aligned_parity_margin": aligned,
        "parity_margin_cosine_to_oracle": aligned / rms if rms > 0 else None,
        "degree5_coefficient_signed_in_mask_basis": coeff[31],
        "degree5_fraction_of_centered_walsh_power": (
            coeff[31] ** 2 / energy if energy > 1e-20 else None
        ),
        "zero_variance_context": energy <= 1e-20,
        "walsh_power_by_degree": [
            sum(coeff[k] ** 2 for k in range(1, 32) if k.bit_count() == degree)
            for degree in range(1, 6)
        ],
    }


def checked_original_rows(index: int, ref: dict, folder: Path) -> list[dict]:
    world, _ = gym.calibration_world(index)
    require(ref["root_index"] == index
            and ref["world_id"] == world["world_id"]
            and ref["world_sha256"] == world["world_sha256"]
            and ref["program_sha256"] == world["program_sha256"]
            and ref["original_truth_4096_sha256"] == world["truth_table_4096_sha256"]
            and ref["matched_native_seed"] == seed_for_root(index),
            "historical NEC04A calibration root or seed identity changed")
    f = folder / ref["training_file"]
    require(f.name == f"nec04c_training_root_{index}.jsonl"
            and digest(f) == ref["training_file_sha256"],
            "original source-owned validation bytes changed")
    rows = [json.loads(line) for line in f.read_text().splitlines()]
    require(len(rows) == SOURCE + VALIDATION, "historical source split incomplete")
    source, partitions = gym.contexts_for_world(index)
    require(len(source) == 80 and len(partitions["development"]) == 24
            and len(partitions["reserved"]) == 24
            and not set(source) & set(partitions["development"])
            and not set(source) & set(partitions["reserved"])
            and not set(partitions["development"]) & set(partitions["reserved"]),
            "source, owned-validation and protected reserved contexts overlap")
    selector = world["context_positions"][0]
    require(world["canonical_program"][:2] == ["MUX", ["VAR", selector]],
            "original NEC04A parity selector program modified")
    seen = set()
    for k, row in enumerate(rows):
        training = k < SOURCE
        group = k // 32 if training else (k - SOURCE) // 32
        mask = k % 32
        group_context = source[group] if training else partitions["development"][group]
        bits = [0] * 12
        for j, pos in enumerate(world["focal_positions"]):
            bits[pos] = (mask >> j) & 1
        for j, pos in enumerate(world["context_positions"]):
            bits[pos] = group_context[j]
        require(row["world_id"] == world["world_id"]
                and row["root_index"] == index
                and row["split"] == ("source_train" if training
                                      else "calibration_world_validation")
                and row["context_group"] == (group if training else 80 + group)
                and row["focal_mask"] == mask
                and row["history_bits12"] == bits
                and row["mux_selector"] == bits[selector]
                and row["composed_label"] == int(oracle(world["canonical_program"], bits)),
                "original source/oracle row or context ownership modified")
        require(tuple(bits) not in seen, "duplicate old source/validation history")
        seen.add(tuple(bits))
    require(len(seen) == SOURCE + VALIDATION, "source/validation input panel truncated")
    return rows


def original_split_score(metrics: dict, rows: list[dict]) -> tuple[list[list[float]], list[int]]:
    raw = metrics["native_logits"]
    predictions = metrics["class_predictions"]
    require(len(raw) == len(predictions) == len(rows) == metrics["cases"],
            "incomplete archived native checkpoint logits or model predictions")
    correct = 0
    positives = 0
    true_positive_correct = true_negative_correct = 0
    for row, logit, prediction in zip(rows, raw, predictions):
        require(len(logit) == 2 and all(isfinite(float(v)) for v in logit),
                "malformed archived native two-class logit")
        expected = int(logit[1] > logit[0])
        truth = int(row["composed_label"])
        require(expected == prediction, "stored original prediction differs from logit")
        correct += int(expected == truth)
        positives += truth
        true_positive_correct += int(expected == truth == 1)
        true_negative_correct += int(expected == truth == 0)
    require(0 < positives < len(rows), "original source/validation labels degenerate")
    bacc = .5 * (true_positive_correct / positives
                 + true_negative_correct / (len(rows) - positives))
    require(correct == metrics["correct"]
            and positives == metrics["truth_positive_cases"]
            and true_positive_correct == metrics["correct_true_positives"]
            and true_negative_correct == metrics["correct_true_negatives"]
            and abs(correct / len(rows) - metrics["accuracy"]) < 1e-12
            and abs(bacc - metrics["balanced_accuracy"]) < 1e-12,
            "original audited native whole-split correctness changed")
    return raw, predictions


def spectral_split(rows: list[dict], raw_logits: list[list[float]],
                   original_correct: int) -> dict:
    require(len(rows) == len(raw_logits)
            and len(rows) % 32 == 0,
            "original complete source/validation context groups required")
    contexts = []
    correct_total = 0
    parity_correct = 0
    parity_cases = 0
    for start in range(0, len(rows), 32):
        block = rows[start:start + 32]
        scores = raw_logits[start:start + 32]
        require(all(row["mux_selector"] == block[0]["mux_selector"] for row in block),
                "MUX branch differs within frozen original 32-focal-mask context")
        prediction = [int(z[1] > z[0]) for z in scores]
        correct_total += sum(prediction[i] == block[i]["composed_label"]
                             for i in range(32))
        if block[0]["mux_selector"] == 1:
            context = context_spectrum(block, scores)
            parity_cases += 32
            parity_correct += context["correct"]
            contexts.append(context)
    require(correct_total == original_correct, "context-wise outcome total changed")
    require(parity_cases > 0 and parity_cases % 32 == 0,
            "original parity context coverage empty")
    fields = (
        "correct",
        "predicted_majority_fraction",
        "logit_margin_rms",
        "logit_margin_focal_std",
        "logit_margin_mean",
        "oracle_aligned_parity_margin",
        "parity_margin_cosine_to_oracle",
        "degree5_fraction_of_centered_walsh_power",
    )
    summary = {}
    for key in fields:
        values = [c[key] for c in contexts if c[key] is not None]
        summary["mean_" + key] = sum(values) / len(values) if values else None
    summary.update({
        "context_count": len(contexts),
        "parity_correct": parity_correct,
        "parity_cases": parity_cases,
        "parity_accuracy": parity_correct / parity_cases,
        "all_contexts_32_original_focal_masks": True,
        "zero_variance_context_count": sum(c["zero_variance_context"] for c in contexts),
        "contexts": contexts,
    })
    return summary


def audit(old_roots: Path, pretraining: Path, output: Path) -> dict:
    _, lock_sha = verify_lock()
    ledger_file = pretraining / "nec04c_pretraining_oracles.json"
    ledger = json.loads(ledger_file.read_text())
    require(
        ledger["schema"] == PRETRAINING_SCHEMA
        and ledger["lock_sha256"] == lock_sha
        and ledger["original_source_registry_sha256"] == EXPECTED_REGISTRY
        and ledger["original_calibration_source_corpus_sha256"] == EXPECTED_CORPUS
        and ledger["root_indices"] == list(ROOTS)
        and ledger["training_arms"] == list(ARMS)
        and ledger["checkpoint_steps"] == list(CHECKPOINTS)
        and ledger["code_sha256_before_training"] == code_identity()
        and ledger["optimizer_steps_before_preflight"] == 0
        and ledger["preflight_source_commit_sha"]
        == "b6e7a9a2e2b530e738967aa15d37c265643e98b4"
        and len(ledger["roots"]) == len(ROOTS)
        and ledger["native_SAE_training_updates"] == 0
        and ledger["new_development_world_specs_materialized"] == 0
        and ledger["new_confirmation_world_specs_materialized"] == 0
        and ledger["original_reserved_contexts_scored"] == 0
        and ledger["FSI_01_gate"] == "REDESIGN_OR_PILOT_ONLY",
        "historical preregistered root/code/oracle identity changed",
    )
    require({p.name for p in old_roots.iterdir() if p.is_dir()}
            == {f"nec04c-root-{i}" for i in ROOTS},
            "historical six original calibration root artifacts incomplete or extra")
    roots = []
    for ordinal, index in enumerate(ROOTS):
        ref = ledger["roots"][ordinal]
        rows = checked_original_rows(index, ref, pretraining)
        folder = old_roots / f"nec04c-root-{index}"
        score_file = folder / "nec04c_native_root_result.json"
        replay_file = folder / "nec04c_independent_native_audit.json"
        score = json.loads(score_file.read_text())
        replay = json.loads(replay_file.read_text())
        require(
            score["schema"] == SCORING_SCHEMA
            and replay["schema"] == REPLAY_SCHEMA
            and score["root_index"] == replay["root_index"] == index
            and score["world_id"] == replay["world_id"] == ref["world_id"]
            and score["world_sha256"] == ref["world_sha256"]
            and score["pretraining_ledger_sha256"] == digest(ledger_file)
            and replay["pretraining_teacher_ledger_sha256"] == digest(ledger_file)
            and replay["original_root_scoring_receipt_sha256"] == digest(score_file)
            and replay["original_source_validation_file_sha256"]
            == score["original_source_validation_file_sha256"]
            == ref["training_file_sha256"]
            and replay["source_lock_sha256"] == score["source_lock_sha256"] == lock_sha
            and score["native_initial_weights_shared_between_arms"] is True
            and replay["same_native_initial_weights_across_treatment_arms_verified"] is True
            and all(replay[k] is True for k in (
                "four_original_checkpoints_byte_SHA256_verified",
                "four_original_optimizer_step_counters_verified",
                "all_four_native_source_validation_logits_independently_replayed",
            ))
            and replay["training_trajectory_independently_reoptimized"] is False
            and score["source_only_training_updates_per_arm"] == 900
            and score["native_SAE_training_updates"] == 0
            and score["FSI_01_gate"] == replay["FSI_01_gate"]
            == "REDESIGN_OR_PILOT_ONLY",
            "historical independently replayed native root receipt changed",
        )
        require(all(score[k] == 0 for k in (
            "original_reserved_contexts_scored",
            "development_world_roots_materialized",
            "confirmation_world_roots_materialized",
            "old_NEC03P_worlds_scored"
        )), "protected root outcomes present in previous native result")
        arms = {}
        for arm in ARMS:
            native = score["arms"][arm]
            checks = replay["arms"][arm]
            manifest_path = folder / f"nec04c.root{index}.{arm}.checkpoints.json"
            require(digest(manifest_path) == native["sealed_checkpoint_manifest_sha256"],
                    "original two-checkpoint optimizer manifest SHA differs")
            manifest = json.loads(manifest_path.read_text())
            require(manifest == native["checkpoint_manifest"]
                    and manifest["total_updates"] == 900
                    and manifest["optimizer_resets"] == 0
                    and len(manifest["checkpoints"]) == len(CHECKPOINTS)
                    and len(checks) == len(CHECKPOINTS)
                    and len(native["scored_checkpoints"]) == len(CHECKPOINTS)
                    and manifest["native_initial_model_sha256"]
                    == score["native_initial_weights_sha256"],
                    "historical model/optimizer checkpoint seal differs")
            observations = []
            for j, step in enumerate(CHECKPOINTS):
                old = native["scored_checkpoints"][j]
                checked = checks[j]
                checkpoint = manifest["checkpoints"][j]
                pt_file = folder / checkpoint["file"]
                require(
                    old["step"] == checked["step"] == checkpoint["step"] == step
                    and pt_file.name == f"nec04c.root{index}.{arm}.step{step}.pt"
                    and digest(pt_file) == old["checkpoint_sha256"]
                    == checked["checkpoint_sha256"] == checkpoint["checkpoint_sha256"]
                    and old["native_model_sha256"] == checked["native_model_sha256"]
                    == checkpoint["native_model_sha256"],
                    "historical independent native 240/900 checkpoint SHA modified",
                )
                panels = {}
                for name, subset, previous_metric, independent_metric in (
                    ("source", rows[:SOURCE],
                     old["source_context_training_metrics"], checked["source"]),
                    ("owned_validation", rows[SOURCE:],
                     old["owned_calibration_world_validation_metrics"],
                     checked["owned_validation"]),
                ):
                    logits, _ = original_split_score(previous_metric, subset)
                    require(independent_metric["correct"] == previous_metric["correct"]
                            and independent_metric["cases"] == previous_metric["cases"],
                            "original independent full native source/validation accuracy differs")
                    panels[name] = spectral_split(subset, logits, previous_metric["correct"])
                    if name == "owned_validation":
                        require(
                            panels[name]["parity_correct"] ==
                            checked["owned_validation_branches"]["parity"]["correct"]
                            and panels[name]["parity_cases"] ==
                            checked["owned_validation_branches"]["parity"]["cases"],
                            "old independent parity branch audit differs",
                        )
                observations.append({
                    "step": step,
                    "original_checkpoint_sha256": checkpoint["checkpoint_sha256"],
                    "source": panels["source"],
                    "owned_validation": panels["owned_validation"],
                })
            arms[arm] = observations
        roots.append({
            "root_index": index,
            "world_id": ref["world_id"],
            "original_source_sha256": ref["training_file_sha256"],
            "original_score_receipt_sha256": digest(score_file),
            "original_independent_replay_receipt_sha256": digest(replay_file),
            "arms": arms,
        })
    macro = {}
    for arm in ARMS:
        outcomes = []
        for j, step in enumerate(CHECKPOINTS):
            panels = {}
            for name in ("source", "owned_validation"):
                vals = [root["arms"][arm][j][name] for root in roots]
                keys = (
                    "parity_accuracy",
                    "mean_oracle_aligned_parity_margin",
                    "mean_parity_margin_cosine_to_oracle",
                    "mean_degree5_fraction_of_centered_walsh_power",
                    "mean_predicted_majority_fraction",
                    "mean_logit_margin_focal_std",
                )
                panel = {
                    "root_count": len(vals),
                    "root_macro_" + key: (
                        sum(v[key] for v in vals) / len(vals)
                        if all(v[key] is not None for v in vals) else None
                    )
                    for key in keys
                }
                panel.update({
                    "parity_pooled_cases_descriptive": sum(v["parity_cases"] for v in vals),
                    "parity_pooled_correct_descriptive": sum(v["parity_correct"] for v in vals),
                    "context_counts_by_root": [v["context_count"] for v in vals],
                    "zero_variance_context_count": sum(v["zero_variance_context_count"]
                                                       for v in vals),
                })
                panels[name] = panel
            outcomes.append({"step": step, "panels": panels})
        macro[arm] = outcomes
    final = macro["vanilla"][-1]["panels"]["owned_validation"]
    treat = macro["parity_curriculum"][-1]["panels"]["owned_validation"]
    require(
        (final["parity_pooled_correct_descriptive"],
         final["parity_pooled_cases_descriptive"]) == (1072, 2144)
        and (treat["parity_pooled_correct_descriptive"],
             treat["parity_pooled_cases_descriptive"]) == (1071, 2144),
        "previously audited primary parity validation totals changed",
    )
    result = {
        "schema": SCHEMA,
        "evidence_class": "exploratory-read-only-original-six-calibration-world-native-logits",
        "original_pretraining_ledger_sha256": digest(ledger_file),
        "root_indices": list(ROOTS),
        "arms": list(ARMS),
        "original_checkpoint_budgets": list(CHECKPOINTS),
        "old_native_checkpoint_byte_SHA_verified": len(ROOTS)*len(ARMS)*len(CHECKPOINTS),
        "original_independent_native_replay_receipts_checked": len(ROOTS),
        "new_native_model_optimizer_updates": 0,
        "new_model_inferences": 0,
        "new_source_SAE_training_updates": 0,
        "original_reserved_contexts_scored": 0,
        "development_world_roots_instantiated": 0,
        "confirmation_world_roots_instantiated": 0,
        "old_NEC03P_worlds_scored": 0,
        "FSI_01_gate": "REDESIGN_OR_PILOT_ONLY",
        "root_results": roots,
        "root_macro_and_descriptive_case_counts": macro,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "nec04d_original_parity_walsh_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        "NEC04D_ORIGINAL_SIX_ROOT_24_CHECKPOINT_WALSH_AUDIT_PASS",
        "vanilla_d5_fraction", final["root_macro_mean_degree5_fraction_of_centered_walsh_power"],
        "curriculum_d5_fraction", treat["root_macro_mean_degree5_fraction_of_centered_walsh_power"],
        flush=True,
    )
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--old-roots", required=True, type=Path)
    p.add_argument("--pretraining", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    audit(a.old_roots, a.pretraining, a.out)


if __name__ == "__main__":
    main()

"""PEF-05R: read-only cross-run audit of two sealed PEF-05 CPU runs.

Compare original runs 43687506414 and 43688425647, where the
implementation blob, source split, starting weights and batch stream
agree but step-125 checkpoints already diverge numerically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from research.pef05.pef05 import ARMS, SEEDS


def read(path):
    return json.loads(path.read_text())


def checked_checkpoint(root, seed, stage, receipt):
    path = root / f"seed{seed}.{stage}.pt"
    if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["checkpoint_sha256"]:
        raise AssertionError(f"source checkpoint SHA256 mismatch: {path}")
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if obj["seed"] != seed or obj["weights_sha256"] != receipt["weights_sha256"]:
        raise AssertionError("checkpoint contains wrong model tensors")
    if obj["optimizer_sha256"] != receipt["optimizer_sha256"]:
        raise AssertionError("checkpoint contains wrong AdamW moments")
    return obj


def parameter_vector(model):
    return torch.cat([model[key].detach().double().reshape(-1)
                      for key in sorted(model)])


def optimizer_moments(state, name):
    vals = []
    for key in sorted(state["state"]):
        row = state["state"][key]
        if name not in row:
            raise AssertionError(f"missing AdamW {name}")
        vals.append(row[name].detach().double().reshape(-1))
    return torch.cat(vals)


def drift(left, right):
    if left.shape != right.shape or not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise AssertionError("different tensor shapes or nonfinite values")
    x, y = left.double().reshape(-1), right.double().reshape(-1)
    delta = x - y
    scale = (x.norm() + y.norm()).item() / 2
    return {
        "relative_l2": float(delta.norm()) / max(scale, 1e-30),
        "absolute_l2": float(delta.norm()),
        "maximum_absolute_element": float(delta.abs().max()),
        "exactly_equal": bool(torch.equal(x, y)),
    }


def staged_receipt(ledger, stage):
    if stage == "common.step125":
        return ledger["common"]
    arm, moment = stage.split(".")
    return ledger["arms"][arm]["intermediate" if moment == "step150" else "final"]


def compare(left, right, output):
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    roots = (left, right)
    sums = [read(root / "summary.json") for root in roots]
    manifests = [read(root / "split_manifest.json") for root in roots]
    if manifests[0] != manifests[1]:
        raise AssertionError("registered split differs")
    if any(s["registered_decision"] != "criterion_not_met" for s in sums):
        raise AssertionError("original registered negative verdict changed")
    if any(s["registered_seed_order"] != list(SEEDS) for s in sums):
        raise AssertionError("seed selection changed")
    results = []
    for seed in SEEDS:
        ledgers = [read(root / f"seed{seed}.sealed_ledger.json") for root in roots]
        scores = [read(root / f"seed{seed}.evaluation.json") for root in roots]
        if ledgers[0]["original_initial_weights_sha256"] != ledgers[1]["original_initial_weights_sha256"]:
            raise AssertionError("same-seed model initialization differs")
        if ledgers[0]["supervised_batch_stream_sha256"] != ledgers[1]["supervised_batch_stream_sha256"]:
            raise AssertionError("supervised batch stream differs")
        if any(l["source_split_sha256"] != manifests[0]["sha256"] for l in ledgers):
            raise AssertionError("source-manifest mismatch")
        stages = {}
        for stage in ("common.step125",) + tuple(
            f"{arm}.{at}" for at in ("step150","step250") for arm in ARMS
        ):
            models = [
                checked_checkpoint(root, seed, stage, staged_receipt(ledger, stage))
                for root, ledger in zip(roots, ledgers)
            ]
            stages[stage] = {
                "model_parameters": drift(
                    parameter_vector(models[0]["model"]),
                    parameter_vector(models[1]["model"])
                ),
                "optimizer_exp_avg": drift(
                    optimizer_moments(models[0]["optimizer"], "exp_avg"),
                    optimizer_moments(models[1]["optimizer"], "exp_avg")
                ),
                "optimizer_exp_avg_sq": drift(
                    optimizer_moments(models[0]["optimizer"], "exp_avg_sq"),
                    optimizer_moments(models[1]["optimizer"], "exp_avg_sq")
                ),
                "source_checkpoint_SHA256s": [
                    staged_receipt(ledger, stage)["checkpoint_sha256"]
                    for ledger in ledgers
                ],
            }
        if stages["common.step125"]["model_parameters"]["exactly_equal"]:
            raise AssertionError("expected original source divergence before intervention")
        arm_outcomes = {}
        for arm in ARMS:
            rows = []
            for ledger, score in zip(ledgers, scores):
                row = ledger["arms"][arm]
                evaluated = score["arms"][arm]
                rows.append({
                    "correct_oldest_history": evaluated["h10_01"]["correct_histories"],
                    "fully_correct_pairs": evaluated["h10_01"]["both_members_correct_pairs"],
                    "supervised_task_correct": evaluated["original_task_correct_out_of_48"],
                    "protected_training_margin_step150": row["source_only_margin_step150"],
                    "protected_training_margin_step250": row["source_only_margin_step250"],
                    "active_gradient_surgery_steps": [
                        x["step"] for x in row["surgery"] if x["active"]
                    ]
                })
            act0 = set(rows[0]["active_gradient_surgery_steps"])
            act1 = set(rows[1]["active_gradient_surgery_steps"])
            arm_outcomes[arm] = {
                "runs": rows,
                "surgery_step_symmetric_difference": sorted(act0 ^ act1),
                "parameter_relative_drift_growth_since_step125":
                    stages[f"{arm}.step250"]["model_parameters"]["relative_l2"] /
                    stages["common.step125"]["model_parameters"]["relative_l2"]
            }
        result = {
            "seed": seed,
            "same_original_initial_weight_SHA256": ledgers[0]["original_initial_weights_sha256"],
            "same_training_batch_SHA256": ledgers[0]["supervised_batch_stream_sha256"],
            "different_step125_checkpoints": True,
            "stages": stages,
            "arms": arm_outcomes,
        }
        (output / f"seed{seed}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        results.append(result)
    summary = {
        "campaign": "PEF-05R",
        "compared_run_ids": [43687506414,43688425647],
        "source_script_blob_SHA": "87c414464a2d2e0a73f6329d9dbdc43ce8561832",
        "registered_decisions": [s["registered_decision"] for s in sums],
        "source_split_identical": True,
        "original_seeded_model_weights_identical": True,
        "source_minibatch_stream_identical": True,
        "step125_numerical_divergence_present_for_every_seed": True,
        "no_model_training": True,
        "per_seed": [{
            "seed": s["seed"],
            "prefix125_rel_L2": s["stages"]["common.step125"]["model_parameters"]["relative_l2"],
            "baseline250_rel_L2": s["stages"]["baseline.step250"]["model_parameters"]["relative_l2"],
            "baseline_drift_amplification": s["arms"]["baseline"]["parameter_relative_drift_growth_since_step125"],
            "oldest_scores": {
                arm: [r["correct_oldest_history"] for r in s["arms"][arm]["runs"]]
                for arm in ARMS
            },
        } for s in results],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--left", type=Path, required=True)
    p.add_argument("--right", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    compare(a.left, a.right, a.out)


if __name__ == "__main__":
    main()

"""PEF-06T: threshold-only versus nonlinear old-bit coalition response.

Read-only, posthoc analysis of the original PEF-05 registered frozen
checkpoint family. Compare true paired-bit output logits with the
additive interpolation of their TWO independently observed singleton
logit effects, at the same original history:

  additive_pair_logits = logits(x^i)+logits(x^j)-logits(x).

An actual pair-only class change compatible with the surrogate
can arise through the final argmax even when the full logit map is
additive. An actual class change *not* predicted by that surrogate
requires a nonzero logit four-corner difference on the tested input
and its intervention quartet, but alone identifies no unique circuit.

The original supervised head was trained on h8, so h8 is the primary
descriptive output and continued h10(01) is explicitly secondary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import torch

from research.pef03.pef03 import BitMemoryLooper, hash_weights
from research.pef04.pef04 import _canonical_bytes
from research.pef05.pef05 import SEEDS, ARMS, pair_split
from research.pef06.frozen_coalitions import (
    SOURCE_RUN_ID, ALL_MASKS, finite_difference_audit, source_histories,
    forward_natural_01
)


def threshold_surrogate(logits: torch.Tensor, hidden: torch.Tensor) -> dict:
    """Exact finite classification audit on all sixteen-class-logit branches.

    Uses logits for all 256 source histories, *not* a local derivative.
    Counts are across 15 unordered bit pairs times 256 source cases.
    """
    if tuple(logits.shape) != (256, 4) or tuple(hidden.shape) != (256, 16):
        raise ValueError("expected 256x4 logits and 256x16 hidden memory")
    if not torch.isfinite(logits).all() or not torch.isfinite(hidden).all():
        raise AssertionError("nonfinite frozen observable")
    torch.set_num_threads(1)
    words = torch.arange(256, dtype=torch.long)
    target = 2*((words >> 6) & 1) + ((words >> 7) & 1)
    classes = logits.argmax(dim=1)
    correct_base = classes == target
    two_largest = logits.topk(2, dim=1).values
    margin = two_largest[:, 0] - two_largest[:, 1]
    per_pair = []
    for i,j in combinations(range(6), 2):
        fi = words ^ (1 << i)
        fj = words ^ (1 << j)
        fij = words ^ (1 << i) ^ (1 << j)
        if not torch.equal(target[fij], target):
            raise AssertionError("old-input intervention changed target")
        pair_only = (
            (classes[fi] == classes) &
            (classes[fj] == classes) &
            (classes[fij] != classes)
        )
        # This surrogate has EXACTLY zero second logit difference,
        # but nonlinear argmax decisions can still change by a pair.
        additive = logits[fi] + logits[fj] - logits
        additive_class = additive.argmax(dim=1)
        compatible = pair_only & (additive_class == classes[fij])
        nonlinear_class = pair_only & (additive_class != classes[fij])
        additive_pair = (
            (classes[fi] == classes) &
            (classes[fj] == classes) &
            (additive_class != classes)
        )
        delta_logits = logits[fij] - logits[fi] - logits[fj] + logits
        delta_hidden = hidden[fij] - hidden[fi] - hidden[fj] + hidden
        count = int(pair_only.sum())
        pair = {
            "history_bit_pair": [i,j],
            "pair_only_actual_task_class_changes": count,
            "pair_only_from_correct_base_task_prediction": int(
                (pair_only & correct_base).sum()),
            "pair_only_from_incorrect_base_task_prediction": int(
                (pair_only & ~correct_base).sum()),
            "pair_only_exact_class_reproduced_by_additive_logit_surrogate":
                int(compatible.sum()),
            "pair_only_class_differs_from_additive_logit_surrogate":
                int(nonlinear_class.sum()),
            "threshold_only_surrogate_pair_only_class_changes":
                int(additive_pair.sum()),
            "mean_four_corner_logit_l2": float(delta_logits.norm(dim=1).mean()),
            "mean_four_corner_hidden_l2": float(delta_hidden.norm(dim=1).mean()),
            "mean_original_top_two_logit_margin_given_pair_only":
                float(margin[pair_only].mean()) if count else None,
            "pair_only_cases_by_supervised_target_class": {
                str(k): int((pair_only & (target == k)).sum())
                for k in range(4)
            },
            "denominator": 256,
        }
        if (pair["pair_only_exact_class_reproduced_by_additive_logit_surrogate"] +
            pair["pair_only_class_differs_from_additive_logit_surrogate"] != count):
            raise AssertionError("additive and nonadditive cases did not partition")
        if (pair["pair_only_from_correct_base_task_prediction"] +
            pair["pair_only_from_incorrect_base_task_prediction"] != count):
            raise AssertionError("target-conditional cases did not partition")
        if int(nonlinear_class.sum()) and float(delta_logits[nonlinear_class].abs().max()) == 0:
            raise AssertionError("non-additive class change with exact zero logit delta")
        per_pair.append(pair)

    result = {
        "source_histories": 256,
        "old_bit_intervention_pairs": 15,
        "count_denominator_all_source_pair_cases": 256 * 15,
        "task_correct_on_complete_256_source_cube": int(correct_base.sum()),
        "pair_only_actual_class_change_events": sum(
            r["pair_only_actual_task_class_changes"] for r in per_pair),
        "pair_only_events_from_correct_baselines": sum(
            r["pair_only_from_correct_base_task_prediction"] for r in per_pair),
        "pair_only_events_from_incorrect_baselines": sum(
            r["pair_only_from_incorrect_base_task_prediction"] for r in per_pair),
        "pair_only_events_reproduced_by_additive_surrogate": sum(
            r["pair_only_exact_class_reproduced_by_additive_logit_surrogate"]
            for r in per_pair),
        "pair_only_events_not_reproduced_by_additive_surrogate": sum(
            r["pair_only_class_differs_from_additive_logit_surrogate"]
            for r in per_pair),
        "additive_surrogate_pair_only_class_change_events": sum(
            r["threshold_only_surrogate_pair_only_class_changes"]
            for r in per_pair),
        "mean_of_pair_average_four_corner_logit_l2": (
            sum(r["mean_four_corner_logit_l2"] for r in per_pair)/15),
        "mean_of_pair_average_four_corner_hidden_l2": (
            sum(r["mean_four_corner_hidden_l2"] for r in per_pair)/15),
        "original_supervised_task_label_remains_invariant": True,
        "additive_surrogate_is_local_on_discrete_intervention_quartet": True,
        "source_distribution": "uniform across all 256 eight-bit histories",
        "posthoc_source_family_reused": True,
        "per_pair": per_pair,
    }
    assert (
        result["pair_only_events_reproduced_by_additive_surrogate"] +
        result["pair_only_events_not_reproduced_by_additive_surrogate"]
        == result["pair_only_actual_class_change_events"])
    assert (
        result["pair_only_events_from_correct_baselines"] +
        result["pair_only_events_from_incorrect_baselines"]
        == result["pair_only_actual_class_change_events"])
    return result


@torch.no_grad()
def _h8_h10(model: BitMemoryLooper, full: torch.Tensor):
    h8 = model.memory(full)
    l8 = model.head(h8)
    h10,l10 = forward_natural_01(model,full)
    return (h8,l8),(h10,l10)


def pinned_audit(archive: Path, output: Path):
    """Recompute all twelve model quartets with immutable checkpoint audit."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    summary = json.loads((archive / "summary.json").read_text())
    if summary["registered_decision"] != "criterion_not_met":
        raise AssertionError("PEF-05 original negative verdict changed")
    if summary["registered_seed_order"] != list(SEEDS):
        raise AssertionError("wrong frozen registered model set")
    split,manifest = pair_split()
    if manifest != json.loads((archive / "split_manifest.json").read_text()):
        raise AssertionError("source partition was changed")
    output.mkdir(parents=True, exist_ok=True)
    full = source_histories()
    records = []
    for seed in SEEDS:
        ledger = json.loads(
            (archive / f"seed{seed}.sealed_ledger.json").read_text()
        )
        if not ledger["all_three_arms_frozen_before_downstream_probe"]:
            raise AssertionError("unsealed original PEF-05 source")
        for arm in ARMS:
            checkpoint_meta = ledger["arms"][arm]["final"]
            checkpoint = archive / checkpoint_meta["file"]
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if digest != checkpoint_meta["checkpoint_sha256"]:
                raise AssertionError("frozen PEF-05 checkpoint SHA256 mismatch")
            data = torch.load(checkpoint, weights_only=True, map_location="cpu")
            if (data["seed"], data["arm"], data["step"]) != (seed,arm,250):
                raise AssertionError("wrong checkpoint seed/treatment/step")
            model = BitMemoryLooper()
            model.load_state_dict(data["model"],strict=True)
            model.eval()
            before = hash_weights(model)
            if before != checkpoint_meta["weights_sha256"]:
                raise AssertionError("frozen model content hash mismatch")
            (h8,l8),(h10,l10) = _h8_h10(model,full)
            h8_out = threshold_surrogate(l8,h8)
            h10_out = threshold_surrogate(l10,h10)
            h8_parent = finite_difference_audit(h8,l8)
            h10_parent = finite_difference_audit(h10,l10)
            for metric,parent in ((h8_out,h8_parent),(h10_out,h10_parent)):
                if (metric["pair_only_actual_class_change_events"] !=
                    parent["number_of_pair_only_class_changes_across_all_15_pairs"]):
                    raise AssertionError("independent parent pair-only measurement mismatched")
            if hash_weights(model) != before:
                raise AssertionError("frozen checkpoint weights changed through analysis")
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != digest:
                raise AssertionError("frozen original checkpoint bytes changed")
            record = {
                "seed": seed, "arm": arm,
                "original_PE05_checkpoint_sha256": digest,
                "original_registered_negative_preserved": True,
                "original_weights_unchanged": True,
                "h8_trained_task_head": h8_out,
                "h10_same_head_after_unsupervised_two_visit_extension": h10_out,
                "h8_is_primary_descriptive_trained_head": True,
                "h10_is_secondary_out_of_original_training_horizon": True,
            }
            (output / f"seed{seed}.{arm}.threshold.json").write_bytes(
                _canonical_bytes(record))
            records.append({
                "seed": seed, "arm": arm,
                "original_PE05_checkpoint_sha256": digest,
                "trained_h8_task_correct_256":
                    h8_out["task_correct_on_complete_256_source_cube"],
                "trained_h8_pair_only":
                    h8_out["pair_only_actual_class_change_events"],
                "trained_h8_pair_only_compatible_additive":
                    h8_out["pair_only_events_reproduced_by_additive_surrogate"],
                "trained_h8_pair_only_different_from_additive":
                    h8_out["pair_only_events_not_reproduced_by_additive_surrogate"],
                "trained_h8_pair_only_from_correct_base":
                    h8_out["pair_only_events_from_correct_baselines"],
                "continued_h10_pair_only":
                    h10_out["pair_only_actual_class_change_events"],
            })
    result = {
        "campaign": "PEF-06T",
        "original_source_run_id": SOURCE_RUN_ID,
        "source_sha256_verified_all_twelve": True,
        "source_PE05_negative_unchanged": True,
        "posthoc_heldout_universe_reused": True,
        "all_twelve_loaded_from_archived_frozen_checkpoints": True,
        "pair_only_event_denominator_each_model": 15 * 256,
        "h8_trained_head_separated_from_h10_extended_head": True,
        "surrogate_is_threshold_compatibility_control_not_causal_attribution": True,
        "all_models": records,
    }
    (output / "threshold_summary.json").write_bytes(_canonical_bytes(result))
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    pinned_audit(args.archive,args.out)


if __name__ == "__main__":
    main()

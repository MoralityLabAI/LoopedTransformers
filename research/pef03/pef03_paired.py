"""PEF-03P: explicitly posthoc paired counterfactual decoder replay.

Uses the immutable PEF-03 train/probe/heldout split and frozen checkpoints.
This follow-up was selected after inspecting the registered run. It cannot
be treated as independently preregistered or as a new model trial.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from research.pef03.pef03 import (
    SEEDS, fit_ridge, hash_weights, labels, load_frozen, predict_ridge,
    split_catalog, tensor,
)


@torch.no_grad()
def paired_replay(seed: int, output_dir: Path) -> dict:
    receipt_path = output_dir / f"seed{seed}.json"
    checkpoint = output_dir / f"seed{seed}.frozen.pt"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_sha = receipt["training"]["checkpoint_sha256"]
    actual_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if expected_sha != actual_sha:
        raise AssertionError("original frozen checkpoint receipt digest mismatch")
    model = load_frozen(checkpoint, seed)
    old_hash = hash_weights(model)
    catalog = split_catalog()
    probe = tensor(catalog["probe"])
    test = tensor(catalog["heldout"])
    flipped = test.clone()
    flipped[:, 0] = 1 - flipped[:, 0]
    if not torch.equal(flipped[:, 1:], test[:, 1:]):
        raise AssertionError("paired intervention did not hold all newer bits fixed")
    if not torch.equal(labels(test), labels(flipped)):
        raise AssertionError("paired intervention changed supervised task labels")

    # The classifier was fitted on the previously sealed probe-only pool.
    fit = fit_ridge(model.memory(probe), probe[:, 0])
    original_pred = predict_ridge(model.memory(test), fit)
    flipped_pred = predict_ridge(model.memory(flipped), fit)
    correct_original = original_pred == test[:, 0]
    correct_flipped = flipped_pred == flipped[:, 0]

    task_classes = labels(test)
    classes_unchanged = (
        model.logits(test).argmax(1) == model.logits(flipped).argmax(1)
    )
    # Independently quantify the information explainable by the
    # supervised *label alone*. This target-only feature is identical
    # for every controlled old-bit counterfactual pair.
    target_only_fit = fit_ridge(
        F.one_hot(labels(probe), 4).double(), probe[:, 0]
    )
    target_code = F.one_hot(labels(test), 4).double()
    target_only_orig = predict_ridge(target_code, target_only_fit)
    target_only_flipped = predict_ridge(
        F.one_hot(labels(flipped), 4).double(), target_only_fit
    )
    if not torch.equal(target_only_orig, target_only_flipped):
        raise AssertionError("last-two-bit-only control changed under old-bit intervention")

    observed_orig_acc = float(correct_original.float().mean())
    recorded_orig_acc = receipt["task_and_uniform_history_probes"][
        "history_bit_probes"
    ][0]["heldout_accuracy"]
    if abs(observed_orig_acc - recorded_orig_acc) > 1e-8:
        raise AssertionError("independent decoder replay disagrees with sealed verdict")
    head_change = float((~classes_unchanged).float().mean())
    recorded_head_change = receipt["causal_discrete_bit_interventions"][
        "fp32"
    ][0]["head_class_change_rate"]
    if abs(head_change - recorded_head_change) > 1e-8:
        raise AssertionError("paired head counterfactual diverged from prior recorded receipt")

    result = {
        "campaign": "PEF-03P",
        "posthoc_selected_after_frozen_scored_result": True,
        "seed": seed,
        "original_frozen_checkpoint_sha256": actual_sha,
        "frozen_model_parameters_unchanged": True,
        "heldout_pairs": len(test),
        "changed_source_history_positions": [0],
        "supervised_task_targets_unchanged": True,
        "oldest_bit_probe_original_correct": int(correct_original.sum()),
        "oldest_bit_probe_flipped_correct": int(correct_flipped.sum()),
        "oldest_bit_probe_both_directions_correct": int(
            (correct_original & correct_flipped).sum()
        ),
        "oldest_bit_probe_prediction_flips": int(
            (original_pred != flipped_pred).sum()
        ),
        "last_two_bit_task_head_class_unchanged": int(
            classes_unchanged.sum()
        ),
        "last_two_bit_target_only_baseline_correct": int(
            (target_only_orig == test[:, 0]).sum()
        ),
        "last_two_bit_target_only_probe_flip_count": int(
            (target_only_orig != target_only_flipped).sum()
        ),
        "conditional_by_task_class": [{
            "class": k,
            "n": int((task_classes == k).sum()),
            "old_bit_probe_original_correct": int(
                (correct_original & (task_classes == k)).sum()
            ),
            "paired_old_bit_probe_both_correct": int(
                (correct_original & correct_flipped & (task_classes == k)).sum()
            ),
        } for k in range(4)],
        "older_bit_source_probe_fitted_only_on_56_preexisting_probe_histories": True,
        "counterfactual_histories_can_belong_to_other_original_splits": True,
        "causal_probe_output_a_posthoc_diagnostic": True,
    }
    if hash_weights(model) != old_hash:
        raise AssertionError("read-only replay mutated model state")
    (output_dir / f"seed{seed}.paired.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    rows = [paired_replay(seed, args.dir) for seed in SEEDS]
    summary = {
        "campaign": "PEF-03P",
        "posthoc": True,
        "seeds": list(SEEDS),
        "paired_results": [{
            "seed": r["seed"],
            "n": r["heldout_pairs"],
            "both_probe_predictions_correct":
                r["oldest_bit_probe_both_directions_correct"],
            "probe_prediction_flips": r["oldest_bit_probe_prediction_flips"],
            "task_head_class_unchanged":
                r["last_two_bit_task_head_class_unchanged"],
            "target_only_control_flips":
                r["last_two_bit_target_only_probe_flip_count"],
        } for r in rows],
    }
    (args.dir / "paired_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

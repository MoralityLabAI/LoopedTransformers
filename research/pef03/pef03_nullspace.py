"""PEF-03N: posthoc final-head nullspace of frozen recurrent memory.

The analysis was chosen after the preregistered PEF-03 and posthoc PEF-03P
results. It is diagnostic, not a new heldout preregistration. Decompose
the final 16D memory into the 4D row space and 12D kernel of the frozen
4-class linear head. Probes train ONLY on the original 56 probe-training
histories. Two-member old-bit counterfactual tests reuse the immutable 52
heldout histories. The projection invariance applies at the *final* head;
it makes no claim about earlier computational interventions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from research.pef03.pef03 import (
    SEEDS, fit_ridge, hash_weights, load_frozen, predict_ridge,
    split_catalog, tensor,
)


def frozen_nullspace_audit(seed: int, root: Path) -> dict:
    receipt_path = root / f"seed{seed}.json"
    checkpoint = root / f"seed{seed}.frozen.pt"
    original = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_sha = original["training"]["checkpoint_sha256"]
    actual_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if expected_sha != actual_sha:
        raise AssertionError("frozen checkpoint digest differs from preregistered receipt")
    model = load_frozen(checkpoint, seed)
    checksum = hash_weights(model)
    catalog = split_catalog()
    train_bits = tensor(catalog["probe"])
    eval_bits = tensor(catalog["heldout"])
    opposite_bits = eval_bits.clone()
    opposite_bits[:, 0] = 1 - opposite_bits[:, 0]
    if not torch.equal(eval_bits[:, 1:], opposite_bits[:, 1:]):
        raise AssertionError("counterfactual changed non-oldest bit")

    with torch.no_grad():
        w = model.head.weight.detach().double()
        u, singular, vh = torch.linalg.svd(w, full_matrices=True)
        rank = int((singular > 1e-8).sum().item())
        if w.shape != (4, 16) or rank != 4:
            raise AssertionError("unexpected final head rank or geometry")
        row = vh[:rank].T.contiguous()
        null = vh[rank:].T.contiguous()
        if row.shape != (16, 4) or null.shape != (16, 12):
            raise AssertionError("projection dimension mismatch")
        zero = float((w @ null).abs().max().item())
        orth = float((row.T @ null).abs().max().item())
        if zero > 1e-10 or orth > 1e-10:
            raise AssertionError("head row/kernel decomposition failed")
        project_row = row @ row.T
        project_null = null @ null.T
        identity = torch.eye(16, dtype=torch.float64)
        if float((project_row + project_null - identity).abs().max()) > 1e-10:
            raise AssertionError("orthogonal head-space decomposition is incomplete")

        h_train = model.memory(train_bits).double()
        h_test = model.memory(eval_bits).double()
        h_flip = model.memory(opposite_bits).double()
        if not (torch.isfinite(h_train).all() and
                torch.isfinite(h_test).all() and
                torch.isfinite(h_flip).all()):
            raise AssertionError("nonfinite frozen hidden state")

        full_logit = model.head(h_test.float())
        projected_logit = model.head((h_test @ project_row).float())
        max_logit_error = float((full_logit - projected_logit).abs().max())
        if max_logit_error > 1e-4:
            raise AssertionError("nullspace deletion changed the final task logits")

        # Nullspace component replacement from an oldest-bit-flipped
        # donor acts as a *state-space intervention at the final head*.
        # Every newer bit and therefore the supervised label are fixed.
        patched = h_test @ project_row + h_flip @ project_null
        patched_logits = model.head(patched.float())
        max_patch_logit_error = float((full_logit - patched_logits).abs().max())
        if max_patch_logit_error > 1e-4:
            raise AssertionError("nullspace donor patch changed task logits")

        spaces = {}
        basis_options = {
            "full_16d": None,
            "head_rowspace_4d": row,
            "head_nullspace_12d": null,
        }
        for name, basis in basis_options.items():
            tr = h_train if basis is None else h_train @ basis
            te = h_test if basis is None else h_test @ basis
            cf = h_flip if basis is None else h_flip @ basis
            fit = fit_ridge(tr, train_bits[:, 0])
            original_predictions = predict_ridge(te, fit)
            changed_predictions = predict_ridge(cf, fit)
            original_correct = original_predictions == eval_bits[:, 0]
            flipped_correct = changed_predictions == opposite_bits[:, 0]
            spaces[name] = {
                "feature_dimension": tr.shape[1],
                "oldest_bit_correct_on_original_heldout": int(original_correct.sum()),
                "oldest_bit_correct_on_flipped_heldout": int(flipped_correct.sum()),
                "both_members_oldest_bit_correct": int(
                    (original_correct & flipped_correct).sum()
                ),
                "prediction_flips_on_oldest_bit_intervention": int(
                    (original_predictions != changed_predictions).sum()
                ),
                "denominator": len(eval_bits),
                "ridge_penalty": 1.0,
                "ridge_standardization_is_subspace_specific": True,
            }
        expected = original["task_and_uniform_history_probes"][
            "history_bit_probes"
        ][0]["heldout_accuracy"]
        actual = spaces["full_16d"][
            "oldest_bit_correct_on_original_heldout"
        ] / len(eval_bits)
        if abs(expected - actual) > 1e-8:
            raise AssertionError("full-space probe contradicts frozen scored receipt")
        paired_path = root / f"seed{seed}.paired.json"
        if paired_path.exists():
            paired = json.loads(paired_path.read_text(encoding="utf-8"))
            if (spaces["full_16d"]["both_members_oldest_bit_correct"] !=
                    paired["oldest_bit_probe_both_directions_correct"]):
                raise AssertionError("full-space result disagrees with posthoc paired replay")

        # The projection of each true old-bit intervention onto the
        # final logit-sensitive and exactly logit-invisible components.
        differences = h_flip - h_test
        null_delta = differences @ project_null
        row_delta = differences @ project_row
        residual = float((differences - null_delta - row_delta).abs().max())
        if residual > 1e-9:
            raise AssertionError("old-bit intervention did not decompose")

        # Decoding a donor after nullspace patch must equal its nullspace
        # feature, because the final rowspace and nullspace are orthogonal.
        patched_null_features = patched @ null
        donor_null_features = h_flip @ null
        if float((patched_null_features - donor_null_features).abs().max()) > 1e-9:
            raise AssertionError("nullspace patch did not transfer donor feature")

        result = {
            "campaign": "PEF-03N",
            "posthoc_selected_after_pef03_paired_results": True,
            "seed": seed,
            "source": "frozen PEF-03 original two-seed training; no further training",
            "checkpoint_sha256": actual_sha,
            "frozen_weights_unchanged": True,
            "train_only_probe_size": len(train_bits),
            "original_heldout_size": len(eval_bits),
            "oldest_bit_counterfactual_size": len(opposite_bits),
            "head_weight_shape": list(w.shape),
            "head_weight_rank": rank,
            "rowspace_dimension": row.shape[1],
            "nullspace_dimension": null.shape[1],
            "max_abs_head_times_null_basis": zero,
            "max_abs_row_null_orthogonality_error": orth,
            "max_abs_final_logit_change_after_nullspace_deletion": max_logit_error,
            "max_abs_final_logit_change_after_nullspace_donor_patch": max_patch_logit_error,
            "oldest_bit_intervention_mean_total_hidden_l2": float(
                differences.norm(dim=1).mean()),
            "oldest_bit_intervention_mean_nullspace_hidden_l2": float(
                null_delta.norm(dim=1).mean()),
            "oldest_bit_intervention_mean_rowspace_hidden_l2": float(
                row_delta.norm(dim=1).mean()),
            "ridge_probes": spaces,
            "hidden_nullspace_ablation_does_not_re-run_recurrent_model": True,
            "nullspace_intervention_is_at_final_linear_head_only": True,
            "independent_preregistration_for_pef03n": False,
            "not_a_pretrained_language_model": True,
        }
        if hash_weights(model) != checksum:
            raise AssertionError("analysis modified frozen model weights")
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != actual_sha:
            raise AssertionError("analysis modified frozen checkpoint")
    destination = root / f"seed{seed}.nullspace.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    results = [frozen_nullspace_audit(seed, args.dir) for seed in SEEDS]
    summary = {
        "campaign": "PEF-03N",
        "posthoc": True,
        "source_frozen_seeds": list(SEEDS),
        "results": [{
            "seed": r["seed"],
            "checkpoint_sha256": r["checkpoint_sha256"],
            "head_rank": r["head_weight_rank"],
            "nullspace_dimension": r["nullspace_dimension"],
            "max_logit_change_after_nullspace_patch":
                r["max_abs_final_logit_change_after_nullspace_donor_patch"],
            "original_oldest_correct_by_subspace": {
                key: row["oldest_bit_correct_on_original_heldout"]
                for key, row in r["ridge_probes"].items()
            },
            "paired_oldest_correct_by_subspace": {
                key: row["both_members_oldest_bit_correct"]
                for key, row in r["ridge_probes"].items()
            },
        } for r in results],
    }
    (args.dir / "nullspace_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""NRF-06 fresh-fold transfer of the NRF-04 learned-vs-random observation tradeoff.

The lambda=1 objective was selected after inspecting NRF-04 evaluation
data and is an EXPLORATORY, preregistered new-input transfer. The model,
training fold, projection rank, ridge solution and null seeds are frozen.
Only the evaluation input fold (5) and leakage shadow fold (6) are new.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from research_gym.analysis.nrf03_checkpoint_probe import manifest_entries, load_pinned_weights
from research_gym.analysis.nrf04_semantic_fiber import (
    N, K, TRAIN_N, gaussian_stream, teacher, blocks, split_data, half_states,
    supervised_basis, coordinate_basis, random_basis, ridge_head,
    heldout_mse, leakage_at,
)

FRESH_EVAL_FOLD = 5
FRESH_SHADOW_FOLD = 6
EVAL_N = 96
SHADOW_N = 12
LAMBDA = 1.0


def fresh_split(parameters, seed, rounds, fold, size):
    """Generate independent LCG32 Box-Muller inputs from a sealed new fold."""
    if fold not in (FRESH_EVAL_FOLD, FRESH_SHADOW_FOLD) or size < 1:
        raise ValueError("only fresh fold 5 or 6 is permitted")
    if seed not in (101, 103, 107):
        raise ValueError("task seed is outside the frozen cohort")
    stream = gaussian_stream(20261115 + 7919 * seed + 104729 * fold)
    inputs = torch.tensor(
        [[next(stream) for _ in range(N)] for _ in range(size)],
        dtype=torch.float64,
    )
    perm, signs = teacher(seed)
    targets = torch.tanh(inputs[:, perm] * signs)
    return inputs, half_states(parameters, inputs, rounds), targets


def _median(values):
    if len(values) != 8:
        raise ValueError("eight matched random objectives required")
    ordered = sorted(values)
    return (ordered[3] + ordered[4]) / 2


def evaluate_transfer(parameters, task_seed: int, rounds: int):
    parameters = blocks(parameters)
    _, train_h, train_y = split_data(
        parameters, task_seed, rounds, 1, TRAIN_N)
    _, test_h, test_y = fresh_split(
        parameters, task_seed, rounds, FRESH_EVAL_FOLD, EVAL_N)
    _, shadow_h, _ = fresh_split(
        parameters, task_seed, rounds, FRESH_SHADOW_FOLD, SHADOW_N)
    mean_h, mean_y = train_h.mean(0), train_y.mean(0)
    bases = {
        "supervised": supervised_basis(train_h, train_y, mean_h, mean_y),
        "label_shifted_train_null": supervised_basis(
            train_h, train_y, mean_h, mean_y, shifted=True),
        "first_eight_coordinates": coordinate_basis(),
    }
    for k in range(8):
        bases[f"random_{k}"] = random_basis(400000 + 31*k + 17*task_seed)
    arms = {}
    for name, basis in bases.items():
        head = ridge_head(train_h, train_y, basis, mean_h, mean_y)
        error = heldout_mse(test_h, test_y, basis, mean_h, mean_y, head)
        local_leakage = sum(
            leakage_at(parameters, h, rounds // 2, basis) for h in shadow_h
        ) / SHADOW_N
        arms[name] = {
            "heldout_mse": error,
            "local_fiber_leakage": local_leakage,
            "lambda_one_objective": error + LAMBDA*local_leakage**2,
        }
    supervised_objective = arms["supervised"]["lambda_one_objective"]
    random_median = _median([
        arms[f"random_{k}"]["lambda_one_objective"] for k in range(8)
    ])
    return {
        "arms": arms,
        "supervised_objective": supervised_objective,
        "random_median_objective": random_median,
        "supervised_wins": supervised_objective < random_median,
    }


def run_manifest(checkpoint_dir: Path, events: Path):
    torch.set_num_threads(1)
    entries = manifest_entries(checkpoint_dir, events)
    rows = []
    for entry in entries:
        result = evaluate_transfer(load_pinned_weights(entry),
                                   entry["seed"], entry["rounds"])
        rows.append({
            "id": f"R{entry['rounds']}_S{entry['seed']}_E{entry['exposure']}",
            "checkpoint_sha256": entry["sha256"],
            "rounds": entry["rounds"], "seed": entry["seed"],
            "exposure": entry["exposure"], **result,
        })
    finals = [r for r in rows if r["exposure"] == 4096]
    r16 = [r for r in finals if r["rounds"] == 16]
    r32 = [r for r in finals if r["rounds"] == 32]
    assert len(r16) == len(r32) == 3
    return {
        "study_id": "NRF-06-regional-observation-transfer-v1",
        "new_evaluation_fold": FRESH_EVAL_FOLD,
        "new_shadow_fold": FRESH_SHADOW_FOLD,
        "objective_lambda": LAMBDA,
        "rows": rows,
        "precommitted_transfer_gate": {
            "r16_supervised_wins_final": sum(r["supervised_wins"] for r in r16),
            "r32_supervised_wins_final": sum(r["supervised_wins"] for r in r32),
            "expected_r16_three_of_three": all(r["supervised_wins"] for r in r16),
            "expected_r32_zero_of_three": not any(r["supervised_wins"] for r in r32),
        },
        "claim_boundary": (
            "New input split for already-used model checkpoints and post-hoc "
            "selected lambda. Neither a fully independent preregistered "
            "discovery nor a proof of global observability."
        ),
    }


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=(
        root / "experiments/loop_schedule_kappa_surface_v1/checkpoints"))
    parser.add_argument("--events", type=Path, default=(
        root / "experiments/loop_schedule_kappa_surface_v1/surface_events.jsonl"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = run_manifest(args.checkpoint_dir, args.events)
    payload = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()

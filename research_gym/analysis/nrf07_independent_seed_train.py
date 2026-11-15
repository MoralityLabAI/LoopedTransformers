"""NRF-07: independent CPU training-seed replication of the NRF-06 decoder tradeoff.

Freezes seeds 109, 113, 127, tied ToyLoop depths 16/32, state-visit
exposures 2048/4096, held-out task fold 5, leakage fold 6, and lambda=1.
Run with --seed 109 --rounds 16. Existing checkpoint files are untouched.

A full run writes checkpoint binaries and all eleven arm-level measurements
under --outdir. Compare against the compact sealed receipt in NRF07_RESULTS.md.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
from pathlib import Path

SEEDS = (109, 113, 127)
DEPTHS = (16, 32)
N = 128
K = 8


def gaussian_inputs(task_seed: int, fold: int, count: int):
    import torch
    from research_gym.analysis.nrf04_semantic_fiber import gaussian_stream
    if task_seed not in SEEDS or fold not in (1, 5, 6) or count <= 0:
        raise ValueError("unregistered NRF-07 task seed, fold or size")
    stream = gaussian_stream(20261115 + 7919 * task_seed + 104729 * fold)
    x = torch.tensor([[next(stream) for _ in range(N)] for _ in range(count)],
                     dtype=torch.float64)
    permutation = torch.randperm(
        N, generator=torch.Generator().manual_seed(task_seed + 17))[:K]
    signs = torch.where(torch.rand(
        N, generator=torch.Generator().manual_seed(task_seed + 29)) > .5,
        1., -1.)[:K].to(torch.float64)
    return x, torch.tanh(x[:, permutation] * signs)


def train_one(seed: int, rounds: int, outdir: Path):
    import torch
    from lsa.kappa_probe import ToyLoop
    from research_gym.scripts.bench_loop_schedule_algebra_v0_1 import (
        signed_permutation_batch,
    )
    if seed not in SEEDS or rounds not in DEPTHS:
        raise ValueError("unregistered NRF-07 model seed or depth")
    torch.manual_seed(seed)
    model = ToyLoop(N, rounds, tied=True, alpha=1., beta=.25).to("cpu")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=.001, weight_decay=0.)
    outdir.mkdir(parents=True, exist_ok=True)
    snapshots = []
    for step in range(1, 4096 // (8 * rounds) + 1):
        x, y = signed_permutation_batch(
            seed, 8, N, torch.device("cpu"),
            stream=step - 1, data_seed=seed)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        exposure = step * 8 * rounds
        if exposure not in (2048, 4096):
            continue
        snap = copy.deepcopy(model.state_dict())
        path = outdir / f"nrf07_tied_R{rounds}_S{seed}_E{exposure}.pt"
        torch.save({
            "cell_id": f"nrf07_tied_R{rounds}_S{seed}",
            "step": step, "exposure": exposure,
            "state_dict": snap,
            "optimizer": copy.deepcopy(optimizer.state_dict()),
        }, path)
        snapshots.append({
            "rounds": rounds, "seed": seed, "exposure": exposure,
            "step": step, "checkpoint_sha256": hashlib.sha256(
                path.read_bytes()).hexdigest(),
            "train_loss_latest": float(loss),
            "state_dict": snap,
        })
    return snapshots


def evaluate(snapshot):
    import torch
    from research_gym.analysis.nrf04_semantic_fiber import (
        blocks, half_states, supervised_basis, random_basis,
        coordinate_basis, ridge_head, heldout_mse, leakage_at,
    )
    rounds, task_seed = snapshot["rounds"], snapshot["seed"]
    state = snapshot["state_dict"]
    weights = blocks(tuple(state[key] for key in (
        "blocks.0.in_proj.weight", "blocks.0.in_proj.bias",
        "blocks.0.out_proj.weight", "blocks.0.out_proj.bias")))
    def dataset(fold: int, size: int):
        inputs, targets = gaussian_inputs(task_seed, fold, size)
        return half_states(weights, inputs, rounds), targets
    with torch.no_grad():
        train_h, train_y = dataset(1, 160)
        eval_h, eval_y = dataset(5, 96)
        shadow_h, _ = dataset(6, 12)
        mean_h, mean_y = train_h.mean(0), train_y.mean(0)
        bases = {
            "supervised": supervised_basis(
                train_h, train_y, mean_h, mean_y),
            "label_shifted_train_null": supervised_basis(
                train_h, train_y, mean_h, mean_y, shifted=True),
            "first_eight_coordinates": coordinate_basis(),
        }
        for i in range(8):
            bases[f"random_{i}"] = random_basis(
                400000 + 31*i + 17*task_seed)
        arms = {}
        for name, basis in bases.items():
            head = ridge_head(
                train_h, train_y, basis, mean_h, mean_y)
            error = heldout_mse(
                eval_h, eval_y, basis, mean_h, mean_y, head)
            leakage = sum(
                leakage_at(weights, h, rounds // 2, basis)
                for h in shadow_h) / len(shadow_h)
            arms[name] = {
                "heldout_mse": error,
                "local_fiber_leakage": leakage,
                "lambda_one_objective": error + leakage**2,
            }
        ordered = sorted(
            arms[f"random_{i}"]["lambda_one_objective"]
            for i in range(8))
        ref = (ordered[3] + ordered[4]) / 2
        sup = arms["supervised"]["lambda_one_objective"]
    return {
        "id": f"R{rounds}_S{task_seed}_E{snapshot['exposure']}",
        "rounds": rounds, "seed": task_seed,
        "exposure": snapshot["exposure"],
        "checkpoint_sha256": snapshot["checkpoint_sha256"],
        "train_loss_latest": snapshot["train_loss_latest"],
        "arms": arms,
        "supervised_objective": sup,
        "random_median_objective": ref,
        "supervised_wins": sup < ref,
    }


def run(seed: int, rounds: int, outdir: Path):
    import torch
    torch.set_num_threads(1)
    snapshots = train_one(seed, rounds, outdir / "checkpoints")
    results = [evaluate(s) for s in snapshots]
    if len(results) != 2:
        raise AssertionError("missing training exposure")
    receipt = {
        "study_id": "NRF-07-independent-training-seed-transfer-v1",
        "registered_config": "configs/nrf07_independent_seeds_v1.json",
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "rows": results,
        "claim_boundary": (
            "Independent training seeds and new teacher instances within "
            "the same synthetic family; lambda was selected after NRF-04, "
            "and this study does not establish a global quotient."),
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"nrf07_R{rounds}_S{seed}.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--rounds", type=int, choices=DEPTHS, required=True)
    parser.add_argument("--outdir", type=Path, default=Path(
        "experiments/numerical_range_fixed_point_v1/nrf07_generated"))
    args = parser.parse_args()
    output = run(args.seed, args.rounds, args.outdir)
    for r in output["rows"]:
        print(
            r["id"], "J_supervised", f"{r['supervised_objective']:.6f}",
            "J_random", f"{r['random_median_objective']:.6f}",
            "supervised_wins", r["supervised_wins"])


if __name__ == "__main__":
    main()

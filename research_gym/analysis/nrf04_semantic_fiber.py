"""NRF-04: preregistered task-grounded projection and finite-sample fiber-leakage audit.

Uses frozen, hash-pinned tied ToyLoop checkpoints and source-compatible
signed-permutation targets. Projection is train-only; task MSE is held out.
All 8-dimensional arms receive the identical ridge protocol.
Pointwise Jacobian leakage is a measurement, not a uniform certificate.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from research_gym.analysis.nrf03_checkpoint_probe import manifest_entries, load_pinned_weights

N, K = 128, 8
TRAIN_N, EVAL_N, SHADOW_N = 160, 96, 12
TEACHER_PERM8 = {
    101: (0, 81, 111, 24, 47, 19, 46, 87),
    103: (39, 20, 32, 47, 90, 127, 59, 124),
    107: (78, 104, 93, 2, 38, 102, 46, 22),
}
TEACHER_SIGN8 = {
    101: (-1, -1, 1, -1, -1, 1, -1, -1),
    103: (1, -1, 1, -1, -1, 1, -1, 1),
    107: (-1, 1, 1, 1, 1, 1, -1, 1),
}


def gaussian_stream(seed: int):
    """LCG32 plus fresh Box-Muller pair per sample; mirrors the frozen JS audit."""
    state = seed & 0xffffffff
    while True:
        state = (1664525 * state + 1013904223) & 0xffffffff
        u = (state + .5) / 4294967296
        state = (1664525 * state + 1013904223) & 0xffffffff
        v = (state + .5) / 4294967296
        yield math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * v)


def teacher(task_seed: int):
    """Match the source PyTorch signed-permutation teacher exactly."""
    if task_seed not in TEACHER_PERM8:
        raise ValueError("outside preregistered task-seed cohort")
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(task_seed + 17))[:K]
    signs = torch.where(
        torch.rand(N, generator=torch.Generator().manual_seed(task_seed + 29)) > .5,
        1, -1,
    )[:K]
    if tuple(perm.tolist()) != TEACHER_PERM8[task_seed]:
        raise ValueError("source task permutation differs from preregistered teacher")
    if tuple(signs.tolist()) != TEACHER_SIGN8[task_seed]:
        raise ValueError("source task signs differ from preregistered teacher")
    return perm, signs.to(torch.float64)


def array_inputs(task_seed: int, fold: int, size: int):
    if task_seed not in TEACHER_PERM8 or fold not in (1, 3, 4) or size <= 0:
        raise ValueError("outside preregistered split")
    stream = gaussian_stream(20261115 + 7919 * task_seed + 104729 * fold)
    return torch.tensor(
        [[next(stream) for _ in range(N)] for _ in range(size)],
        dtype=torch.float64,
    )


def blocks(parameters):
    w1, b1, w2, b2 = parameters
    if (w1.shape != (N, N) or w2.shape != (N, N)
            or b1.shape != (N,) or b2.shape != (N,)):
        raise ValueError("pinned hidden_size=128 checkpoint required")
    return tuple(x.detach().cpu().to(torch.float64) for x in parameters)


def step_batch(parameters, h):
    w1, b1, w2, b2 = parameters
    return h + .25 * (torch.tanh(h @ w1.T + b1) @ w2.T + b2)


def half_states(parameters, x, rounds):
    if rounds not in (16, 32):
        raise ValueError("outside pinned tied-loop depths")
    h = x.clone()
    for _ in range(rounds // 2):
        h = step_batch(parameters, h)
    return h


def split_data(parameters, seed, rounds, fold, n):
    x = array_inputs(seed, fold, n)
    perm, signs = teacher(seed)
    y = torch.tanh(x[:, perm] * signs)
    return x, half_states(parameters, x, rounds), y


def orthonormal_columns(columns):
    out = []
    for column in columns:
        v = column.clone()
        for _ in range(2):
            for q in out:
                v -= q * torch.dot(q, v)
        length = float(v.norm())
        if length < 1e-10:
            raise ValueError("degenerate supervised or null projection")
        out.append(v / length)
    return torch.stack(out)


def supervised_basis(h, y, mean_h, mean_y, shifted=False):
    train_labels = torch.roll(y, shifts=-1, dims=0) if shifted else y
    covariance = (h - mean_h).T @ (train_labels - mean_y)
    return orthonormal_columns([covariance[:, j] for j in range(K)])


def random_basis(seed):
    stream = gaussian_stream(seed)
    cols = torch.tensor(
        [[next(stream) for _ in range(N)] for _ in range(K)],
        dtype=torch.float64,
    )
    return orthonormal_columns(cols)


def coordinate_basis():
    return torch.eye(N, dtype=torch.float64)[:K]


def ridge_head(train_h, train_y, basis, mean_h, mean_y):
    z = (train_h - mean_h) @ basis.T
    gram = z.T @ z + torch.eye(K, dtype=torch.float64)
    return torch.linalg.solve(gram, z.T @ (train_y - mean_y))


def heldout_mse(test_h, test_y, basis, mean_h, mean_y, head):
    prediction = mean_y + (test_h - mean_h) @ basis.T @ head
    return float(torch.mean((prediction - test_y) ** 2))


def suffix_derivatives(parameters, h, count):
    w1, b1, w2, b2 = parameters
    derivatives = []
    for _ in range(count):
        a = torch.tanh(w1 @ h + b1)
        derivatives.append(1 - a * a)
        h = h + .25 * (w2 @ a + b2)
    return derivatives


def leakage_at(parameters, h, suffix_length, basis):
    """Frobenius(U J_suffix (I-P)) / sqrt(8) via reverse JVPs."""
    w1, b1, w2, b2 = parameters
    derivatives = suffix_derivatives(parameters, h, suffix_length)
    squared = 0.
    for u in basis:
        v = u.clone()
        for derivative in reversed(derivatives):
            v = v + .25 * w1.T @ (derivative * (w2.T @ v))
        dark = v - basis.T @ (basis @ v)
        squared += float(torch.dot(dark, dark))
    return math.sqrt(max(0., squared / K))


def evaluate_one(parameters, task_seed: int, rounds: int):
    parameters = blocks(parameters)
    _, train_h, train_y = split_data(parameters, task_seed, rounds, 1, TRAIN_N)
    _, test_h, test_y = split_data(parameters, task_seed, rounds, 3, EVAL_N)
    _, shadow_h, _ = split_data(parameters, task_seed, rounds, 4, SHADOW_N)
    mean_h = train_h.mean(0)
    mean_y = train_y.mean(0)
    bases = {
        "supervised": supervised_basis(train_h, train_y, mean_h, mean_y),
        "label_shifted_train_null": supervised_basis(
            train_h, train_y, mean_h, mean_y, shifted=True),
        "first_eight_coordinates": coordinate_basis(),
    }
    for i in range(8):
        bases[f"random_{i}"] = random_basis(400000 + 31 * i + 17 * task_seed)
    results = {}
    for name, basis in bases.items():
        head = ridge_head(train_h, train_y, basis, mean_h, mean_y)
        error = heldout_mse(test_h, test_y, basis, mean_h, mean_y, head)
        leakage = sum(leakage_at(parameters, h, rounds // 2, basis)
                      for h in shadow_h) / SHADOW_N
        results[name] = {
            "heldout_mse": error, "local_fiber_leakage": leakage,
        }
    return results


def run_manifest(checkpoints: Path, events: Path):
    torch.set_num_threads(1)
    rows = []
    for entry in manifest_entries(checkpoints, events):
        measurements = evaluate_one(
            load_pinned_weights(entry), entry["seed"], entry["rounds"])
        rows.append({
            "id": f"R{entry['rounds']}_S{entry['seed']}_E{entry['exposure']}",
            "sha256": entry["sha256"],
            "rounds": entry["rounds"], "seed": entry["seed"],
            "exposure": entry["exposure"], "arms": measurements,
        })
    return {
        "study_id": "NRF-04-semantic-fiber-decoder-v1",
        "checkpoint_count": len(rows), "rows": rows,
        "claim_boundary": (
            "Frozen trained toy loop weights and train-only rank-eight "
            "projections. Heldout signed-permutation task MSE and finite "
            "sample Jacobian leakage are separate outcomes; neither "
            "constitutes a uniform quotient or causal model improvement."
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
    output = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

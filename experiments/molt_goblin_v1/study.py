"""Development-only budget panels and append-only prefix evaluation."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from .pilot import PrefixReasoner, batch


def append_only(model: PrefixReasoner, x):
    """Topological evaluation with one projected K/V write per used prefix node.

    No detach: within a single autograd graph the earlier states retain gradients.
    This dense Python reference still concatenates prefix tensors.
    """
    base = model.embedding(x)
    block = model.block
    b, t, d = base.shape
    keys = base.new_zeros((b, 1, d))
    values = base.new_zeros((b, 1, d))
    states = []
    for i in range(t):
        query = base[:, i:i + 1]
        q = block.q(query)
        weights = ((q @ keys.transpose(-1, -2)) * d ** -0.5).softmax(dim=-1)
        z = block.norm(query + block.o(weights @ values))
        h = z + block.ff(z)
        states.append(h)
        if i + 1 < t:
            keys = torch.cat((keys, block.k(h)), dim=1)
            values = torch.cat((values, block.v(h)), dim=1)
    return torch.cat(states, dim=1)


def objective(model: PrefixReasoner, x, y, arm: str, loops: int, loss_schedule: str):
    if arm not in {"prefix_kv", "residual", "one_pass"} or loss_schedule not in {"last", "all"}:
        raise ValueError("unknown arm or objective")
    base = model.embedding(x)
    h = base
    actual = 1 if arm == "one_pass" else loops
    losses = []
    for r in range(actual):
        h = model.block(h if arm == "residual" else base, h)
        if loss_schedule == "all" or r + 1 == actual:
            logits = model.head(h)
            losses.append(nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten()))
    return torch.stack(losses).mean()


def plan(steps: int, loops: int, schedules: tuple[str, ...]):
    cells = []
    for schedule in schedules:
        for arm in ("prefix_kv", "residual"):
            cells.append((arm, steps, schedule, "fixed_data"))
    cells.extend([("one_pass", steps, "last", "fixed_data"),
                  ("one_pass", steps * loops, "last", "reinvest_forward_calls")])
    return cells


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate(model: PrefixReasoner, arm: str, length: int, requested: int, seed: int, device: str):
    model.eval()
    x, y = batch(seed, 64, length, device)
    sync(device)
    start = time.perf_counter()
    logits = model(x, requested, arm)
    sync(device)
    elapsed = time.perf_counter() - start
    ok = logits.argmax(-1).eq(y)
    return {
        "length": length, "requested_loops": requested,
        "actual_loops": 1 if arm == "one_pass" else requested,
        "token_accuracy": ok.float().mean().item(),
        "sequence_accuracy": ok.all(1).float().mean().item(),
        "position_accuracy": ok.float().mean(0).tolist(),
        "sequence_correct": ok.all(1).int().tolist(),
        "current_symbol_baseline": x.eq(y).float().mean().item(),
        "eval_wall_s": elapsed,
    }


def fit(arm: str, steps: int, schedule: str, panel: str, loops: int, seed: int, device: str):
    torch.manual_seed(seed)
    model = PrefixReasoner().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    sync(device)
    wall = time.perf_counter()
    cpu = time.process_time()
    for step in range(steps):
        model.train()
        x, y = batch(seed * 100000 + step, 32, 8, device)
        optimizer.zero_grad(set_to_none=True)
        loss = objective(model, x, y, arm, loops, schedule)
        loss.backward()
        optimizer.step()
    sync(device)
    elapsed = time.perf_counter() - wall
    cpu_elapsed = time.process_time() - cpu
    peak = torch.cuda.max_memory_allocated() if device == "cuda" else None
    actual = 1 if arm == "one_pass" else loops
    tokens = steps * 32 * 8
    rows = [evaluate(model, arm, n, r, seed + 2000000000 + n, device)
            for n in (8, 16) for r in sorted({1, loops, n})]
    parity = None
    if arm == "prefix_kv":
        with torch.no_grad():
            x, _ = batch(seed + 2000000016, 64, 16, device)
            fixed = model.states(x, 16, arm)
            cached = append_only(model, x)
            reference = model.sequential(x)
            parity = {"length": 16, "max_abs_vs_unroll": (fixed - cached).abs().max().item(),
                      "max_abs_vs_sequential": (reference - cached).abs().max().item(),
                      "projected_prefix_tokens_per_sequence": 15,
                      "reference_projected_prefix_tokens_per_sequence": 120}
    return {
        "arm": arm, "panel": panel, "loss_schedule": schedule, "training_seed": seed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "optimizer_updates": steps, "processed_tokens": tokens,
        "forward_block_calls": steps * actual, "forward_token_block_visits": tokens * actual,
        "supervised_token_head_visits": tokens * (actual if schedule == "all" else 1),
        "train_wall_s": elapsed, "train_process_cpu_s": cpu_elapsed,
        "peak_cuda_allocated_bytes": peak, "last_batch_loss": loss.item(),
        "evaluations": rows, "append_only_parity": parity,
    }


def run(steps: int, loops: int, seeds: tuple[int, ...], schedules: tuple[str, ...], device: str):
    if not 1 <= steps or not 2 <= loops <= 8 or steps * loops > 2000:
        raise ValueError("study exceeds step bound")
    if not 1 <= len(seeds) <= 8 or len(set(seeds)) != len(seeds) or not all(0 <= s <= 10000 for s in seeds):
        raise ValueError("invalid development seeds")
    if not schedules or len(set(schedules)) != len(schedules) or set(schedules) - {"last", "all"}:
        raise ValueError("invalid schedules")
    return {"schema": "molt-study-v2", "status": "development_only", "torch": torch.__version__,
            "device": device, "threads": torch.get_num_threads(), "train_loops": loops,
            "promotion_authorized": False, "actual_flops": None, "process_tree_peak_rss_bytes": None,
            "results": [fit(a, s, l, p, loops, root, device) for root in seeds
                        for a, s, l, p in plan(steps, loops, schedules)]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--loops", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[31])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--objective", choices=("last", "all", "both"), default="both")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.threads <= 8:
        parser.error("thread count outside pilot bounds")
    if args.out.exists():
        parser.error("refuse to overwrite report")
    torch.set_num_threads(args.threads)
    schedules = ("last", "all") if args.objective == "both" else (args.objective,)
    result = run(args.steps, args.loops, tuple(args.seeds), schedules, args.device)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({"status": result["status"], "cells": len(result["results"]), "report": str(args.out)}))


if __name__ == "__main__":
    main()

"""Token-input attention TRM using the actual LOOPSCOPE-02 TRMBlock.

New task adapter, trained from scratch; no Sudoku checkpoint or pretrained claim.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from mechinterp.loopscope02.scripts.trm_sudoku import TRMBlock
from research_gym.interp.aux_gym_v2 import (
    FAMILIES_V2, digest_file, dump, majority, metrics, rows, write_rows,
)


@dataclass
class TRMTrajectory:
    logits: Tensor
    high_states: tuple[Tensor, ...]
    low_states: tuple[Tensor, ...]
    block_calls: int


class AuxTokenTRM(nn.Module):
    """One shared attention TRM block; every high commit follows one or more low updates."""

    def __init__(self, width=128, heads=4, low_cycles=1, vocab_size=2048, seq_len=64):
        super().__init__()
        if width % heads or (width // heads) % 2 or low_cycles < 1:
            raise ValueError("invalid TRM dimensions")
        self.low_cycles, self.width, self.seq_len = low_cycles, width, seq_len
        self.embedding = nn.Embedding(vocab_size, width)
        nn.init.normal_(self.embedding.weight, std=width**-0.5)
        self.block = TRMBlock(width, heads)
        self.head_norm = nn.LayerNorm(width)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, width // heads, 2).float() / (width // heads)))
        freq = torch.outer(torch.arange(seq_len).float(), inv_freq)
        emb = torch.cat((freq, freq), dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def lens(self, state):
        return F.linear(self.head_norm(state), self.embedding.weight)

    def forward(self, tokens, *, commits):
        if tokens.ndim != 2 or tokens.shape[1] != self.seq_len or commits < 1:
            raise ValueError("invalid token shape or commit budget")
        x = self.embedding(tokens) * self.width**0.5
        zh, zl = torch.zeros_like(x), torch.zeros_like(x)
        high, low = [], []
        calls = 0
        for _ in range(commits):
            for _ in range(self.low_cycles):
                zl = self.block(zl + zh + x, self.cos, self.sin)
                low.append(zl)
                calls += 1
            zh = self.block(zh + zl, self.cos, self.sin)
            high.append(zh)
            calls += 1
        return TRMTrajectory(self.lens(zh[:, -1]), tuple(high), tuple(low), calls)


def trace_positions(depth, commits):
    # Every commit advances one supervised symbolic step; no labels from future depth.
    return [min(i, depth) for i in range(1, commits + 1)]


def run(args):
    if args.out.exists() or min(args.steps, args.batch, args.commits, args.low_cycles, args.threads) < 1:
        raise ValueError("fresh output and positive budgets required")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed)
    manifest = json.loads((args.data / "manifest.json").read_text())
    def checked(split):
        name = f"{split}/{args.family}.jsonl"
        path = args.data / name
        if digest_file(path) != manifest["files"][name]["sha256"]:
            raise ValueError("shard hash mismatch")
        return rows(path)
    train, val = checked("train"), checked("validation")
    device = torch.device(args.device)
    model = AuxTokenTRM(width=args.width, low_cycles=args.low_cycles).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    tokens = torch.tensor([r["tokens"] for r in train])
    labels = torch.tensor([r["target_token"] for r in train])
    targets = torch.tensor([[r["symbolic_trace_tokens"][j] for j in trace_positions(r["difficulty"], args.commits)]
                            for r in train])
    depths = torch.tensor([r["difficulty"] for r in train])
    buckets = [torch.where(depths == d)[0] for d in sorted(set(depths.tolist()))]
    start = time.perf_counter()
    for _ in range(args.steps):
        choices = torch.randint(len(buckets), (args.batch,), generator=rng)
        idx = torch.tensor([int(buckets[c][torch.randint(len(buckets[c]), (1,), generator=rng)])
                            for c in choices.tolist()])
        optimizer.zero_grad()
        o = model(tokens[idx].to(device), commits=args.commits)
        y = labels[idx].to(device)
        loss = F.cross_entropy(o.logits, y)
        aux = []
        for i, state in enumerate(o.high_states[:-1]):
            if args.mode == "final":
                break
            t = y if args.mode == "repeated_final" else targets[idx, i].to(device)
            aux.append(F.cross_entropy(model.lens(state[:, -1]), t))
        if aux:
            loss = loss + args.weight * torch.stack(aux).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    model.eval()
    pred = []
    visit_records = []
    with torch.no_grad():
        for offset in range(0, len(val), args.batch):
            chunk = val[offset:offset + args.batch]
            o = model(torch.tensor([r["tokens"] for r in chunk], device=device), commits=args.commits)
            pred.extend(o.logits.argmax(-1).cpu().tolist())
            high = [model.lens(s[:, -1]).argmax(-1).cpu().tolist() for s in o.high_states]
            low = [model.lens(s[:, -1]).argmax(-1).cpu().tolist() for s in o.low_states]
            for j, r in enumerate(chunk):
                visit_records.append({"example_id": r["example_id"], "depth": r["difficulty"],
                    "target": r["target_token"], "prediction": int(o.logits[j].argmax()),
                    "high_predictions": [s[j] for s in high], "low_predictions": [s[j] for s in low]})
    args.out.mkdir(parents=True)
    torch.save(model.state_dict(), args.out / "model.pt")
    write_rows(args.out / "validation_predictions.jsonl", visit_records)
    dump(args.out / "receipt.json", {
        "arm": "AUX-ATTN-TRM", "family": args.family, "mode": args.mode,
        "seed": args.seed, "steps": args.steps, "batch": args.batch, "width": args.width,
        "commits": args.commits, "low_cycles": args.low_cycles,
        "physical_block_calls_per_forward": args.commits * (args.low_cycles + 1),
        "parameters": sum(p.numel() for p in model.parameters()), "lr": args.lr,
        "aux_weight": args.weight, "device": str(device), "training_seconds": elapsed,
        "manifest_sha256": digest_file(args.data / "manifest.json"),
        "checkpoint_sha256": digest_file(args.out / "model.pt"),
        "predictions_sha256": digest_file(args.out / "validation_predictions.jsonl"),
        "validation": metrics(val, pred, majority(train)),
    })


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--family", choices=FAMILIES_V2, required=True)
    p.add_argument("--mode", choices=("final", "repeated_final", "symbolic_trace"), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--commits", type=int, default=4)
    p.add_argument("--low-cycles", type=int, default=1)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight", type=float, default=0.5)
    run(p.parse_args())


if __name__ == "__main__":
    main()

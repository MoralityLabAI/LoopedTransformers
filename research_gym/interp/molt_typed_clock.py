"""MOLT-12: typed primitive-operation clock using the native tied decoder block.

Structured competence control for modular recurrence. This changes the input
interface. It is not an accuracy improvement of the original AUX-V2 model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import resource
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from research_gym.interp.models import build_model

MOD = 17
OPERATIONS = 33
VOCAB = 2048
WIDTH = 64
QUERY = 17
OP_START = 32
ADDRESS_START = 128
ARMS = ("separable", "addressed")
DEPTHS = (1, 2, 4, 8, 12)
SITE = "decoder.blocks.0.mlp.2"


def target_step(state, op):
    """Host-only oracle; never called by the model's forward transition."""
    return (state * (op + 1)) % MOD if op < 16 else (state + op - 16) % MOD


def trace_of(world):
    value = world[0]
    trace = []
    for op in world[1:]:
        value = target_step(value, op)
        trace.append(value)
    return tuple(trace)


def make_worlds(seed, depth, count, excluded=()):
    rng = random.Random(seed)
    seen = set(excluded)
    worlds = []
    while len(worlds) < count:
        ops = tuple(op for _ in range(depth) for op in (rng.randrange(16), 16 + rng.randrange(MOD)))
        row = (rng.randrange(MOD), *ops)
        if row not in seen:
            seen.add(row)
            worlds.append(row)
    return worlds


def bank_digest(rows):
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def primitive_bank():
    x = torch.tensor([(s, op) for s in range(MOD) for op in range(OPERATIONS)], dtype=torch.long)
    y = torch.tensor([target_step(s, op) for s, op in x.tolist()], dtype=torch.long)
    return x, y


class TypedClockDecoder(nn.Module):
    """One shared physical attention/MLP block, one continuous carry register."""

    def __init__(self, arm):
        super().__init__()
        if arm not in ARMS:
            raise ValueError("unknown clock arm")
        self.arm = arm
        self.decoder = build_model("LOOP-T", hidden_size=WIDTH, num_heads=4, visits=1, vocab_size=VOCAB)

    def stage(self, state, operation):
        w = self.decoder.embedding.weight
        state_embed = state @ w[:MOD]
        op_embed = self.decoder.embedding(operation + OP_START)
        query = w[QUERY].expand(len(state), -1)
        if self.arm == "addressed":
            # This is a learned lookup of the public operation/state address.
            # It contains no arithmetic target. Soft carry mixes rows.
            ids = ADDRESS_START + MOD * operation[:, None] + torch.arange(MOD, device=state.device)
            query = query + torch.einsum("bs,bsd->bd", state, self.decoder.embedding(ids))
        raw = torch.stack((state_embed, op_embed, query), dim=1)
        h = self.decoder.input_norm(raw)
        h = h + self.decoder._position_encoding(3, device=h.device, dtype=h.dtype)
        h = self.decoder.block_at(0)(h)
        logits = self.decoder.lens(h[:, -1])
        return logits, logits[:, :MOD].softmax(-1)

    def rollout(self, tokens, passes=None):
        if tokens.ndim != 2 or tokens.shape[1] < 2:
            raise ValueError("expected start and public operation sequence")
        if passes is None:
            passes = tokens.shape[1] - 1
        if not 1 <= passes < tokens.shape[1]:
            raise ValueError("invalid stage count")
        state = F.one_hot(tokens[:, 0], MOD).to(self.decoder.embedding.weight)
        readouts, boundaries = [], []
        for visit in range(passes):
            logits, state = self.stage(state, tokens[:, visit + 1])
            readouts.append(logits)
            boundaries.append(state)
        return torch.stack(readouts, dim=1), torch.stack(boundaries, dim=1)


def tensor_worlds(worlds):
    return torch.tensor(worlds, dtype=torch.long), torch.tensor([trace_of(w) for w in worlds], dtype=torch.long)


@torch.no_grad()
def evaluate(model, worlds, batch=32):
    model.eval()
    x, y = tensor_worlds(worlds)
    legal, full, confidence = [], [], []
    for i in range(0, len(x), batch):
        logits, probs = model.rollout(x[i:i + batch])
        legal.append(probs.argmax(-1))
        full.append(logits.argmax(-1))
        confidence.append(probs.max(-1).values)
    legal, full, confidence = torch.cat(legal), torch.cat(full), torch.cat(confidence)
    return {
        "worlds": len(worlds), "terminal_exact": float((legal[:, -1] == y[:, -1]).double().mean()),
        "unrestricted_terminal_exact": float((full[:, -1] == y[:, -1]).double().mean()),
        "complete_trace_exact": float((legal == y).all(1).double().mean()),
        "unrestricted_trace_exact": float((full == y).all(1).double().mean()),
        "stage_exact": (legal == y).double().mean(0).tolist(),
        "mean_legal_confidence": float(confidence.double().mean()),
        "predictions": legal[:, -1].tolist(),
        "unrestricted_predictions": full[:, -1].tolist(),
    }


@torch.no_grad()
def primitive_exact(model):
    model.eval()
    x, y = primitive_bank()
    correct, full = 0, 0
    for tokens, target in zip(x.split(64), y.split(64), strict=True):
        logits, _ = model.stage(F.one_hot(tokens[:, 0], MOD).to(model.decoder.embedding.weight), tokens[:, 1])
        correct += int((logits[:, :MOD].argmax(-1) == target).sum())
        full += int((logits.argmax(-1) == target).sum())
    return {"examples": len(x), "legal_correct": correct, "unrestricted_correct": full}


def fit(arm, seed, train, primitive_updates, rollout_updates):
    torch.manual_seed(seed)
    model = TypedClockDecoder(arm)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.004)
    primitive_x, primitive_y = primitive_bank()
    train_x, train_y = tensor_worlds(train)
    rng = torch.Generator().manual_seed(seed + 41000)
    start = time.perf_counter()
    model.train()
    losses = []
    for _ in range(primitive_updates):
        idx = torch.randint(len(primitive_x), (64,), generator=rng)
        x = primitive_x[idx]
        p = F.one_hot(x[:, 0], MOD).float()
        logits, _ = model.stage(p, x[:, 1])
        loss = F.cross_entropy(logits, primitive_y[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    before = primitive_exact(model)
    model.train()
    continuation = []
    for _ in range(rollout_updates):
        idx = torch.randint(len(train_x), (16,), generator=rng)
        logits, _ = model.rollout(train_x[idx])
        loss = F.cross_entropy(logits.flatten(0, 1), train_y[idx].flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        continuation.append(float(loss.detach()))
    return model.eval(), {
        "arm": arm, "seed": seed, "primitive_updates": primitive_updates,
        "rollout_updates": rollout_updates, "primitive_before_rollout": before,
        "primitive_after_rollout": primitive_exact(model),
        "primitive_final_loss": losses[-1],
        "rollout_final_loss": continuation[-1] if continuation else None,
        "train_stage_examples": primitive_updates * 64 + rollout_updates * 16 * 8,
        "parameters": sum(p.numel() for p in model.parameters()),
        "training_seconds": time.perf_counter() - start,
    }


def competent(row):
    p = row["primitive_after_rollout"]
    return (p["unrestricted_correct"] / p["examples"] >= 0.98
            and row["development"]["4"]["unrestricted_terminal_exact"] >= 0.90
            and row["development"]["8"]["unrestricted_terminal_exact"] >= 0.80)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--primitive-updates", type=int, default=512)
    p.add_argument("--rollout-updates", type=int, default=64)
    p.add_argument("--roots", type=int, default=128)
    p.add_argument("--seeds", nargs="+", type=int, default=[157, 163])
    args = p.parse_args()
    if (args.out.exists() or not 1 <= args.primitive_updates <= 2048
            or not 0 <= args.rollout_updates <= 256 or not 1 <= args.roots <= 256
            or not 1 <= len(args.seeds) <= 4 or len(set(args.seeds)) != len(args.seeds)
            or any(not 0 <= s <= 100000 for s in args.seeds)):
        p.error("outside development bounds or output exists")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    train = make_worlds(121001, 4, 512)
    development = {d: make_worlds(121100 + d, d, args.roots, train) for d in DEPTHS}
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for seed in args.seeds:
        for arm in ARMS:
            model, row = fit(arm, seed, train, args.primitive_updates, args.rollout_updates)
            row["development"] = {str(d): evaluate(model, worlds) for d, worlds in development.items()}
            row["competence_gate_passed"] = competent(row)
            checkpoint = args.out / f"{arm}-{seed}.pt"
            torch.save(model.state_dict(), checkpoint)
            row["checkpoint_file"] = checkpoint.name
            row["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            rows.append(row)
    report = {
        "schema": "molt12-clock-v1", "status": "development_only", "torch": torch.__version__,
        "model_kind": "typed_clock_native_shared_decoder", "physical_site": SITE,
        "word": "(multiply.add)^depth", "primitive_corpus": 17 * 33,
        "train_worlds": len(train), "train_worlds_sha256": bank_digest(train),
        "development_roster_sha256": {str(d): bank_digest(w) for d, w in development.items()},
        "gate": {"primitive_unrestricted": 0.98, "depth4_unrestricted": 0.90, "depth8_unrestricted": 0.80},
        "rows": rows, "native_bpd_fit_performed": False,
        "confirmation_ready": False, "promotion_authorized": False,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
    }
    with (args.out / "receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
    print(json.dumps({"schema": report["schema"], "rows": [
        {k: r[k] for k in ("arm", "seed", "primitive_after_rollout", "competence_gate_passed",
                           "training_seconds")} | {"development": {
            d: {k: v[k] for k in ("terminal_exact", "unrestricted_terminal_exact", "complete_trace_exact")}
            for d, v in r["development"].items()}}
        for r in rows]}, sort_keys=True))


if __name__ == "__main__":
    main()

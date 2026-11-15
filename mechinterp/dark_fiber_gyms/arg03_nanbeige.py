#!/usr/bin/env python3
"""ARG-03-N: local Nanbeige context-fiber gym reusing the pinned ARG-02 adapter.

Readout-preserving causal patches operate on context positions BEFORE the
three scored positions. Produces model-only answer tapes; labels/futures stay
with the separate EvalMath scorer. This runner never imports EvalMath.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

import torch

from .core import (
    arg02_program_scores, nanbeige_context_swap, sha256_file, write_json,
)

HERE = Path(__file__).resolve().parent


def public_rows(path: Path) -> list[dict]:
    required = {"case_id", "topology", "observed_degree", "demos", "query", "public_sha256"}
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if set(row) != required:
            raise ValueError(f"line {i}: runner received hidden or missing fields: {sorted(set(row)^required)}")
        if len(row["demos"]) != 2 or row["observed_degree"] != 12:
            raise ValueError(f"line {i}: unexpected ARG-02 record")
        rows.append(row)
    if len({r["case_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate case IDs")
    return rows


def load_engine(root: Path, device: str):
    # Requires Compactification-Theory checked out at ARG-02 Nanbeige branch,
    # with local checkpoint lock/snapshots. No runtime downloads.
    sys.path.insert(0, str(root.resolve()))
    from research.algernon.algebraic_reasoning_gym.arg02 import runner
    from research.algernon.algebraic_reasoning_gym.arg02.settings import load_lock
    entry = load_lock()["nanbeige"]
    model, tok = runner.load_model(entry, device)
    model.eval()
    engine = runner.Engine(model, tok, block=(0, len(model.model.layers)), shots=[], device=device)
    return engine, entry


@torch.inference_mode()
def evaluate_case(engine, row: dict, *, sigma: float, width: int, seed: int, depths=(1, 2, 4, 8)):
    loop = engine.loop
    ids, prompt_tokens = engine.rows(row)
    emb, ctx = loop.prepare(ids)
    h1 = loop.run(emb, 0, loop.n, ctx)
    lp1 = loop.readout(h1, ctx)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(tuple(emb.shape), generator=gen, dtype=torch.float32).to(emb.device)
    emb_scale = emb.float().square().mean().sqrt()
    donor = loop.run((emb.float() + sigma * emb_scale * noise).to(emb.dtype), 0, loop.n, ctx)
    patched, patch_meta = nanbeige_context_swap(h1, donor, width=width, readout_tail=3)
    lp1_after = loop.readout(patched, ctx)
    leak = float((lp1 - lp1_after).abs().max().item())
    if leak > 1e-6:
        raise AssertionError(f"full-vocabulary output changed before continuation: {leak}")
    row_stats = {
        "case_id": row["case_id"],
        "public_sha256": row["public_sha256"],
        "prompt_tokens": prompt_tokens,
        "baseline": {"1": arg02_program_scores(lp1, engine.op_ids)},
        "context_swap": {"1": arg02_program_scores(lp1_after, engine.op_ids)},
        "patch": patch_meta,
        "max_precontinuation_logprob_error": leak,
        "postloop_readout_distance": {},
    }
    clean, intervention = h1, patched
    for t in range(2, max(depths) + 1):
        clean = loop.run(clean, 0, loop.n, ctx)
        intervention = loop.run(intervention, 0, loop.n, ctx)
        if t in depths:
            lc, li = loop.readout(clean, ctx), loop.readout(intervention, ctx)
            row_stats["baseline"][str(t)] = arg02_program_scores(lc, engine.op_ids)
            row_stats["context_swap"][str(t)] = arg02_program_scores(li, engine.op_ids)
            row_stats["postloop_readout_distance"][str(t)] = float((lc.float()-li.float()).norm().item())
    return row_stats


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--public-jsonl", type=Path, required=True, help="Fresh ARG-03 public-only split")
    p.add_argument("--compactification-root", type=Path, required=True, help="Checkout containing arg02 runner/lock")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=4, help="Local pilot count; 0 means entire supplied split")
    p.add_argument("--sigma", type=float, default=0.02, help="Donor embedding-noise RMS scale")
    p.add_argument("--width", type=int, default=8, help="Context positions immediately before scored tail")
    p.add_argument("--seed", type=int, default=20261115)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-small-vram", action="store_true")
    args = p.parse_args(argv)
    if args.limit < 0 or args.width < 1 or not (0 < args.sigma < 1):
        p.error("invalid limit, width, or sigma")
    rows = public_rows(args.public_jsonl)
    chosen = rows[:args.limit] if args.limit else rows
    metadata = {
        "schema": "darkfibers.arg03.nanbeige.local.v1",
        "public_jsonl_sha256": sha256_file(args.public_jsonl),
        "source_runner": str(args.compactification_root),
        "chosen_count": len(chosen), "total_public_count": len(rows),
        "sigma": args.sigma, "width": args.width, "seed": args.seed, "device": args.device,
        "code_sha256": sha256_file(Path(__file__)),
        "status": "dry-run" if args.dry_run else "running",
    }
    if args.dry_run:
        print(json.dumps(metadata, indent=2))
        return 0
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device unavailable; use --dry-run or --device cpu")
        memory = torch.cuda.get_device_properties(args.device).total_memory
        if memory < 12 * 1024**3 and not args.allow_small_vram:
            raise RuntimeError("BF16 Nanbeige adapter needs ~10+ GiB VRAM; RTX 3050 4GB: run TRM gym instead")
    engine, entry = load_engine(args.compactification_root, args.device)
    metadata["checkpoint_revision"] = entry["revision"]
    metadata["checkpoint_lock_sha256"] = sha256_file(
        args.compactification_root / "research/algernon/algebraic_reasoning_gym/arg02/config/checkpoint_lock.json"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir/"manifest.json", metadata)
    tape = args.out_dir/"answer_tape.jsonl"
    existing = {}
    if tape.is_file():
        for line in tape.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                existing[record["case_id"]] = record
    start = time.perf_counter()
    for idx, row in enumerate(chosen, 1):
        if row["case_id"] in existing:
            if existing[row["case_id"]]["public_sha256"] != row["public_sha256"]:
                raise ValueError("resume identity mismatch")
            continue
        stable_seed = args.seed ^ int(row["public_sha256"][:8], 16)
        result = evaluate_case(engine, row, sigma=args.sigma, width=args.width, seed=stable_seed)
        with tape.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, allow_nan=False, separators=(",", ":"))+"\n")
        print(f"{idx}/{len(chosen)} {row['case_id']}: full-readout parity {result['max_precontinuation_logprob_error']:.2e}", flush=True)
    metadata["status"] = "completed"
    metadata["elapsed_seconds"] = round(time.perf_counter() - start, 2)
    if args.device.startswith("cuda"):
        metadata["peak_allocated_vram_bytes"] = torch.cuda.max_memory_allocated()
    metadata["answer_tape_sha256"] = sha256_file(tape)
    write_json(args.out_dir/"manifest.json", metadata)
    print("wrote", tape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

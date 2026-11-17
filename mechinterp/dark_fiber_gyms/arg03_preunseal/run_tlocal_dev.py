"""ARG-03 development-only *time-local* H2a experiment on pinned Nanbeige.

The historical intervention enters at loop 1 and is allowed to propagate through
t later loops. This independent new arm starts from an exact clean state at t,
takes the *context-only* coordinates from that structured trajectory, and
compares a fresh random context direction at the same t and SAME input-state
L2 dose. The three scored-token states are identical before continuation.
All 64 program scores come from model-only inference, never EvalMath labels.

Run from repo root with:
  python -m mechinterp.dark_fiber_gyms.arg03_preunseal.run_tlocal_dev --help

HARD GATE: only ARG-03 development case IDs and dev-named public JSONL files
are accepted. This module cannot unseal the frozen test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from .. import arg03_nanbeige as NB
from ..core import arg02_program_scores, nanbeige_context_swap, sha256_file, write_json
from .controls import match_random_direction


DEPTHS = (2, 4, 8)
PREFIXES = ("arg03:development:", "arg03:development_finalR_fit:", "arg03:development_finalR_valid:")


def check_dev_path(path: Path) -> None:
    parts = [part.lower() for part in path.parts]
    if any(part in {"test", "test.jsonl", "heldout", "test_full168"} for part in parts):
        raise ValueError("sealed held-out test path is forbidden")
    if not path.name.startswith("development") or path.suffix != ".jsonl":
        raise ValueError("time-local runner only accepts development*.jsonl")


def load_development(path: Path) -> list[dict]:
    check_dev_path(path)
    rows = NB.public_rows(path)
    if not rows or any(not row["case_id"].startswith(PREFIXES) for row in rows):
        raise ValueError("only ARG-03 development case IDs are allowed")
    return rows


def _candidate_scores(logprob: torch.Tensor, engine) -> list[float]:
    return arg02_program_scores(logprob, engine.op_ids)


def _norm_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max())


@torch.inference_mode()
def evaluate_tlocal_case(engine, row: dict, *, sigma: float = 0.02, width: int = 8,
                         seed: int = 20261117, depths: tuple[int, ...] = DEPTHS) -> dict:
    if any(not (1 < depth <= 8) for depth in depths):
        raise ValueError("depths must be 2..8")
    NB.assert_oracle_isolated()
    start = time.perf_counter()
    loop = engine.loop
    ids, prompt_tokens = engine.rows(row)
    emb, ctx = loop.prepare(ids)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(tuple(emb.shape), dtype=torch.float32, generator=generator).to(emb.device)
    donor_input = (emb.float() + sigma * emb.float().square().mean().sqrt() * noise).to(emb.dtype)
    clean, perturbed = NB._run_pairs(loop, [(emb, ctx), (donor_input, ctx)])
    structured, initial_patch = nanbeige_context_swap(clean, perturbed, width=width)
    result = {
        "schema": "darkfibers.arg03.h2a_tlocal_dev.v1",
        "case_id": row["case_id"],
        "public_sha256": row["public_sha256"],
        "prompt_tokens": prompt_tokens,
        "source_split": "development_only",
        "sigma": sigma,
        "window": width,
        "depths": {},
        "historical_h2a_unchanged": True,
        "initial_patch": initial_patch,
    }
    for depth in range(2, max(depths) + 1):
        clean, structured = NB._run_pairs(loop, [(clean, ctx), (structured, ctx)])
        if depth not in depths:
            continue
        # Previous-state differences are allowed to spread over the sequence.
        # The time-local intervention reanchors every other token to clean t.
        local, _ = nanbeige_context_swap(clean, structured, width=width)
        random_generator = torch.Generator(device="cpu").manual_seed(seed + 1009 * depth)
        random_direction = torch.randn(tuple(clean.shape), generator=random_generator,
                                       dtype=torch.float32).to(clean.device)
        try:
            random_local, certificate = match_random_direction(
                clean, local, random_direction, prefix_width=width, scored_tail=3,
            )
        except ValueError as exc:
            if "Zero structured" not in str(exc):
                raise
            result["depths"][str(depth)] = {
                "skipped_reason": "structured context displacement rounds to zero at this depth",
            }
            continue

        current = loop.readout(clean, ctx)
        pre_structured = loop.readout(local, ctx)
        pre_random = loop.readout(random_local, ctx)
        parity_struct = _norm_error(current, pre_structured)
        parity_random = _norm_error(current, pre_random)
        if max(parity_struct, parity_random) > 1e-6:
            raise AssertionError(f"current full-vocabulary logit parity violated at depth {depth}")
        if not torch.equal(local[:, -3:, :], clean[:, -3:, :]):
            raise AssertionError("structured patch modified scored states")

        # Each continuation is one identical stationary native 22-layer loop.
        next_clean, next_struct, next_random = NB._run_pairs(
            loop, [(clean, ctx), (local, ctx), (random_local, ctx)],
        )
        q_clean = loop.readout(next_clean, ctx)
        q_struct = loop.readout(next_struct, ctx)
        q_random = loop.readout(next_random, ctx)
        norm_struct = float((local.float() - clean.float()).norm())
        norm_random = float((random_local.float() - clean.float()).norm())
        scores_clean = _candidate_scores(q_clean, engine)
        scores_struct = _candidate_scores(q_struct, engine)
        scores_rand = _candidate_scores(q_random, engine)
        result["depths"][str(depth)] = {
            "pre_head_logprob_error": {
                "structured": parity_struct, "random_matched": parity_random,
            },
            "matched_l2_dose": {
                "structured": norm_struct,
                "random_matched": norm_random,
                "relative_error": certificate["max_relative_norm_error"],
            },
            "next_full_vocab_l2": {
                "structured": float((q_struct.float() - q_clean.float()).norm()),
                "random_matched": float((q_random.float() - q_clean.float()).norm()),
            },
            "next_64_program_scores": {
                "clean": scores_clean,
                "structured": scores_struct,
                "random_matched": scores_rand,
            },
            "ranking_change": {
                "structured": NB.ranking_comparison(scores_clean, scores_struct),
                "random_matched": NB.ranking_comparison(scores_clean, scores_rand),
            },
            "certificate": certificate,
        }
    if emb.device.type == "cuda":
        torch.cuda.synchronize(emb.device)
    result["elapsed_seconds"] = round(time.perf_counter() - start, 4)
    NB.assert_oracle_isolated()
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--public-jsonl", type=Path, required=True)
    p.add_argument("--compactification-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=2, help="2-case pilot by default; 0 = all development")
    p.add_argument("--width", type=int, default=8)
    p.add_argument("--sigma", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=20261117)
    p.add_argument("--min-free-gib", type=float, default=6.0)
    p.add_argument("--vram-reserve-gib", type=float, default=2.2)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    if a.limit < 0 or a.width < 1 or not (0 < a.sigma < 1):
        p.error("invalid limit, width or sigma")
    rows = load_development(a.public_jsonl)
    chosen = rows[:a.limit] if a.limit else rows
    manifest = {
        "schema": "darkfibers.arg03.h2a_tlocal_manifest.v1",
        "split": "development_only",
        "input_sha256": sha256_file(a.public_jsonl),
        "code_sha256": sha256_file(Path(__file__)),
        "compactification_root_name": a.compactification_root.name,
        "chosen_count": len(chosen),
        "total_count": len(rows),
        "width": a.width, "sigma": a.sigma, "seed": a.seed, "device": a.device,
        "depths": list(DEPTHS),
        "historical_protocol_untouched": True,
        "status": "dry-run" if a.dry_run else "running",
    }
    if a.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0
    old_manifest = a.out_dir / "manifest.json"
    if old_manifest.is_file():
        old = json.loads(old_manifest.read_text(encoding="utf-8"))
        for k in ("schema", "split", "input_sha256", "code_sha256", "chosen_count",
                  "width", "sigma", "seed", "device", "depths"):
            if old.get(k) != manifest[k]:
                raise RuntimeError(f"resume configuration mismatch: {k}")
    if a.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.cuda.init()
        free, total = NB._true_cuda_free_and_total_bytes(torch.device(a.device))
        if total < 12 * 1024**3 or free < a.min_free_gib * 1024**3:
            raise RuntimeError("insufficient global free VRAM for 3B Nanbeige")
    engine, lock = NB.load_engine(
        a.compactification_root, a.device, vram_reserve_gib=a.vram_reserve_gib,
    )
    manifest["checkpoint_revision"] = lock["revision"]
    manifest["checkpoint_lock_sha256"] = sha256_file(
        a.compactification_root / "research/algernon/algebraic_reasoning_gym/arg02/config/checkpoint_lock.json"
    )
    a.out_dir.mkdir(parents=True, exist_ok=True)
    tape = a.out_dir / "tlocal_dev_tape.jsonl"
    prior = {}
    if tape.is_file():
        for line in tape.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec["case_id"] in prior:
                    raise ValueError("duplicate case in existing t-local tape")
                prior[rec["case_id"]] = rec["public_sha256"]
    write_json(a.out_dir / "manifest.json", manifest)
    start = time.perf_counter()
    for i, row in enumerate(chosen, 1):
        cid = row["case_id"]
        if cid in prior:
            if prior[cid] != row["public_sha256"]:
                raise ValueError("resume case digest mismatch")
            continue
        seed = a.seed ^ int(row["public_sha256"][:8], 16)
        result = evaluate_tlocal_case(engine, row, sigma=a.sigma, width=a.width, seed=seed)
        with tape.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, allow_nan=False, separators=(",", ":")) + "\n")
        print(f"{i}/{len(chosen)} {cid}: t-local depths {list(result['depths'])}", flush=True)
    manifest["status"] = "completed"
    manifest["elapsed_seconds"] = round(time.perf_counter() - start, 2)
    manifest["answer_tape_sha256"] = sha256_file(tape)
    if a.device.startswith("cuda"):
        manifest["peak_allocated_vram_bytes"] = int(torch.cuda.max_memory_allocated(a.device))
    write_json(a.out_dir / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

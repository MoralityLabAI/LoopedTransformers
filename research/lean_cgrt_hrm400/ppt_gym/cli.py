"""PPT Gym CLI: fresh verified-pool training, depth sweeps and DRAM support/query trials.

Run from research/lean_cgrt_hrm400:
  python -m ppt_gym fixture --out runs/ppt-fixture
  python -m ppt_gym train --fixture runs/ppt-fixture --allow-synthetic --plan a --out runs/ppt-a
  python -m ppt_gym compare --fixture runs/ppt-fixture --allow-synthetic \
      --checkpoint runs/ppt-a/checkpoint.pt --out runs/ppt-a-compare

Use --pack PATH to load the checked-candidate pack exported from DataGoblin.
The CLI never executes Lean or turns synthetic results into Lean proof claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch

from .core import Config, FastHead, PlasticProofTransformer, RetrievalControl, select_candidate, supervised_loss
from .data import Case, canonical, ensure_no_overlap, load_checked_pack, load_fixture, make_fixture, sha256_file


def source_cases(args, split):
    if bool(args.pack) == bool(args.fixture):
        raise ValueError("choose exactly one source: --pack or --fixture")
    if args.fixture:
        if not args.allow_synthetic:
            raise ValueError("synthetic fixtures require --allow-synthetic")
        return load_fixture(args.fixture, split, args.max_cases)
    return load_checked_pack(args.pack, split, args.max_cases)


def save_json(path: Path, value):
    path.write_text(canonical(value) + "\n", encoding="utf-8")


def checkpoint_load(path: Path):
    item = torch.load(path, map_location="cpu", weights_only=True)
    if item.get("schema") != "ppt_gym.checkpoint.v1":
        raise ValueError("unexpected PPT checkpoint schema")
    cfg = Config(**item["config"]).validate()
    model = PlasticProofTransformer(cfg)
    model.load_state_dict(item["weights"], strict=True)
    model.eval()
    return model, item


def train(args):
    cases, source = source_cases(args, "train")
    if not cases:
        raise ValueError("empty train split")
    usable = [c for c in cases if any(x >= 0 for x in c.labels)]
    if not usable:
        raise ValueError("training requires definitive candidate outcomes")
    cfg = Config(args.plan, args.width, args.heads, 8, args.max_bytes, args.mlp_ratio).validate()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    torch.set_num_threads(args.threads)
    model = PlasticProofTransformer(cfg)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    logs = []
    for step in range(args.steps):
        if time.perf_counter() - started > args.max_seconds:
            break
        case = rng.choice(usable)
        depth = rng.choice(args.depths)
        outputs = model(case.goal, case.candidates, depth=depth, return_all=True,
                        middle_order=args.middle_order)
        loss = supervised_loss(outputs, case.labels)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        logs.append({"step": step + 1, "root": case.root,
                     "depth": depth, "loss": round(float(loss.detach()), 6)})
    model.eval()
    checkpoint = args.out / "checkpoint.pt"
    torch.save({"schema": "ppt_gym.checkpoint.v1", "config": asdict(cfg),
                "weights": model.state_dict(), "source_manifest_sha256": source["manifest_sha256"],
                "source_evidence_class": source["evidence_class"],
                "train_steps": len(logs), "seed": args.seed,
                "train_order": args.middle_order}, checkpoint)
    with (args.out / "train_steps.jsonl").open("x", encoding="utf-8") as stream:
        for row in logs:
            stream.write(canonical(row) + "\n")
    result = {"schema": "ppt_gym.train.v1", "model": model.architecture(),
              "source": source, "train_roots": len({c.lineage for c in usable}),
              "updates_completed": len(logs), "requested_steps": args.steps,
              "training_seconds": round(time.perf_counter() - started, 3),
              "train_order": args.middle_order,
              "checkpoint_sha256": sha256_file(checkpoint),
              "train_steps_sha256": sha256_file(args.out / "train_steps.jsonl"),
              "scientific_eligible": source["scientific_eligible"],
              "claim": "offline checked-candidate training" if source["scientific_eligible"]
                       else "synthetic engineering smoke"}
    save_json(args.out / "manifest.json", result)
    return result


@torch.no_grad()
def predictions(model, case: Case, depth: int, message_mode: str):
    outputs = model(case.goal, case.candidates, depth=depth, message_mode=message_mode)
    return outputs[-1]


def _groups(cases: list[Case], seed: int, max_support: int, query_cases=None):
    groups = defaultdict(list)
    for case in cases:
        groups[case.family].append(case)
    query_groups = defaultdict(list)
    if query_cases is not None:
        ensure_no_overlap(cases, query_cases)
        for case in query_cases:
            query_groups[case.family].append(case)
    rng = random.Random(seed)
    for family, rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda row: row.lineage)
        rng.shuffle(ordered)
        if query_cases is None:
            support_size = min(max_support, max(1, len(ordered) // 2))
            support, query = ordered[:support_size], ordered[support_size:]
        else:
            support, query = ordered[:max_support], query_groups[family]
        if support and query:
            ensure_no_overlap(support, query)
            yield family, support, query


def compare(args):
    model, checkpoint = checkpoint_load(args.checkpoint)
    support_cases, src = source_cases(args, args.support_split)
    query_cases = None
    qsrc = src
    if args.query_split != args.support_split:
        query_cases, qsrc = source_cases(args, args.query_split)
    if checkpoint["source_manifest_sha256"] != src["manifest_sha256"]:
        raise ValueError("checkpoint and support source manifest mismatch")
    if query_cases is not None and src["manifest_sha256"] != qsrc["manifest_sha256"]:
        raise ValueError("query and support must originate from the same frozen pack")
    if any(c.evidence != checkpoint["source_evidence_class"] for c in support_cases):
        raise ValueError("checkpoint trained with a different evidence class")
    model.eval()
    args.out.mkdir(parents=True, exist_ok=False)
    totals = {arm: {"definitive_queries": 0, "verified_choice": 0, "censored_choices": 0}
              for arm in ("frozen", "dram_fast", "retrieval", "inverted_feedback")}
    rows_out = []
    for family, support, query in _groups(support_cases, args.seed, args.support, query_cases):
        fast = FastHead(model.cfg.width, args.fast_lr)
        inverted = FastHead(model.cfg.width, args.fast_lr)
        retrieval = RetrievalControl(model.cfg.width)
        updates = 0
        # One predeclared tactic check per support goal, identical across arms.
        # Other archived candidate labels stay hidden from the online updater.
        for case in support:
            out = predictions(model, case, args.depth, args.message_mode)
            idx = int(out["logits"].argmax())
            label = case.labels[idx]
            if label not in (0, 1):
                continue
            feat = out["features"][idx]
            fast.update(feat, label)
            inverted.update(feat, 1 - label)
            retrieval.update(feat, label)
            updates += 1
        for case in query:
            out = predictions(model, case, args.depth, args.message_mode)
            base = out["logits"]
            feat = out["features"]
            policy = {
                "frozen": base, "dram_fast": base + args.fast_scale * fast.predict(feat),
                "retrieval": base + args.fast_scale * retrieval.predict(feat),
                "inverted_feedback": base + args.fast_scale * inverted.predict(feat),
            }
            decisions = {}
            for arm, logits in policy.items():
                index, success = select_candidate(logits, case.labels)
                verdict = case.labels[index]
                tally = totals[arm]
                if verdict == -1:
                    tally["censored_choices"] += 1
                else:
                    tally["definitive_queries"] += 1
                    tally["verified_choice"] += int(success)
                decisions[arm] = {"candidate_index": index, "receipt_label": verdict}
            rows_out.append({"root": case.root, "lineage": case.lineage, "family": family,
                             "split": case.split, "support_checks": updates,
                             "candidate_count": len(case.candidates), "decisions": decisions})
    for tally in totals.values():
        count = tally["definitive_queries"]
        tally["verified_choice_rate"] = round(tally["verified_choice"] / count, 6) if count else None
    path = args.out / "queries.jsonl"
    with path.open("x", encoding="utf-8") as stream:
        for row in rows_out:
            stream.write(canonical(row) + "\n")
    result = {"schema": "ppt_gym.support_query.v1", "plan": model.cfg.plan,
              "depth": args.depth, "message_mode": args.message_mode,
              "source": src, "query_source": qsrc, "seed": args.seed,
              "fast_lr": args.fast_lr, "fast_scale": args.fast_scale,
              "support_split": args.support_split, "query_split": args.query_split,
              "support_limit_per_family": args.support, "query_roots": len(rows_out),
              "arms": totals, "query_sha256": sha256_file(path),
              "scientific_eligible": src["scientific_eligible"] and qsrc["scientific_eligible"],
              "metric": "offline checked candidate selection, not fresh end-to-end Lean proof search",
              "updates_from_query_labels": 0}
    save_json(args.out / "manifest.json", result)
    return result


def sweep(args):
    model, checkpoint = checkpoint_load(args.checkpoint)
    cases, source = source_cases(args, args.split)
    if checkpoint["source_manifest_sha256"] != source["manifest_sha256"]:
        raise ValueError("checkpoint and evaluation source manifest mismatch")
    modes = ("normal", "zero", "roll") if model.cfg.plan == "b" else ("normal",)
    model.eval()
    args.out.mkdir(parents=True, exist_ok=False)
    entries = []
    for mode in modes:
        for depth in args.depths:
            known = successes = censored = witness = 0
            for case in cases:
                out = predictions(model, case, depth, mode)
                idx = int(out["logits"].argmax())
                verdict = case.labels[idx]
                known += int(verdict in (0, 1))
                successes += int(verdict == 1)
                censored += int(verdict == -1)
                witness += int(1 in case.labels)
            entries.append({"plan": model.cfg.plan, "message_mode": mode, "depth": depth,
                            "cases": len(cases), "definitive": known, "certified_choice": successes,
                            "censored": censored, "witness_pool_roots": witness,
                            "certified_choice_rate": round(successes / known, 6) if known else None})
    if args.adaptive_threshold is not None:
        if not checkpoint.get("exit_fit_steps"):
            raise ValueError("adaptive exits require a checkpoint with a trained continue head")
        maximum = max(args.depths)
        for mode in modes:
            known = successes = censored = passes = 0
            for case in cases:
                out = model(case.goal, case.candidates, depth=maximum,
                            message_mode=mode, exit_threshold=args.adaptive_threshold)[-1]
                verdict = case.labels[int(out["logits"].argmax())]
                known += int(verdict in (0, 1))
                successes += int(verdict == 1)
                censored += int(verdict == -1)
                passes += int(out["pass"])
            entries.append({"plan": model.cfg.plan, "message_mode": mode,
                            "depth_policy": "trained_exit_head", "max_depth": maximum,
                            "threshold": args.adaptive_threshold, "cases": len(cases),
                            "definitive": known, "certified_choice": successes,
                            "censored": censored, "total_passes_executed": passes,
                            "mean_passes": round(passes / len(cases), 4) if cases else None,
                            "certified_choice_rate": round(successes / known, 6) if known else None,
                            "compute_caveat": "full coda is evaluated on each visited pass"})
    result = {"schema": "ppt_gym.depth_sweep.v1", "source": source, "split": args.split,
              "checkpoint_sha256": sha256_file(args.checkpoint), "results": entries,
              "metric": "offline candidate ranking against collected Lean receipts; no new Lean calls",
              "scientific_eligible": source["scientific_eligible"]}
    save_json(args.out / "manifest.json", result)
    return result


def fit_exit(args):
    """Learn a *development-only* continuation head from train checked menus.

    A pass is worth continuing when the final depth can select a certified
    tactic and the current pass cannot. This is a proxy, not a Bellman oracle.
    """
    model, checkpoint = checkpoint_load(args.checkpoint)
    cases, source = source_cases(args, "train")
    if checkpoint["source_manifest_sha256"] != source["manifest_sha256"]:
        raise ValueError("checkpoint and gate-training manifest mismatch")
    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.continue_head.parameters():
        param.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.continue_head.parameters(), lr=args.lr)
    rng = random.Random(args.seed)
    examples = [c for c in cases if all(y != -1 for y in c.labels) and 1 in c.labels]
    if not examples:
        raise ValueError("gate fitting requires certified train witnesses")
    log = []
    started = time.perf_counter()
    for step in range(args.steps):
        if time.perf_counter() - started > args.max_seconds:
            break
        case = rng.choice(examples)
        outs = model(case.goal, case.candidates, depth=args.depth, return_all=True)
        closed = [case.labels[int(o["logits"].argmax())] == 1 for o in outs]
        gain = [float((not closed[i]) and any(closed[i + 1:])) for i in range(len(outs) - 1)]
        if not gain:
            raise ValueError("depth must exceed one for gate fitting")
        targets = torch.tensor(gain, device=outs[0]["logits"].device)
        predicts = torch.stack([o["continue_logit"] for o in outs[:-1]])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(predicts, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        log.append({"step": step + 1, "loss": round(float(loss.detach()), 6)})
    args.out.mkdir(parents=True, exist_ok=False)
    path = args.out / "checkpoint.pt"
    torch.save({"**note": "exit-head fitted on train menus only", "schema": "ppt_gym.checkpoint.v1",
                "config": asdict(model.cfg), "weights": model.state_dict(),
                "source_manifest_sha256": source["manifest_sha256"],
                "source_evidence_class": source["evidence_class"],
                "train_steps": checkpoint["train_steps"],
                "exit_fit_steps": len(log), "exit_fit_depth": args.depth,
                "seed": args.seed}, path)
    result = {"schema": "ppt_gym.fit_exit.v1", "source": source,
              "checkpoint_sha256": sha256_file(path),
              "updates": len(log), "depth": args.depth,
              "target": "future gain of tactic ranker on train menus",
              "status": "calibration-only depth-proxy fit, not end-to-end adaptive-halting validation"}
    save_json(args.out / "manifest.json", result)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    fixture = sub.add_parser("fixture")
    fixture.add_argument("--out", type=Path, required=True)
    fixture.add_argument("--per-family", type=int, default=24)
    fixture.add_argument("--seed", type=int, default=20261121)

    def common(cmd):
        source = cmd.add_mutually_exclusive_group(required=True)
        source.add_argument("--pack", type=Path)
        source.add_argument("--fixture", type=Path)
        cmd.add_argument("--allow-synthetic", action="store_true")
        cmd.add_argument("--max-cases", type=int, default=256)
        cmd.add_argument("--out", type=Path, required=True)
        cmd.add_argument("--seed", type=int, default=17)

    train_p = sub.add_parser("train")
    common(train_p)
    train_p.add_argument("--plan", choices=("a", "b"), required=True)
    train_p.add_argument("--middle-order", choices=("ab", "ba"), default="ab")
    train_p.add_argument("--width", type=int, default=48)
    train_p.add_argument("--heads", type=int, default=4)
    train_p.add_argument("--max-bytes", type=int, default=160)
    train_p.add_argument("--mlp-ratio", type=int, default=2)
    train_p.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 5])
    train_p.add_argument("--steps", type=int, default=32)
    train_p.add_argument("--lr", type=float, default=0.0005)
    train_p.add_argument("--threads", type=int, default=1)
    train_p.add_argument("--max-seconds", type=float, default=120.0)

    compare_p = sub.add_parser("compare")
    common(compare_p)
    compare_p.add_argument("--checkpoint", type=Path, required=True)
    compare_p.add_argument("--support-split", default="train")
    compare_p.add_argument("--query-split", default="train")
    compare_p.add_argument("--support", type=int, default=4)
    compare_p.add_argument("--depth", type=int, default=3)
    compare_p.add_argument("--message-mode", choices=("normal", "zero", "roll"), default="normal")
    compare_p.add_argument("--fast-lr", type=float, default=1.0)
    compare_p.add_argument("--fast-scale", type=float, default=2.0)

    sweep_p = sub.add_parser("sweep")
    common(sweep_p)
    sweep_p.add_argument("--checkpoint", type=Path, required=True)
    sweep_p.add_argument("--split", default="calibration")
    sweep_p.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 5, 8, 10])
    sweep_p.add_argument("--adaptive-threshold", type=float,
                         help="Use a fitted continuation head to execute genuine early exit")

    exit_p = sub.add_parser("fit-exit")
    common(exit_p)
    exit_p.add_argument("--checkpoint", type=Path, required=True)
    exit_p.add_argument("--depth", type=int, default=5)
    exit_p.add_argument("--steps", type=int, default=64)
    exit_p.add_argument("--lr", type=float, default=0.001)
    exit_p.add_argument("--max-seconds", type=float, default=120.0)
    acc = sub.add_parser("accrete", help="Accumulate train-only DRAM weights and checked ALEPH rows")
    common(acc)
    acc.add_argument("--checkpoint", type=Path, required=True)
    acc.add_argument("--depth", type=int, default=3)
    acc.add_argument("--fast-lr", type=float, default=1.0)
    acc.add_argument("--fast-scale", type=float, default=2.0)

    replay = sub.add_parser("replay-accretion", help="Replay a versioned online learning history")
    common(replay)
    replay.add_argument("--checkpoint", type=Path, required=True)
    replay.add_argument("--session", type=Path, required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "fixture":
        result = make_fixture(args.out, per_family=args.per_family, seed=args.seed)
    else:
        if args.max_cases < 1 or args.max_cases > 20000:
            raise ValueError("max-cases must be 1..20000")
        if args.command in ("train", "fit-exit"):
            if args.steps < 1 or args.steps > 5000 or args.max_seconds <= 0:
                raise ValueError("bounded train steps and wall time required")
            if not 0 < args.lr <= 0.1:
                raise ValueError("invalid learning rate")
        if args.command == "train":
            if args.threads not in (1, 2, 4) or any(d < 1 or d > 16 for d in args.depths):
                raise ValueError("invalid thread/depth configuration")
            if args.plan == "a" and args.middle_order != "ab":
                raise ValueError("Plan A has no two-block order to permute")
            result = train(args)
        elif args.command == "compare":
            if not 1 <= args.support <= 128 or not 1 <= args.depth <= 16:
                raise ValueError("invalid support or depth budget")
            result = compare(args)
        elif args.command == "sweep":
            if any(d < 1 or d > 16 for d in args.depths):
                raise ValueError("invalid sweep depth")
            result = sweep(args)
        elif args.command == "fit-exit":
            if not 2 <= args.depth <= 16:
                raise ValueError("gate fit depth must be 2..16")
            result = fit_exit(args)
        elif args.command == "accrete":
            from .accretion import collect
            model, checkpoint = checkpoint_load(args.checkpoint)
            cases, source = source_cases(args, "train")
            if source["manifest_sha256"] != checkpoint["source_manifest_sha256"]:
                raise ValueError("source manifest differs from the trained model")
            result = collect(model, cases, source, args.checkpoint, args.out,
                             maximum=args.max_cases, depth=args.depth,
                             learning_rate=args.fast_lr, score_scale=args.fast_scale)
        elif args.command == "replay-accretion":
            from .accretion import audit
            model, checkpoint = checkpoint_load(args.checkpoint)
            cases, source = source_cases(args, "train")
            if source["manifest_sha256"] != checkpoint["source_manifest_sha256"]:
                raise ValueError("source manifest differs from the trained model")
            result = audit(model, cases, source, args.checkpoint, args.session)
            args.out.mkdir(parents=True, exist_ok=False)
            save_json(args.out / "replay.json", result)
        else:
            raise AssertionError("unknown command")
    print(canonical(result))
    return result


if __name__ == "__main__":
    main()

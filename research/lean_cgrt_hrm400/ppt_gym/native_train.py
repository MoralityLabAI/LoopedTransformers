"""LSA-02B CPU-only Plan B training on independently compiled Lean Horn proofs.

Two phases:
  collect: native Lean 4.19 shortest-proof extraction; no neural model or learning.
  fit: offline exact-reference policy supervision on train-only exported packets.

Each training frame is a *shortest-continuation* label, not a one-step closure.
It is never presented as a separately certified one-tactic solution. A native
teacher is authoritative only after full reference proof compilation; synthetic
teacher fixtures may exercise mechanics with an explicit allow-synthetic flag.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import io
import json
import math
from pathlib import Path
import random
import time

import torch

from lcgrt import config
from lcgrt.certify import certify, decl_name
from lcgrt.extract import explore, goal_key
from lcgrt.lean_repl import LeanRepl
from .core import Config, PlasticProofTransformer, supervised_loss
from .data import canonical, sha256_file
from .live_lean import candidate_menu, fingerprint, public_state, state_input
from .live_suite import load_suite
from .typed_state import encode_public_packet, schema_for_view, VIEWS


SCHEMA = "lsa02b.native_teacher.v1"
EVIDENCE = "lean_kernel_verified_reference_path"
FRAME_SCHEMA = "lsa02b.native_shortest_tactic_frame.v1"
CHECKS = 8


def _write(path: Path, data: dict) -> None:
    path.write_text(canonical(data) + "\n", encoding="utf-8")


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _reference_optimal_actions(goal: str, menu: tuple[str, ...], table: dict, sizes: dict) -> tuple[int, ...]:
    """Exact minimum-proof cost from native Lean-explored AND/OR transitions.

    0 means 'not shortest' or 'unreachable', not 'one-tactic Lean rejection'.
    """
    goal_size = sizes.get(goal)
    if goal_size is None or not isinstance(goal_size, int) or goal_size < 1:
        raise ValueError("source goal has no finite shortest proof")
    actions = table.get(goal)
    if not isinstance(actions, dict):
        raise ValueError("missing native Lean goal transition table")
    labels = []
    for tactic in menu:
        resulting = actions.get(tactic)
        if resulting is None:
            labels.append(0)
            continue
        lengths = [sizes.get(target) for target in resulting]
        if any(x is None or not isinstance(x, int) or x < 0 for x in lengths):
            labels.append(0)
        else:
            labels.append(int(1 + sum(lengths) == goal_size))
    if not any(labels):
        raise ValueError("public menu loses all native shortest continuations")
    return tuple(labels)


def _snapshot_trajectory(repl, case, oracle: dict) -> list[dict]:
    """Recreate model-visible packets from public Lean states along a checked proof."""
    proof = oracle["proof"]["script"]
    if not proof or oracle["proof"]["repl_kernel"] != {"ok": True, "axioms": []}:
        raise ValueError("Lean extraction omitted kernel-checked reference proof")
    repl.reset()
    current = repl.init(case.statement, pp=True)
    frames: list[dict] = []
    for index, selected in enumerate(proof):
        state = public_state(repl, current)
        if not state["goals"]:
            raise ValueError("shortest reference path terminated prematurely")
        front = goal_key(state["goals"][0])
        menu = candidate_menu(repl, current)
        if selected not in menu:
            raise ValueError("shortest proof contains tactic absent from public candidate menu")
        labels = _reference_optimal_actions(front, menu, oracle["table"], oracle["size"])
        if labels[menu.index(selected)] != 1:
            raise ValueError("selected native reference tactic is not shortest for current goal")
        remaining = max(1, CHECKS - index)
        packet, state_hash = state_input(repl, current, (), remaining)
        frames.append({
            "schema": FRAME_SCHEMA,
            "root": case.root,
            "lineage": case.lineage,
            "family": case.family,
            "step": index,
            "goal": packet,
            "public_state_sha256": state_hash,
            "candidates": list(menu),
            "labels": list(labels),
            "chosen_reference_tactic": selected,
            "checks_remaining": remaining,
            "reference_proof_size": len(proof),
            "label_semantics": "one_plus_Lean_minimum_successor_size_equals_Lean_goal_minimum",
            "evidence": EVIDENCE,
        })
        response = repl.run(current["sid"], selected)
        if response.get("ok") is not True:
            raise ValueError("Lean replay rejected an extracted reference tactic")
        current = response["state"]
    if current.get("goals"):
        raise ValueError("the extracted shortest proof still leaves live Lean goals")
    receipt = repl.kernel(current["sid"])
    if receipt.get("ok") is not True or receipt.get("axioms") != []:
        raise ValueError("native proof replay failed terminal Lean kernel or axiom gate")
    return frames


def collect(
    suite: Path,
    out: Path,
    *,
    roots: int = 32,
    request_timeout: int = 60,
    max_seconds: float = 480,
    cert_timeout: int = 180,
) -> dict:
    """Build an immutable, train-only corpus using the actual Lean server."""
    if out.exists() or not (1 <= roots <= 1024) or max_seconds <= 0:
        raise ValueError("fresh output, 1..1024 roots and positive wall budget required")
    if not config.REPL_EXE.is_file() or not config.LEAN_EXE.is_file():
        raise RuntimeError("pinned native Lean 4.19 lgym executable is unavailable")
    selected, source = load_suite(suite, "train", roots)
    out.mkdir(parents=True)
    begun = time.perf_counter()
    frames: list[dict] = []
    references: list[tuple] = []
    census: list[dict] = []
    with LeanRepl(
        timeout=request_timeout, exe=config.REPL_EXE,
        cwd=config.LEAN_DIR, env=config.lean_env(),
    ) as repl:
        if not str(repl.version).startswith("4.19.0"):
            raise ValueError("native teacher requires Lean 4.19.0")
        backend = fingerprint(config.REPL_EXE, config.LEAN_EXE, str(repl.version))
        for case in selected:
            if time.perf_counter() - begun >= max_seconds:
                raise TimeoutError("native source collection exceeded registered wall budget")
            oracle = explore(repl, case.theorem())
            trajectory = _snapshot_trajectory(repl, case, oracle)
            frames.extend(trajectory)
            reference_script = oracle["proof"]["script"]
            references.append((case.tid, case.theorem(), reference_script))
            census.append({
                "root": case.root, "lineage": case.lineage,
                "family": case.family, "proof_size": len(reference_script),
                "train_frames": len(trajectory),
                "within_eight_checks": len(reference_script) <= CHECKS,
                "public_menu_complete": True,
            })
    if len(census) != len(selected):
        raise RuntimeError("partial train-only corpus prohibited")
    # The independent compiler supplies a second proof authority boundary.
    receipt = certify(references, timeout=cert_timeout, keep=out / "standalone.lean")
    if receipt.get("timed_out") or any(
        receipt.get("status", {}).get(decl_name(tid)) != "passed" for tid, _, _ in references
    ):
        raise ValueError("native teacher includes a proof that failed independent compilation")
    if len({f["root"] for f in frames}) != len(census):
        raise ValueError("some root has no valid shortest-continuation frames")
    frame_file = out / "teacher.jsonl"
    with frame_file.open("x", encoding="utf-8") as writer:
        for frame in frames:
            writer.write(canonical(frame) + "\n")
    census_file = out / "train_census.jsonl"
    with census_file.open("x", encoding="utf-8") as writer:
        for root in census:
            writer.write(canonical(root) + "\n")
    _write(out / "compiler_receipt.json", receipt)
    report = {
        "schema": SCHEMA,
        "source": source,
        "split": "train",
        "evidence_class": EVIDENCE,
        "backend": backend,
        "reference_proofs_compiled": len(census),
        "independent_lineages": len({entry["lineage"] for entry in census}),
        "train_examples": len(frames),
        "roots_with_eight_check_reference": sum(c["within_eight_checks"] for c in census),
        "query_roots_exposed_to_fit": 0,
        "locked_test_opened": False,
        "source_compiler_passed": True,
        "scientific_eligible": False,
        "claim": "Lean-verified native shortest-continuation training corpus; downstream model competence untested",
        "artifacts": {
            name: sha256_file(out / name) for name in (
                "teacher.jsonl", "train_census.jsonl",
                "compiler_receipt.json", "standalone.lean",
            )
        },
    }
    _write(out / "manifest.json", report)
    return report


def _load_teacher(directory: Path, *, allow_synthetic: bool = False) -> tuple[list[dict], dict]:
    meta_path = directory / "manifest.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("schema") != SCHEMA or meta.get("split") != "train":
        raise ValueError("unsupported teacher schema or nontraining split")
    for filename, hash_ in meta.get("artifacts", {}).items():
        if filename != Path(filename).name or sha256_file(directory / filename) != hash_:
            raise ValueError("teacher dataset or compiler evidence SHA-256 changed")
    if not meta.get("artifacts") or "teacher.jsonl" not in meta["artifacts"]:
        raise ValueError("teacher corpus lacks an attested source file")
    native = (
        meta.get("evidence_class") == EVIDENCE
        and meta.get("backend", {}).get("actual_lean_process") is True
        and meta.get("source_compiler_passed") is True
        and meta.get("query_roots_exposed_to_fit") == 0
    )
    if not native and not allow_synthetic:
        raise ValueError("training needs native compiled Lean sources; synthetic input requires explicit test flag")
    rows = _records(directory / "teacher.jsonl")
    if len(rows) != meta.get("train_examples"):
        raise ValueError("teacher row count differs from the frozen manifest")
    roots = set()
    for f in rows:
        if f.get("schema") != FRAME_SCHEMA or f.get("evidence") != EVIDENCE:
            if not allow_synthetic:
                raise ValueError("teacher row has unsupported label provenance")
        if not 1 <= len(f.get("candidates", [])) <= 64:
            raise ValueError("teacher menu size outside model grammar")
        labels = f.get("labels", [])
        if len(labels) != len(f["candidates"]) or not any(x == 1 for x in labels):
            raise ValueError("teacher frame lacks shortest native tactic witness")
        if any(type(x) is not int or x not in (0, 1) for x in labels):
            raise ValueError("invalid native source-label type")
        if not (1 <= f.get("checks_remaining", 0) <= 8):
            raise ValueError("invalid training tactic-budget feature")
        roots.add(f["lineage"])
    if len(roots) != meta.get("independent_lineages"):
        raise ValueError("native source lineage census changed")
    return rows, meta


def _model_fingerprint(model) -> str:
    # Round-trip the same named state_dict without NumPy or GPU.
    bio = io.BytesIO()
    torch.save(model.state_dict(), bio)
    return hashlib.sha256(bio.getvalue()).hexdigest()


def fit(
    teacher: Path, out: Path,
    *,
    order: str,
    seed: int,
    steps: int = 320,
    depth: int = 2,
    width: int = 32,
    max_bytes: int = 384,
    learning_rate: float = 0.001,
    threads: int = 2,
    max_seconds: float = 300,
    allow_synthetic: bool = False,
    input_view: str = "raw",
    weight_mode: str = "plain",
) -> dict:
    """Train matched source/seed runs under a selected public-state view."""
    if out.exists() or order not in ("ab", "ba") or input_view not in VIEWS:
        raise ValueError("fresh fit output, AB/BA schedule and raw/typed input view required")
    if weight_mode not in ("plain", "fork4", "sham4"):
        raise ValueError("unknown training-only fork emphasis mode")
    if not (1 <= steps <= 5000 and 1 <= depth <= 16 and
            width in (16, 32, 48, 64, 96) and
            max_bytes in (64, 160, 256, 384) and
            1 <= threads <= 4 and 0 < learning_rate <= 0.01 and max_seconds > 0):
        raise ValueError("outside the registered CPU training shape or compute envelope")
    dataset, source = _load_teacher(teacher, allow_synthetic=allow_synthetic)
    by_root: dict[str, list[dict]] = defaultdict(list)
    source_bytes = []
    view_bytes = []
    encoded_collision_labels: dict[tuple, tuple[int, ...]] = {}
    contradictory_collisions = 0
    for frame in dataset:
        encoded = encode_public_packet(
            frame["goal"], frame["candidates"], view=input_view,
            max_bytes=max_bytes,
        )
        row = dict(frame)
        row["_input_view_text"] = encoded
        by_root[frame["lineage"]].append(row)
        source_bytes.append(len(frame["goal"].encode("utf-8")))
        view_bytes.append(len(encoded.encode("utf-8")))
        visible = encoded.encode("utf-8")[-max_bytes:]
        key = (visible, tuple(frame["candidates"]), frame["checks_remaining"])
        outcome = tuple(frame["labels"])
        if key in encoded_collision_labels and encoded_collision_labels[key] != outcome:
            contradictory_collisions += 1
        encoded_collision_labels[key] = outcome
    available = sorted(by_root)
    # Only the frozen native train teacher supplies fork labels. Same-seed
    # runs receive identical reference-path states and optimizer draws.
    indexed = sorted((r["lineage"], r["step"]) for r in dataset)
    if len(set(indexed)) != len(indexed):
        raise ValueError("duplicate train-lineage/step reference frame")
    fork_ids = {
        (r["lineage"], r["step"]) for r in dataset
        if len(r["labels"]) > 1 and 0 < sum(r["labels"]) < len(r["labels"])
    }
    if not fork_ids:
        raise ValueError("native train teacher lacks discriminating candidate forks")
    boosted = set()
    if weight_mode == "fork4":
        boosted = fork_ids.copy()
    elif weight_mode == "sham4":
        sham_rng = random.Random(seed ^ 0x02F42026)
        boosted = set(sham_rng.sample(indexed, len(fork_ids)))
    # The two boosted arms receive equal aggregate source loss mass.
    normalizer = (1 + 3 * len(fork_ids) / len(indexed)
                  if weight_mode != "plain" else 1.0)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    cfg = Config(plan="b", width=width, heads=4, max_bytes=max_bytes).validate()
    model = PlasticProofTransformer(cfg).cpu().float()
    initial_sha = _model_fingerprint(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    out.mkdir(parents=True)
    started = time.perf_counter()
    draws = []
    for index in range(steps):
        if time.perf_counter() - started >= max_seconds:
            break
        root = available[rng.randrange(len(available))]
        row = by_root[root][rng.randrange(len(by_root[root]))]
        result = model(
            row["_input_view_text"], row["candidates"], depth=depth,
            checks_remaining=row["checks_remaining"],
            return_all=True, middle_order=order,
        )
        key = (row["lineage"], row["step"])
        weight = 4.0 if key in boosted else 1.0
        loss = supervised_loss(result, tuple(row["labels"])) * (weight / normalizer)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite source-label loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError("nonfinite source-label gradient")
        optimizer.step()
        draws.append({
            "step": index + 1, "root": root,
            "teacher_frame_index": row["step"],
            "loss": float(loss.detach()),
            "discriminating_fork": key in fork_ids,
            "emphasis_weight": weight,
        })
    if len(draws) < 1:
        raise TimeoutError("CPU fit stopped without a single optimizer update")
    model.eval()
    target = out / "checkpoint.pt"
    # Checkpoint format is the same one used by native_order and order_probe.
    torch.save({
        "schema": "ppt_gym.checkpoint.v1",
        "config": asdict(cfg),
        "weights": model.state_dict(),
        "source_manifest_sha256": source["source"]["suite_manifest_sha256"],
        "source_evidence_class": source["evidence_class"],
        "native_teacher_sha256": sha256_file(teacher / "manifest.json"),
        "train_steps": len(draws),
        "seed": seed,
        "train_order": order,
        "weight_mode": weight_mode,
        "weight_mode_schema": "lsa02g.source_fork_weighting.v1",
        "input_view": input_view,
        "input_view_schema": schema_for_view(input_view),
    }, target)
    log = out / "optimizer_steps.jsonl"
    with log.open("x", encoding="utf-8") as writer:
        for entry in draws:
            writer.write(canonical(entry) + "\n")
    observed = [x["loss"] for x in draws]
    report = {
        "schema": "lsa02b.native_fit.v1",
        "plan": "b",
        "train_order": order,
        "weight_mode": weight_mode,
        "weight_mode_schema": "lsa02g.source_fork_weighting.v1",
        "native_train_fork_frames": len(fork_ids),
        "source_boosted_frames": len(boosted),
        "source_boosted_true_fork_overlap": len(boosted & fork_ids),
        "source_loss_mass_normalizer": normalizer,
        "optimizer_draws_on_forks": sum(r["discriminating_fork"] for r in draws),
        "optimizer_draws_on_boosted_frames": sum(r["emphasis_weight"] > 1.0 for r in draws),
        "input_view": input_view,
        "input_view_schema": schema_for_view(input_view),
        "mean_raw_public_bytes": sum(source_bytes) / len(source_bytes),
        "mean_view_public_bytes": sum(view_bytes) / len(view_bytes),
        "raw_input_over_budget": sum(b > max_bytes for b in source_bytes),
        "view_input_over_budget": sum(b > max_bytes for b in view_bytes),
        "conflicting_training_view_collisions": contradictory_collisions,
        "seed": seed,
        "model": model.architecture(),
        "train_split": "train",
        "native_teacher_manifest_sha256": sha256_file(teacher / "manifest.json"),
        "source_manifest_sha256": source["source"]["suite_manifest_sha256"],
        "training_source_native": source["backend"].get("actual_lean_process") is True,
        "train_roots": len(available),
        "teacher_frames": len(dataset),
        "optimizer_steps": len(draws),
        "requested_steps": steps,
        "depth": depth,
        "unique_parameter_count": model.parameter_count(),
        "initial_weights_sha256": initial_sha,
        "final_weights_sha256": _model_fingerprint(model),
        "initial_training_loss": observed[0],
        "last_training_loss": observed[-1],
        "mean_final_32_loss": sum(observed[-32:]) / min(32, len(observed)),
        "optimizer_calls_on_calibration_or_test": 0,
        "training_clock_seconds": round(time.perf_counter() - started, 3),
        "checkpoint_sha256": sha256_file(target),
        "optimizer_steps_sha256": sha256_file(log),
        "scientific_eligible": False,
        "claim": "matched CPU Plan B source-based training; native proof competence unverified",
    }
    _write(out / "manifest.json", report)
    return report


def cli(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    cmd = parser.add_subparsers(dest="command", required=True)
    collect_cmd = cmd.add_parser("collect")
    collect_cmd.add_argument("--suite", type=Path, required=True)
    collect_cmd.add_argument("--out", type=Path, required=True)
    collect_cmd.add_argument("--roots", type=int, default=32)
    collect_cmd.add_argument("--request-timeout", type=int, default=60)
    collect_cmd.add_argument("--max-seconds", type=float, default=480)
    collect_cmd.add_argument("--cert-timeout", type=int, default=180)
    fit_cmd = cmd.add_parser("fit")
    fit_cmd.add_argument("--teacher", type=Path, required=True)
    fit_cmd.add_argument("--out", type=Path, required=True)
    fit_cmd.add_argument("--middle-order", choices=("ab", "ba"), required=True)
    fit_cmd.add_argument("--seed", type=int, default=101)
    fit_cmd.add_argument("--steps", type=int, default=320)
    fit_cmd.add_argument("--depth", type=int, default=2)
    fit_cmd.add_argument("--width", type=int, default=32)
    fit_cmd.add_argument("--max-bytes", type=int, default=384)
    fit_cmd.add_argument("--learning-rate", type=float, default=0.001)
    fit_cmd.add_argument("--threads", type=int, default=2)
    fit_cmd.add_argument("--max-seconds", type=float, default=300)
    fit_cmd.add_argument("--allow-synthetic", action="store_true")
    fit_cmd.add_argument("--input-view", choices=VIEWS, default="raw")
    fit_cmd.add_argument("--weight-mode", choices=("plain", "fork4", "sham4"), default="plain")
    args = parser.parse_args(argv)
    if args.command == "collect":
        result = collect(
            args.suite, args.out, roots=args.roots,
            request_timeout=args.request_timeout,
            max_seconds=args.max_seconds, cert_timeout=args.cert_timeout,
        )
    else:
        result = fit(
            args.teacher, args.out, order=args.middle_order, seed=args.seed,
            steps=args.steps, depth=args.depth, width=args.width,
            max_bytes=args.max_bytes, learning_rate=args.learning_rate,
            threads=args.threads, max_seconds=args.max_seconds,
            allow_synthetic=args.allow_synthetic,
            input_view=args.input_view,
            weight_mode=args.weight_mode,
        )
    print(canonical(result))
    return result


if __name__ == "__main__":
    cli()

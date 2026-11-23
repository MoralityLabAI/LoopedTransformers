"""LSA-02J: native Lean training-only tactic viability and recovery-cost labels.

Each row is an existing public reference-trajectory state from a frozen
Lean-certified TRAIN theorem. We independently explore that theorem's Lean
action graph, then annotate each *public candidate* with its exact minimal
successor proof cost (or null for a rejected/unprovable successor).

A candidate's finite cost includes all still-pending other goals. The
minimum effort requirement is the native tactic count, not an estimate of
the student's actions or a promise of remaining 16-unit search feasibility.
No calibration or locked theorem labels are read or written.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

from lcgrt import config
from lcgrt.extract import explore, goal_key
from lcgrt.lean_repl import LeanRepl

from .data import canonical, sha256_file
from .live_suite import load_suite
from .native_train import _load_teacher
from .live_lean import fingerprint

SCHEMA = "lsa02j.lean_train_branch_utility.v1"
MAX_PROOF_COST = 16


def _write(path: Path, obj: dict) -> None:
    path.write_text(canonical(obj) + "\n", encoding="utf-8")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def action_cost(oracle: dict, state: dict, action: str) -> int | None:
    """Minimal whole-state *tactic* cost after applying a public action.

    The oracle is permitted only inside TRAIN source generation and in
    post-scoring diagnostic jobs. It never enters the live model selector.
    """
    goals = state.get("goals")
    if not isinstance(goals, list) or not goals:
        raise ValueError("source state has no visible goals")
    root = goal_key(goals[0])
    successors = oracle["table"].get(root, {}).get(action)
    if successors is None:
        return None
    remainder = [goal_key(g) for g in goals[1:]]
    lengths = [oracle["size"].get(key) for key in [*successors, *remainder]]
    if any(not isinstance(v, int) or v < 1 for v in lengths):
        return None
    return 1 + sum(lengths)


def label_frame(frame: dict, oracle: dict) -> dict:
    packet = json.loads(frame["goal"])
    if set(packet) != {"state", "failed_choices", "checks_remaining"}:
        raise ValueError("native source packet contains nonpublic model fields")
    state = packet["state"]
    candidates = frame["candidates"]
    costs = [action_cost(oracle, state, candidate) for candidate in candidates]
    finite = [int(x is not None) for x in costs]
    bounded = [int(x is not None and x <= MAX_PROOF_COST) for x in costs]
    if not any(finite) or not any(bounded):
        raise ValueError("compiled reference frame has no bounded viable candidate")
    if len(costs) != len(frame["labels"]):
        raise ValueError("native utility menu and shortest-path labels disagree")
    if any(frame["labels"][i] and (
        costs[i] is None or costs[i] != min(x for x in costs if x is not None)
    ) for i in range(len(costs))):
        raise ValueError("native shortest-continuation labels contradict finite utility costs")
    return {
        "schema": SCHEMA,
        "root": frame["root"], "lineage": frame["lineage"],
        "step": frame["step"], "goal": frame["goal"],
        "public_state_sha256": frame["public_state_sha256"],
        "candidates": candidates,
        "checks_remaining": frame["checks_remaining"],
        "finite_viability": finite,
        "bounded_viability": bounded,
        "minimum_whole_state_tactic_cost": costs,
        "has_mixed_finite_actions": 0 < sum(finite) < len(finite),
        "has_mixed_bounded_actions": 0 < sum(bounded) < len(bounded),
        "source_only": True,
        "labels": frame["labels"],
    }


def collect(suite: Path, teacher: Path, out: Path, *,
            roots: int = 64, max_seconds: float = 900, request_timeout: int = 60) -> dict:
    if out.exists() or not 32 <= roots <= 1024 or max_seconds <= 0:
        raise ValueError("fresh training-only branch utility output required")
    frames, metadata = _load_teacher(teacher)
    cases, suite_meta = load_suite(suite, "train", roots)
    if metadata["independent_lineages"] != roots or (
        metadata["source"]["suite_manifest_sha256"] != suite_meta["suite_manifest_sha256"]
    ):
        raise ValueError("native training teacher differs from registered suite")
    by_root = {}
    for f in frames:
        by_root.setdefault(f["root"], []).append(f)
    if len(by_root) != roots or {c.root for c in cases} != set(by_root):
        raise ValueError("training root census mismatches native teacher")
    t0 = time.perf_counter()
    rows = []
    with LeanRepl(
        timeout=request_timeout, exe=config.REPL_EXE,
        cwd=config.LEAN_DIR, env=config.lean_env(),
    ) as repl:
        if str(repl.version) != "4.19.0":
            raise ValueError("expected native Lean 4.19.0")
        backend = fingerprint(config.REPL_EXE, config.LEAN_EXE, str(repl.version))
        for case in cases:
            if time.perf_counter() - t0 > max_seconds:
                raise TimeoutError("native train-only branch viability collection exceeded cap")
            oracle = explore(repl, case.theorem())
            source_rows = by_root[case.root]
            if not oracle["proof"]["repl_kernel"]["ok"]:
                raise ValueError("native shortest proof lacks Lean kernel authority")
            for frame in source_rows:
                if frame["lineage"] != case.lineage:
                    raise ValueError("source lineage crossed frozen theorem")
                rows.append(label_frame(frame, oracle))
    mix = sum(row["has_mixed_finite_actions"] for row in rows)
    bounded_mix = sum(row["has_mixed_bounded_actions"] for row in rows)
    if mix < 16 or len({r["root"] for r in rows if r["has_mixed_finite_actions"]}) < 8:
        raise ValueError("insufficient training-only viable/dead branches for a ranker")
    if len(rows) != len(frames):
        raise ValueError("missing frozen native teacher frames")
    out.mkdir(parents=True)
    corpus = out / "branch_labels.jsonl"
    with corpus.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(canonical(row) + "\n")
    manifest = {
        "schema": SCHEMA,
        "teacher_sha256": sha256_file(teacher / "manifest.json"),
        "suite_manifest_sha256": suite_meta["suite_manifest_sha256"],
        "split": "train",
        "backend": backend,
        "training_lineages": roots,
        "public_frames": len(rows),
        "candidate_actions": sum(len(r["candidates"]) for r in rows),
        "finite_candidate_actions": sum(sum(r["finite_viability"]) for r in rows),
        "bounded_candidate_actions": sum(sum(r["bounded_viability"]) for r in rows),
        "dead_or_rejected_candidate_actions": sum(len(r["candidates"]) - sum(r["finite_viability"]) for r in rows),
        "mixed_finite_action_frames": mix,
        "mixed_finite_action_roots": len({r["root"] for r in rows if r["has_mixed_finite_actions"]}),
        "mixed_bounded_action_frames": bounded_mix,
        "minimum_cost_definition": "1 + Lean shortest lengths for successor goals and pending goals; None for rejected or unprovable",
        "source_reference_independently_compiled": True,
        "calibration_model_updates": 0,
        "no_ordinary_or_ood_opened": True,
        "model_calls": 0,
        "scientific_eligible": False,
        "artifacts": {"branch_labels.jsonl": sha256_file(corpus)},
    }
    _write(out / "manifest.json", manifest)
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--roots", type=int, default=64)
    p.add_argument("--max-seconds", type=float, default=900)
    p.add_argument("--request-timeout", type=int, default=60)
    a = p.parse_args(argv)
    report = collect(a.suite, a.teacher, a.out,
                     roots=a.roots, max_seconds=a.max_seconds,
                     request_timeout=a.request_timeout)
    print(canonical({
        "source": report["teacher_sha256"],
        "roots": report["training_lineages"],
        "frames": report["public_frames"],
        "candidates": report["candidate_actions"],
        "finite": report["finite_candidate_actions"],
        "dead": report["dead_or_rejected_candidate_actions"],
        "mixed": report["mixed_finite_action_frames"],
    }))
    return report


if __name__ == "__main__":
    main()

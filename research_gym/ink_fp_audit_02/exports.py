"""DataGoblin / Aleph corpus exports from verified rollouts.

Row shape follows DataGoblin rlt_gym (corpus.py / inkling.py _aleph_rows):
  {id, kind, root, family, split, task_digest, input{task, history}, target, loss, source_model,
   provider_execution_attested, verification}
Streams: sft (verified solutions), repairs (failed attempt -> verified solution on the same
problem), outcomes (every checked attempt with the verifier verdict), routing (cheapest effort
that verified), research (show-your-work explanations). Only the train partition is exported.
Model-written explanations live under target.model_explanation with model_generated=True and
are never merged into the verifier certificate.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from .ledger import file_sha256, iter_jsonl

SCHEMA = "rlt_gym.aleph_corpus.v1"
EFFORT_RANK = {"low": 0, "medium": 1, "high": 2, "native_default": 1}
SHOW_WORK_FIELDS = ("objective", "invariants", "method", "why", "failure_diagnosis",
                    "verification", "cost", "alternatives")
SHOW_WORK_TEMPLATE = ("Before the final line, write these labelled lines: " +
                      "; ".join(f"{f.upper()}:" for f in SHOW_WORK_FIELDS) + ".")


def sha256_json(x) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def explanation(text: str) -> dict:
    """Parse labelled show-your-work lines; returns only fields actually present."""
    out = {}
    for f in SHOW_WORK_FIELDS:
        m = re.search(rf"(?im)^{f.upper()}:\s*(.+)$", text)
        if m:
            out[f] = m.group(1).strip()
    return out


def _groups(path: Path):
    """Consecutive records sharing (generation, task_id) form one sampled group."""
    cur, key = [], None
    for rec in iter_jsonl(path):
        k = (rec["generation"], rec["task"]["task_id"])
        if key is not None and k != key:
            yield cur
            cur = []
        key = k
        cur.append(rec)
    if cur:
        yield cur


def _row(kind, rec, context, target, verification):
    t = rec["task"]
    root = t["source_world"]
    return {"id": sha256_json([kind, root, t["task_id"], context, target]), "kind": kind, "root": root,
            "family": t["family"], "split": "train", "task_digest": t["prompt_sha256"],
            "semantic_task_id": t["task_id"], "input": context, "target": target,
            "loss": {"boundary": "response_only", "field": "target", "eligible": kind != "outcomes"},
            "source_model": {"provider": "tinker" if not rec.get("simulated") else "simulation",
                             "model": rec.get("model", "thinkingmachines/Inkling-2-Mini:peft:65536:free"),
                             "policy_checkpoint": rec["policy_checkpoint"], "effort": rec["effort"]},
            "provider_execution_attested": False, "verification": verification,
            "token_cost": {"completion_tokens": rec.get("completion_tokens", 0)},
            "difficulty": t["difficulty"], "generator": t["generator"],
            "generator_version": t["generator_version"]}


def build(rollout_paths: list[Path], out_dir: Path, prompts: dict[str, str] | None = None) -> dict:
    """prompts: optional task_id -> prompt text (rollouts store only the prompt digest)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    streams = {k: out_dir / f"{k}.jsonl" for k in ("sft", "repairs", "outcomes", "routing", "research")}
    for p in streams.values():
        p.write_text("", encoding="utf-8")
    seen, counts, roots, fams = set(), Counter(), {k: set() for k in streams}, {k: Counter() for k in streams}
    sem_seen: dict[str, set] = {k: set() for k in ("sft", "routing")}

    def emit(kind, row):
        if row["id"] in seen:
            return
        seen.add(row["id"])
        with streams[kind].open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        counts[kind] += 1
        roots[kind].add(row["root"])
        fams[kind][row["family"]] += 1

    sources = []
    for path in rollout_paths:
        sources.append({"path": Path(path).name, "sha256": file_sha256(path)})
        for group in _groups(Path(path)):
            t = group[0]["task"]
            if t["partition"] != "train":
                continue
            task_ctx = {"task_id": t["task_id"], "prompt": (prompts or {}).get(t["task_id"]),
                        "prompt_sha256": t["prompt_sha256"]}
            failures = []
            for rec in group:
                ver = {"verifier": t["verifier"], "passed": bool(rec["correct"]), "completed": rec["completed"],
                       "repetition": rec["repetition"], "expected_answer": t["answer"],
                       "certificate": t.get("certificate", {}), "evidence_class": "exact_verifier"}
                cand = {"candidate": rec["text"]}
                emit("outcomes", _row("outcomes", rec, {"task": task_ctx, "history": [], **cand},
                                      {"passed": ver["passed"]}, ver))
                if rec["correct"]:
                    # Deduplicate verified demonstrations by semantic task identity and root.
                    key = (t["task_id"], t["source_world"])
                    if key not in sem_seen["sft"]:
                        sem_seen["sft"].add(key)
                        emit("sft", _row("sft", rec, {"task": task_ctx, "history": []}, cand, ver))
                    if failures:
                        hist = [{"candidate": f["text"], "verification": {"passed": False,
                                 "completed": f["completed"], "repetition": f["repetition"]}} for f in failures[:2]]
                        emit("repairs", _row("repairs", rec, {"task": task_ctx, "history": hist}, cand, ver))
                    expl = explanation(rec["text"])
                    if expl:
                        emit("research", _row("research", rec, {"task": task_ctx, "exercise": "show_your_work"},
                                              {"model_explanation": expl, "model_generated": True}, ver))
                else:
                    failures.append(rec)
            solved = [r for r in group if r["correct"]]
            if solved and t["task_id"] not in sem_seen["routing"]:
                sem_seen["routing"].add(t["task_id"])
                cheapest = min(solved, key=lambda r: (EFFORT_RANK.get(r["effort"], 1), r.get("completion_tokens", 0)))
                by_effort = Counter((r["effort"], bool(r["correct"])) for r in group)
                emit("routing", _row("routing", cheapest, {"task": task_ctx, "family": t["family"],
                                     "difficulty": t["difficulty"]},
                                     {"route_effort": cheapest["effort"],
                                      "observed": {f"{e}:{'pass' if ok else 'fail'}": n for (e, ok), n in sorted(by_effort.items())}},
                                     {"verifier": t["verifier"], "passed": True, "evidence_class": "exact_verifier"}))
    manifest = {"schema_version": SCHEMA, "split": "train",
                "streams": {k: {"sha256": file_sha256(p), "unique_examples": counts[k], "roots": len(roots[k]),
                                "families": dict(fams[k])} for k, p in streams.items()},
                "sources": sources, "provider_execution_attested": False,
                "explanations_are_model_generated": True,
                "sampling_plan": {"max_presentations_per_root_across_streams": 4,
                                  "accounting_lane_by_family": {"experiment_design": "research_mechinterp",
                                                               "bpd_nullspace": "research_mechinterp",
                                                               "default": "games_procedural"}}}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest

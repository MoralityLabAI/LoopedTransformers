"""Frozen-base Inky evaluation on curriculum tasks (Stage C base arm), free tier only.

Reads task receipts (JSONL with prompts), sends each (task, effort) once through the audit-01
OpenAI-compatible runner, verifies with the task's registered verifier, and streams one record
per request. Resumable: (task_id, effort) pairs already recorded are skipped. Concurrency 1,
fixed delay, no retries; the first HTTP or transport error writes failure.json and stops.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from . import resources as res
from . import gyms  # noqa: F401  registers generators and the storyworld verifier
from .curriculum import Task, verify
from .ledger import append_jsonl, iter_jsonl
from .tinker_backend import OAISampler, Usage

EFFORTS = ("low", "medium", "high")


def load_tasks(paths: list[Path]) -> list[Task]:
    fields = Task.__dataclass_fields__
    out = []
    for p in paths:
        for rec in iter_jsonl(p):
            out.append(Task(**{k: v for k, v in rec.items() if k in fields}))
    return out


def run(tasks: list[Task], out: Path, efforts=EFFORTS, max_tokens: int = 512, delay: float = 1.0,
        max_requests: int | None = None, transport=None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    recs = out / "records.jsonl"
    done = {(r["task_id"], r["effort"]) for r in iter_jsonl(recs)} if recs.exists() else set()
    key = os.environ.get("TINKER_API_KEY")
    if not key:
        raise RuntimeError("TINKER_API_KEY not present in environment")
    usage = Usage()
    sampler = OAISampler(usage, transport=transport)
    monitor = res.ResourceMonitor()
    sent = 0
    for t in tasks:
        for e in efforts:
            if (t.task_id, e) in done:
                continue
            if max_requests is not None and sent >= max_requests:
                return {"sent": sent, "stopped": "max_requests"}
            monitor.gate()
            try:
                s = sampler.sample(t.prompt, e, max_tokens, key)
            except Exception as exc:
                (out / "failure.json").write_text(json.dumps({
                    "task_id": t.task_id, "effort": e, "sent_before": sent, "error_type": type(exc).__name__,
                    "http_status": getattr(exc, "code", None), "automatic_retries": 0, "paid_fallback": False,
                    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2) + "\n", encoding="utf-8")
                return {"sent": sent, "stopped": f"{type(exc).__name__} {getattr(exc, 'code', '')}".strip()}
            sent += 1
            v = verify(t, s["text"])
            append_jsonl(recs, {"task_id": t.task_id, "family": t.family, "partition": t.partition,
                                "source_world": t.source_world, "effort": e, "expected": t.answer,
                                **v, "finish_reason": s["finish_reason"], "usage": s["usage"],
                                "response_model": s["response_model"], "seconds": s["seconds"],
                                "final_text": s["text"][-2000:]})
            if delay:
                time.sleep(delay)
    return {"sent": sent, "stopped": "complete", "resources": monitor.receipt()}


def summarize(recs: Path) -> dict:
    by: dict = {}
    for r in iter_jsonl(recs):
        k = (r["family"], r["effort"])
        a = by.setdefault(k, {"n": 0, "correct": 0, "completed": 0, "repetition": 0, "completion_tokens": 0})
        a["n"] += 1
        a["correct"] += r["correct"]
        a["completed"] += r["completed"]
        a["repetition"] += r["repetition"]
        a["completion_tokens"] += int((r.get("usage") or {}).get("completion_tokens") or 0)
    fams = sorted({f for f, _ in by})
    table = {f: {e: by.get((f, e)) for e in EFFORTS} for f in fams}
    # rescue: tasks wrong at low but right at high (paired by task)
    paired: dict = {}
    for r in iter_jsonl(recs):
        paired.setdefault(r["task_id"], {})[r["effort"]] = r["correct"]
    full = [v for v in paired.values() if all(e in v for e in EFFORTS)]
    return {"by_family_effort": table, "paired_tasks": len(full),
            "rescued_high_over_low": sum(1 for v in full if v["high"] and not v["low"]),
            "lost_high_vs_low": sum(1 for v in full if v["low"] and not v["high"]),
            "totals": {e: {"correct": sum(1 for v in full if v[e]), "n": len(full)} for e in EFFORTS}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--efforts", nargs="+", default=list(EFFORTS), choices=EFFORTS)
    ap.add_argument("--max-requests", type=int)
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    a = ap.parse_args()
    res.set_below_normal_priority()
    r = run(load_tasks(a.tasks), a.out, tuple(a.efforts), max_tokens=a.max_tokens, delay=a.delay,
            max_requests=a.max_requests)
    s = summarize(a.out / "records.jsonl")
    (a.out / "summary.json").write_text(json.dumps({"run": r, **s}, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"run": {k: v for k, v in r.items() if k != "resources"}, "totals": s["totals"],
                      "rescued": s["rescued_high_over_low"], "lost": s["lost_high_vs_low"]}))


if __name__ == "__main__":
    main()

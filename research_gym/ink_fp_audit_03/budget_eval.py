"""INK-BUDGET-01 data collection: one long-cap call per (task, effort) on the OpenAI-compatible route.

At temperature 0 decoding is deterministic, so the outcome at any smaller completion cap c is
implied by the long run: correct@c iff the long run finished within c tokens and was correct;
otherwise the cap-c run truncates. `prefix_check` spends a few extra calls to test that claim
on the live endpoint instead of assuming it. Policy evaluation (budget_policies.py) then runs
offline on these receipts.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from research_gym.ink_fp_audit_02 import inky_eval
from research_gym.ink_fp_audit_02.ledger import append_jsonl, iter_jsonl
from research_gym.ink_fp_audit_02.tinker_backend import OAISampler, Usage
from . import frontier  # noqa: F401  registers frontier verifiers
from research_gym.ink_fp_audit_02.curriculum import verify


def prefix_check(tasks_path: Path, records: Path, out: Path, caps=(512, 1024), max_calls: int = 8) -> dict:
    """Re-query selected (task, effort) pairs at smaller caps; compare with the long-cap implication."""
    key = os.environ["TINKER_API_KEY"]
    tasks = {t.task_id: t for t in inky_eval.load_tasks([tasks_path])}
    long = [r for r in iter_jsonl(records)]
    picks, seen = [], set()
    for cap in caps:  # one finished-within-cap and one exceeding-cap example per cap and effort
        for want_fit in (True, False):
            for r in long:
                ct = int((r.get("usage") or {}).get("completion_tokens") or 0)
                fits = r["finish_reason"] == "stop" and ct <= cap
                k = (r["task_id"], r["effort"], cap)
                if fits == want_fit and k not in seen and (ct > cap or fits):
                    picks.append((r, cap))
                    seen.add(k)
                    break
    picks = picks[:max_calls]
    sampler = OAISampler(Usage())
    rows = []
    for r, cap in picks:
        s = sampler.sample(tasks[r["task_id"]].prompt, r["effort"], cap, key)
        v = verify(tasks[r["task_id"]], s["text"])
        ct_long = int((r.get("usage") or {}).get("completion_tokens") or 0)
        implied_trunc = not (r["finish_reason"] == "stop" and ct_long <= cap)
        row = {"task_id": r["task_id"], "effort": r["effort"], "cap": cap, "long_completion_tokens": ct_long,
               "implied_truncated": implied_trunc, "observed_finish": s["finish_reason"],
               "observed_truncated": s["finish_reason"] == "length",
               "implied_correct": (not implied_trunc) and bool(r["correct"]), "observed_correct": v["correct"],
               "text_equal_when_finished": (not implied_trunc) and s["text"] == r["final_text"][-2000:][-len(s["text"]):]
               if s["text"] else None,
               "observed_completion_tokens": s["completion_tokens"]}
        row["agree"] = row["implied_truncated"] == row["observed_truncated"] and row["implied_correct"] == row["observed_correct"]
        rows.append(row)
        append_jsonl(out, row)
    return {"calls": len(rows), "agree": sum(r["agree"] for r in rows)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--prefix-check", action="store_true")
    ap.add_argument("--efforts", nargs="+", default=list(inky_eval.EFFORTS), choices=inky_eval.EFFORTS)
    a = ap.parse_args()
    if a.prefix_check:
        print(json.dumps(prefix_check(a.tasks, a.out / "records.jsonl", a.out / "prefix_check.jsonl")))
        return
    # 8192-token generations can exceed the audit-02 default client timeout of 120 s.
    r = inky_eval.run(inky_eval.load_tasks([a.tasks]), a.out, tuple(a.efforts), max_tokens=a.max_tokens, delay=1.0,
                      timeout=600)
    s = inky_eval.summarize(a.out / "records.jsonl")
    (a.out / "summary.json").write_text(json.dumps({"run": r, **s}, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"run": {k: v for k, v in r.items() if k != "resources"}, "totals": s["totals"]}))


if __name__ == "__main__":
    main()

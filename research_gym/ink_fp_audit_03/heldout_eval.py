"""Evaluate registered adapter heads and the base on fresh held-out tasks (Linux venv).

    python heldout_eval.py --dir receipts/adapter_heldout_20261128
Reads registration.json (committed before sampling), runs greedy decoding at the registered effort
and cap for each checkpoint, and writes one JSONL per checkpoint plus score.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adapter_pilot as ap  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", type=Path, required=True)
    a = p.parse_args()
    reg = json.loads((a.dir / "registration.json").read_text(encoding="utf-8"))
    tasks = ap.load_tasks(a.dir / reg["tasks_file"])
    nat = ap.Native()
    monitor = ap.res.ResourceMonitor(reader=ap.host_free_bytes, poll_seconds=30)
    score = {}
    for name, ck in reg["checkpoints"].items():
        monitor.gate()
        path = None if ck is None else ck["sampler_path"]
        cache = a.dir / f"eval_{name}.jsonl"
        cases = ap.evaluate(nat, path, tasks, cache)
        toks = sum(r["completion_tokens"] for r in ap.iter_jsonl(cache))
        ok = sum(c["correct"] for c in cases)
        score[name] = {"correct": ok, "n": len(cases), "completion_tokens": toks,
                       "success_per_1k_tokens": 1000 * ok / toks if toks else None,
                       "repetition": sum(c["repetition"] for c in cases), "completed": sum(c["completed"] for c in cases)}
    paired = {}
    for name in score:
        for r in ap.iter_jsonl(a.dir / f"eval_{name}.jsonl"):
            paired.setdefault(r["task_id"], {"family": r["family"]})[name] = int(r["correct"])
    out = {"registration_sha256_of_file": __import__("hashlib").sha256((a.dir / "registration.json").read_bytes()).hexdigest(),
           "score": score, "usage": {"requests": nat.requests, "sample_tokens": nat.sample_tokens},
           "paired_correct": paired}
    (a.dir / "score.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(score))


if __name__ == "__main__":
    main()

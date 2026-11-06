"""Lean-certified one-step completion on held-out Mathlib states, at a fixed budget of Lean calls.

    python -m lcgrt.mathlib_eval --data mathlib-snap1 --ckpt RUN_A RUN_B ... --out EVAL_DIR

States: terminal steps (the source tactic closed the main goal) of certified declarations,
sampled from each held-out split with a fixed seed, at most `--per-file` per file, so every
model is scored on the same states. For each state, every tactic any policy would try is run
by Lean at that exact source position (`lgym_trace` select mode, the environment as it stood
there, the theorem being proved not yet in scope). A tactic *completes* the state when Lean
accepts it, no goals remain, the closed proof passes the kernel as a standalone theorem over
the goal's context, and its axioms are within {propext, Classical.choice, Quot.sound}.

Policies, each an ordered list of tactic texts; close@k = some tactic among the first k completes:
  auto        the 12 automation candidates, ordered by how often each closed a training state
  <run>       the model's beam (k = 8), forbidden texts removed, duplicates merged
  <run>+auto  model and automation interleaved (m1, a1, m2, a2, ...)
  human       the source tactic itself (whitespace-normalised): a check of the pipeline, not a policy
Each file is elaborated once for all policies and models.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import config
from .mathlib import CANDIDATES, STANDARD_AXIOMS, _lower_priority, select_one
from .tactic_lm import goal_bytes, load_model, make_batch, norm_tactic, usable, CTX

FORBIDDEN = re.compile(r"sorry|admit|native_decide|run_tac|run_cmd|run_meta|by_elab|set_option|#|unsafe|"
                       r"implemented_by|extern|IO\.|ofReduceBool|trustCompiler")
KS = (1, 2, 4, 8)


def sample_states(data, split, n, per_file, seed):
    by_file = collections.defaultdict(dict)
    with gzip.open(config.DATA_ROOT / data / f"{split}.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            term = r.get("terminal", len(r["goals_after"]) < len(r["goals_before"]))   # v1 traces lack the field
            if term and usable(r):
                by_file[r["file"]].setdefault((r["line"], r["col"]), r)
    files = sorted(by_file, key=lambda p: hashlib.sha1(f"{seed}:{p}".encode()).hexdigest())
    out = []
    for p in files:
        rows = [by_file[p][k] for k in sorted(by_file[p])]
        rng = np.random.default_rng(int(hashlib.sha1(f"{seed}:{p}".encode()).hexdigest()[:8], 16))
        pick = sorted(rng.choice(len(rows), min(per_file, len(rows)), replace=False))
        out += [rows[i] | {"split": split} for i in pick]
        if len(out) >= n:
            return out[:n]
    return out


def auto_order(data):
    """Automation candidates by closing frequency on training states (ties: listed order)."""
    closes = collections.Counter()
    with gzip.open(config.DATA_ROOT / data / "train.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            for c in json.loads(line)["candidates"]:
                closes[c["tactic"]] += bool(c["ok"] and c.get("goals_after") == 0)
    return sorted(CANDIDATES, key=lambda t: (-closes[t], CANDIDATES.index(t))), dict(closes)


def completes(res):
    if not res or not res.get("ok") or res.get("goals_after") != 0:
        return False
    cert = res.get("cert") or {}
    return bool(cert.get("kernel")) and set(cert.get("axioms", [])) <= STANDARD_AXIOMS


def proposals(model, states, device, beam, fast_steps=None, bs=32):
    out = []
    for i in range(0, len(states), bs):
        chunk = states[i:i + bs]
        enc = np.zeros((len(chunk), CTX), np.uint8); el = np.zeros(len(chunk), np.int16)
        for j, r in enumerate(chunk):
            g = goal_bytes(r); enc[j, :len(g)] = np.frombuffer(g, np.uint8); el[j] = len(g)
        b = make_batch(enc, el, device)
        for row in model.beam(b["enc"], b["enc_mask"], k=beam, fast_steps=fast_steps):
            seen, keep = set(), []
            for text, lp in row:
                t = norm_tactic(text)
                if t and t not in seen and not FORBIDDEN.search(t):
                    seen.add(t); keep.append(t)
            out.append(keep)
    return out


def interleave(a, b):
    out = []
    for i in range(max(len(a), len(b))):
        out += [x for x in (a[i:i + 1] + b[i:i + 1]) if x not in out]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--splits", nargs="+", default=["test_file", "test_long", "test_family"])
    ap.add_argument("--per-split", type=int, default=200)
    ap.add_argument("--per-file", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--fast-steps", type=int, default=None)
    ap.add_argument("--heartbeats", type=int, default=5000, help="per tactic, in thousands")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    order, closes = auto_order(a.data)
    states = [s for sp in a.splits for s in sample_states(a.data, sp, a.per_split, a.per_file, a.seed)]
    print(json.dumps({"event": "states", "n": len(states), "auto_order": order}), flush=True)
    runs = {}
    for ck in a.ckpt:
        model, meta = load_model(ck, a.device)
        runs[Path(ck).name] = proposals(model, states, a.device, a.beam, a.fast_steps)
        print(json.dumps({"event": "proposals", "run": Path(ck).name, "s": round(time.time() - t0)}), flush=True)
    # One select pass per file: the union of every tactic any policy tries on each state.
    plan = collections.defaultdict(list)
    for i, s in enumerate(states):
        tacs = list(dict.fromkeys(order + [norm_tactic(s["tactic"])] + [t for p in runs.values() for t in p[i]]))
        plan[s["file"]].append((i, {"line": s["line"], "col": s["col"], "tactics": tacs}))
    tasks = [(f, [sel for _, sel in v], str(out / "lean" / (f.replace("/", "__") + ".jsonl")), a.heartbeats)
             for f, v in sorted(plan.items())]
    (out / "lean").mkdir(exist_ok=True)
    file_log = []
    with mp.get_context("spawn").Pool(a.workers, initializer=_lower_priority) as pool:
        for r in pool.imap_unordered(select_one, tasks):
            file_log.append(r)
            print(json.dumps({k: r.get(k) for k in ("file", "selected", "seconds", "failure")}), flush=True)
    results = {}
    for f, sels, path, _ in tasks:
        if Path(path).exists():
            for line in open(path, encoding="utf-8"):
                row = json.loads(line)
                results[(f, row["line"], row["col"])] = {x["tactic"]: x for x in row["results"]}
    rows, summary = [], collections.defaultdict(dict)
    for i, s in enumerate(states):
        res = results.get((s["file"], s["line"], s["col"]))
        pol = {"auto": order, "human": [norm_tactic(s["tactic"])]}
        for name, p in runs.items():
            pol[name] = p[i]; pol[name + "+auto"] = interleave(p[i], order)
        rows.append({"file": s["file"], "line": s["line"], "col": s["col"], "split": s["split"],
                     "decl": s["decl"], "reached": res is not None,
                     "policies": {k: v[:max(KS)] for k, v in pol.items()},
                     "results": res})
    for sp in a.splits + ["all"]:
        sub = [r for r in rows if sp == "all" or r["split"] == sp]
        reached = [r for r in sub if r["reached"]]
        summary[sp]["states"] = len(sub); summary[sp]["reached"] = len(reached)
        for name in rows[0]["policies"] if rows else []:
            ks = {}
            for k in KS:
                ks[f"close@{k}"] = round(sum(any(completes(r["results"].get(t)) for t in r["policies"][name][:k])
                                             for r in reached) / max(1, len(reached)), 4)
            ks["valid@1"] = round(sum(bool(r["policies"][name]) and bool((r["results"].get(r["policies"][name][0]) or {}).get("ok"))
                                      for r in reached) / max(1, len(reached)), 4)
            summary[sp][name] = ks
    with gzip.open(out / "states.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = {"data": a.data, "ckpts": a.ckpt, "splits": a.splits, "per_split": a.per_split, "per_file": a.per_file,
              "seed": a.seed, "beam": a.beam, "fast_steps": a.fast_steps, "heartbeats_k": a.heartbeats,
              "auto_order": order, "auto_train_closes": closes, "summary": summary,
              "files": len(tasks), "failed_files": [r for r in file_log if not r.get("ok", True)],
              "seconds": round(time.time() - t0, 1)}
    (out / "eval.json").write_text(json.dumps(report, indent=1) + "\n")
    (out / "complete.json").write_text(json.dumps({"event": "complete", "seconds": report["seconds"]}) + "\n")
    print(json.dumps({"summary": summary}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

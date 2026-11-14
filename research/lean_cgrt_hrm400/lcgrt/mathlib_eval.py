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
  human       the source tactic itself, as written (`human_text`): a check of the pipeline, not a policy
Scored states are the reached ones whose main goal is certifiable (a proposition with a closed
statement); for the others no tactic can pass the kernel check, so they are counted, not scored.
Each file is elaborated once for all policies and models.
"""
from __future__ import annotations

import argparse
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
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

from . import config, hw
from .mathlib import CANDIDATES, STANDARD_AXIOMS, _lower_priority, mathlib_commit, select_one
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
    guard = hw.Guard(hw.Limits(use_gpu=True)) if device.startswith("cuda") else None
    for i in range(0, len(states), bs):
        chunk = states[i:i + bs]
        enc = np.zeros((len(chunk), CTX), np.uint8); el = np.zeros(len(chunk), np.int16)
        for j, r in enumerate(chunk):
            g = goal_bytes(r); enc[j, :len(g)] = np.frombuffer(g, np.uint8); el[j] = len(g)
        b = make_batch(enc, el, device)
        extra = {"guard": guard} if guard is not None else {}
        for row in model.beam(b["enc"], b["enc_mask"], k=beam, fast_steps=fast_steps, **extra):
            seen, keep = set(), []
            for text, lp in row:
                t = norm_tactic(text)
                if t and t not in seen and not FORBIDDEN.search(t):
                    seen.add(t); keep.append(t)
            out.append(keep)
    return out


def states_sha256(states):
    """Identity of the sampled states: position and the exact goal bytes the model reads."""
    h = hashlib.sha256()
    for s in states:
        h.update(json.dumps([s["file"], s["line"], s["col"]]).encode()); h.update(goal_bytes(s)); h.update(b"\0")
    return h.hexdigest()


def frozen_states(path, splits):
    """Use a preregistered cohort verbatim; reject duplicate or mislabeled positions."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    positions = [(s["file"], s["line"], s["col"]) for s in rows]
    if not rows or len(set(positions)) != len(rows):
        raise ValueError("frozen cohort is empty or contains duplicate positions")
    if set(s["split"] for s in rows) != set(splits) or not all(usable(s) for s in rows):
        raise ValueError("frozen cohort has unexpected split labels or unusable states")
    return rows


def select_files(tasks, workers):
    """Each task already launches an independent Lean process; avoid Python spawn.

    Windows pool respawns can fail during DuplicateHandle and leave the pool stuck
    after a worker dies. Threads only coordinate subprocess I/O; Lean still runs
    in separate processes and retains select_one's startup retries and timeouts.
    """
    if workers < 1:
        raise ValueError("Lean workers must be positive")
    if workers == 1:
        yield from map(select_one, tasks)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(select_one, task) for task in tasks]):
            yield future.result()


def proposal_key(ck, digest, beam, fast_steps, threads, device):
    """What a cached beam must match to be reused: the best-on-val weights (best.json's hash),
    the states, the beam width, fast_steps, the torch thread count and the device. None when the
    run has no best.json (its proposals are then always computed)."""
    bp = Path(ck) / "best.json"
    if not bp.exists():
        return None
    return {"ckpt_sha256": json.loads(bp.read_text())["sha256"], "states_sha256": digest, "beam": beam,
            "fast_steps": fast_steps, "threads": threads, "device": device}


def _proposal_job(job):
    """One checkpoint's proposals in its own process (`--proposal-jobs`): read from the cache
    when the key matches, else computed and stored atomically. Returns (run, proposals, hit)."""
    ck, states, key, cache, beam, fast_steps, threads, device, *alias = job
    run = alias[0] if alias else Path(ck).name
    torch.set_num_threads(threads)
    path = Path(cache) / f"{run}.json" if cache and key else None
    if path and path.exists():
        entry = json.loads(path.read_text(encoding="utf-8"))
        if entry.get("key") == key:
            return run, entry["proposals"], True
    model, _ = load_model(ck, device)
    props = proposals(model, states, device, beam, fast_steps)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        tmp.write_text(json.dumps({"key": key, "proposals": props}), encoding="utf-8")
        tmp.replace(path)
    return run, props, False


def interleave(a, b):
    out = []
    for i in range(max(len(a), len(b))):
        out += [x for x in (a[i:i + 1] + b[i:i + 1]) if x not in out]
    return out


def human_text(s):
    """The source tactic as written, continuation lines dedented by its own column. Collapsing
    whitespace (norm_tactic) breaks layout-sensitive blocks (`classical` over indented lines)
    and lets a line comment swallow the lines after it."""
    first, *rest = s["tactic"].split("\n")
    return "\n".join([first] + [l[min(s["col"], len(l) - len(l.lstrip(" "))):] for l in rest])


def certifiable(row):
    """A reached state any tactic could complete: its main goal is a proposition with a closed
    statement (Lean's `main_prop`, `main_closed`). Rows from runs before these fields: True."""
    return bool(row.get("main_prop", True) and row.get("main_closed", True))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--splits", nargs="+", default=["test_file", "test_long", "test_family"])
    ap.add_argument("--states-file", default=None, help="frozen JSONL cohort; no resampling or outcome-based filtering")
    ap.add_argument("--per-split", type=int, default=200)
    ap.add_argument("--per-file", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--fast-steps", type=int, default=None)
    ap.add_argument("--depths", type=int, nargs="+", default=None,
                    help="evaluate multiple encoder budgets in one Lean pass; first keeps the base policy name")
    ap.add_argument("--heartbeats", type=int, default=5000, help="per tactic, in thousands")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--proposal-cache", default=None,
                    help="directory of per-run beam proposals, reused only on an exact key match (proposal_key)")
    ap.add_argument("--proposal-jobs", type=int, default=1,
                    help="checkpoints whose proposals are computed at once, each in its own process with --threads")
    ap.add_argument("--proposals-only", action="store_true", help="fill --proposal-cache and stop before Lean")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.depths and (a.fast_steps is not None or min(a.depths) < 1 or len(set(a.depths)) != len(a.depths)):
        ap.error("--depths must be distinct positive budgets and cannot be combined with --fast-steps")
    torch.set_num_threads(a.threads)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    order, closes = auto_order(a.data)
    states = (frozen_states(a.states_file, a.splits) if a.states_file else
              [s for sp in a.splits for s in sample_states(a.data, sp, a.per_split, a.per_file, a.seed)])
    digest = states_sha256(states)
    print(json.dumps({"event": "states", "n": len(states), "states_sha256": digest, "auto_order": order}), flush=True)
    jobs, order_runs, policy_depths = [], [], {}
    for ck in a.ckpt:
        for index, budget in enumerate(a.depths or [a.fast_steps]):
            name = Path(ck).name + (f"_T{budget}" if index else "")
            order_runs.append(name); policy_depths[name] = budget
            jobs.append((ck, states, proposal_key(ck, digest, a.beam, budget, a.threads, a.device),
                         a.proposal_cache, a.beam, budget, a.threads, a.device, name))
    runs, cache_hits = {}, []
    if a.proposal_jobs > 1:
        with mp.get_context("spawn").Pool(min(a.proposal_jobs, len(jobs)), initializer=_lower_priority) as pool:
            done = list(pool.imap_unordered(_proposal_job, jobs))
    else:
        done = map(_proposal_job, jobs)
    got = {}
    for run, props, hit in done:
        got[run] = props
        cache_hits += [run] if hit else []
        print(json.dumps({"event": "proposals", "run": run, "cached": hit, "s": round(time.time() - t0)}), flush=True)
    runs = {name: got[name] for name in order_runs}                 # requested order, whatever finished first
    if a.device.startswith("cuda"):
        torch.cuda.empty_cache()                                 # release unused buffers before CPU Lean checking
    if a.proposals_only:
        print(json.dumps({"event": "proposals_only", "runs": list(runs), "cache": a.proposal_cache}), flush=True)
        return 0
    # One select pass per file: the union of every tactic any policy tries on each state.
    plan = collections.defaultdict(list)
    for i, s in enumerate(states):
        tacs = list(dict.fromkeys(order + [human_text(s)] + [t for p in runs.values() for t in p[i]]))
        plan[s["file"]].append((i, {"line": s["line"], "col": s["col"], "tactics": tacs}))
    tasks = [(f, [sel for _, sel in v], str(out / "lean" / (f.replace("/", "__") + ".jsonl")), a.heartbeats)
             for f, v in sorted(plan.items())]
    (out / "lean").mkdir(exist_ok=True)
    file_log = []
    for r in select_files(tasks, a.workers):
        file_log.append(r)
        print(json.dumps({k: r.get(k) for k in ("file", "selected", "seconds", "failure")}), flush=True)
    results, meta = {}, {}
    for f, sels, path, _ in tasks:
        if Path(path).exists():
            for line in open(path, encoding="utf-8"):
                row = json.loads(line)
                results[(f, row["line"], row["col"])] = {x["tactic"]: x for x in row["results"]}
                meta[(f, row["line"], row["col"])] = {k: row[k] for k in ("goals", "main_prop", "main_closed") if k in row}
    rows, summary = [], collections.defaultdict(dict)
    for i, s in enumerate(states):
        res = results.get((s["file"], s["line"], s["col"]))
        pol = {"auto": order, "human": [human_text(s)]}
        for name, p in runs.items():
            pol[name] = p[i]; pol[name + "+auto"] = interleave(p[i], order)
        rows.append({"file": s["file"], "line": s["line"], "col": s["col"], "split": s["split"],
                     "decl": s["decl"], "reached": res is not None,
                     **meta.get((s["file"], s["line"], s["col"]), {}),
                     "policies": {k: v[:max(KS)] for k, v in pol.items()},
                     "results": res})
    for sp in a.splits + ["all"]:
        sub = [r for r in rows if sp == "all" or r["split"] == sp]
        reached = [r for r in sub if r["reached"] and certifiable(r)]
        summary[sp]["states"] = len(sub); summary[sp]["reached"] = sum(r["reached"] for r in sub)
        summary[sp]["certifiable"] = len(reached)
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
              "lean": config.LEAN_VERSION, "mathlib_commit": mathlib_commit(),
              "lean_execution": "independent Lean subprocesses, coordinated without Python process pools",
              "seed": a.seed, "beam": a.beam, "fast_steps": a.fast_steps, "heartbeats_k": a.heartbeats,
              "states_file": a.states_file,
              "states_file_sha256": hashlib.sha256(Path(a.states_file).read_bytes()).hexdigest() if a.states_file else None,
              "requested_counts": dict(collections.Counter(s["split"] for s in states)),
              "depths": a.depths, "policy_depths": policy_depths,
              "states_sha256": digest, "threads": a.threads, "device": a.device,
              "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32 if a.device.startswith("cuda") else None,
              "proposal_cache": a.proposal_cache,
              "proposals_from_cache": cache_hits,
              "auto_order": order, "auto_train_closes": closes, "summary": summary,
              "files": len(tasks), "failed_files": [r for r in file_log if not r.get("ok", True)],
              "seconds": round(time.time() - t0, 1)}
    (out / "eval.json").write_text(json.dumps(report, indent=1) + "\n")
    (out / "complete.json").write_text(json.dumps({"event": "complete", "seconds": report["seconds"]}) + "\n")
    print(json.dumps({"summary": summary}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except hw.ResourcePause as exc:
        print(json.dumps({"event": "pause", "reason": str(exc)}), flush=True)
        sys.exit(75)

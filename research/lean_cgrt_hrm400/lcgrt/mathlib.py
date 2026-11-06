"""Mathlib tactic-state extraction (pilot): real proofs, traced by Lean.

    python -m lcgrt.mathlib trace --name mathlib-pilot --workers 2

For each selected Mathlib source file, `lgym_trace` (lean/Trace.lean) re-elaborates the file
with Lean's frontend and writes every source tactic with the goals before and after; on a
sample of steps it also runs a fixed candidate vocabulary on the same goal, in the
environment as it stood at that point of the file. One process per file keeps RAM bounded
(the file's imports are loaded from Mathlib's oleans and freed when it exits).

Per step: Mathlib commit, file, declaration, position, tactic text, goals before/after,
candidate outcomes, a proof-state hash, and the declaration's checker status:
  certified  elaborated without errors and its axioms are within {propext, Classical.choice,
             Quot.sound} (Lean's addDecl kernel-checks every declaration it elaborates)
  rejected   an error inside the declaration, or sorryAx / any other axiom
Dedup on the proof-state hash happens before splitting. Holdouts:
  test_family  every file under a held-out top-level directory (Mathlib/Algebra)
  test_file    whole held-out files (their lemmas never appear in training)
  test_long    declarations whose traced proof is in the longest 10% (proof depth)
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

from . import config

STANDARD_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}
CANDIDATES = ["rfl", "simp", "simp_all", "assumption", "trivial", "decide", "omega", "tauto",
              "constructor", "intro", "exact?", "aesop"]
PILOT = {  # file -> family (top-level Mathlib directory)
    "Mathlib/Logic/Basic.lean": "Logic", "Mathlib/Logic/Function/Basic.lean": "Logic",
    "Mathlib/Logic/Relation.lean": "Logic", "Mathlib/Logic/Equiv/Defs.lean": "Logic",
    "Mathlib/Order/Basic.lean": "Order", "Mathlib/Order/Lattice.lean": "Order",
    "Mathlib/Order/BoundedOrder/Basic.lean": "Order", "Mathlib/Order/Monotone/Basic.lean": "Order",
    "Mathlib/Data/Nat/Basic.lean": "Data", "Mathlib/Data/List/Basic.lean": "Data",
    "Mathlib/Data/Set/Basic.lean": "Data", "Mathlib/Data/Option/Basic.lean": "Data",
    "Mathlib/Algebra/Group/Basic.lean": "Algebra", "Mathlib/Algebra/Group/Defs.lean": "Algebra",
    "Mathlib/Algebra/Order/Monoid/Basic.lean": "Algebra",
}
HELDOUT_FAMILY = "Algebra"
HELDOUT_FILES = {"Mathlib/Order/Lattice.lean", "Mathlib/Data/Option/Basic.lean"}


def lean_path():
    m = config.MATHLIB_DIR
    dirs = [m / ".lake" / "build" / "lib" / "lean"]
    pk = m / ".lake" / "packages"
    if pk.exists():
        dirs += [p / ".lake" / "build" / "lib" / "lean" for p in sorted(pk.iterdir()) if p.is_dir()]
    return os.pathsep.join(str(d) for d in dirs)


def mathlib_commit():
    """Tarball installs record the tag's commit (git ls-remote) in MATHLIB_COMMIT."""
    for f in (config.MATHLIB_DIR / "MATHLIB_COMMIT", config.MATHLIB_DIR / ".git" / "shallow"):
        if f.exists():
            return f.read_text().split()[0]
    return None


def _lower_priority():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:
        pass


def _trace_one(task):
    from .errors import LeanError
    from .lean_repl import LeanRepl
    rel, family, out_dir, cand_every = task
    out = Path(out_dir) / (rel.replace("/", "__").replace(".lean", "") + ".steps.jsonl")
    summary_p = out.with_suffix(".summary.json")
    if summary_p.exists():
        return json.loads(summary_p.read_text())
    env = config.lean_env() | {"LEAN_PATH": lean_path()}
    t0 = time.time()
    try:
        import psutil
    except ImportError:
        psutil = None
    peak = 0
    with LeanRepl(exe=config.TRACE_EXE, env=env, cwd=str(config.MATHLIB_DIR), timeout=3600) as r:
        if psutil:
            proc = psutil.Process(r.proc.pid)
        try:
            res = r.request(op="trace", path=str(config.MATHLIB_DIR / rel), out=str(out),
                            candidates=CANDIDATES, cand_every=cand_every, cand_heartbeats=5000)
        except LeanError as e:          # recorded per file; one bad file does not stop the build
            res = {"ok": False, "failure": e.code, "message": str(e)[:400], "steps": 0, "decls": [], "errors": []}
        if psutil:
            try:
                peak = proc.memory_info().peak_wset if hasattr(proc.memory_info(), "peak_wset") else proc.memory_info().rss
            except Exception:
                peak = 0
    res.update(file=rel, family=family, seconds=round(time.time() - t0, 1), peak_rss_gib=round(peak / 2 ** 30, 2))
    summary_p.write_text(json.dumps(res) + "\n")
    return res


def decl_status(summary):
    errs = summary.get("errors", [])
    status = {}
    for d in summary["decls"]:
        ax = set(d["axioms"])
        ok = d["in_env"] and "sorryAx" not in ax and ax <= STANDARD_AXIOMS
        status[d["decl"]] = {"certified": ok, "axioms": sorted(ax)}
    return status, errs


def family_of(rel):
    return PILOT.get(rel, rel.split("/")[1] if rel.count("/") > 1 else "Mathlib")


def heldout_file(rel, full):
    """Pilot: a fixed list. Full Mathlib: ~3% of files outside the held-out family, by a
    stable hash of the path (no dependence on run order)."""
    if not full:
        return rel in HELDOUT_FILES
    return int(hashlib.sha1(rel.encode()).hexdigest(), 16) % 33 == 0


def all_files():
    base = config.MATHLIB_DIR
    return sorted(p.relative_to(base).as_posix() for p in (base / "Mathlib").rglob("*.lean"))


def build(name, workers, cand_every, files=None, full=False):
    root = config.DATA_ROOT / name
    raw = root / "raw"; raw.mkdir(parents=True, exist_ok=True)
    files = files or (all_files() if full else list(PILOT))
    tasks = [(f, family_of(f), str(raw), cand_every) for f in files]
    t0 = time.time()
    summaries = []
    with mp.get_context("spawn").Pool(workers, initializer=_lower_priority) as pool:
        for s in pool.imap_unordered(_trace_one, tasks):
            summaries.append(s)
            print(json.dumps({k: s.get(k) for k in ("file", "steps", "seconds", "elab_ms", "peak_rss_gib")} |
                             {"errors": len(s.get("errors", [])), "decls": len(s.get("decls", [])),
                              "failure": s.get("failure")}), flush=True)
    summaries.sort(key=lambda s: s["file"])          # deterministic dedup and split order
    commit = mathlib_commit()
    failed = [{"file": s["file"], "failure": s.get("failure"), "message": s.get("message")}
              for s in summaries if not s.get("ok", True)]
    ok_sums = [s for s in summaries if s.get("ok", True)]
    # Pass 1: proof length per declaration (from Lean's per-declaration step counts).
    lens = sorted(d["steps"] for s in ok_sums for d in s["decls"])
    long_cut = lens[int(0.9 * (len(lens) - 1))] if lens else 0
    # Pass 2: stream steps file by file into split files.
    seen = set()
    stats = {"steps": 0, "dedup_dropped": 0, "rejected_steps": 0, "candidate_trials": 0, "candidate_successes": 0}
    split_counts, per_file, outs = {}, [], {}
    try:
        for s in ok_sums:
            rel = s["file"]
            status, errs = decl_status(s)
            dlen = {d["decl"]: d["steps"] for d in s["decls"]}
            steps_p = raw / (rel.replace("/", "__").replace(".lean", "") + ".steps.jsonl")
            spans = {}
            with open(steps_p, encoding="utf-8") as f:
                for line in f:
                    st = json.loads(line)
                    a, b = spans.get(st["decl"], (st["line"], st["line"]))
                    spans[st["decl"]] = (min(a, st["line"]), max(b, st["line"]))
            bad = {d for d, (a, b) in spans.items() if any(a <= e["line"] <= b for e in errs)}
            per_file.append({"file": rel, "family": s["family"], "steps": s["steps"], "errors": len(errs),
                             "decls": len(spans), "seconds": s["seconds"], "elab_ms": s.get("elab_ms"),
                             "peak_rss_gib": s.get("peak_rss_gib"),
                             "certified_decls": sum(1 for d in spans if status.get(d, {}).get("certified") and d not in bad)})
            with open(steps_p, encoding="utf-8") as f:
                for line in f:
                    st = json.loads(line)
                    st.update(file=rel, family=s["family"], mathlib_commit=commit, lean=config.LEAN_VERSION)
                    ok = status.get(st["decl"], {}).get("certified", False) and st["decl"] not in bad
                    st["checker"] = "certified" if ok else "rejected"
                    st["axioms"] = status.get(st["decl"], {}).get("axioms")
                    st["state_id"] = hashlib.sha256("\n\n".join(st["goals_before"]).encode()).hexdigest()[:24]
                    stats["steps"] += 1
                    stats["candidate_trials"] += len(st["candidates"])
                    stats["candidate_successes"] += sum(c["ok"] for c in st["candidates"])
                    if not ok:
                        stats["rejected_steps"] += 1
                    if st["state_id"] in seen:
                        stats["dedup_dropped"] += 1
                        continue
                    seen.add(st["state_id"])
                    split = ("test_family" if s["family"] == HELDOUT_FAMILY else
                             "test_file" if heldout_file(rel, full) else
                             "test_long" if dlen.get(st["decl"], 0) > long_cut else "train")
                    st["split"] = split
                    if split not in outs:
                        outs[split] = gzip.open(root / f"{split}.jsonl.gz", "wt", encoding="utf-8")
                    outs[split].write(json.dumps(st, ensure_ascii=False) + "\n")
                    split_counts[split] = split_counts.get(split, 0) + 1
    finally:
        for f in outs.values():
            f.close()
    manifest = {
        "format": "lcgrt-mathlib-trace-v1", "name": name, "lean_version": config.LEAN_VERSION,
        "mathlib_commit": commit, "mathlib_tag": "v4.19.0", "files": per_file,
        "candidates": CANDIDATES, "cand_every": cand_every, "full_mathlib": full,
        "heldout_family": HELDOUT_FAMILY,
        "heldout_files": "sha1(path) % 33 == 0" if full else sorted(HELDOUT_FILES),
        "long_proof_cut_steps": long_cut, "split_counts": split_counts, **stats, "failed_files": failed,
        "split_sha256": {s: hashlib.sha256((root / f"{s}.jsonl.gz").read_bytes()).hexdigest() for s in outs},
        "seconds": round(time.time() - t0, 1),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: manifest[k] for k in ("split_counts", "steps", "dedup_dropped", "rejected_steps",
                                               "candidate_trials", "candidate_successes", "seconds")}))
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("trace")
    t.add_argument("--name", default="mathlib-pilot")
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--cand-every", type=int, default=8)
    t.add_argument("--files", nargs="*", default=None)
    t.add_argument("--all", action="store_true", help="every file under Mathlib/")
    a = ap.parse_args(argv)
    build(a.name, a.workers, a.cand_every, a.files, full=a.all)


if __name__ == "__main__":
    main()

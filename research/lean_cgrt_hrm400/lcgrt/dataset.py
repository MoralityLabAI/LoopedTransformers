"""Build a Lean-extracted, deduplicated, split, certified dataset as gzip JSONL shards.

    python -m lcgrt.dataset build --name lcgrt-v1 --workers 3

Order of operations (each step is recorded in manifest.json):
  1. plan: generate candidate theorems per split, held-out splits first;
  2. dedup: drop any theorem whose canonical hash (names, hypothesis order and fact
     order factored out) was already taken by any split, before extraction;
  3. extract: Lean explores every theorem (lcgrt.extract); failures are classified;
  4. certify: each shard's shortest proofs are compiled as one Lean file (lcgrt.certify);
     only `passed` theorems are visible to training/evaluation loaders.
Shards are resumable: a shard whose file and certificate exist is skipped.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

from . import config
from . import fragment as F

# split -> (families, depths, arity3, count). Held-out splits are planned first.
SPEC = {
    "test_family": (F.HELDOUT_FAMILIES, range(2, 7), False, 600),
    "test_depth": (F.TRAIN_FAMILIES, range(7, 13), False, 600),
    "test_dep": (F.TRAIN_FAMILIES, range(2, 7), True, 600),
    "test_id": (F.TRAIN_FAMILIES, range(2, 7), False, 600),
    "val": (F.TRAIN_FAMILIES, range(2, 7), False, 400),
    "train": (F.TRAIN_FAMILIES, range(2, 7), False, 24000),
}
# Reasoning-depth holdout, added after campaign c1 saturated test_depth: proof depth stays in
# range while each decision needs deeper reasoning. Deep-decoy dead chains of 6-12 steps
# (training: 2-4) and cycle loops with 4-8 extra atoms (training: 0-1).
EXTRA_SPECS = {
    "reason": {"test_reason": (("chain-deep", "alt-deep", "chain-cycle", "tree-cycle"), range(4, 9), False, 400,
                               {"deep_len": (6, 12), "cycle_extra": (4, 8)})},
    # Mirror decoys: structural copies of the real derivation with dead leaves (see fragment).
    "mirror": {"test_mirror": (F.TRAIN_FAMILIES, range(3, 9), False, 420, {"mirror_p": 0.6})},
}
# Whole-dataset specs other than SPEC. Campaign c2 (after c1's test_mirror result): every
# split carries mirror decoys; train on depths 2-5, hold out depths 6-9.
DATASET_SPECS = {
    "lcgrt-mirror": {
        "test_mirror_deep": (F.TRAIN_FAMILIES, range(6, 10), False, 400, {"mirror_p": 0.6}),
        "test_mirror_id": (F.TRAIN_FAMILIES, range(2, 6), False, 300, {"mirror_p": 0.6}),
        "val": (F.TRAIN_FAMILIES, range(2, 6), False, 300, {"mirror_p": 0.6}),
        "train": (F.TRAIN_FAMILIES, range(2, 6), False, 12000, {"mirror_p": 0.6}),
    },
}
SHARD = 500
SEED_BASE = {"train": 0, "val": 10_000_000, "test_id": 20_000_000, "test_dep": 30_000_000,
             "test_depth": 40_000_000, "test_family": 50_000_000, "test_reason": 60_000_000,
             "test_mirror": 70_000_000, "test_mirror_id": 80_000_000, "test_mirror_deep": 90_000_000}


def plan(spec=None, scale: float = 1.0, taken: dict | None = None):
    spec = SPEC if spec is None else spec
    taken = {} if taken is None else taken
    rows, dropped = [], {}
    for split, entry in spec.items():
        fams, depths, a3, count = entry[:4]
        kw = entry[4] if len(entry) > 4 else {}
        count = max(1, int(count * scale))
        depths = list(depths)
        i = kept = 0
        while kept < count:
            fam = fams[i % len(fams)]
            depth = depths[(i // len(fams)) % len(depths)]
            seed = SEED_BASE[split] + i
            i += 1
            t = F.generate(seed, depth, fam, arity3=a3, tid=f"{split}_{kept:05d}", **kw)
            h = F.canonical_hash(t)
            if h in taken:
                dropped[split] = dropped.get(split, 0) + 1
                continue
            taken[h] = split
            rows.append({"split": split, "tid": t.tid, "family": fam, "depth": depth, "seed": seed,
                         "arity3": a3, "canonical_hash": h, "stmt_sha256": F.stmt_sha(t),
                         **({"gen_kwargs": {k: (list(v) if isinstance(v, (tuple, list)) else v) for k, v in kw.items()}} if kw else {})})
            kept += 1
    return rows, dropped


def gen_kwargs(row):
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in (row.get("gen_kwargs") or {}).items()}


def _theorem(row):
    return F.generate(row["seed"], row["depth"], row["family"], arity3=row["arity3"], tid=row["tid"],
                      **gen_kwargs(row))


_REPL = None


def _work(task):
    global _REPL
    from .certify import certify
    from .errors import GymError
    from .extract import explore
    from .lean_repl import LeanRepl
    split, idx, rows, out_dir = task
    out = Path(out_dir) / split
    out.mkdir(parents=True, exist_ok=True)
    shard = out / f"shard_{idx:05d}.jsonl.gz"
    cert_path = out / f"shard_{idx:05d}.cert.json"
    if shard.exists() and cert_path.exists():
        return json.loads(cert_path.read_text())["summary"]
    if _REPL is None:
        _REPL = LeanRepl()
    t0 = time.time()
    recs, failures = [], []
    for row in rows:
        thm = _theorem(row)
        try:
            rec = explore(_REPL, thm)
        except GymError as e:
            failures.append({"tid": row["tid"], "code": e.code, "message": str(e)[:300]})
            if e.code in ("lean_timeout", "lean_crashed"):
                _REPL.restart()
            continue
        rec["split"] = split
        rec["gen_kwargs"] = row.get("gen_kwargs")
        recs.append((rec, thm))
    receipt = certify([(r["tid"], thm, r["proof"]["script"]) for r, thm in recs])
    from .certify import decl_name
    tmp = shard.with_suffix(".part")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for r, _ in recs:
            r["certificate"] = {"status": receipt["status"].get(decl_name(r["tid"]), "missing"),
                                "source_sha256": receipt["source_sha256"],
                                "lean_version": receipt["lean_version"]}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, shard)
    summary = {"split": split, "shard": shard.name, "sha256": sha256(shard), "theorems": len(recs),
               "certified": receipt["passed"], "rejected": len(receipt["rejected"]),
               "failures": failures, "seconds": round(time.time() - t0, 1),
               "lean_calls": sum(r["lean"]["calls"] for r, _ in recs),
               "states": sum(len(r["path"]) for r, _ in recs)}
    receipt["summary"] = summary
    receipt.pop("status")
    tmpc = cert_path.with_suffix(".part")
    tmpc.write_text(json.dumps(receipt, indent=1) + "\n")
    os.replace(tmpc, cert_path)
    return summary


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(name: str, workers: int, scale: float = 1.0, extra: str | None = None):
    """`extra` (a key of EXTRA_SPECS) adds those splits to an existing dataset, deduplicated
    against every theorem already planned there; it writes plan_<extra>.jsonl and
    manifest_<extra>.json and leaves the original plan and manifest untouched."""
    root = config.DATA_ROOT / name
    root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    spec = EXTRA_SPECS[extra] if extra is not None else DATASET_SPECS.get(name, SPEC)
    suffix = "" if extra is None else f"_{extra}"
    taken: dict = {}
    # Dedup against this dataset's other plans and, for a new dataset, every other dataset's.
    plans = [f for f in root.glob("plan*.jsonl") if f.name != f"plan{suffix}.jsonl"]
    if name in DATASET_SPECS:
        plans += [f for d in config.DATA_ROOT.iterdir() if d.is_dir() and d.name != name
                  for f in d.glob("plan*.jsonl")]
    if extra is not None or name in DATASET_SPECS:
        for f in plans:
            for line in f.read_text().splitlines():
                r = json.loads(line); taken[r["canonical_hash"]] = r["split"]
    n_prior = len(taken)
    rows, dropped = plan(spec, scale=scale, taken=taken)
    (root / f"plan{suffix}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    tasks = []
    for split in spec:
        srows = [r for r in rows if r["split"] == split]
        for k in range(0, len(srows), SHARD):
            tasks.append((split, k // SHARD, srows[k:k + SHARD], str(root)))
    print(json.dumps({"event": "plan", "theorems": len(rows), "dedup_dropped": dropped,
                      "shards": len(tasks)}), flush=True)
    summaries = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_lower_priority) as pool:
        for s in pool.imap_unordered(_work, tasks):
            summaries.append(s)
            print(json.dumps({k: v for k, v in s.items() if k != "failures"} |
                             {"failures": len(s["failures"])}), flush=True)
    manifest = {
        "format": "lcgrt-shards-v1", "name": name, "created_unix": int(time.time()),
        "lean_version": config.LEAN_VERSION, "lean_commit": config.LEAN_COMMIT,
        "generator": "lcgrt.fragment (seed string lcgrt-v1)", "spec": {
            k: {"families": list(v[0]), "depths": list(v[1]), "arity3": v[2], "count": int(v[3] * scale),
                **({"gen_kwargs": {a: (list(b) if isinstance(b, (tuple, list)) else b) for a, b in v[4].items()}} if len(v) > 4 else {})}
            for k, v in spec.items()},
        "heldout_families": list(F.HELDOUT_FAMILIES), "dedup": "canonical_hash, held-out splits first",
        "dedup_against_prior_theorems": n_prior,
        "dedup_dropped": dropped, "plan_sha256": sha256(root / f"plan{suffix}.jsonl"),
        "shards": sorted(summaries, key=lambda s: (s["split"], s["shard"])),
        "seconds": round(time.time() - t0, 1),
    }
    tot = {}
    for s in summaries:
        d = tot.setdefault(s["split"], {"theorems": 0, "certified": 0, "states": 0, "failures": 0})
        d["theorems"] += s["theorems"]; d["certified"] += s["certified"]
        d["states"] += s["states"]; d["failures"] += len(s["failures"])
    manifest["totals"] = tot
    (root / f"manifest{suffix}.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({"event": "done", "totals": tot, "seconds": manifest["seconds"]}), flush=True)
    return manifest


def _lower_priority():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if sys.platform == "win32" else 10)
    except Exception:
        pass


def iter_records(name: str, split: str, certified_only: bool = True):
    """Stream records of one split, one shard at a time (bounded RAM)."""
    for shard in sorted((config.DATA_ROOT / name / split).glob("shard_*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if certified_only and r["certificate"]["status"] != "passed":
                    continue
                yield r


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--name", default="lcgrt-v1")
    b.add_argument("--workers", type=int, default=3)
    b.add_argument("--scale", type=float, default=1.0)
    b.add_argument("--extra", default=None, choices=sorted(EXTRA_SPECS))
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build(a.name, a.workers, a.scale, a.extra)


if __name__ == "__main__":
    main()

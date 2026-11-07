"""Publish a traced Mathlib dataset to a private Hugging Face dataset repo, then verify it.

    python -m lcgrt.hf_upload --data mathlib-c3 --repo AlephFunk/mathlib-v4.19.0-tactic-states
    python -m lcgrt.hf_upload --data mathlib-c3 --wait      # waits for the dataset's manifest first

Uploads every split file (`*.jsonl.gz`), the manifest, per-file trace summaries (declarations,
axioms, Lean errors; one gzip file) and a dataset card generated from the manifest, in one commit
under `--prefix` (default: the root for the main dataset). Afterwards every uploaded file's
sha256 (LFS) or size is compared with the local file; a mismatch is an error. The receipt goes
to `<data>/hf_upload.json`. Uses the Hugging Face login already on this machine.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import time
from pathlib import Path

from . import config

CODE_URL = "https://github.com/MoralityLabAI/LoopedTransformers/tree/main/research/lean_cgrt_hrm400"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def summaries_blob(m) -> bytes:
    """Per-file trace summaries of the files in the manifest (from the source trace's raw dir)."""
    raw = config.DATA_ROOT / m.get("source", m["name"]) / "raw"
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as gz:
        for f in m["files"]:
            p = raw / (f["file"].replace("/", "__").replace(".lean", "") + ".steps.summary.json")
            if p.exists():
                gz.write(p.read_bytes().rstrip(b"\n") + b"\n")
    return buf.getvalue()


def card(m, prefix, splits) -> str:
    sc = m["split_counts"]
    rows = "\n".join(f"| `{s}` | {sc.get(s, 0):,} |" for s in splits)
    cfg = "\n".join(f"  - split: {s}\n    path: {prefix}{s}.jsonl.gz" for s in splits)
    held = m["heldout_files"] if isinstance(m["heldout_files"], str) else ", ".join(m["heldout_files"])
    return f"""---
license: apache-2.0
pretty_name: Mathlib v4.19.0 tactic states traced by Lean
tags: [lean4, mathlib, theorem-proving, tactic-prediction]
configs:
- config_name: {m['name']}
  data_files:
{cfg}
---

# Mathlib v4.19.0 tactic states, traced by Lean

Every source tactic in Mathlib `{m.get('mathlib_tag')}` (commit `{m.get('mathlib_commit')}`), traced by
Lean `{m.get('lean_version')}`. One row per tactic: the goals before and after as Lean pretty-printed
them, the tactic text, its source position and declaration, and the checker status of that
declaration. On every {'/'.join(str(c) for c in (m.get('cand_every') or []) if c) or '16'}-th step, 12 automation tactics were run on the same goal in the environment
as it stood at that point of the file (the theorem being proved not yet in scope), with each outcome.

Built by [`lcgrt.mathlib`]({CODE_URL}) (tracer: `lean/Trace.lean`). Files traced: {m.get('files_traced'):,};
failed files: {len(m.get('failed_files', []))}.

| split | rows |
|---|---:|
{rows}

Splits (deduplicated on a hash of the goal state before splitting):
- `test_family`: every file under `Mathlib/{m['heldout_family']}`;
- `test_file`: whole held-out files ({held});
- `val`: whole files, sha1(path) % 33 == 1 (full Mathlib only), for training-time monitoring;
- `test_long`: declarations in the longest 10% by traced proof length (more than {m.get('long_proof_cut_steps')} steps);
- `train`: the rest.

Fields: `decl`, `file`, `line`, `col`, `tactic`, `kind`, `elaborator`, `goals_before`, `goals_after`,
`candidates` (`tactic`, `ok`, `goals_after` or `error`), `checker` (`certified` = elaborated without
errors, kernel-checked by `addDecl`, axioms within propext / Classical.choice / Quot.sound;
otherwise `rejected`), `axioms`, `state_id`, `terminal` (the source tactic closed the main goal),
`family`, `split`, `mathlib_commit`, `lean`. Bracket and separator tokens (`]`, `;`, `<;>`) are
dropped: {m.get('atom_rows_dropped', 0):,} rows; duplicates dropped: {m.get('dedup_dropped', 0):,}.

`{prefix}manifest.json` holds per-file statistics and split sha256 sums; `{prefix}file_summaries.jsonl.gz`
holds each file's declarations with their axioms and every Lean error. Mathlib is Apache-2.0.
"""


def upload(data, repo, prefix, private):
    from huggingface_hub import CommitOperationAdd, HfApi
    root = config.DATA_ROOT / data
    m = json.loads((root / "manifest.json").read_text())
    splits = sorted(m["split_counts"], key=lambda s: ("train", "val", "test_file", "test_long", "test_family").index(s)
                    if s in ("train", "val", "test_file", "test_long", "test_family") else 9)
    files = {f"{prefix}{s}.jsonl.gz": root / f"{s}.jsonl.gz" for s in splits}
    files[f"{prefix}manifest.json"] = root / "manifest.json"
    blobs = {f"{prefix}file_summaries.jsonl.gz": summaries_blob(m)}
    if not prefix:
        blobs["README.md"] = card(m, prefix, splits).encode("utf-8")
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    ops = [CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(v)) for k, v in files.items()]
    ops += [CommitOperationAdd(path_in_repo=k, path_or_fileobj=v) for k, v in blobs.items()]
    t0 = time.time()
    info = api.create_commit(repo, repo_type="dataset", operations=ops,
                             commit_message=f"{data}: {m.get('files_traced')} files, splits {m['split_counts']}")
    # Verify against the remote tree: LFS files by sha256, small files by size.
    want = {k: (sha256(v), v.stat().st_size) for k, v in files.items()}
    want |= {k: (hashlib.sha256(b).hexdigest(), len(b)) for k, b in blobs.items()}
    remote = {e.path: e for e in api.get_paths_info(repo, list(want), repo_type="dataset", expand=True)}
    bad = []
    for k, (h, n) in want.items():
        e = remote.get(k)
        lfs = getattr(e, "lfs", None) if e else None
        ok = e is not None and (lfs.sha256 == h if lfs else e.size == n)
        if not ok:
            bad.append(k)
    receipt = {"repo": repo, "private": private, "prefix": prefix, "commit": info.oid, "url": info.commit_url,
               "files": {k: {"sha256": h, "bytes": n} for k, (h, n) in want.items()}, "verified": not bad,
               "mismatch": bad, "seconds": round(time.time() - t0, 1), "unix": int(time.time())}
    (root / "hf_upload.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps({k: receipt[k] for k in ("repo", "commit", "verified", "mismatch", "seconds")}), flush=True)
    if bad:
        raise SystemExit(f"remote files differ from local: {bad}")
    return receipt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--repo", default="AlephFunk/mathlib-v4.19.0-tactic-states")
    ap.add_argument("--prefix", default="", help="folder in the repo, e.g. 'pilot/' (empty: root, with the card)")
    ap.add_argument("--public", action="store_true", help="default: private")
    ap.add_argument("--wait", action="store_true", help="wait for <data>/manifest.json")
    ap.add_argument("--poll-s", type=int, default=300)
    a = ap.parse_args(argv)
    man = config.DATA_ROOT / a.data / "manifest.json"
    while a.wait and not man.exists():
        time.sleep(a.poll_s)
    upload(a.data, a.repo, a.prefix, private=not a.public)


if __name__ == "__main__":
    main()

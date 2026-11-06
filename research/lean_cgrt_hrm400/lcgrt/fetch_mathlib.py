"""Fetch a Mathlib source tree file by file and verify it against git's blob hashes.

    python -m lcgrt.fetch_mathlib --commit c44e0c8ee63ca166450922a373c7409c5d26b00b

Used when git clones and codeload tarballs from GitHub are throttled or cut mid-transfer
(2026-11-06: ~20 KB/s and repeated `early EOF` on this machine, while CDN-served files
came at full speed). One GitHub tree-API request lists every blob with its SHA-1; curl
downloads them in parallel from raw.githubusercontent.com; every file is accepted only if
sha1("blob <size>\\0" + content) equals the tree's hash, so the result is the commit's
exact tree. Re-running skips verified files. Then run `lake exe cache get` in the tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from . import config

REPO = "leanprover-community/mathlib4"


def blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", default="c44e0c8ee63ca166450922a373c7409c5d26b00b")   # tag v4.19.0
    ap.add_argument("--out", default=str(config.MATHLIB_DIR))
    ap.add_argument("--parallel", type=int, default=16)
    ap.add_argument("--listing", default=None, help="git ls-tree -r output (avoids the tree API)")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.listing:
        # `git ls-tree -r <commit>` from a blobless clone: "mode blob sha\tpath" per file.
        blobs = []
        for line in Path(a.listing).read_text(encoding="utf-8").splitlines():
            meta, path = line.split("\t", 1)
            mode, kind, sha = meta.split()
            if kind == "blob":
                blobs.append({"path": path, "sha": sha})
    else:
        tree_p = out / ".tree.json"
        if not tree_p.exists():
            subprocess.run(["curl", "-sSL", "--fail", "--retry", "5", "-o", str(tree_p),
                            f"https://api.github.com/repos/{REPO}/git/trees/{a.commit}?recursive=1"], check=True)
        tree = json.loads(tree_p.read_text(encoding="utf-8"))
        if tree.get("truncated"):
            sys.exit("tree listing truncated by the API; cannot guarantee a complete checkout")
        blobs = [t for t in tree["tree"] if t["type"] == "blob"]
    for attempt in range(4):
        todo = []
        for b in blobs:
            p = out / b["path"]
            if not (p.exists() and blob_sha1(p.read_bytes()) == b["sha"]):
                todo.append(b)
        print(json.dumps({"attempt": attempt, "files": len(blobs), "missing_or_bad": len(todo)}), flush=True)
        if not todo:
            break
        cfg = out / ".curl.cfg"
        with cfg.open("w", encoding="utf-8") as f:
            for b in todo:
                p = out / b["path"]; p.parent.mkdir(parents=True, exist_ok=True)
                f.write(f'url = "https://raw.githubusercontent.com/{REPO}/{a.commit}/{b["path"]}"\n')
                f.write(f'output = "{p.as_posix()}"\n')
        subprocess.run(["curl", "-sSL", "--fail", "--retry", "5", "--retry-all-errors", "-Z",
                        "--parallel-max", str(a.parallel), "-K", str(cfg)])
    else:
        sys.exit("files still failing verification after 4 attempts")
    (out / "MATHLIB_COMMIT").write_text(a.commit + "\n")
    print(json.dumps({"commit": a.commit, "files": len(blobs), "verified": True,
                      "bytes": sum((out / b["path"]).stat().st_size for b in blobs)}))


if __name__ == "__main__":
    main()

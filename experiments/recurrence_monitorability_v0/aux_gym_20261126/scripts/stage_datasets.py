"""Stage the aux-gym datasets (validation / ood_stress shards, gzipped train pools, receipts, stats) into the push worktree."""
import collections
import gzip
import hashlib
import json
import shutil
from pathlib import Path

SRC = Path(r"C:\Users\patri\rmp-aux-20261125\experiments\recurrence_monitorability_v0\datasets")
POOL = Path(r"C:\Users\patri\Documents\Codex\rmp_v0_aux\pool")
DEST = Path(r"C:\Users\patri\lt-main-push-20261126\experiments\recurrence_monitorability_v0\aux_gym_20261126\datasets")
DEST.mkdir(parents=True, exist_ok=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


files = {}
stats = {}
for split in ("validation", "ood_stress"):
    (DEST / split).mkdir(exist_ok=True)
    for src in sorted((SRC / split).glob("*.jsonl")):
        dst = DEST / split / src.name
        shutil.copyfile(src, dst)
        files[f"{split}/{src.name}"] = {"bytes": dst.stat().st_size, "sha256": sha(dst)}
        rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
        fam = src.stem
        depth = collections.Counter(r["difficulty"] for r in rows)
        targets = collections.Counter(r["target_token"] for r in rows)
        lens = [len(r["tokens"]) for r in rows]
        top = targets.most_common(1)[0]
        stats[f"{split}/{fam}"] = {
            "examples": len(rows),
            "by_difficulty": dict(sorted(depth.items())),
            "distinct_target_tokens": len(targets),
            "most_common_target_share": round(top[1] / len(rows), 4),
            "distinct_fingerprints": len({r["fingerprint"] for r in rows}),
            "distinct_groups": len({r["group_id"] for r in rows}),
            "token_length_min_max": [min(lens), max(lens)],
        }

pool_dir = DEST / "train_pools"
pool_dir.mkdir(exist_ok=True)
receipt = json.loads((POOL / "pool_receipt.json").read_text(encoding="utf-8"))
shutil.copyfile(POOL / "pool_receipt.json", pool_dir / "pool_receipt.json")
for key, meta in receipt.items():
    src = POOL / meta["path"]
    assert sha(src) == meta["sha256"], key
    dst = pool_dir / (meta["path"] + ".gz")
    with open(src, "rb") as fin, open(dst, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as fout:
        shutil.copyfileobj(fin, fout, 1 << 20)
    with gzip.open(dst, "rb") as g:
        h = hashlib.sha256()
        for block in iter(lambda: g.read(1 << 20), b""):
            h.update(block)
    assert h.hexdigest() == meta["sha256"], f"round trip {key}"
    files[f"train_pools/{dst.name}"] = {"bytes": dst.stat().st_size, "sha256": sha(dst), "uncompressed_sha256": meta["sha256"], "uncompressed_bytes": src.stat().st_size, "examples": meta["size"]}

(DEST / "dataset_index.json").write_text(json.dumps({"files": files, "stats": stats}, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
print(json.dumps(stats, indent=1)[:3500])
print(sum(v["bytes"] for v in files.values()))

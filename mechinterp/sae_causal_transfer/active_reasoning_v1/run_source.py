"""LS-SAE12 source-only stage: fetch the 48 pinned pool files and admit them.

No checkpoint, model prediction or intervention is touched. Every pool slot must
pass (blob SHA, 81 digits, unique solution under two independent exact solvers
that agree, singles-stall, disjointness from LS-SAE08 and LS-SAE11) or the pool
aborts with no replacement.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from mechinterp.sae_causal_transfer.active_reasoning_v1 import families as F

SCHEMA = "LS-SAE12/source-only-pool-48-v1"


def prior_puzzles(ls_sae11_source, ls_sae08_full=None):
    s11 = json.loads(Path(ls_sae11_source).read_text())
    rows = [r for g in s11["rosters"].values() for r in g]
    inputs = [tuple(r["x"]) for r in rows]
    targets = [tuple(r["target"]) for r in rows]
    prov = {"ls_sae11_source_20_sha256": hashlib.sha256(Path(ls_sae11_source).read_bytes()).hexdigest(),
            "ls_sae11_rows": len(rows)}
    if ls_sae08_full is not None:
        raw = Path(ls_sae08_full).read_bytes()
        full = json.loads(raw)
        old = [r for g in ("test_52", "test_54") for r in full["families"][g]["source_roster"]]
        inputs += [tuple(r["x"]) for r in old]
        targets += [tuple(r["target"]) for r in old]
        prov.update({"ls_sae08_full_sha256": hashlib.sha256(raw).hexdigest(), "ls_sae08_rows": len(old)})
    return inputs, targets, prov


def build(ls_sae11_source, ls_sae08_full=None, raw_dir=None):
    inputs, targets, prov = prior_puzzles(ls_sae11_source, ls_sae08_full)
    pool = []
    for k, (name, digest) in enumerate(F.POOL_FILES):
        if raw_dir is not None:
            raw = (Path(raw_dir) / name).read_bytes()
            x = F.parse(name, digest, raw)
        else:
            x, raw = F.fetch(name, digest, F.POOL_FILES)
        pool.append(F.admit_one(F.POOL_START + k, name, digest, x, raw))
        print("LS_SAE12_SOURCE", F.POOL_START + k, name, "holes", pool[-1]["holes"],
              "algx_nodes", pool[-1]["solver_algx_search_nodes"], "dfs_nodes", pool[-1]["solver_dfs_nodes"],
              "singles_deduced", pool[-1]["visible_structural_difficulty"]["deduced_missing_cells"], flush=True)
    F.check_disjoint(pool, inputs, targets)
    return {"schema": SCHEMA, "external_repo": F.REPO, "external_repo_commit": F.COMMIT,
            "external_tree_sha": F.TREE_SHA, "pool_start": F.POOL_START,
            "pool_files": [list(p) for p in F.POOL_FILES], "pool": pool,
            "pool_fingerprint": F.fingerprint(pool), "prior_cohorts": prov,
            "model_inference_executed": False}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ls-sae11-source", type=Path, required=True)
    p.add_argument("--ls-sae08-full", type=Path)
    p.add_argument("--raw-dir", type=Path, help="read pinned files from disk instead of the network")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = build(a.ls_sae11_source, a.ls_sae08_full, a.raw_dir)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes((json.dumps(result, sort_keys=True, indent=1) + "\n").encode())
    print(json.dumps({"admitted": len(result["pool"]), "fingerprint": result["pool_fingerprint"],
                      "prior": result["prior_cohorts"]}, indent=2))


if __name__ == "__main__":
    main()

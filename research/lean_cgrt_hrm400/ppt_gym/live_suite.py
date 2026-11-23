"""PPT-G03 immutable, lineage-separated Lean Horn theorem suites.

This generator produces *new* theorem statements for independent Lean execution.
Its Python reference proof is never included in model-facing packets. Benchmark
families may overlap earlier public LC-GRT datasets; the manifest only certifies
disjointness within this newly frozen suite.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from lcgrt import fragment as F
from .data import canonical, digest, sha256_file

SCHEMA = "ppt_gym.live_horn_suite.v1"
SPLITS = {
    "train": (F.TRAIN_FAMILIES, (2, 3, 4)),
    "calibration": (F.TRAIN_FAMILIES, (3, 4, 5)),
    "ordinary_test": (F.TRAIN_FAMILIES, (4, 5, 6)),
    "ood_family": (F.HELDOUT_FAMILIES, (4, 5, 6)),
}


@dataclass(frozen=True)
class LiveCase:
    root: str
    lineage: str
    split: str
    family: str
    seed: int
    depth: int
    tid: str
    statement: str
    equal_alt_p: float = 0.0

    def theorem(self) -> F.Theorem:
        t = F.generate(self.seed, self.depth, self.family, arity3=False, tid=self.tid,
                       equal_alt_p=self.equal_alt_p)
        if F.canonical_hash(t) != self.lineage or F.statement(t) != self.statement:
            raise ValueError("frozen Lean theorem differs from current generator")
        if digest([self.split, self.lineage, self.statement]) != self.root:
            raise ValueError("frozen Lean root digest changed")
        return t


def _seeds(seed: int, split: str, index: int, attempt: int) -> int:
    raw = hashlib.sha256(f"ppt-g03/{seed}/{split}/{index}/{attempt}".encode()).digest()
    return int.from_bytes(raw[:8], "big")


def create_suite(directory: Path, *, seed: int = 20261122, per_split: int = 24,
                 equal_alt_p: float = 0.0):
    if directory.exists() or not 2 <= per_split <= 1024 or not 0 <= equal_alt_p <= 1:
        raise ValueError("suite directory must be fresh; use 2..1024 roots per split")
    directory.mkdir(parents=True)
    files = {}
    seen_lineages = set()
    for split, (families, depths) in SPLITS.items():
        path = directory / f"{split}.jsonl"
        with path.open("x", encoding="utf-8") as stream:
            for i in range(per_split):
                fam = families[i % len(families)]
                depth = depths[(i // len(families)) % len(depths)]
                tid = f"ppt_g03_{split}_{seed}_{i:05d}"
                for retry in range(256):
                    index_seed = _seeds(seed, split, i, retry)
                    thm = F.generate(index_seed, depth, fam, arity3=False, tid=tid,
                                     equal_alt_p=equal_alt_p)
                    lineage = F.canonical_hash(thm)
                    if lineage not in seen_lineages:
                        break
                else:
                    raise RuntimeError("could not allocate a unique theorem lineage")
                seen_lineages.add(lineage)
                statement = F.statement(thm)
                row = {
                    "root": digest([split, lineage, statement]),
                    "lineage": lineage, "split": split, "family": fam,
                    "seed": index_seed, "depth": depth, "tid": tid, "statement": statement,
                }
                if equal_alt_p:
                    row["equal_alt_p"] = equal_alt_p
                stream.write(canonical(row) + "\n")
        files[split] = {"file": path.name, "sha256": sha256_file(path), "rows": per_split}
    manifest = {
        "schema": SCHEMA, "seed": seed, "files": files,
        "generator_sha256": sha256_file(Path(F.__file__)),
        "toolchain_expected": "4.19.0",
        "evidence": "new_generated_horn_problems_unverified_until_live_execution",
        "benchmark_ancestry_status": "external_dataset_overlap_unchecked",
        "source_splits": {"train": "train_only",
                          "calibration": "development",
                          "ordinary_test": "held_out_roots",
                          "ood_family": "held_out_generator_families"},
    }
    if equal_alt_p:
        manifest["equal_alt_p"] = equal_alt_p
        manifest["source_family"] = "lbf09_true_equal_cost_premise_twins.v1"
    (directory / "manifest.json").write_text(canonical(manifest) + "\n", encoding="utf-8")
    return manifest


def load_suite(directory: Path, split: str, max_cases: int = 1024):
    if split not in SPLITS or not 1 <= max_cases <= 1024:
        raise ValueError("invalid split or bounded root count")
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema") != SCHEMA or
            manifest.get("generator_sha256") != sha256_file(Path(F.__file__))):
        raise ValueError("Lean generator source or suite schema changed")
    all_seen = set()
    selected = []
    for current in SPLITS:
        info = manifest["files"][current]
        if info["file"] != f"{current}.jsonl":
            raise ValueError("unexpected Lean suite shard")
        shard = directory / info["file"]
        if sha256_file(shard) != info["sha256"]:
            raise ValueError("frozen Lean suite bytes changed")
        count = 0
        with shard.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                case = LiveCase(**row)
                if case.split != current or case.lineage in all_seen:
                    raise ValueError("duplicate or crossed theorem lineage")
                if case.equal_alt_p != manifest.get("equal_alt_p", 0.0):
                    raise ValueError("theorem source equal-cost augmentation differs from frozen manifest")
                if case.family not in SPLITS[current][0] or case.depth not in SPLITS[current][1]:
                    raise ValueError("case crosses registered family or depth split")
                case.theorem()
                all_seen.add(case.lineage)
                count += 1
                if current == split and len(selected) < max_cases:
                    selected.append(case)
        if count != info["rows"]:
            raise ValueError("suite shard row count changed")
    if not selected:
        raise ValueError("empty frozen Lean suite split")
    source = {"schema": SCHEMA, "suite_manifest_sha256": sha256_file(manifest_path),
              "generator_sha256": manifest["generator_sha256"],
              "split": split, "total_roots": manifest["files"][split]["rows"],
              "selected_roots": len(selected),
              "benchmark_ancestry_status": manifest["benchmark_ancestry_status"]}
    if manifest.get("equal_alt_p", 0.0):
        source["equal_alt_p"] = manifest["equal_alt_p"]
        source["source_family"] = manifest.get("source_family")
    return selected, source

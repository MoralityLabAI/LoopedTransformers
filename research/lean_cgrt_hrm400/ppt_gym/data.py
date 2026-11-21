"""Strict DataGoblin Lean-pool importer and clearly marked synthetic fixtures.

A checked-candidate pack is a *retrospective* menu of tactics verified by Lean
during collection. This loader never certifies a newly generated proof.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path

PACK_SCHEMA = "datagoblin.checked-candidate-pack.v1"
FIXTURE_SCHEMA = "ppt.synthetic_fixture.v1"
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class Case:
    root: str
    lineage: str
    family: str
    split: str
    goal: str
    candidates: tuple[str, ...]
    labels: tuple[int, ...]
    evidence: str

    def validate(self):
        if not all(isinstance(x, str) and x for x in (self.root, self.lineage, self.family, self.split, self.goal)):
            raise ValueError("case identity/family/goal must be nonempty strings")
        if not 1 <= len(self.candidates) <= 64 or len(self.candidates) != len(self.labels):
            raise ValueError("case must have 1..64 aligned candidate tactics")
        if not all(isinstance(x, str) and x for x in self.candidates):
            raise ValueError("empty tactic")
        if not all(type(x) is int and x in (-1, 0, 1) for x in self.labels):
            raise ValueError("labels must be -1, 0, or 1")
        if self.evidence not in ("lean_kernel_receipt_replay", "synthetic_fixture"):
            raise ValueError("unsupported evidence class")
        return self


def _filename(directory: Path, entry: str):
    if not isinstance(entry, str) or not entry or entry != Path(entry).name:
        raise ValueError("pack manifest may reference only files within its directory")
    return directory / entry


def _rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_checked_pack(directory: Path, split: str, max_cases: int = 2000):
    """Verify pack and each declared tactic receipt before extracting examples.

    The CLI only updates fast weights on the train split. Eval splits are
    consumed as scoring-only data, with root/lineage overlap checks elsewhere.
    """
    if not 1 <= max_cases <= 20000:
        raise ValueError("max_cases outside supported bounded range")
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != PACK_SCHEMA or split not in manifest.get("splits", {}):
        raise ValueError("expected a DataGoblin checked-candidate pack and a listed split")
    info = manifest["splits"][split]
    public_path = _filename(directory, info["public"])
    private_path = _filename(directory, info["receipts"])
    if sha256_file(public_path) != info["public_sha256"] or sha256_file(private_path) != info["receipts_sha256"]:
        raise ValueError("checked pack bytes have changed since the manifest")
    cases = []
    seen_roots = set()
    count = 0
    for public, private in itertools.zip_longest(_rows(public_path), _rows(private_path)):
        if public is None or private is None:
            raise ValueError("public and receipt rows differ in count")
        count += 1
        if public.get("id") != private.get("id") or public.get("epoch") != private.get("epoch"):
            raise ValueError("public/receipt identities differ")
        root = public["id"]
        if root in seen_roots:
            raise ValueError("duplicate goal root in one split")
        seen_roots.add(root)
        if count > max_cases:
            continue
        options = public["candidates"]
        receipts = private["outcomes"]
        if {item["id"] for item in options} != set(receipts) or len(set(item["id"] for item in options)) != len(options):
            raise ValueError("candidate receipt set and public pool differ")
        labels, tactics = [], []
        for item in options:
            if item.get("action") != "lean_tactic":
                raise ValueError("unsupported action kind")
            row = receipts[item["id"]]
            verdict = row.get("label")
            if verdict is not None and (type(verdict) is not int or verdict not in (0, 1)):
                raise ValueError("invalid verifier label")
            kernel = (row.get("closed") is True and row.get("kernel_ok") is True
                      and type(row.get("kernel_calls")) is int and row["kernel_calls"] == 1
                      and isinstance(row.get("axioms"), list)
                      and all(isinstance(x, str) and x in ALLOWED_AXIOMS for x in row["axioms"]))
            if (verdict == 1) != kernel:
                raise ValueError("inconsistent Lean closure receipt")
            tactics.append(item["tactic"])
            labels.append(-1 if verdict is None else verdict)
        statement = public.get("statement") or ""
        pretty = public.get("context") or ""
        if not isinstance(statement, str) or not isinstance(pretty, str):
            raise ValueError("goal text must be a string")
        if public.get("task") != "lean_candidate_proof":
            raise ValueError("expected a Lean candidate proof goal")
        cases.append(Case(
            root=root, lineage=public["lineage"], family=public["family"],
            split=split, goal=f"<STATEMENT> {statement}\n<GOALS> {pretty}",
            candidates=tuple(tactics), labels=tuple(labels),
            evidence="lean_kernel_receipt_replay",
        ).validate())
    if count != info["goals"]:
        raise ValueError("goal count differs from signed pack manifest")
    return cases, {"evidence_class": "lean_kernel_receipt_replay",
                   "scientific_eligible": True, "manifest_sha256": sha256_file(manifest_path),
                   "source_evidence_sha256": manifest.get("source_evidence_sha256"),
                   "environment": manifest.get("environment", {}),
                   "split": split, "rows": count, "loaded": len(cases)}


def make_fixture(directory: Path, *, per_family: int = 24, seed: int = 20261121):
    """Generate deterministic fake labels. Never call these kernel-verified."""
    if not 3 <= per_family <= 512 or directory.exists():
        raise ValueError("fresh fixture directory and 3..512 cases per family required")
    directory.mkdir(parents=True)
    rng = random.Random(seed)
    files = {}
    for split in ("train", "calibration"):
        target = directory / f"{split}.jsonl"
        count = 0
        with target.open("x", encoding="utf-8") as stream:
            for family, lemma in (("nat_add", "Nat.add_zero"), ("list_append", "List.append_nil"),
                                  ("bool_and", "Bool.and_true")):
                for n in range(per_family if split == "train" else max(3, per_family // 3)):
                    token = f"{family}_{split}_{n:04d}"
                    options = ["rfl", f"exact {lemma}", "simp", "omega"]
                    rng.shuffle(options)
                    winner = f"exact {lemma}" if n % 7 else None
                    labels = [int(option == winner) for option in options]
                    row = {
                        "root": digest(token), "lineage": digest(["fixture", token]),
                        "family": family, "split": split,
                        "goal": f"SYNTHETIC FIXTURE (NO LEAN): prove {lemma} for {n} in {family}",
                        "candidates": options, "labels": labels,
                        "evidence": "synthetic_fixture",
                    }
                    stream.write(canonical(row) + "\n")
                    count += 1
        files[split] = {"file": target.name, "sha256": sha256_file(target), "rows": count}
    manifest = {"schema": FIXTURE_SCHEMA, "source_kind": "synthetic_fixture",
                "scientific_eligible": False, "student_training_eligible": False,
                "seed": seed, "files": files}
    (directory / "manifest.json").write_text(canonical(manifest) + "\n", encoding="utf-8")
    return manifest


def load_fixture(directory: Path, split: str, max_cases: int = 2000):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != FIXTURE_SCHEMA or split not in manifest.get("files", {}):
        raise ValueError("missing PPT synthetic fixture manifest or split")
    info = manifest["files"][split]
    path = _filename(directory, info["file"])
    if sha256_file(path) != info["sha256"]:
        raise ValueError("fixture bytes have changed")
    cases = [Case(**row).validate() for row in itertools.islice(_rows(path), max_cases)]
    if any(c.split != split or c.evidence != "synthetic_fixture" for c in cases):
        raise ValueError("unexpected split or evidence in fixture")
    return cases, {"evidence_class": "synthetic_fixture", "scientific_eligible": False,
                   "manifest_sha256": sha256_file(manifest_path),
                   "split": split, "loaded": len(cases), "rows": info["rows"]}


def ensure_no_overlap(train: list[Case], query: list[Case]):
    roots = {c.root for c in train}
    lineages = {c.lineage for c in train}
    if any(c.root in roots or c.lineage in lineages for c in query):
        raise ValueError("query lineage overlaps support/training")

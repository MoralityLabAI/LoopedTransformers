"""Independently compile an original MathCan TRAIN witness in Lean 4.35."""
import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

from ppt_gym.ram_limits import TreeBudget

ARCHIVE_SHA = "a2ac363547858871c4e066c96bac67253dfeb7ff4e2a0050bb5886c8ca9edcfc"
TRAIN_COMMIT = "c84e2025334e9a967c1297425a490f98d5b3d189"
MODULE = "MathCan/FiniteHorizonGap.lean"
THEOREM = "epsilon_sixth_equiv"
SCRIPT = ("constructor", "omega", "intro", "omega")
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}


def sha(content):
    return hashlib.sha256(content).hexdigest()


def audit_axioms(stdout):
    name = re.escape("MathCan.FiniteHorizonGap." + THEOREM)
    inventory = re.findall(rf"^'{name}' (does not depend on any axioms|depends on axioms: \[[^\]]*\])$",
                           stdout, flags=re.MULTILINE)
    if len(inventory) != 1:
        raise ValueError("expected one fully qualified MathCan axiom inventory")
    axioms = (sorted(part.strip() for part in inventory[0].split("[", 1)[1][:-1].split(","))
              if "[" in inventory[0] else [])
    if not set(axioms) <= ALLOWED_AXIOMS:
        raise ValueError(f"unexpected MathCan theorem axioms: {axioms}")
    return axioms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("original", type=Path)
    parser.add_argument("lean", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if sha(args.archive.read_bytes()) != ARCHIVE_SHA:
        raise ValueError("unexpected original training archive")
    with zipfile.ZipFile(args.archive) as source:
        transitions = (json.loads(line) for line in source.read("data/transitions.jsonl").splitlines())
        witness = next(row for row in transitions if
                       row["source_file"] == MODULE and row["theorem"] == THEOREM
                       and row["development"] is False and (*row["prefix"], row["script"]) == SCRIPT
                       and row["after"]["accepted"] and row["after"]["goals"] == [])
    original = args.original.read_bytes()
    if sha(original) != witness["source_sha256"]:
        raise ValueError("source module is not the original MathCan TRAIN blob")
    start = original.index(f"theorem {THEOREM} ".encode())
    body = original.index(b":= by\n", start) + len(b":= by\n")
    end = original.index(b"\n\n", body)
    replacement = "\n".join("  " + tactic for tactic in SCRIPT).encode("ascii")
    if original[body:end] != b"  omega":
        raise ValueError("unexpected original target proof boundary")
    revised = original[:body] + replacement + original[end:]
    budget = TreeBudget(6, 2)
    try:
        lean = str(args.lean.resolve())
        version = subprocess.check_output([lean, "--version"], text=True, encoding="utf-8", timeout=15).strip()
        if "version 4.35.0" not in version:
            raise ValueError("expected pinned MathCan Lean 4.35.0")
        receipts = []
        with tempfile.TemporaryDirectory() as scratch:
            module = Path(scratch) / "Replay.lean"
            module.write_bytes(revised)
            for _ in range(2):
                result = subprocess.run([lean, str(module)], timeout=60, text=True, encoding="utf-8",
                                        capture_output=True, check=True)
                receipts.append({"returncode": result.returncode, "axioms": audit_axioms(result.stdout)})
        receipt = {"source_commit": TRAIN_COMMIT, "module": MODULE, "theorem": THEOREM,
                   "archive_sha256": ARCHIVE_SHA, "source_sha256": sha(original),
                   "replay_sha256": sha(revised), "steps": SCRIPT, "training_only": True,
                   "lean": version, "lean_sha256": sha(args.lean.read_bytes()), "compiles": receipts,
                   "memory": budget.stats()}
        args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    finally:
        budget.close()


if __name__ == "__main__":
    main()

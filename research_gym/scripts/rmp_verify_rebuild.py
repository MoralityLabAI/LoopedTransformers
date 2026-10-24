"""Check a rebuilt training cell against its committed run record, checkpoint by checkpoint.

The heavy artifact directory was lost on or before 2026-10-24 and rebuilt by retraining (training is
deterministic on this machine). A rebuilt cell is accepted only if every checkpoint in the committed
`result.json` exists in the rebuilt record with the same step and SHA-256, the file on disk hashes to it,
and the final status and step count agree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def verify(cell: str, committed_runs: Path, rebuilt_runs: Path, artifacts: Path) -> dict:
    committed = json.loads((committed_runs / cell / "result.json").read_text(encoding="utf-8"))
    rebuilt_path = rebuilt_runs / cell / "result.json"
    if not rebuilt_path.exists():
        return {"cell": cell, "ok": False, "problems": ["no rebuilt result.json"]}
    rebuilt = json.loads(rebuilt_path.read_text(encoding="utf-8"))
    problems = []
    for key in ("cell_hash", "status", "optimizer_steps"):
        if committed.get(key) != rebuilt.get(key):
            problems.append(f"{key}: committed {committed.get(key)!r}, rebuilt {rebuilt.get(key)!r}")
    found = {entry["label"]: entry for entry in rebuilt["checkpoints"]}
    for entry in committed["checkpoints"]:
        twin = found.get(entry["label"])
        if twin is None or (twin["step"], twin["sha256"]) != (entry["step"], entry["sha256"]):
            problems.append(f"checkpoint {entry['label']} differs or is missing")
            continue
        path = artifacts / cell / "checkpoints" / entry["path"]
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            problems.append(f"checkpoint file {entry['path']} missing or altered")
    extra = sorted(set(found) - {entry["label"] for entry in committed["checkpoints"]})
    if extra:
        problems.append(f"checkpoints not in the committed record: {extra}")
    return {"cell": cell, "ok": not problems, "checkpoints": len(committed["checkpoints"]), "problems": problems}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", nargs="+", required=True)
    parser.add_argument("--committed-runs", type=Path, default=Path("experiments/recurrence_monitorability_v0/runs"))
    parser.add_argument("--rebuilt-runs", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    outcomes = [verify(cell, args.committed_runs, args.rebuilt_runs, args.artifact_dir) for cell in args.cells]
    for outcome in outcomes:
        print(json.dumps(outcome), flush=True)
    if not all(outcome["ok"] for outcome in outcomes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

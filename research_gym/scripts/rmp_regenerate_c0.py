"""Regenerate a cell's step-0 checkpoint (`C0`) from its recorded config and check it against the recorded hash.

`C0` is the seeded initialization, built exactly as `research_gym.interp.training` builds it, so it can be
recreated when the heavy artifact directory is lost. A checkpoint whose hash does not match is removed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from research_gym.architecture_discovery.checkpoint import save_paced_checkpoint
from research_gym.interp.analysis import cell_config
from research_gym.interp.models import build_model


def regenerate(runs: Path, artifacts: Path, cell: str, device: torch.device) -> dict:
    result = json.loads((runs / cell / "result.json").read_text(encoding="utf-8"))
    recorded = next(entry for entry in result["checkpoints"] if entry["label"] == "C0")
    config = cell_config(runs, cell)
    torch.manual_seed(config.seed)
    model = build_model(
        config.arm, hidden_size=config.hidden_size, num_heads=config.num_heads, visits=config.visits, beta=config.beta
    ).to(device)
    path = artifacts / cell / "checkpoints" / "C0.pt"
    receipt = save_paced_checkpoint(
        {"cell_hash": result["cell_hash"], "label": "C0", "step": 0, "model": model.state_dict()}, path
    )
    match = receipt["sha256"] == recorded["sha256"]
    if not match:
        path.unlink()
        path.with_suffix(path.suffix + ".receipt.json").unlink()
    return {"cell": cell, "match": match, "sha256": receipt["sha256"], "recorded": recorded["sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", nargs="+", required=True)
    parser.add_argument("--runs", type=Path, default=Path("experiments/recurrence_monitorability_v0/runs"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    outcomes = [regenerate(args.runs, args.artifact_dir, cell, torch.device(args.device)) for cell in args.cells]
    for outcome in outcomes:
        print(json.dumps(outcome), flush=True)
    if not all(outcome["match"] for outcome in outcomes):
        raise SystemExit("some C0 checkpoints did not match their recorded hash")


if __name__ == "__main__":
    main()

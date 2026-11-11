"""Diagnostic: how many loop visits does each algorithmic step cost?

For a cell's latest checkpoint, reads the answer through the final norm and tied
unembedding after every visit (16 visits, so the trained 8 are followed by 8
extra), and reports per depth the first visit from which the answer is correct
for at least ``threshold`` of examples. CPU only, forward only.

Usage: python visit_cost.py <cell_id> <families...> [--per-depth N] [--threshold T]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from research_gym.interp.bundle import read_shard  # noqa: E402
from research_gym.interp.models import build_model  # noqa: E402
from research_gym.interp.training import _eval_tensors, _subset  # noqa: E402

ARTIFACTS = Path(r"C:\Users\patri\Documents\Codex\rmp_v0\artifacts")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cell")
    parser.add_argument("families", nargs="+")
    parser.add_argument("--per-depth", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--visits", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(3)
    saved = torch.load(ARTIFACTS / args.cell / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)
    arm = "-".join(args.cell.split("-")[:2])
    model = build_model(arm, beta=1.0)
    model.load_state_dict(saved["model"])
    model.eval()
    bundle = REPO / "experiments" / "recurrence_monitorability_v0" / "datasets"
    report = {"cell": args.cell, "step": saved["state"]["step"], "threshold": args.threshold, "families": {}}
    with torch.no_grad():
        for family in args.families:
            data = _subset(_eval_tensors(read_shard(bundle, "validation", family)), args.per_depth)
            states = []
            for start in range(0, data["tokens"].shape[0], 200):
                output = model(data["tokens"][start : start + 200], depth_visits=args.visits)
                states.append(torch.stack([model.lens(s[:, -1]).argmax(-1) for s in output.visit_outputs]))
            predictions = torch.cat(states, dim=1)  # [visits, examples]
            correct = (predictions == data["targets"]).float()
            rows = {}
            for depth in sorted(set(data["depths"].tolist())):
                mask = data["depths"] == depth
                by_visit = [round(float(correct[v][mask].mean()), 3) for v in range(args.visits)]
                first = next((v + 1 for v, value in enumerate(by_visit) if value >= args.threshold), None)
                rows[str(depth)] = {"first_visit_at_threshold": first, "accuracy_by_visit": by_visit}
                print(f"{family:<20} depth {depth}: first visit >= {args.threshold}: {first}  "
                      + " ".join(f"{value:.2f}" for value in by_visit), flush=True)
            report["families"][family] = rows
    out = REPO / "experiments" / "recurrence_monitorability_v0" / "diagnostics" / f"visit_cost_{args.cell}.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()

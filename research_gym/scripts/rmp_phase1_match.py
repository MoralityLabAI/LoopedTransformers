"""Choose the accuracy-matched LOOP-T checkpoint for Phase 1 by the rule in docs/rmp_phase1_plan.md."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from research_gym.interp.analysis import evaluate, load_analysis_model
from research_gym.interp.bundle import read_shard


FAMILIES = ("pointer_chase", "gated_pointer_chase", "reachability", "sudoku4")
TARGET, TOLERANCE, FLOOR = 0.960, 0.010, 0.98


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("experiments/recurrence_monitorability_v0/runs"))
    parser.add_argument("--bundle-dir", type=Path, default=Path("experiments/recurrence_monitorability_v0/datasets"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("experiments/recurrence_monitorability_v0/phase1/matched_checkpoint.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    shards = {family: read_shard(args.bundle_dir, "validation", family) for family in FAMILIES}
    candidates = []
    for seed in (1, 2, 3):
        for cell, stage in ((f"LOOP-T-core4-s{seed}", "plain"), (f"LOOP-T-core4-ds-s{seed}", "deep_supervision")):
            directory = args.artifact_dir / cell / "checkpoints"
            for path in sorted(directory.glob("*.pt")):
                model = load_analysis_model(args.runs, args.artifact_dir, cell, path.stem, device=args.device)
                report = evaluate(model, shards)
                exact = {family: report[family]["exact"] for family in FAMILIES}
                others = min(value for family, value in exact.items() if family != "reachability")
                # A deep-supervision cell's step-0 checkpoints are copies of the plain stage's final weights.
                trained_in = stage if model.step > 0 or stage == "plain" else "plain"
                candidates.append({
                    "cell_id": cell, "label": path.stem, "seed": seed, "stage": trained_in, "step": model.step,
                    "exact": exact, "by_depth": {family: report[family]["by_depth"] for family in FAMILIES},
                    "qualifies": abs(exact["reachability"] - TARGET) <= TOLERANCE and others >= FLOOR,
                })
                print(cell, path.stem, model.step, {k: round(v, 4) for k, v in exact.items()}, flush=True)

    qualifying = [item for item in candidates if item["qualifies"]]
    qualifying.sort(key=lambda item: (
        item["stage"] != "deep_supervision", abs(item["exact"]["reachability"] - TARGET), item["seed"],
    ))
    chosen = qualifying[0] if qualifying else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "rule": {"target_reachability": TARGET, "tolerance": TOLERANCE, "other_families_floor": FLOOR,
                 "order": ["deep-supervision stage first", "smallest reachability distance", "lowest seed"]},
        "chosen": chosen, "qualifying": len(qualifying), "candidates": candidates,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print("chosen:", None if chosen is None else (chosen["cell_id"], chosen["label"], chosen["exact"]))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_gym.interp.phase0 import cell_summary, gate_summary


def _row(values) -> str:
    return "| " + " | ".join(str(value) for value in values) + " |"


def _share(value) -> str:
    return "-" if value is None else f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize RMP Phase 0 cells against gate G0.")
    parser.add_argument("--runs", type=Path, default=Path("experiments/recurrence_monitorability_v0/runs"))
    parser.add_argument(
        "--bundle-dir", type=Path, default=Path("experiments/recurrence_monitorability_v0/datasets")
    )
    parser.add_argument("--variant", required=True, help="summarize the cells of this variant")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    manifest = json.loads((args.bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    results = [
        json.loads(path.read_text(encoding="utf-8")) for path in sorted(args.runs.glob("*/result.json"))
    ]
    cells = [
        cell_summary(result, manifest) for result in results if result["config"]["variant"] == args.variant
    ]
    summary = {"variant": args.variant, "cells": cells, "gate": gate_summary(cells)}
    out = args.out or args.runs.parent / f"phase0_summary_{args.variant}.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

    print(_row(["Cell", "Status", "Steps", "C3 step", "Gate", "Learned"]))
    print(_row(["---"] * 6))
    for cell in cells:
        print(_row([cell["cell_id"], cell["status"], cell["optimizer_steps"], cell["c3_step"] or "-",
                    "pass" if cell["gate_passed"] else "fail", "yes" if cell["learned"] else "no"]))
    for cell in cells:
        print(f"\n{cell['cell_id']}")
        print(_row(["Family", "Train loss", "Target entropy", "Best constant", "Validation", "By depth", "OOD by depth"]))
        print(_row(["---"] * 7))
        for name, family in cell["families"].items():
            depths = " ".join(_share(v) for _, v in sorted((family["validation_by_depth"] or {}).items(), key=lambda kv: int(kv[0])))
            ood = " ".join(_share(v) for _, v in sorted((family["ood_by_depth"] or {}).items(), key=lambda kv: int(kv[0])))
            print(_row([name, _share(family["train_loss"]), _share(family["target_entropy_nats"]),
                        _share(family["best_constant"]), _share(family["validation_exact"]), depths, ood or "-"]))
    print("\n" + json.dumps(summary["gate"], sort_keys=True))


if __name__ == "__main__":
    main()

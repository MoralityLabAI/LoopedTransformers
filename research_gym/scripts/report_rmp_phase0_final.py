"""Resolve every gate-G0 cell across its ladder stages and print the Phase 0 report tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_gym.interp.phase0 import MINIMUM_SEEDS, STAGE_SUFFIXES, resolve_seed


ARMS = ("LOOP-T", "FF-U")
SEEDS = (1, 2, 3)
FAMILIES = ("pointer_chase", "gated_pointer_chase", "reachability", "sudoku4")


def _share(value) -> str:
    return "-" if value is None else f"{value:.3f}"


def _depths(by_depth) -> str:
    if not by_depth:
        return "-"
    return " ".join(_share(value) for _, value in sorted(by_depth.items(), key=lambda kv: int(kv[0])))


def _row(values) -> str:
    return "| " + " | ".join(str(value) for value in values) + " |"


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve RMP Phase 0 gate cells across ladder stages.")
    parser.add_argument("--runs", type=Path, default=Path("experiments/recurrence_monitorability_v0/runs"))
    parser.add_argument(
        "--bundle-dir", type=Path, default=Path("experiments/recurrence_monitorability_v0/datasets")
    )
    parser.add_argument("--variant", default="core4")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    manifest = json.loads((args.bundle_dir / "manifest.json").read_text(encoding="utf-8"))

    seeds = []
    for arm in ARMS:
        for seed in SEEDS:
            stages = {}
            for suffix in STAGE_SUFFIXES:
                path = args.runs / f"{arm}-{args.variant}{suffix}-s{seed}" / "result.json"
                stages[suffix] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
            seeds.append(resolve_seed(arm, seed, stages, manifest))

    arms = {}
    for arm in ARMS:
        verdicts = {entry["seed"]: entry["verdict"] for entry in seeds if entry["arm"] == arm}
        passing = [seed for seed, verdict in verdicts.items() if verdict == "passed"]
        failed = [seed for seed, verdict in verdicts.items() if verdict == "failed"]
        arms[arm] = {
            "passing_seeds": passing,
            "failed_seeds": failed,
            "g0_passed": len(passing) >= MINIMUM_SEEDS,
            # Decided once three seeds pass, or once too few undecided seeds remain to reach three.
            "decided": len(passing) >= MINIMUM_SEEDS or len(SEEDS) - len(failed) < MINIMUM_SEEDS,
        }
    for entry in seeds:
        # A seed never started in an arm that is already decided cannot change the outcome.
        if entry["verdict"] == "not_started" and arms[entry["arm"]]["decided"]:
            entry["verdict"] = "not_run_arm_decided"
    resolved = all(entry["verdict"] in ("passed", "failed", "not_run_arm_decided") for entry in seeds)
    summary = {
        "variant": args.variant,
        "all_cells_resolved": resolved,
        "g0_passed": all(value["g0_passed"] for value in arms.values()),
        "arms": arms,
        "seeds": seeds,
    }
    out = args.out or args.runs.parent / f"phase0_final_{args.variant}.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

    print(_row(["Arm", "Seed", "Verdict", "Stages (steps, status)"]))
    print(_row(["---"] * 4))
    for entry in seeds:
        stages = "; ".join(
            f"{stage['stage']} {stage['optimizer_steps']:,} {stage['status']}" for stage in entry["stages"]
        )
        print(_row([entry["arm"], entry["seed"], entry["verdict"], stages or "-"]))

    for entry in seeds:
        cell = entry["deciding"]
        if cell is None:
            continue
        print(f"\n{entry['arm']} seed {entry['seed']}: {entry['verdict']} "
              f"(deciding stage {cell['stage']}, {cell['cell_id']}, step {cell['optimizer_steps']:,})")
        print(_row(["Family", "Validation", "Depths 1-8", "Depths 9-12", "Train loss", "Target entropy", "Best constant"]))
        print(_row(["---"] * 7))
        for name in FAMILIES:
            family = cell["families"].get(name)
            if family is None:
                continue
            print(_row([
                name,
                _share(family["validation_exact"]),
                _depths(family["validation_by_depth"]),
                _depths(family["ood_by_depth"]),
                _share(family["train_loss"]),
                _share(family["target_entropy_nats"]),
                _share(family["best_constant"]),
            ]))
    print("\n" + json.dumps({key: summary[key] for key in ("all_cells_resolved", "g0_passed", "arms")}, sort_keys=True))


if __name__ == "__main__":
    main()

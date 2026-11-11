"""Phase 2, E4: feature-variable association for the Phase 1 trace variables, by select-then-confirm.

Each trace variable is tested at every algorithmic step it has, on the examples deep enough to have it:
`cur_node(s)`, `latch(s)`, `goal_found(s)` and the queried cell's candidates. A binary variable is one test; a
categorical one is one test per class against the rest, for classes with at least `--min-class` examples on
each side in both halves. The pipeline picks the (site, feature) with the largest |AUROC - 0.5| on the
selection half of `monitor_calibration` and "finds" the variable if that feature reaches the plan's bar on the
reporting half (docs/rmp_phase2_plan.md). Run on the trained models for E4, on the `C0` models for control 1,
and with `--kind gauss` (structure-free dictionaries) for control 2.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from research_gym.interp.association import answer_codes, select_then_confirm
from research_gym.interp.dictionaries import load_dictionary
from research_gym.interp.phase2 import CORE, adjacent_width, core_examples, load_model, select_config, selection_half
from research_gym.interp.probes import trace_labels

MAIN = ["LOOP-T s1", "LOOP-T s2", "LOOP-T s3", "FF-U s3"]


def binary_tests(labels: np.ndarray, select: np.ndarray, min_class: int) -> list[tuple[int | None, np.ndarray]]:
    """(class, binary target) pairs worth testing; ``labels`` is -1 where a row has no such step."""

    valid = labels >= 0
    classes = np.unique(labels[valid])
    if len(classes) < 2:
        return []
    candidates = [(None, labels == classes[-1])] if len(classes) == 2 else [(int(c), labels == c) for c in classes]
    tests = []
    for value, target in candidates:
        counts = [int((target & valid & half).sum()) for half in (select, ~select)]
        others = [int((~target & valid & half).sum()) for half in (select, ~select)]
        if min(counts + others) >= min_class:
            tests.append((value, target))
    return tests


def associate(model, dictionary, calibration: dict, half: np.ndarray, *, rounds: int, min_class: int,
              device: torch.device) -> list[dict]:
    tests = []
    for family in CORE:
        index = np.flatnonzero(calibration["families"] == family)
        rows = [calibration["rows"][i] for i in index]
        codes = answer_codes(model, dictionary, calibration["tokens"][torch.as_tensor(index)])
        select = half[index]
        for variable, by_step in trace_labels(family, rows).items():
            for step, labels in sorted(by_step.items()):
                valid = labels >= 0
                for value, target in binary_tests(labels, select, min_class):
                    pick, report = valid & select, valid & ~select
                    outcome = select_then_confirm(codes[torch.as_tensor(pick)], target[pick].astype(int),
                                                  codes[torch.as_tensor(report)], target[report].astype(int),
                                                  rounds=rounds, device=device)
                    tests.append({"family": family, "variable": variable, "step": int(step), "class": value,
                                  "n_select": int(pick.sum()), "n_report": int(report.sum()), **outcome})
    return tests


def summarize(tests: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for test in tests:
        groups[f"{test['family']}/{test['variable']}"].append(test)
    summary = {}
    for key, members in sorted(groups.items()):
        found = [test for test in members if test["found"]]
        summary[key] = {
            "tests": len(members), "found": len(found), "found_fraction": len(found) / len(members),
            "mean_report_deviation": float(np.mean([abs(test["report_auroc"] - 0.5) for test in members])),
            "found_by_step": {str(step): sum(t["found"] for t in members if t["step"] == step)
                              for step in sorted({t["step"] for t in members})},
            "found_sites_by_step": {str(step): sorted(t["site"] for t in found if t["step"] == step)
                                    for step in sorted({t["step"] for t in found})},
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=MAIN + [f"{name} C0" for name in MAIN])
    parser.add_argument("--kind", default="shared", choices=("shared", "gauss"),
                        help="gauss: control 2's dictionary at the model's selected D-shared size")
    parser.add_argument("--adjacent", action="store_true",
                        help="control 4: the D-shared dictionary at the adjacent width, same k (key '<model> | adjacent')")
    parser.add_argument("--runs", type=Path, default=Path("experiments/recurrence_monitorability_v0/runs"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--monitor-bundle", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, default=Path("experiments/recurrence_monitorability_v0/phase2/dictionaries.json"))
    parser.add_argument("--out", type=Path, default=Path("experiments/recurrence_monitorability_v0/phase2/association.json"))
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--min-class", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    calibration = core_examples(args.monitor_bundle, "monitor_calibration")
    half = selection_half(calibration["groups"])
    report = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    for name in args.models:
        if name not in metrics:
            print("no dictionaries yet, skipped:", name, flush=True)
            continue
        chosen, met = select_config(metrics[name])
        parent = name.removesuffix(" C0")
        if parent != name and parent in metrics:
            # Control 1: the selection rule is undefined on a random model; use its trained model's size.
            chosen, met = select_config(metrics[parent])[0], None
        if args.adjacent:
            width = int(chosen.split("_w")[1].split("_")[0])
            chosen = chosen.replace(f"_w{width}_", f"_w{adjacent_width(width)}_", 1)
            if chosen not in metrics[name]:
                print("no adjacent dictionary, skipped:", name, chosen, flush=True)
                continue
        if args.kind == "gauss":
            chosen = chosen.replace("shared_", "gauss_", 1)
            if chosen not in metrics[name]:
                print("no gauss dictionary yet, skipped:", name, chosen, flush=True)
                continue
        dictionary, _ = load_dictionary(Path(metrics[name][chosen]["path"]), device=device)
        model = load_model(name, args.runs, args.artifact_dir, device)
        tests = associate(model, dictionary, calibration, half, rounds=args.rounds, min_class=args.min_class,
                          device=device)
        key = name if args.kind == "shared" else f"{name} | gauss"
        key = f"{key} | adjacent" if args.adjacent else key
        report[key] = {"dictionary": chosen, "selection_bar_met": met, "summary": summarize(tests), "tests": tests}
        print(name, args.kind, chosen, json.dumps({k: (v["found"], v["tests"]) for k, v in report[key]["summary"].items()}),
              flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()

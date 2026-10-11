"""Print the Phase 1 report's E2 tables from experiments/recurrence_monitorability_v0/phase1/e2_probes.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TRAINED = ("LOOP-T s1", "LOOP-T s2", "LOOP-T s3", "FF-U s1 (below gate)", "FF-U s3", "LOOP-T s1 matched")


def _row(values) -> str:
    return "| " + " | ".join(str(value) for value in values) + " |"


def _constant(result: dict) -> bool:
    """The reporting half's label never varies: every probe, the floor included, scores 1."""

    return min(result["accuracy"]) >= 0.999 and min(result.get("floor", [1.0])) >= 0.999


def _steps(probes: dict, variable: str) -> list[tuple[int, dict]]:
    items = [(int(key.split("@")[1]), result) for key, result in probes.items() if key.split("@")[0] == variable]
    return sorted((step, result) for step, result in items if not _constant(result))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes", type=Path, default=Path("experiments/recurrence_monitorability_v0/phase1/e2_probes.json"))
    args = parser.parse_args()
    models = json.loads(args.probes.read_text(encoding="utf-8"))["models"]
    present = [name for name in TRAINED if name in models]

    # Where each step lives: the site of peak accuracy for cur_node, and for the monotone flags the first
    # site within 0.01 of their peak. Site k > 0 is the state after visit k - 1.
    print("\nStep-to-site alignment (site of peak `cur_node(s)` accuracy; first site within 0.01 of the peak for `latch(s)` and `goal_found(s)`; peak in brackets)\n")
    print(_row(["Model", "Variable", *[f"s={step}" for step in range(1, 9)]]))
    print(_row(["---"] * 10))
    for name in present:
        for family, variable in (("pointer_chase", "cur_node"), ("gated_pointer_chase", "cur_node"),
                                 ("gated_pointer_chase", "latch"), ("reachability", "goal_found")):
            cells = {}
            for step, result in _steps(models[name]["families"][family], variable):
                accuracy = np.array(result["accuracy"])
                if variable == "cur_node":
                    cells[step] = f"{int(accuracy.argmax())} ({accuracy.max():.2f})"
                else:
                    first = int(np.flatnonzero(accuracy >= accuracy.max() - 0.01)[0])
                    cells[step] = f"{first} ({accuracy.max():.2f})"
            print(_row([name, f"`{family}` {variable}", *[cells.get(step, "-") for step in range(1, 9)]]))

    for name in present:
        print(f"\n### {name}\n")
        print(_row(["Probe", "Sites 0-8", "Best baseline at peak", "Computed at", "MLP peak", "Code ratio, min"]))
        print(_row(["---"] * 6))
        for family, probes in models[name]["families"].items():
            for key, result in probes.items():
                if _constant(result):
                    continue
                accuracy = np.array(result["accuracy"])
                peak = int(accuracy.argmax())
                computed = [site for site, flag in enumerate(result.get("computed", [])) if flag]
                missing = result.get("missing_baselines", [])
                ratio = [value for value in result.get("code_ratio", []) if value == value]
                print(_row([
                    f"`{family}` {key}",
                    " ".join(f"{value:.2f}" for value in accuracy),
                    f"{result['best_baseline'][peak]:.2f}" if "best_baseline" in result else "-",
                    (", ".join(str(site) for site in computed) or "none") + (f" (without {', '.join(missing)})" if missing else ""),
                    f"{max(result['mlp']):.2f}" if "mlp" in result else "-",
                    f"{min(ratio):.2f}" if ratio else "-",
                ]))


if __name__ == "__main__":
    main()

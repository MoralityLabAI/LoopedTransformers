"""Use certified TRAIN-only FC-12 incidence to propose BPD Lean probes.

Missing action-family cells are UNKNOWN. A verified positive is evidence
for one feasible edge, not an assertion that other scripts or tactics fail.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.refutation_transfer.tactic_fiber_balance import (
    Fibre, hall_deficit, critical_new_positive_edges,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "experiments/refutation_transfer/data/tfh01_fc12_verified_incidence.json"


def plan(data: dict, *, cap: int = 40) -> dict:
    fibre = Fibre.from_dict(data)
    if type(cap) is not int or cap <= 0:
        raise ValueError("positive cap required")
    known = sum(len(row) for row in fibre.legal)
    potential = len(fibre.states) * len(fibre.families)
    deficit = hall_deficit(fibre)
    critical = critical_new_positive_edges(fibre)
    counts = {name: sum(j in row for row in fibre.legal)
              for j, name in enumerate(fibre.families)}
    prioritized = []
    for row in critical.get("candidate_edges", []):
        # TFH-01's candidate listing is advisory, NOT a successful compiler probe.
        prioritized.append({"state_id": row["state_id"], "family": row["family"],
                            "reason": "hypothetically closes every currently maximal Hall cut",
                            "status": "unknown_requires_original_lean_compile"})
    if not prioritized:
        for i, state in enumerate(fibre.states):
            for j, family in enumerate(fibre.families):
                if j not in fibre.legal[i]:
                    prioritized.append({"state_id": state, "family": family,
                        "reason": "unknown action family; positive coverage exploration",
                        "status": "unknown_requires_original_lean_compile"})
    return {
        "schema": "BPD-LG-01-TFH-v1",
        "data_split": "train",
        "states": len(fibre.states),
        "families": list(fibre.families),
        "known_positive_edges": known,
        "unknown_action_family_cells": potential - known,
        "verified_negative_cells": 0,
        "per_family_known_positives": counts,
        "hall_lower_bound_current_certified_edges": deficit,
        "structurally_critical_edge_count": critical["structurally_critical_missing_positive_edges"],
        "candidate_probes": prioritized[:cap],
        "kernel_statement": "Existing positive edges were certified upstream; no new Lean compiler action was run by this planner",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incidence", type=Path, default=DEFAULT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cap", type=int, default=40)
    args = parser.parse_args()
    data = json.loads(args.incidence.read_text())
    result = plan(data, cap=args.cap)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("states", "known_positive_edges",
           "unknown_action_family_cells", "structurally_critical_edge_count")}, sort_keys=True))


if __name__ == "__main__":
    main()

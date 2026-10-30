"""Phase 2, audit of the query-block patch: does it transfer the intermediate node or just copy the start?

The query block includes the source's start node. A patch that only copied the source start, letting the
model recompute the chase on the base graph, predicts y_start = f_B^K(a_S). Node transfer predicts
y_node = f_B^(K-s)(f_S^s(a_S)).

For each model's best early-hop cell (step 1 or 2, before the answer forms), this regenerates the exact
evaluation pairs of `rmp_phase2_query_patch` (same seeds, same sampling loop) and reports:
- the collision rate, i.e. the share of pairs where y_node == y_start;
- IIA for y_node on the non-colliding pairs;
- the rate at which the model instead outputs y_start on those pairs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import research_gym.scripts.rmp_phase2_n2 as n2
from research_gym.interp.phase2 import core_examples, load_model
from research_gym.interp.tasks import POINTER_TARGET_BASE, START_MARKER
from research_gym.scripts.rmp_phase2_n1 import DEEP
from research_gym.scripts.rmp_phase2_query_patch import block_states, patched

_offset = n2.offset
n2.offset = lambda abstraction, visit: int(abstraction[4:]) if abstraction.startswith("step") else _offset(abstraction, visit)
ORIGINAL = {"LOOP-T s1", "LOOP-T s2", "LOOP-T s3", "FF-U s3"}


def walk(mapping, node: int, hops: int) -> int:
    for _ in range(hops):
        node = int(mapping[node])
    return node


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["LOOP-T s1", "LOOP-T s2", "LOOP-T s3", "FF-U s3", "LOOP-T s4",
                                                         "LOOP-T s5", "LOOP-T s6", "FF-U s4", "FF-U s6"])
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, default=Path("experiments/recurrence_monitorability_v0/datasets"))
    parser.add_argument("--results", type=Path, default=Path("experiments/recurrence_monitorability_v0/phase2"))
    parser.add_argument("--out", type=Path, default=Path("experiments/recurrence_monitorability_v0/phase2/query_collision.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    validation = core_examples(args.bundle_dir, "validation", families=("pointer_chase",))
    rows = validation["rows"]
    start = list(rows[0].tokens).index(START_MARKER)
    positions = [start, start + 1, start + 2, start + 3, len(rows[0].tokens) - 1]
    by_hops = {}
    for i, row in enumerate(rows):
        by_hops.setdefault(int(row.semantic["hops"]), []).append(i)
    deep_pool = [i for h in DEEP for i in by_hops.get(h, [])]
    report = {}
    for name in args.models:
        runs = Path("experiments/recurrence_monitorability_v0/runs" if name in ORIGINAL else "experiments/recurrence_monitorability_v0/phase3/runs")
        table = json.loads((args.results / f"query_patch_{name.replace(' ', '_')}.json").read_text(encoding="utf-8"))[name]["pointer_chase"]
        cells = [(table[str(v)][str(s)], v, s) for v in range(1, 9) for s in (1, 2)
                 if str(s) in table[str(v)] and table[str(v)]["source_answer"] <= 0.2]
        _, visit, step = max(cells)
        model = load_model(name, runs, args.artifact_dir, device)
        rng = np.random.default_rng(1000 * visit + step)  # the exact sampling of rmp_phase2_query_patch
        bases, sources, nodes, starts = [], [], [], []
        for _ in range(20000):
            b = int(rng.choice(deep_pool))
            s = int(rng.choice(by_hops[int(rows[b].semantic["hops"])]))
            if s == b:
                continue
            target = n2.counterfactual(rows[b], rows[s], f"step{step}", visit)
            if target is None or target == rows[b].target_token:
                continue
            hops = int(rows[b].semantic["hops"])
            bases.append(b), sources.append(s), nodes.append(target)
            starts.append(POINTER_TARGET_BASE + walk(rows[b].semantic["mapping"], int(rows[s].semantic["start"]), hops))
            if len(bases) == 1000:
                break
        source = block_states(model, torch.tensor([rows[i].tokens for i in sources], device=device), visit, positions)
        prediction = patched(model, torch.tensor([rows[i].tokens for i in bases], device=device), source, visit, positions).cpu().numpy()
        nodes, starts = np.array(nodes), np.array(starts)
        collide = nodes == starts
        apart = ~collide
        report[name] = {"visit": visit, "step": step, "pairs": int(len(nodes)), "collision_rate": float(collide.mean()),
                        "iia_all": float((prediction == nodes).mean()),
                        "iia_noncolliding": float((prediction[apart] == nodes[apart]).mean()),
                        "start_copy_rate_noncolliding": float((prediction[apart] == starts[apart]).mean())}
        print(name, json.dumps(report[name]), flush=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()

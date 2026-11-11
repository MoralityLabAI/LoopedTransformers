"""Phase 2, E6, post hoc: is the chase's intermediate node carried by the query block rather than the answer
position alone?

At the answer position, Phase 1's probes decode `cur_node(1)` and `cur_node(2)` at 0.76 to 1.00, but
patching that position alone transfers at most 0.22 of counterfactuals mid-loop (`n2_schedule_*.json`).

- **Patch.** This patches the four query tokens (start marker, start node, hops marker, hop count) and the answer position
  at visit `v` from a source example with the same number of hops. The query then differs only in its start
  node.
- **Scoring.** Each visit and step `s` is scored against the counterfactual that continues the base graph
  from the source's `cur_node(s)`.
- **Control.** The source's own answer (the counterfactual "the whole query was swapped") is scored too.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import research_gym.scripts.rmp_phase2_n2 as n2
from research_gym.interp.tasks import START_MARKER
from research_gym.interp.phase2 import core_examples, load_model
from research_gym.scripts.rmp_phase2_n1 import DEEP

_offset = n2.offset
n2.offset = lambda abstraction, visit: int(abstraction[4:]) if abstraction.startswith("step") else _offset(abstraction, visit)


@torch.no_grad()
def block_states(model, tokens, visit, positions):
    captured = {}

    def hook(context):
        if context.visit == visit - 1:
            captured["state"] = context.after[:, positions].clone()
        return None

    model.run(tokens, hook)
    return captured["state"]


@torch.no_grad()
def patched(model, tokens, source, visit, positions):
    def hook(context):
        if context.visit != visit - 1:
            return None
        after = context.after.clone()
        after[:, positions] = source
        return after

    return model.run(tokens, hook).logits[:, -1].argmax(dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--runs", type=Path, default=Path("experiments/recurrence_monitorability_v0/runs"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, default=Path("experiments/recurrence_monitorability_v0/datasets"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = load_model(args.model, args.runs, args.artifact_dir, device)
    validation = core_examples(args.bundle_dir, "validation", families=n2.FAMILIES)
    report = {}
    for family in n2.FAMILIES:
        rows = [r for r, f in zip(validation["rows"], validation["families"]) if f == family]
        starts = {list(row.tokens).index(START_MARKER) for row in rows}
        assert len(starts) == 1, starts  # the query block sits at one fixed place per family
        start = starts.pop()
        positions = [start, start + 1, start + 2, start + 3, len(rows[0].tokens) - 1]
        by_hops = {}
        for i, row in enumerate(rows):
            by_hops.setdefault(int(row.semantic["hops"]), []).append(i)
        deep_pool = [i for h in DEEP for i in by_hops.get(h, [])]
        table = {"positions": positions}
        for visit in range(1, model.config.visits + 1):
            row_out = {}
            for step in list(range(1, 9)) + ["source_answer"]:
                rng = np.random.default_rng(1000 * visit + (0 if step == "source_answer" else step))
                bases, sources, targets = [], [], []
                for _ in range(20000):
                    b = int(rng.choice(deep_pool))
                    s = int(rng.choice(by_hops[int(rows[b].semantic["hops"])]))
                    if s == b:
                        continue
                    target = rows[s].target_token if step == "source_answer" else n2.counterfactual(rows[b], rows[s], f"step{step}", visit)
                    if target is None or target == rows[b].target_token:
                        continue
                    bases.append(b), sources.append(s), targets.append(target)
                    if len(bases) == 1000:
                        break
                if len(bases) < 100:
                    continue
                source = block_states(model, torch.tensor([rows[i].tokens for i in sources], device=device), visit, positions)
                predictions = patched(model, torch.tensor([rows[i].tokens for i in bases], device=device), source, visit, positions)
                row_out[str(step)] = float((predictions == torch.tensor(targets, device=device)).float().mean())
            table[str(visit)] = row_out
            steps = {k: v for k, v in row_out.items() if k != "source_answer"}
            best = max(steps, key=steps.get) if steps else None
            print(args.model, family, "v", visit, "best step", best, round(steps.get(best, 0), 3),
                  "source answer", round(row_out.get("source_answer", 0), 3), flush=True)
        report[family] = table
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({args.model: report}, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()

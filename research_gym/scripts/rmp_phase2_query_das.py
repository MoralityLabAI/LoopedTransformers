"""Phase 2, E6, post hoc: a low-rank subspace over the query block for the hop variable.

For each visit, the step with the best full query-block patch is taken from `query_patch_<model>.json`.

- **Fit.** A rank-`r` subspace of the flattened 5-position block (four query tokens plus the answer
  position) is fitted by distributed alignment search. Pairs come from the selection half of
  `monitor_calibration`, with the same hop count, as in `rmp_phase2_query_patch`.
- **Report.** IIA on `validation` pairs whose base is at depth 5 to 8, at ranks 8, 32 and 128.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import research_gym.scripts.rmp_phase2_n2 as n2
from research_gym.interp.phase2 import core_examples, load_model, selection_half
from research_gym.interp.tasks import START_MARKER
from research_gym.scripts.rmp_phase2_n1 import DEEP
from research_gym.scripts.rmp_phase2_query_patch import block_states

_offset = n2.offset
n2.offset = lambda abstraction, visit: int(abstraction[4:]) if abstraction.startswith("step") else _offset(abstraction, visit)


def pairs(rows, step, visit, rng, count, deep_only):
    by_hops = {}
    for i, row in enumerate(rows):
        by_hops.setdefault(int(row.semantic["hops"]), []).append(i)
    pool = [i for h, members in by_hops.items() if not deep_only or h in DEEP for i in members]
    bases, sources, targets = [], [], []
    for _ in range(count * 20):
        b = int(rng.choice(pool))
        s = int(rng.choice(by_hops[int(rows[b].semantic["hops"])]))
        target = None if s == b else n2.counterfactual(rows[b], rows[s], f"step{step}", visit)
        if target is None or target == rows[b].target_token:
            continue
        bases.append(b), sources.append(s), targets.append(target)
        if len(bases) == count:
            break
    return bases, sources, targets


def logits_with(model, tokens, source, visit, positions, q):
    def hook(context):
        if context.visit != visit - 1:
            return None
        after = context.after.clone()
        base = after[:, positions].reshape(len(tokens), -1)
        flat = source.reshape(len(tokens), -1)
        after[:, positions] = (base + ((flat - base) @ q) @ q.T).reshape(after[:, positions].shape)
        return after

    return model.run(tokens, hook).logits[:, -1].float()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--runs", type=Path, default=Path("experiments/recurrence_monitorability_v0/runs"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--monitor-bundle", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, default=Path("experiments/recurrence_monitorability_v0/datasets"))
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = load_model(args.model, args.runs, args.artifact_dir, device)
    for parameter in model.model.parameters():
        parameter.requires_grad_(False)
    scan = json.loads(args.scan.read_text(encoding="utf-8"))[args.model]
    calibration = core_examples(args.monitor_bundle, "monitor_calibration", families=n2.FAMILIES)
    half = selection_half(calibration["groups"])
    validation = core_examples(args.bundle_dir, "validation", families=n2.FAMILIES)
    report = {}
    for family in n2.FAMILIES:
        fit_rows = [r for r, h, f in zip(calibration["rows"], half, calibration["families"]) if h and f == family]
        test_rows = [r for r, f in zip(validation["rows"], validation["families"]) if f == family]
        start = list(test_rows[0].tokens).index(START_MARKER)
        positions = [start, start + 1, start + 2, start + 3, len(test_rows[0].tokens) - 1]
        tensor = lambda rows, index: torch.tensor([rows[i].tokens for i in index], device=device)  # noqa: E731
        out = {}
        for visit in range(1, model.config.visits + 1):
            cells = {k: v for k, v in scan[family][str(visit)].items() if k != "source_answer"}
            step = int(max(cells, key=cells.get))
            rng = np.random.default_rng(visit)
            b, s, t = pairs(fit_rows, step, visit, rng, 3000, deep_only=False)
            vb, vs, vt = pairs(test_rows, step, visit, rng, 1000, deep_only=True)
            fit_tokens, test_tokens = tensor(fit_rows, b), tensor(test_rows, vb)
            fit_source = block_states(model, tensor(fit_rows, s), visit, positions).reshape(len(b), -1)
            test_source = block_states(model, tensor(test_rows, vs), visit, positions).reshape(len(vb), -1)
            fit_targets, test_targets = torch.tensor(t, device=device), torch.tensor(vt, device=device)
            cell = {"step": step, "full_block_iia": cells[str(step)], "iia": {}}
            for rank in args.ranks:
                generator = torch.Generator().manual_seed(visit * 1000 + rank)
                weight = torch.randn(fit_source.shape[1], rank, generator=generator).to(device).requires_grad_()
                optimizer = torch.optim.Adam([weight], lr=1e-2)
                for _ in range(args.steps):
                    index = torch.randint(0, len(b), (64,), generator=generator).to(device)
                    q = torch.linalg.qr(weight).Q
                    loss = torch.nn.functional.cross_entropy(
                        logits_with(model, fit_tokens[index], fit_source[index], visit, positions, q), fit_targets[index])
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                with torch.no_grad():
                    q = torch.linalg.qr(weight.detach()).Q
                    predictions = logits_with(model, test_tokens, test_source, visit, positions, q).argmax(dim=-1)
                cell["iia"][str(rank)] = float((predictions == test_targets).float().mean())
            out[str(visit)] = cell
            print(args.model, family, "v", visit, json.dumps(cell), flush=True)
        report[family] = out
        args.out.write_text(json.dumps({args.model: report}, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()

"""Phase 2 successor experiment (registered 2026-10-31): do sparse features carry the named hop variable?

The rules are in docs/rmp_phase2_e5_e6_plan.md, "Successor experiment".

- **Cell and features.** On the selection half of `monitor_calibration` (`pointer_chase`), pick the
  query-block cell and rank `D-shared` features by their source-minus-base code difference.
- **Patching.** On the reporting half, patch only the top-`m` features' code differences at the five block
  positions.
- **Scoring.** Score against the `cur_node(s)` counterfactual, beside the following:
  - random alive features;
  - the all-features (reconstruction difference) patch;
  - the full-block patch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import research_gym.scripts.rmp_phase2_n2 as n2
from research_gym.interp.dictionaries import load_dictionary, rms
from research_gym.interp.phase2 import core_examples, load_model, select_config, selection_half
from research_gym.interp.tasks import POINTER_TARGET_BASE, START_MARKER
from research_gym.scripts.rmp_phase2_n1 import DEEP

_offset = n2.offset
n2.offset = lambda abstraction, visit: int(abstraction[4:]) if abstraction.startswith("step") else _offset(abstraction, visit)
ORIGINAL = {"LOOP-T s1", "LOOP-T s2", "LOOP-T s3", "FF-U s3"}
SIZES = (8, 16, 32, 64)


def walk(mapping, node: int, hops: int) -> int:
    for _ in range(hops):
        node = int(mapping[node])
    return node


def make_pairs(rows, step, visit, rng, count, deep_only):
    by_hops = {}
    for i, row in enumerate(rows):
        by_hops.setdefault(int(row.semantic["hops"]), []).append(i)
    pool = [i for h, members in by_hops.items() if (not deep_only or h in DEEP) for i in members]
    bases, sources, targets = [], [], []
    for _ in range(count * 30):
        b = int(rng.choice(pool))
        s = int(rng.choice(by_hops[int(rows[b].semantic["hops"])]))
        if s == b:
            continue
        target = n2.counterfactual(rows[b], rows[s], f"step{step}", visit)
        hops = int(rows[b].semantic["hops"])
        start_copy = POINTER_TARGET_BASE + walk(rows[b].semantic["mapping"], int(rows[s].semantic["start"]), hops)
        if target is None or target == rows[b].target_token or target == start_copy:
            continue
        bases.append(b), sources.append(s), targets.append(target)
        if len(bases) == count:
            break
    return bases, sources, targets


@torch.no_grad()
def states(model, tokens, visit, positions):
    captured = {}

    def hook(context):
        if context.visit == visit - 1:
            captured["s"] = context.after[:, positions].clone()
        return None

    model.run(tokens, hook)
    return captured["s"]


@torch.no_grad()
def run_with(model, tokens, visit, positions, delta):
    def hook(context):
        if context.visit != visit - 1:
            return None
        after = context.after.clone()
        after[:, positions] = after[:, positions] + delta
        return after

    return model.run(tokens, hook).logits[:, -1].argmax(dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["LOOP-T s1", "LOOP-T s2", "LOOP-T s3", "LOOP-T s4", "LOOP-T s5",
                                                         "LOOP-T s6", "FF-U s3", "FF-U s4", "FF-U s6"])
    parser.add_argument("--runs", type=Path, default=None, help="unused; run records are chosen per model")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--monitor-bundle", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, default=Path("experiments/recurrence_monitorability_v0/phase2/dictionaries.json"))
    parser.add_argument("--out", type=Path, default=Path("experiments/recurrence_monitorability_v0/phase2/feature_link.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    data = core_examples(args.monitor_bundle, "monitor_calibration", families=("pointer_chase",))
    half = selection_half(data["groups"])
    select_rows = [r for r, h in zip(data["rows"], half) if h]
    report_rows = [r for r, h in zip(data["rows"], half) if not h]
    start = list(select_rows[0].tokens).index(START_MARKER)
    positions = [start, start + 1, start + 2, start + 3, len(select_rows[0].tokens) - 1]
    report = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    tok = lambda rows, idx: torch.tensor([rows[i].tokens for i in idx], device=device)  # noqa: E731
    for name in args.models:
        if name in report:
            continue
        runs = Path("experiments/recurrence_monitorability_v0/runs" if name in ORIGINAL
                    else "experiments/recurrence_monitorability_v0/phase3/runs")
        model = load_model(name, runs, args.artifact_dir, device)
        if name in metrics:
            chosen, _ = select_config(metrics[name])
            shared = load_dictionary(Path(metrics[name][chosen]["path"]), device=device)[0]
        else:
            print("no D-shared dictionary yet, skipped:", name, flush=True)
            continue
        # Cell choice on the selection half: full-block patch, hop 1 or 2, visits 1-6, source-answer control <= 0.2.
        best = None
        for visit in range(1, 7):
            for step in (1, 2):
                rng = np.random.default_rng(10 * visit + step)
                b, s, t = make_pairs(select_rows, step, visit, rng, 500, deep_only=True)
                if len(t) < 100:
                    continue
                src = states(model, tok(select_rows, s), visit, positions)
                base = states(model, tok(select_rows, b), visit, positions)
                pred = run_with(model, tok(select_rows, b), visit, positions, src - base).cpu().numpy()
                iia = float((pred == np.array(t)).mean())
                control = float((pred == np.array([select_rows[i].target_token for i in s])).mean())
                if control <= 0.2 and (best is None or iia > best[0]):
                    best = (iia, visit, step)
        entry = {"dictionary": chosen, "selection_cell": {"visit": best[1], "step": best[2], "full_block_iia": best[0]}}
        _, visit, step = best
        # Feature ranking on 2,000 selection-half pairs at the cell.
        rng = np.random.default_rng(1)
        b, s, t = make_pairs(select_rows, step, visit, rng, 2000, deep_only=False)
        src, base = states(model, tok(select_rows, s), visit, positions), states(model, tok(select_rows, b), visit, positions)
        # Reporting-half pairs.
        rng = np.random.default_rng(2)
        rb, rs, rt = make_pairs(report_rows, step, visit, rng, 1000, deep_only=True)
        rsrc, rbase = states(model, tok(report_rows, rs), visit, positions), states(model, tok(report_rows, rb), visit, positions)
        targets = np.array(rt)
        score = lambda delta: float((run_with(model, tok(report_rows, rb), visit, positions, delta).cpu().numpy() == targets).mean())  # noqa: E731
        entry["pairs"] = len(rt)
        entry["full_block_iia"] = score(rsrc - rbase)
        if True:
            with torch.no_grad():
                codes = lambda x: shared.encode(x / rms(x))  # noqa: E731
                ranking = (codes(src) - codes(base)).abs().mean(dim=(0, 1))
                alive = torch.nonzero((codes(base) > 0).flatten(0, 1).any(dim=0)).flatten()
                cs, cb, scale = codes(rsrc), codes(rbase), rms(rbase)

                def feature_patch(index):
                    mask = torch.zeros(cs.shape[-1], device=device)
                    mask[index] = 1
                    return ((cs - cb) * mask) @ shared.decoder * scale

            entry["all_features_iia"] = score(feature_patch(torch.arange(cs.shape[-1], device=device)))
            generator = torch.Generator().manual_seed(0)
            entry["by_m"] = {}
            for m in SIZES:
                top = ranking.topk(m).indices
                random = alive[torch.randperm(len(alive), generator=generator)[:m].to(device)]
                entry["by_m"][str(m)] = {"selected": score(feature_patch(top)), "random": score(feature_patch(random))}
            eligible = entry["full_block_iia"] >= 0.6
            linked = eligible and any(v["selected"] >= 0.5 * entry["full_block_iia"] and v["selected"] - v["random"] >= 0.20
                                      for m, v in entry["by_m"].items() if int(m) <= 32)
            entry["eligible"], entry["linked"] = eligible, linked
        report[name] = entry
        print(name, json.dumps(entry), flush=True)
        args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()

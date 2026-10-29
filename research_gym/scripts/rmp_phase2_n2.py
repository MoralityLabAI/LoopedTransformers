"""Phase 2, E6: N2, abstraction mismatch, by distributed alignment search (docs/rmp_phase2_e5_e6_plan.md).

For each pointer family, abstraction, visit `v` and rank `r`:
- an orthonormal subspace `Q` (H x r) is fitted on pairs from the selection half of `monitor_calibration`;
- the answer position's `resid[v]` of a base example takes `Q Q^T (source - base)` from a source example;
- the model's answer is scored against the abstraction's counterfactual prediction.

Abstractions:
- `hop`: the variable after visit v is `cur_node(v)`, the node v hops from the start.
- `doubling`: it is the node `2^(v-1)` hops from the start.

Either way the counterfactual answer continues the base example's own graph from the source's node for the
remaining hops, including the gated family's locks. IIA (the share of answers that match the
counterfactual) is reported on `validation` pairs whose base is at depth 5 to 8, with two guards:
- the full-residual patch's IIA at the same site;
- the share of patched states whose norm exceeds the 99th percentile of clean norms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research_gym.interp.phase2 import core_examples, load_model, selection_half
from research_gym.interp.tasks import POINTER_TARGET_BASE, REFUSE_TOKEN
from research_gym.scripts.rmp_phase2_n1 import DEEP

FAMILIES = ("pointer_chase", "gated_pointer_chase")
RANKS = (1, 2, 4, 8)


def offset(abstraction: str, visit: int) -> int:
    return visit if abstraction == "hop" else 2 ** (visit - 1)


def node_at(row, hops: int) -> int:
    mapping, node = row.semantic["mapping"], int(row.semantic["start"])
    for _ in range(hops):
        node = int(mapping[node])
    return node


def counterfactual(base, source, abstraction: str, visit: int) -> int | None:
    """The answer token the abstraction predicts, or None when the pair does not define one."""

    k = offset(abstraction, visit)
    hops = int(base.semantic["hops"])
    if k > hops or k > int(source.semantic["hops"]):
        return None
    mapping = base.semantic["mapping"]
    prefix = [node_at(base, step) for step in range(k)]  # base sources before the swap
    path = [node_at(source, k)]
    for _ in range(hops - k):
        path.append(int(mapping[path[-1]]))
    if len(set(prefix + path[:-1])) != hops:
        return None  # the continued walk revisits a node; the task never poses that
    if base.family == "gated_pointer_chase":
        locks, grants = base.semantic["locks"], set(base.semantic["grants"]) | {0}
        if any(locks[node] not in grants for node in prefix + path[:-1]):
            return REFUSE_TOKEN
    return POINTER_TARGET_BASE + path[-1]


def make_pairs(rows, abstraction, visit, rng, count, deep_only) -> tuple[list[int], list[int], list[int]]:
    bases, sources, targets = [], [], []
    candidates = [i for i, row in enumerate(rows) if not deep_only or row.difficulty in DEEP]
    for _ in range(count * 20):
        b, s = rng.choice(candidates), rng.integers(len(rows))
        if b == s:
            continue
        target = counterfactual(rows[b], rows[s], abstraction, visit)
        if target is None or target == rows[b].target_token:
            continue
        bases.append(b), sources.append(s), targets.append(target)
        if len(bases) == count:
            break
    return bases, sources, targets


def states_at(model, tokens, visit):
    captured = {}

    def hook(context):
        if context.visit == visit - 1:
            captured["state"] = context.after[:, -1].detach()
        return None

    model.run(tokens, hook)
    return captured["state"]


def patched_logits(model, tokens, source_state, visit, q):
    def hook(context):
        if context.visit != visit - 1:
            return None
        after = context.after.clone()
        base = after[:, -1]
        difference = source_state - base
        after[:, -1] = base + (difference if q is None else (difference @ q) @ q.T)
        return after

    return model.run(tokens, hook).logits[:, -1].float()


def fit(model, tokens, source_states, targets, visit, rank, *, steps, seed) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(source_states.shape[1], rank, generator=generator).to(source_states.device).requires_grad_()
    optimizer = torch.optim.Adam([weight], lr=1e-2)
    for step in range(steps):
        index = torch.randint(0, len(targets), (64,), generator=generator).to(tokens.device)
        q = torch.linalg.qr(weight).Q
        logits = patched_logits(model, tokens[index], source_states[index], visit, q)
        loss = torch.nn.functional.cross_entropy(logits, targets[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return torch.linalg.qr(weight.detach()).Q


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--pairs", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--ranks", type=int, nargs="+", default=list(RANKS))
    parser.add_argument("--runs", type=Path, default=Path("experiments/recurrence_monitorability_v0/runs"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--monitor-bundle", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, default=Path("experiments/recurrence_monitorability_v0/datasets"))
    parser.add_argument("--out", type=Path, default=Path("experiments/recurrence_monitorability_v0/phase2/n2.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = load_model(args.model, args.runs, args.artifact_dir, device)
    for parameter in model.model.parameters():
        parameter.requires_grad_(False)
    calibration = core_examples(args.monitor_bundle, "monitor_calibration", families=FAMILIES)
    half = selection_half(calibration["groups"])
    validation = core_examples(args.bundle_dir, "validation", families=FAMILIES)
    report = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    out = report.setdefault(args.model, {"families": {}})
    for family in FAMILIES:
        fit_rows = [r for r, h, f in zip(calibration["rows"], half, calibration["families"]) if h and f == family]
        test_rows = [r for r, f in zip(validation["rows"], validation["families"]) if f == family]
        test_tokens_all = torch.tensor([r.tokens for r in test_rows], device=device)
        deep = [i for i, r in enumerate(test_rows) if r.difficulty in DEEP]
        with torch.no_grad():
            clean = model.run(test_tokens_all[deep]).logits[:, -1].argmax(dim=-1).cpu().numpy()
        entry = out["families"].setdefault(family, {"clean_deep_accuracy": float(np.mean(
            clean == np.array([test_rows[i].target_token for i in deep]))), "cells": {}})
        with torch.no_grad():
            clean_norms = states_at(model, test_tokens_all[deep], model.config.visits).norm(dim=-1)
        for abstraction in ("hop", "doubling"):
            for visit in range(1, model.config.visits + 1):
                key = f"{abstraction}/v{visit}"
                if key in entry["cells"]:
                    continue
                rng = np.random.default_rng(visit)
                b, s, t = make_pairs(fit_rows, abstraction, visit, rng, args.pairs, deep_only=False)
                vb, vs, vt = make_pairs(test_rows, abstraction, visit, rng, 1000, deep_only=True)
                if len(t) < 100 or len(vt) < 100:
                    entry["cells"][key] = {"status": "too_few_pairs", "fit_pairs": len(t), "test_pairs": len(vt)}
                    continue
                fit_tokens = torch.tensor([fit_rows[i].tokens for i in b], device=device)
                with torch.no_grad():
                    fit_sources = states_at(model, torch.tensor([fit_rows[i].tokens for i in s], device=device), visit)
                    test_tokens = torch.tensor([test_rows[i].tokens for i in vb], device=device)
                    test_sources = states_at(model, torch.tensor([test_rows[i].tokens for i in vs], device=device), visit)
                fit_targets = torch.tensor(t, device=device)
                test_targets = torch.tensor(vt, device=device)
                cell = {"fit_pairs": len(t), "test_pairs": len(vt), "iia": {}, "out_of_norm": {}}
                with torch.no_grad():
                    full = patched_logits(model, test_tokens, test_sources, visit, None)
                cell["full_patch_iia"] = float((full.argmax(dim=-1) == test_targets).float().mean())
                bound = float(torch.quantile(clean_norms, 0.99))
                for rank in args.ranks:
                    q = fit(model, fit_tokens, fit_sources, fit_targets, visit, rank, steps=args.steps, seed=visit * 10 + rank)
                    with torch.no_grad():
                        logits = patched_logits(model, test_tokens, test_sources, visit, q)
                        base = states_at(model, test_tokens, visit)
                        patched = base + ((test_sources - base) @ q) @ q.T
                    cell["iia"][str(rank)] = float((logits.argmax(dim=-1) == test_targets).float().mean())
                    cell["out_of_norm"][str(rank)] = float((patched.norm(dim=-1) > bound).float().mean())
                cell["best_iia"] = max(cell["iia"].values())
                entry["cells"][key] = cell
                print(args.model, family, key, json.dumps({k: round(v, 3) for k, v in cell["iia"].items()}),
                      "full", round(cell["full_patch_iia"], 3), flush=True)
                args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        best = {a: max((c.get("best_iia", 0.0) for k, c in entry["cells"].items() if k.startswith(a)), default=0.0)
                for a in ("hop", "doubling")}
        entry["best_iia"] = best
        # Guard (plan change 2026-10-29): only cells whose full-residual patch reaches 0.6 can carry the
        # variable at all; with none, the site is wrong and N2 is unresolved, not negative.
        usable = [c["best_iia"] for c in entry["cells"].values() if c.get("full_patch_iia", 0.0) >= 0.6]
        entry["usable_cells"] = len(usable)
        if entry["clean_deep_accuracy"] < 0.98:
            entry["n2"] = "not_applicable"
        elif not usable:
            entry["n2"] = "unresolved_site"
        else:
            entry["n2"] = "negative" if max(usable) < 0.6 else "not_negative"
        args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()

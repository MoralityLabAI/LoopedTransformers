"""Phase 2, E6, post hoc: which hop the answer position holds after each visit (a fitted schedule abstraction).

Phase 1's arrival matrices show neither one hop per visit nor doubling: a few hops early, then several per
visit. This scans the step instead of fixing it.

- **Scan.** For visit `v` and step `s`, the answer position's full `resid[v]` is patched from a source
  example. The patch is scored against the counterfactual that continues the base example's graph from the
  source's `cur_node(s)` (`rmp_phase2_n2.counterfactual` with offset `s`).
- **Fit.** For each visit, at the step with the highest full-patch IIA, a rank-32 subspace is fitted by
  distributed alignment search, exactly as in N2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import research_gym.scripts.rmp_phase2_n2 as n2
from research_gym.interp.phase2 import core_examples, load_model, selection_half

_offset = n2.offset
n2.offset = lambda abstraction, visit: int(abstraction[4:]) if abstraction.startswith("step") else _offset(abstraction, visit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--runs", type=Path, default=Path("experiments/recurrence_monitorability_v0/runs"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--monitor-bundle", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, default=Path("experiments/recurrence_monitorability_v0/datasets"))
    parser.add_argument("--out", type=Path, default=Path("experiments/recurrence_monitorability_v0/phase2/n2_schedule.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = load_model(args.model, args.runs, args.artifact_dir, device)
    for parameter in model.model.parameters():
        parameter.requires_grad_(False)
    calibration = core_examples(args.monitor_bundle, "monitor_calibration", families=n2.FAMILIES)
    half = selection_half(calibration["groups"])
    validation = core_examples(args.bundle_dir, "validation", families=n2.FAMILIES)
    report = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    out = report.setdefault(args.model, {})
    as_tokens = lambda rows, index: torch.tensor([rows[i].tokens for i in index], device=device)  # noqa: E731
    for family in n2.FAMILIES:
        if family in out:
            continue
        fit_rows = [r for r, h, f in zip(calibration["rows"], half, calibration["families"]) if h and f == family]
        test_rows = [r for r, f in zip(validation["rows"], validation["families"]) if f == family]
        table, fitted = {}, {}
        for visit in range(1, model.config.visits + 1):
            row = {}
            for step in range(1, 9):
                rng = np.random.default_rng(100 * visit + step)
                b, s, t = n2.make_pairs(test_rows, f"step{step}", visit, rng, 1000, deep_only=True)
                if len(t) < 100:
                    continue
                with torch.no_grad():
                    sources = n2.states_at(model, as_tokens(test_rows, s), visit)
                    logits = n2.patched_logits(model, as_tokens(test_rows, b), sources, visit, None)
                row[str(step)] = float((logits.argmax(dim=-1) == torch.tensor(t, device=device)).float().mean())
            table[str(visit)] = row
            best = max(row, key=row.get)
            rng = np.random.default_rng(visit)
            b, s, t = n2.make_pairs(fit_rows, f"step{best}", visit, rng, 2000, deep_only=False)
            vb, vs, vt = n2.make_pairs(test_rows, f"step{best}", visit, rng, 1000, deep_only=True)
            with torch.no_grad():
                fit_sources = n2.states_at(model, as_tokens(fit_rows, s), visit)
                test_sources = n2.states_at(model, as_tokens(test_rows, vs), visit)
            q = n2.fit(model, as_tokens(fit_rows, b), fit_sources, torch.tensor(t, device=device), visit, args.rank,
                       steps=400, seed=visit)
            with torch.no_grad():
                logits = n2.patched_logits(model, as_tokens(test_rows, vb), test_sources, visit, q)
            fitted[str(visit)] = {"step": int(best), "full_patch_iia": row[best],
                                  "das_iia": float((logits.argmax(dim=-1) == torch.tensor(vt, device=device)).float().mean())}
            print(args.model, family, "v", visit, "best step", best, json.dumps(fitted[str(visit)]), flush=True)
        out[family] = {"full_patch_by_visit_step": table, "das_at_best_step": fitted}
        args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()

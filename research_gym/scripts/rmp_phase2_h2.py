"""Phase 2, E6: H2, "a clock-free tied loop builds a counter it causally uses" (docs/rmp_phase2_e5_e6_plan.md).

**Candidates.**
- These are `D-shared` features (the selected size) whose answer-position activation is monotone in visit.
- A feature's score is the median, over 300 selection-half `pointer_chase` examples where it fires at two or
  more visits, of the Spearman correlation between visit (1..8) and its activation.
- Candidates have a score of 0.9 or more (control 10 found no visit-identity features in any model). The top
  5 by score are tested.

**Clamp.** At visit `v` (3..8), the feature's code at the answer position is set to its value at `v - 2`, by
adding (old code minus new code) times its decoder direction at the state's scale.

**Readout** (changed from the probe readout before any run; see the plan). The final answer "shifts back"
when it equals `cur_node(hops - j)` for some j >= 1, i.e. the walk stopped early.

**Control.** A random direction with the same added norm.

**Data.** Reporting-half `pointer_chase` examples at depth 5 to 8; the best `v` per candidate is reported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from research_gym.interp.dictionaries import load_dictionary, rms
from research_gym.interp.phase2 import core_examples, load_model, select_config, selection_half
from research_gym.interp.tasks import POINTER_TARGET_BASE
from research_gym.scripts.rmp_phase2_n1 import DEEP
from research_gym.scripts.rmp_phase2_n2 import node_at


@torch.no_grad()
def answer_codes(model, shared, tokens) -> torch.Tensor:
    """(B, visits, width) codes of the answer position's resid[1..8]."""

    states = []
    model.run(tokens, lambda c: states.append(c.after[:, -1].float()) or None)
    stacked = torch.stack(states, dim=1)
    return shared.encode(stacked / rms(stacked)), rms(stacked)


@torch.no_grad()
def clamp_run(model, tokens, visit, delta_vectors):
    def hook(context):
        if context.visit != visit - 1:
            return None
        after = context.after.clone()
        after[:, -1] = after[:, -1] + delta_vectors
        return after

    return model.run(tokens, hook).logits[:, -1].argmax(dim=-1)


def shifted_back(predictions, rows) -> float:
    hits = []
    for prediction, row in zip(predictions.tolist(), rows):
        hops = int(row.semantic["hops"])
        earlier = {POINTER_TARGET_BASE + node_at(row, hops - j) for j in range(1, hops + 1)}
        # A walk can end where it started, so only a changed answer that lands on an earlier node counts.
        hits.append(prediction in earlier and prediction != row.target_token)
    return float(np.mean(hits))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["LOOP-T s1", "LOOP-T s2", "LOOP-T s3", "FF-U s3"])
    parser.add_argument("--runs", type=Path, default=Path("experiments/recurrence_monitorability_v0/runs"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--monitor-bundle", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, default=Path("experiments/recurrence_monitorability_v0/phase2/dictionaries.json"))
    parser.add_argument("--out", type=Path, default=Path("experiments/recurrence_monitorability_v0/phase2/h2.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    data = core_examples(args.monitor_bundle, "monitor_calibration", families=("pointer_chase",))
    half = selection_half(data["groups"])
    select_index = np.flatnonzero(half)[:300]
    report_index = [i for i in np.flatnonzero(~half) if data["rows"][i].difficulty in DEEP]
    report = {}
    for name in args.models:
        model = load_model(name, args.runs, args.artifact_dir, device)
        chosen, _ = select_config(metrics[name])
        shared = load_dictionary(Path(metrics[name][chosen]["path"]), device=device)[0]
        codes, _ = answer_codes(model, shared, data["tokens"][select_index].to(device))
        codes = codes.cpu().numpy()
        scores = {}
        for feature in range(codes.shape[2]):
            values = [spearmanr(np.arange(1, codes.shape[1] + 1), codes[i, :, feature]).statistic
                      for i in range(codes.shape[0]) if (codes[i, :, feature] > 0).sum() >= 2]
            values = [v for v in values if np.isfinite(v)]
            if len(values) >= 30:
                scores[feature] = float(np.median(values))
        candidates = sorted((f for f, s in scores.items() if s >= 0.9), key=lambda f: -scores[f])[:5]
        tokens = data["tokens"][report_index].to(device)
        rows = [data["rows"][i] for i in report_index]
        report_codes, scale = answer_codes(model, shared, tokens)
        generator = torch.Generator().manual_seed(0)
        tested = []
        for feature in candidates:
            best = None
            for visit in range(3, model.config.visits + 1):
                change = (report_codes[:, visit - 3, feature] - report_codes[:, visit - 1, feature]) * scale[:, visit - 1, 0]
                vectors = change[:, None] * shared.decoder[feature][None]
                random = torch.randn(vectors.shape, generator=generator).to(device)
                random = random / random.norm(dim=-1, keepdim=True) * vectors.norm(dim=-1, keepdim=True)
                cell = {"visit": visit, "clamp": shifted_back(clamp_run(model, tokens, visit, vectors), rows),
                        "random": shifted_back(clamp_run(model, tokens, visit, random), rows)}
                if best is None or cell["clamp"] > best["clamp"]:
                    best = cell
            tested.append({"feature": feature, "monotonicity": scores[feature], **best})
        clean = shifted_back(model.run(tokens).logits[:, -1].argmax(dim=-1), rows)
        meets = any(t["clamp"] >= 0.5 and t["random"] <= 0.1 for t in tested)
        report[name] = {"dictionary": chosen, "candidates": len([f for f, s in scores.items() if s >= 0.9]),
                        "clean_shifted": clean, "tested": tested, "meets_bar": meets}
        print(name, json.dumps({k: v for k, v in report[name].items() if k != "tested"}),
              [(t["feature"], round(t["clamp"], 3), round(t["random"], 3), t["visit"]) for t in tested], flush=True)
    tied = [report[m]["meets_bar"] for m in report if m.startswith("LOOP-T")]
    report["verdict"] = "supported" if tied and all(tied) else "not_supported"
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print("H2", report["verdict"], flush=True)


if __name__ == "__main__":
    main()

"""Why `D-trans` fails: how much of each visit's update a per-position function of `resid[t]` cannot see.

The program's transcoder reads one position's `resid[t]` and predicts that position's `delta[t]`. A block's
update also carries attention, which reads the other positions, and input re-injection where the cell has
it. This splits the update, in the transcoder's units (divided by the RMS of `resid[t]`), into the attention
term `alpha * beta * attn_out`, the re-injected embedding, and the rest. It reports each one's share of the
update's variance next to the trained transcoders' calibration FVU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research_gym.interp.dictionaries import rms
from research_gym.interp.hooks import record_components
from research_gym.interp.phase2 import core_examples, load_model, selection_half, subset

MODELS = ["LOOP-T s1", "LOOP-T s2", "LOOP-T s3", "FF-U s3"]


def variance(x: torch.Tensor) -> float:
    flat = x.reshape(-1, x.shape[-1]).double()
    return float(((flat - flat.mean(dim=0)) ** 2).sum(dim=1).mean())


@torch.no_grad()
def diagnose(model, tokens: torch.Tensor) -> dict:
    with record_components(model.model) as records:
        model.run(tokens)
    injected = model.model.embed(tokens) if model.config.input_injection else None
    by_visit = {}
    for visit, parts in enumerate(records):
        layer = model.model.block_at(visit)
        state = parts["resid_pre"]
        scale = rms(state)
        attention = layer.alpha * layer.beta * parts["attn_out"] / scale
        update = (parts["resid_post"] - state) / scale + (injected / scale if injected is not None else 0)
        total = variance(update)
        by_visit[f"update{visit}"] = {
            "attention_share": variance(attention) / total,
            "injection_share": variance(injected / scale) / total if injected is not None else 0.0,
            "local_share": variance(update - attention - (injected / scale if injected is not None else 0)) / total,
        }
    return by_visit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("experiments/recurrence_monitorability_v0/runs"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--monitor-bundle", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, default=Path("experiments/recurrence_monitorability_v0/phase2/dictionaries.json"))
    parser.add_argument("--out", type=Path, default=Path("experiments/recurrence_monitorability_v0/phase2/trans_diagnosis.json"))
    parser.add_argument("--examples", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    calibration = core_examples(args.monitor_bundle, "monitor_calibration")
    select = subset(calibration, selection_half(calibration["groups"]))
    tokens = select["tokens"][: args.examples].to(device)
    report = {}
    for name in MODELS:
        model = load_model(name, args.runs, args.artifact_dir, device)
        by_visit = diagnose(model, tokens)
        fvus = {key: value["calibration_fvu"] for key, value in metrics.get(name, {}).items() if key.startswith("trans")}
        report[name] = {
            "input_injection": bool(model.config.input_injection),
            "by_visit": by_visit,
            "mean_attention_share": float(np.mean([v["attention_share"] for v in by_visit.values()])),
            "mean_injection_share": float(np.mean([v["injection_share"] for v in by_visit.values()])),
            "transcoder_fvu_best": min(fvus.values()) if fvus else None,
            "transcoder_fvu_worst": max(fvus.values()) if fvus else None,
        }
        print(name, json.dumps({k: v for k, v in report[name].items() if k != "by_visit"}), flush=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()

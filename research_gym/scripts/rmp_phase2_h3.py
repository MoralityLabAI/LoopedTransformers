"""Phase 2, E5: H3, the MLP transcoders' total description length, per docs/rmp_phase2_e5_e6_plan.md.

For each model, the smallest size in the grid (smallest width, then smallest k) whose replacement model keeps
0.95 or more of clean deep exact match. Its length is counted as (a) alive latents summed over transcoders,
and (b) mean active latents per position per visit times log2(alive latents), summed over transcoders, plus
the skip matrices at 16 bits. Alive and active counts are measured on 256 selection-half calibration examples.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from research_gym.interp.dictionaries import load_dictionary
from research_gym.interp.phase2 import core_examples, load_model, selection_half, subset
from research_gym.scripts.rmp_phase2_mlp_transcoder import mlp_pairs

MODELS = ["LOOP-T s1", "LOOP-T s2", "LOOP-T s3", "FF-U s3"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("experiments/recurrence_monitorability_v0/runs"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--monitor-bundle", type=Path, required=True)
    parser.add_argument("--transcoders", type=Path, default=Path("experiments/recurrence_monitorability_v0/phase2/mlp_transcoders.json"))
    parser.add_argument("--out", type=Path, default=Path("experiments/recurrence_monitorability_v0/phase2/h3.json"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    grid = json.loads(args.transcoders.read_text(encoding="utf-8"))
    calibration = core_examples(args.monitor_bundle, "monitor_calibration")
    tokens = subset(calibration, selection_half(calibration["groups"]))["tokens"][:256].to(device)
    report = {}
    for name in MODELS:
        results = grid[name]["results"]
        passing = sorted((r["width"], r["k"], key) for key, r in results.items() if r["replacement"]["deep_ratio"] >= 0.95)
        if not passing:
            report[name] = {"size": None, "note": "no size reaches 0.95"}
            print(name, "no size reaches 0.95", flush=True)
            continue
        width, k, key = passing[0]
        model = load_model(name, args.runs, args.artifact_dir, device)
        pairs = mlp_pairs(model, tokens)
        dictionaries = [load_dictionary(Path(d["path"]), device=device)[0] for d in results[key]["dictionaries"]]
        alive_total, bits_total, skip_bits = 0, 0.0, 0
        for index, dictionary in enumerate(dictionaries):
            inputs = [pairs[index][0]] if len(dictionaries) > 1 else [x for x, _ in pairs]
            with torch.no_grad():
                codes = torch.cat([dictionary.encode(x) for x in inputs])
            alive = int((codes > 0).any(dim=0).sum())
            active = float((codes > 0).sum(dim=1).float().mean())
            alive_total += alive
            bits_total += active * math.log2(max(alive, 2))
            skip_bits += dictionary.skip.numel() * 16
        report[name] = {"size": key, "replacement_deep_ratio": results[key]["replacement"]["deep_ratio"],
                        "transcoders": len(dictionaries), "alive_latents": alive_total,
                        "bits_per_position_visit": bits_total, "skip_bits": skip_bits,
                        "length_b": bits_total + skip_bits}
        print(name, json.dumps(report[name]), flush=True)
    tied = [report[m] for m in MODELS[:3]]
    untied = report["FF-U s3"]
    if untied.get("size") is None or any(r.get("size") is None for r in tied):
        verdict = "unresolved"
    else:
        below = [r["alive_latents"] < untied["alive_latents"] and r["length_b"] < untied["length_b"] for r in tied]
        above = [r["alive_latents"] > untied["alive_latents"] and r["length_b"] > untied["length_b"] for r in tied]
        verdict = "supported" if all(below) else "contradicted" if all(above) else "mixed"
    report["verdict"] = verdict
    print("H3", verdict, flush=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()

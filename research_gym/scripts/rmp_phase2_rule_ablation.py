"""Phase 2, E6: rule ablation in the MLP-transcoder replacement model (docs/rmp_phase2_e5_e6_plan.md).

Every MLP application is replaced by the w4096 k32 MLP transcoder, with attention live. One latent is
zeroed at all visits, for each of the top 20 latents by mean activation on deep `validation` examples. The
drop in exact match is reported per family and depth, against the unablated replacement model. The
first-error step is not computed; this is recorded in the plan.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research_gym.interp.dictionaries import load_dictionary
from research_gym.interp.phase2 import load_model
from research_gym.interp.replacement import answer_scores
from research_gym.scripts.rmp_phase2_mlp_transcoder import mlp_pairs
from research_gym.scripts.rmp_phase2_n1 import DEEP, GRAPH_FAMILIES, validation_data


class Ablated(torch.nn.Module):
    def __init__(self, dictionary, latent: int | None) -> None:
        super().__init__()
        self.dictionary, self.latent = dictionary, latent

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        shape = z.shape
        x = z.reshape(-1, shape[-1])
        codes = self.dictionary.encode(x)
        if self.latent is not None:
            codes[:, self.latent] = 0
        return self.dictionary.decode(codes, x).reshape(shape)


def by_cell(correct: np.ndarray, data: dict) -> dict:
    return {f"{family}/d{depth}": float(correct[(data["families"] == family) & (data["depths"] == depth)].mean())
            for family in GRAPH_FAMILIES for depth in DEEP}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["LOOP-T s1", "LOOP-T s2", "LOOP-T s3", "FF-U s3"])
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--runs", type=Path, default=Path("experiments/recurrence_monitorability_v0/runs"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, default=Path("experiments/recurrence_monitorability_v0/datasets"))
    parser.add_argument("--transcoders", type=Path, default=Path("experiments/recurrence_monitorability_v0/phase2/mlp_transcoders.json"))
    parser.add_argument("--out", type=Path, default=Path("experiments/recurrence_monitorability_v0/phase2/rule_ablation.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    grid = json.loads(args.transcoders.read_text(encoding="utf-8"))
    data = validation_data(args.bundle_dir)
    deep = np.isin(data["depths"], DEEP)
    report = {}
    for name in args.models:
        model = load_model(name, args.runs, args.artifact_dir, device)
        dictionaries = [load_dictionary(Path(d["path"]), device=device)[0]
                        for d in grid[name]["results"]["mlp_w4096_k32_s0"]["dictionaries"]]
        tied = len(dictionaries) == 1
        sample = data["tokens"][np.flatnonzero(deep)[:512]].to(device)
        pairs = mlp_pairs(model, sample)
        if tied:
            with torch.no_grad():
                activity = torch.cat([dictionaries[0].encode(x) for x, _ in pairs]).mean(dim=0)
            ranked = [(0, int(i)) for i in activity.topk(args.top).indices]
        else:
            # Untied: the top latents over all eight layers' transcoders, ranked together.
            scored = []
            for layer, (x, _) in enumerate(pairs):
                with torch.no_grad():
                    mean = dictionaries[layer].encode(x).mean(dim=0)
                scored += [(float(v), layer, int(i)) for v, i in zip(*mean.topk(args.top))]
            ranked = [(layer, latent) for _, layer, latent in sorted(scored, reverse=True)[: args.top]]

        def run(target_layer, latent):
            originals = {}
            for visit in range(model.config.visits):
                block = model.model.block_at(visit)
                if id(block) not in originals:
                    index = 0 if tied else visit
                    originals[id(block)] = (block, block.mlp)
                    block.mlp = Ablated(dictionaries[index], latent if index == target_layer else None)
            try:
                return answer_scores(model, data["tokens"], data["targets"])[1]
            finally:
                for block, mlp in originals.values():
                    block.mlp = mlp

        base = run(-1, None)
        base_cells = by_cell(base, data)
        rows = []
        for layer, latent in ranked:
            cells = by_cell(run(layer, latent), data)
            drops = {key: base_cells[key] - value for key, value in cells.items()}
            rows.append({"layer": layer, "latent": latent, "drops": drops, "max_drop": max(drops.values()),
                         "worst_cell": max(drops, key=drops.get)})
        report[name] = {"replacement_deep": float(base[deep].mean()), "base_cells": base_cells, "ablations": rows}
        print(name, "replacement deep", round(report[name]["replacement_deep"], 3),
              [(r["layer"], r["latent"], round(r["max_drop"], 3), r["worst_cell"]) for r in rows[:8]], flush=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()

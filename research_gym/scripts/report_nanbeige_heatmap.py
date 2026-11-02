"""Plot the Nanbeige per-pass VPD heat map written by `rmp_nanbeige_vpd_heatmap`.

For each layer:
- left, the ablation map: change in the correct answer's log-probability when a component is removed on
  pass 1, pass 2 or both, sorted by the both-pass effect;
- right, the usage map: mean activation-weighted use of each component per pass, at the answer position.

The 5th–95th percentile of the random rank-one control is printed beside each layer and saved to the
summary JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PASSES = ("pass1", "pass2", "both")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heatmap", type=Path, default=Path(
        "experiments/recurrence_monitorability_v0/nanbeige_heatmap/heatmap.json"))
    args = parser.parse_args()
    report = json.loads(args.heatmap.read_text(encoding="utf-8"))
    hop = sorted(report["clean_mean_logprob"])[0]
    ablation, usage = report["ablation"], report["usage"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), gridspec_kw={"width_ratios": [3, 2]})
    summary = {}
    for row, layer in enumerate((12, 22)):
        n = sum(1 for k in ablation if k.startswith(f"L{layer}_c") and k.endswith("_both"))
        nr = sum(1 for k in ablation if k.startswith(f"L{layer}_rand") and k.endswith("_both"))
        real = np.array([[ablation[f"L{layer}_c{c}_{p}"][f"hops{hop}_dlogp"] for p in PASSES] for c in range(n)])
        rand = np.array([[ablation[f"L{layer}_rand{r}_{p}"][f"hops{hop}_dlogp"] for p in PASSES] for r in range(nr)])
        order = np.argsort(real[:, 2])
        lim = max(np.abs(real).max(), np.abs(rand).max())
        ax = axes[row, 0]
        im = ax.imshow(real[order].T, aspect="auto", cmap="RdBu", vmin=-lim, vmax=lim)
        ax.set_yticks(range(3), ["pass 1", "pass 2", "both"])
        ax.set_xlabel("component (sorted by both-pass effect)")
        ax.set_title(f"layer {layer}: ablation, change in log p(correct)")
        fig.colorbar(im, ax=ax, fraction=0.03)
        use = np.array([usage[f"L{layer}_pass{p}"]["answer_position"] for p in (1, 2)])
        ax = axes[row, 1]
        im = ax.imshow(use[:, order], aspect="auto", cmap="viridis")
        ax.set_yticks(range(2), ["pass 1", "pass 2"])
        ax.set_xlabel("component (same order)")
        ax.set_title(f"layer {layer}: usage at answer position")
        fig.colorbar(im, ax=ax, fraction=0.03)
        band = np.percentile(rand, [5, 95], axis=0)
        summary[f"L{layer}"] = {
            p: {"real_min": float(real[:, i].min()), "real_max": float(real[:, i].max()),
                "real_mean": float(real[:, i].mean()), "random_p5": float(band[0, i]),
                "random_p95": float(band[1, i]),
                "real_below_random_p5": int((real[:, i] < band[0, i]).sum())}
            for i, p in enumerate(PASSES)}
        summary[f"L{layer}"]["usage_pass2_over_pass1"] = float(use[1].sum() / use[0].sum())
    fig.tight_layout()
    out = args.heatmap.with_suffix(".png")
    fig.savefig(out, dpi=130)
    args.heatmap.with_name("heatmap_summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()

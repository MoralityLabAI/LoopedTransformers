"""Plot Huginn results: the depth sweep (accuracy by hop count and recurrence) and, when present, the
per-iteration component heat map written by `rmp_huginn_heatmap`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BASE = Path("experiments/recurrence_monitorability_v0")


def depth(path: Path) -> None:
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    keys = [k.split("_") for k in cells]
    shots = sorted({int(k[0][5:]) for k in keys})
    hops = sorted({int(k[1][4:]) for k in keys})
    steps = sorted({int(k[2][1:]) for k in keys})
    fig, axes = plt.subplots(1, len(shots), figsize=(5 * len(shots), 3.8), squeeze=False)
    for ax, s in zip(axes[0], shots):
        for r in steps:
            acc = [cells.get(f"shots{s}_hops{h}_r{r}", {}).get("acc", np.nan) for h in hops]
            ax.plot(hops, acc, marker="o", label=f"r={r}")
        ax.axhline(1 / 16, color="grey", ls=":", label="chance")
        ax.set_xlabel("hops")
        ax.set_ylabel("accuracy (forced choice)")
        ax.set_title(f"{s}-shot")
        ax.set_ylim(0, 1)
    axes[0, -1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=130)
    print("wrote", path.with_suffix(".png"))


def heatmap(path: Path) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    ablation, usage, windows = report["ablation"], report["usage"], list(report["meta"]["windows"])
    fig, axes = plt.subplots(4, 2, figsize=(12, 13), gridspec_kw={"width_ratios": [3, 2]})
    summary = {}
    for layer in range(4):
        use = np.array(usage[f"core{layer}"])  # [iterations, components]
        ax = axes[layer, 0]
        im = ax.imshow(use.T, aspect="auto", cmap="viridis")
        ax.set_xlabel("iteration")
        ax.set_ylabel("component (singular index)")
        ax.set_title(f"core layer {layer}: usage at answer position")
        fig.colorbar(im, ax=ax, fraction=0.03)
        names = sorted({k.rsplit("_", 1)[0] for k in ablation if k.startswith(f"core{layer}_c")},
                       key=lambda n: int(n.split("_c")[1]))
        rand = sorted({k.rsplit("_", 1)[0] for k in ablation if k.startswith(f"core{layer}_rand")})
        real = np.array([[ablation.get(f"{n}_{w}", {}).get("dlogp", np.nan) for w in windows] for n in names])
        ctrl = np.array([[ablation.get(f"{n}_{w}", {}).get("dlogp", np.nan) for w in windows] for n in rand])
        lim = np.nanmax(np.abs(np.concatenate([real.ravel(), ctrl.ravel()]))) if real.size else 1
        ax = axes[layer, 1]
        im = ax.imshow(real, aspect="auto", cmap="RdBu", vmin=-lim, vmax=lim)
        ax.set_xticks(range(len(windows)), windows, rotation=30, fontsize=8)
        ax.set_yticks(range(len(names)), [n.split("_")[1] for n in names], fontsize=7)
        ax.set_title(f"core layer {layer}: ablation, change in log p(correct)")
        fig.colorbar(im, ax=ax, fraction=0.05)
        band = np.nanpercentile(ctrl, [5, 95], axis=0) if ctrl.size else np.full((2, len(windows)), np.nan)
        summary[f"core{layer}"] = {
            w: {"real_min": float(np.nanmin(real[:, i])), "random_p5": float(band[0, i]),
                "random_p95": float(band[1, i]), "real_below_random_min": int((real[:, i] < np.nanmin(ctrl[:, i])).sum())}
            for i, w in enumerate(windows)}
        summary[f"core{layer}"]["usage_by_iteration"] = use.sum(1).round(3).tolist()
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=130)
    path.with_name("heatmap_summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: {w: v for w, v in s.items() if w != "usage_by_iteration"} for k, s in summary.items()}, indent=1))
    print("wrote", path.with_suffix(".png"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=Path, default=BASE / "huginn_depth/depth_sweep.json")
    parser.add_argument("--heatmap", type=Path, default=BASE / "huginn_heatmap/heatmap.json")
    args = parser.parse_args()
    if args.depth.exists():
        depth(args.depth)
    if args.heatmap.exists():
        heatmap(args.heatmap)


if __name__ == "__main__":
    main()

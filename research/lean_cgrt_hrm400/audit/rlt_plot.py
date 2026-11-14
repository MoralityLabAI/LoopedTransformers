"""Scientific diagnostic figure from the saved pilot receipts."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(sys.argv[1])
report = json.loads((root / "RLT-pilot-receipts.json").read_text(encoding="utf-8"))
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
families = [arch for arch in ("loop", "hrm") if any(n.startswith(arch + "_") for n in report["arms"])]
fig, axes = plt.subplots(len(families), 2, figsize=(12, 3.6 * len(families)), constrained_layout=True, squeeze=False)
colors = {"direct": "#475569", "rlt": "#007f84", "matched_damping": "#c56a15"}
labels = {"direct": "Direct", "rlt": "RLT", "matched_damping": "Matched damping"}
styles = {"direct": "-", "rlt": "--", "matched_damping": ":"}
for row, arch in enumerate(families):
    for feedback in colors:
        candidates = [n for n in report["arms"] if n.startswith(f"{arch}_{feedback}_s")]
        if not candidates:
            continue
        name = candidates[0]
        profile = report["arms"][name]["diagnostics"]["depths"]["32"]["profiles"]["0.001"]
        loops = [p["loop"] for p in profile]
        axes[row, 0].plot(loops, [p["hidden_norm_mean"] for p in profile], lw=2, ls=styles[feedback], color=colors[feedback], label=labels[feedback])
        gains = [p["full_state_perturbation_gain"] for p in profile]
        axes[row, 1].plot(loops, gains, lw=2, ls=styles[feedback], color=colors[feedback], label=labels[feedback])
    axes[row, 0].set_ylabel("Mean valid-token state norm")
    axes[row, 1].set_ylabel("Perturbation gain (full carried state)")
    axes[row, 1].axhline(1, color="#b0b5bd", lw=1, ls="--")
    for col in range(2):
        axes[row, col].set_xlabel("Inference loop")
        axes[row, col].set_xlim(1, 32)
        axes[row, col].set_xticks((1, 8, 16, 24, 32))
        axes[row, col].grid(alpha=.18)
        values = [float(v) for line in axes[row, col].lines for v in line.get_ydata()]
        if values and min(values) > 0 and max(values) / min(values) > 100:
            axes[row, col].set_yscale("log")
        axes[row, col].set_title("Single-clock encoder" if arch == "loop" else "Two-clock HRM encoder", loc="left", fontweight="bold")
        axes[row, col].legend(frameon=False, fontsize=9)
fig.suptitle(f"RLT feedback pilot: 32 inference loops after training at 8\nSeed {report['protocol']['identity']['seed']}; noise 0.001 × input RMS; four fixed val goals", fontsize=14, fontweight="bold")
fig.savefig(root / "RLT-loop-diagnostics.png", dpi=170)
print(root / "RLT-loop-diagnostics.png")

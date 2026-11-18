"""Post-collection display only: plot the frozen LRM-REV-02 summary.

Reads archived aggregate values. It does not generate sources, load weights,
fit models, change outcomes, or compute inferential statistics.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--summary", type=Path, default=here / "lrm_rev02_v1" / "summary.json")
    parser.add_argument("--output-prefix", type=Path, default=here / "lrm_rev02_v1_figure")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    roots = summary["roots"]
    if summary["run_status"] != "COMPLETE" or [r["root"] for r in roots] != [0, 1, 2, 3]:
        raise ValueError("This publication display requires the four complete registered roots.")
    n = len(roots)
    corrected = [math.fsum(r["curve_means"]["corrected"][i]["excess"] for r in roots) / n
                 for i in range(7)]
    random = [math.fsum(math.fsum(r["curve_means"][f"random_{b}"][i]["excess"]
                                 for b in range(4)) / 4 for r in roots) / n
              for i in range(7)]
    effects = [r["means"]["paired_effect"] for r in roots]
    pooled = math.fsum(effects) / n
    cap = summary["registration"]["diagnostics"]["candidate_excess_max"]
    target = summary["registration"]["endpoint"]["scientific_min_effect"]
    failed = sum(not r["diagnostics"]["candidate_competence"] for r in roots)
    if not math.isclose(pooled, 0.008553539883975697, abs_tol=1e-14):
        raise ValueError("The plotted pooled effect differs from the frozen result.")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#81909C", "axes.linewidth": 0.7,
        "xtick.color": "#394B59", "ytick.color": "#394B59",
        "text.color": "#1E303D", "axes.labelcolor": "#243B4B",
        "svg.fonttype": "none", "svg.hashsalt": "lrm-rev02-frozen-display-v1",
    })
    teal, orange, threshold = "#176B87", "#BF652F", "#774657"
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.8, 5.9),
                                     gridspec_kw={"width_ratios": [1.28, 1]})
    fig.subplots_adjust(left=.08, right=.97, top=.79, bottom=.23, wspace=.32)
    fig.suptitle("Posterior recovery with reminders", x=.08, y=.975,
                 ha="left", fontsize=17, fontweight="semibold")
    fig.text(.08, .913, "LRM-REV-02  |  Independent trained models: n = 4  |  Development only",
             fontsize=10.5, color="#516575")

    left.set_title("A   Posterior recovery", loc="left", pad=12, fontweight="semibold")
    left.plot(range(7), corrected, color=teal, marker="o", markersize=5,
              linewidth=2.4, label="Corrected slots", zorder=3)
    left.plot(range(7), random, color=orange, marker="s", markersize=4.5,
              linewidth=2.2, label="Random slots: mean of four controls", zorder=3)
    left.axhline(cap, color=threshold, linestyle=(0, (4, 3)), linewidth=1.3)
    left.text(.1, cap + .003, f"Competence cap {cap:.3f}", color=threshold, fontsize=9.5)
    left.set(xlim=(-.15, 6.15), ylim=(0, .10), xticks=range(7),
             xlabel="Old values repeated after correction (r)",
             ylabel="Excess Brier over full-history Bayes")
    left.yaxis.set_major_formatter(FormatStrFormatter("%.02f"))
    left.grid(axis="y", color="#E4E9ED", linewidth=.65, zorder=0)
    left.legend(loc="lower left", bbox_to_anchor=(.005, .30), frameon=False, fontsize=9.5)
    left.annotate(f"{random[-1]:.4f}", (6, random[-1]), xytext=(-4, 10),
                  textcoords="offset points", ha="right", color=orange, fontsize=9)
    left.annotate(f"{corrected[-1]:.4f}", (6, corrected[-1]), xytext=(-4, -16),
                  textcoords="offset points", ha="right", color=teal, fontsize=9)

    right.set_title("B   Effect by independent model", loc="left", pad=12, fontweight="semibold")
    bars = right.bar(range(n), effects, color=teal, width=.65, alpha=.92, zorder=2)
    right.axhline(target, color=threshold, linewidth=1.4, linestyle=(0, (5, 3)),
                   label=f"Descriptive target {target:.3f}", zorder=3)
    right.axhline(pooled, color="#243B4B", linewidth=1.4, linestyle=(0, (2, 2)),
                   label=f"Equal-root mean {pooled:.5f}", zorder=3)
    for bar, value in zip(bars, effects):
        right.text(bar.get_x() + bar.get_width() / 2, value - .00045,
                    f"{value:.5f}", ha="center", va="top", fontsize=9,
                    color="white", fontweight="medium")
    right.set(xlim=(-.6, 3.6), ylim=(0, .019), xticks=range(n),
              yticks=[0, .005, .01, .015], xlabel="Model root",
              ylabel="Control excess − corrected-slot excess")
    right.yaxis.set_major_formatter(FormatStrFormatter("%.03f"))
    right.grid(axis="y", color="#E4E9ED", linewidth=.65, zorder=0)
    right.legend(loc="upper left", frameon=False, fontsize=9, handlelength=2.4)

    fig.text(.08, .105, f"Candidate competence: failed in {failed}/{n} models against the {cap:.3f} excess cap at r = 6.",
             fontsize=10.5, color=threshold, fontweight="semibold")
    fig.text(.08, .055, "Each model contributes 512 development episodes. Curves and the pooled effect weight models equally.",
             fontsize=9, color="#516575")
    metadata = {"Title": "LRM-REV-02 development pilot: posterior recovery with reminders",
                "Description": "Post-collection display of the four registered model roots; descriptive results only."}
    svg, png = args.output_prefix.with_suffix(".svg"), args.output_prefix.with_suffix(".png")
    fig.savefig(svg, metadata={**metadata, "Date": None}, facecolor="white")
    fig.savefig(png, dpi=180, metadata=metadata, facecolor="white")
    plt.close(fig)
    print(json.dumps({"equal_root_mean_effect": pooled, "corrected_r6_excess": corrected[-1],
                      "control_r6_excess": random[-1], "candidate_failures": failed,
                      "files": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (svg, png, Path(__file__).resolve())}}, indent=2))


if __name__ == "__main__":
    main()

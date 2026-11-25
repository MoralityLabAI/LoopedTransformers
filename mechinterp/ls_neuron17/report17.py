"""Post-run descriptive report for the exposed LS17 allocation ablation."""
from pathlib import Path
import hashlib
import json
from collections import Counter
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

ROOT = Path(__file__).resolve().parent
SEEDS = tuple(range(493, 517))
ARMS = ("beam", "multistart")
LABELS = {"beam": "Dynamic beam", "multistart": "Independent paths"}


def main():
    for line in (ROOT / "lock.sha256").read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    certs = json.loads((ROOT / "certificates.json").read_text())
    audit = json.loads((ROOT / "audit_results.json").read_text())
    stats = json.loads((ROOT / "evidence/results/statistics.json").read_text())
    assert audit["status"] == "PASS" and stats["study_evidence_class"] == "EXPOSED_DEVELOPMENT_ONLY"
    assert stats["comparison"]["sealed_readout"] is False
    assert stats["comparison"]["hypothesis_preregistered"] is False
    rows = {(r["seed"], r["arm"]): r for r in certs["rows"]}
    assert set(rows) == {(s, a) for s in SEEDS for a in ARMS}
    paired = []
    for seed in SEEDS:
        b, m = rows[seed, "beam"], rows[seed, "multistart"]
        new = lambda r: int(r["initial"]["verdict"] == "POINT_VIOLATION" and r["final"]["verdict"] == "CERTIFIED")
        paired.append({"seed": seed, "beam_repair": new(b), "multistart_repair": new(m),
                       "beam_bound": b["scores"]["bound"], "multistart_bound": m["scores"]["bound"],
                       "beam_expanded_codebooks": b["expanded_codebook_count"],
                       "multistart_expanded_codebooks": m["expanded_codebook_count"],
                       "beam_fitted_codebooks": b["unique_codebooks_visited"],
                       "multistart_fitted_codebooks": m["unique_codebooks_visited"],
                       "multistart_winning_branch": m["winning_branch"],
                       "branch_terminal_bounds": [x["terminal_bound"] for x in m["branch_diagnostics"]]})
    counts = {a: sum(p[f"{a}_repair"] for p in paired) for a in ARMS}
    report = {"experiment": "LS-NEURON-17", "scope": "EXPOSED_DEVELOPMENT_ONLY",
        "paired_outcomes": paired, "counts": counts,
        "beam_only_seeds": [p["seed"] for p in paired if p["beam_repair"] and not p["multistart_repair"]],
        "multistart_only_seeds": [p["seed"] for p in paired if p["multistart_repair"] and not p["beam_repair"]],
        "both_seeds": [p["seed"] for p in paired if p["multistart_repair"] and p["beam_repair"]],
        "neither_seeds": [p["seed"] for p in paired if not p["multistart_repair"] and not p["beam_repair"]],
        "prior_LS16_added_seed_outcomes": [p for p in paired if p["seed"] in (501, 516)],
        "winning_bootstrap_branch_counts": dict(Counter(str(p["multistart_winning_branch"]) for p in paired)),
        "codebook_diagnostics": {a: {key: {
             "sum_within_seed_counts": sum(rows[s,a][key] for s in SEEDS),
             "median_per_seed": median(rows[s,a][key] for s in SEEDS)} for key in
             ("unique_codebooks_visited", "unique_decoded_weights_visited", "expanded_codebook_count",
              "repeated_expanded_codebook_count", "retained_codebook_count_all_selections",
              "retained_node_count_all_selections", "final_path_depth")} for a in ARMS},
        "codebook_diagnostics_scope": "Descriptive sums of per-seed unique counts, not globally deduplicated; visited codebooks refer to fitted states, not all proposal profile slots",
        "audit": audit, "official_descriptive_statistics": stats,
        "source_hashes": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
             ("certificates.json", "audit_results.json", "evidence/results/statistics.json", "lock.sha256", "report17.py")},
        "report_code_scope": "Post-run rendering outside the frozen experiment; no new hypothesis tests"}
    (ROOT / "report_data.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none", "svg.hashsalt": "LS-NEURON-17"})
    fig = plt.figure(figsize=(14.5, 8.4), facecolor="white")
    grid = fig.add_gridspec(2, 1, height_ratios=(1, 2.5), left=.12, right=.98, top=.75, bottom=.15, hspace=.48)
    fig.text(.12, .94, "LS17 · Dynamic beam and independent greedy paths", fontsize=20, weight="bold", color="#172235")
    fig.text(.12, .885, f"Exact repairs: dynamic beam {counts['beam']}/24; independent paths {counts['multistart']}/24.", fontsize=14)
    fig.text(.12, .835, "Exposed LS16 cohort · descriptive allocation ablation · 37 expansions charged per arm and seed", fontsize=10, color="#70505e")
    ax = fig.add_subplot(grid[0])
    mapping = {"POINT_VIOLATION": 0, "CERTIFIED": 1, "UNRESOLVED": 2}
    cells = np.array([[mapping[rows[s, a]["final"]["verdict"]] for s in SEEDS] for a in ARMS])
    ax.imshow(cells, cmap=ListedColormap(["#e6ebf0", "#19786d", "#efd38b"]), vmin=0, vmax=2, aspect="auto")
    ax.set_yticks((0, 1), [LABELS[a] for a in ARMS]); ax.set_xticks(range(24), SEEDS); ax.tick_params(axis="both", length=0)
    ax.xaxis.tick_top()
    for i in range(2):
        for j in range(24):
            ax.text(j, i, ("P", "C", "U")[cells[i,j]], ha="center", va="center", weight="bold",
                    color="white" if cells[i,j] == 1 else "#354353")
    ax.set_xticks(np.arange(-.5, 24, 1), minor=True); ax.set_yticks([-.5,.5,1.5], minor=True)
    ax.grid(which="minor", color="white", linewidth=2); ax.tick_params(which="minor", length=0)
    ax.set_title("A  Exact endpoint verdicts   ·   C certified    P point violation", loc="left", pad=26, fontsize=11, weight="bold")
    for spine in ax.spines.values(): spine.set_visible(False)
    bounds = fig.add_subplot(grid[1])
    colors = {"beam": "#d97719", "multistart": "#2c5c96"}
    for j, seed in enumerate(SEEDS):
        values = [rows[seed,a]["scores"]["bound"] for a in ARMS]
        bounds.plot([j-.11,j+.11], values, color="#b7c2cf", linewidth=.8, zorder=1)
    for a, shift, marker in (("beam", -.11, "D"), ("multistart", .11, "o")):
        bounds.scatter(np.arange(24)+shift, [rows[s,a]["scores"]["bound"] for s in SEEDS],
                       color=colors[a], marker=marker, label=LABELS[a], s=28, edgecolors="white", linewidths=.5, zorder=3)
    bounds.axhline(3/32, color="#685d72", linestyle="--", linewidth=1)
    bounds.text(23.25, 3/32, "3/32", ha="right", va="bottom", color="#685d72", fontsize=9,
                bbox={"facecolor":"white","edgecolor":"none","alpha":.85,"pad":1})
    bounds.set_yscale("log"); bounds.set_xticks(range(24), SEEDS); bounds.set_xlim(-.6,23.6)
    bounds.set_xlabel("Perturbation seed"); bounds.set_ylabel("Final numerical interval bound")
    bounds.grid(axis="y", color="#e2e7ed", linewidth=.7)
    bounds.legend(loc="lower left", bbox_to_anchor=(0,1), frameon=False, ncol=2, fontsize=9)
    fig.text(.12, .075, "Both arms reuse the same three bootstrap codebooks. Independent paths keep each lineage's own 12 expansions.", fontsize=10, color="#485668")
    fig.text(.12, .035, "Archived beam outcomes selected this follow-up. Exact recurrence certificates remain separate from population inference.", fontsize=10, color="#485668")
    fig.savefig(ROOT / "figure17.png", dpi=200)
    fig.savefig(ROOT / "figure17.svg", metadata={"Date": None})
    plt.close(fig)
    print(json.dumps({"counts": counts, "beam_only": report["beam_only_seeds"], "multistart_only": report["multistart_only_seeds"],
                      "descriptive_p": stats["comparison"]["p_unadjusted"]}))


if __name__ == "__main__":
    main()

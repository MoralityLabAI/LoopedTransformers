"""Figures for papers/rmp_looped_mechinterp from the committed Phase 2 result JSON (vector PDF)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TIED, UNTIED, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#9a9893", "#0b0b0b", "#52514e"
ORIGINAL = ["LOOP-T s1", "LOOP-T s2", "LOOP-T s3", "FF-U s3"]
FRESH = ["LOOP-T s4", "LOOP-T s5", "LOOP-T s6", "FF-U s4", "FF-U s6"]

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.5, "legend.frameon": False,
    "lines.linewidth": 1.5, "pdf.fonttype": 42,
})


def color(name: str) -> str:
    return TIED if name.startswith("LOOP-T") else UNTIED


def load(root: Path, name: str):
    return json.loads((root / name).read_text(encoding="utf-8"))


def fig_n1(root: Path, out: Path) -> None:
    n1 = load(root, "n1.json")
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    for name in ORIGINAL:
        entries = n1[name]["dictionaries"]
        x = [e["truncation"]["deep_ratio"] for e in entries]
        y = [e["closed_loop"]["deep_ratio"] for e in entries]
        ax.scatter(x, y, s=16, color=color(name), marker="o" if name != "LOOP-T s3" else "s",
                   edgecolors="white", linewidths=0.6, label=name, zorder=3,
                   alpha=1.0 if name in ("LOOP-T s1", "FF-U s3") else 0.75)
    ax.plot([0.3, 1.0], [0.3, 1.0], color=GRAY, lw=1, ls="--", zorder=1)
    ax.axhline(0.90, color=MUTED, lw=0.8, ls=":")
    ax.text(0.31, 0.905, "0.90 bar", color=MUTED, fontsize=7, va="bottom")
    ax.text(0.80, 0.74, "dictionary = PCA", color=GRAY, fontsize=7, rotation=38)
    ax.set_xlim(0.3, 1.0), ax.set_ylim(0.3, 1.0)
    ax.set_xlabel("matched-error PCA, deep EM / clean")
    ax.set_ylabel("closed-loop dictionary, deep EM / clean")
    ax.legend(loc="lower right", fontsize=7, handletextpad=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_n1_closed_loop.pdf")
    plt.close(fig)


def fig_transcoders(root: Path, out: Path) -> None:
    n1, mlp = load(root, "n1.json"), load(root, "mlp_transcoders.json")
    models = ORIGINAL + FRESH
    fig, ax = plt.subplots(figsize=(6.6, 2.3))
    width = 0.38
    for i, name in enumerate(models):
        whole = max((r["deep_ratio"] for r in n1.get(name, {}).get("replacement_model", [])), default=None)
        mlp_ratio = mlp[name]["results"]["mlp_w4096_k32_s0"]["replacement"]["deep_ratio"]
        if whole is not None:
            ax.bar(i - width / 2 - 0.01, whole, width, color=GRAY, label="whole-block transcoder" if i == 0 else None)
        ax.bar(i + (width / 2 + 0.01 if whole is not None else 0), mlp_ratio, width, color=color(name),
               label=("MLP-only, tied" if name == "LOOP-T s1" else "MLP-only, untied" if name == "FF-U s3" else None))
        ax.text(i + (width / 2 + 0.01 if whole is not None else 0), mlp_ratio + 0.02, f"{mlp_ratio:.2f}",
                ha="center", fontsize=6.5, color=INK)
    ax.axvline(3.5, color=GRAY, lw=0.8, ls=":")
    ax.text(3.6, 1.12, "held-out fresh seeds", fontsize=7, color=MUTED)
    ax.set_xticks(range(len(models)), models, fontsize=7)
    ax.set_ylim(0, 1.2), ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("replacement deep EM / clean")
    ax.legend(loc="upper left", fontsize=7, ncol=3, bbox_to_anchor=(0, 1.14))
    fig.tight_layout()
    fig.savefig(out / "fig_transcoders.pdf")
    plt.close(fig)


def fig_e5(root: Path, out: Path) -> None:
    e5 = load(root, "e5_def.json")
    bases = ["features", "mixed", "rules", "heads"]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    width = 0.2
    shades = {"LOOP-T s1": 1.0, "LOOP-T s2": 0.7, "LOOP-T s3": 0.45, "FF-U s3": 1.0}
    for j, name in enumerate(ORIGINAL):
        classes = [c for c in e5[name]["classes"].values() if c["status"] != "no_effect"]
        shares = [sum(c["s90"][b]["accepted"] for c in classes) / len(classes) for b in bases]
        ax.bar([i + (j - 1.5) * width for i in range(len(bases))], shares, width * 0.92,
               color=color(name), alpha=shades[name], label=name)
    ax.set_xticks(range(len(bases)), bases)
    ax.set_ylim(0, 1), ax.set_ylabel("share of transitions accepted")
    ax.legend(fontsize=6.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "fig_e5_acceptance.pdf")
    plt.close(fig)


def fig_schedule(root: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4), sharey=True)
    for ax, arm, models in ((axes[0], "tied (LOOP-T)", ["LOOP-T s1", "LOOP-T s2", "LOOP-T s3", "LOOP-T s4", "LOOP-T s5", "LOOP-T s6"]),
                            (axes[1], "untied (FF-U)", ["FF-U s3", "FF-U s4", "FF-U s6"])):
        for name in models:
            path = root / f"query_patch_{name.replace(' ', '_')}.json"
            table = json.loads(path.read_text(encoding="utf-8"))[name]["pointer_chase"]
            visits = list(range(1, 9))
            hop = [max(v for k, v in table[str(t)].items() if k in ("1", "2", "3", "4")) for t in visits]
            control = [table[str(t)]["source_answer"] for t in visits]
            ax.plot(visits, hop, color=color(name), alpha=0.9 if name in FRESH else 0.55, marker="o", ms=3)
            ax.plot(visits, control, color=GRAY, lw=0.8, ls="--", alpha=0.7)
        ax.axhline(0.6, color=MUTED, lw=0.8, ls=":")
        ax.set_title(arm, fontsize=8, color=INK)
        ax.set_xlabel("visit (layer)")
        ax.set_xticks(range(1, 9))
    axes[0].set_ylabel("IIA, best intermediate hop")
    axes[0].plot([], [], color=TIED, marker="o", ms=3, label="intermediate node, hops 1-4")
    axes[0].plot([], [], color=GRAY, ls="--", lw=0.8, label="control: source's own answer")
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=7, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.0))
    axes[0].set_ylim(0, 1.05)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out / "fig_query_patch_schedule.pdf")
    plt.close(fig)


def fig_rank(root: Path, out: Path) -> None:
    n2, n2r = load(root, "n2.json"), load(root, "n2_rank16_32.json")
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    for name in ORIGINAL:
        a = n2[name]["families"]["pointer_chase"]["cells"]["hop/v8"]
        b = n2r[name]["families"]["pointer_chase"]["cells"]["hop/v8"]
        ranks = [1, 2, 4, 8, 16, 32]
        values = [a["iia"][str(r)] for r in ranks[:4]] + [b["iia"][str(r)] for r in ranks[4:]]
        ax.plot(ranks, values, color=color(name), marker="o", ms=3, alpha=0.8,
                label="answer at the last visit" if name == "LOOP-T s1" else None)
        das = load(root, f"query_das_{name.replace(' ', '_')}.json")[name]["pointer_chase"]
        early = max((c for c in das.values() if c["step"] <= 2), key=lambda c: c["full_block_iia"])
        ax.plot([8, 32, 128], [early["iia"][k] for k in ("8", "32", "128")], color=color(name), ls="--",
                marker="s", ms=3, alpha=0.8, label="early hop, query block" if name == "LOOP-T s1" else None)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 128], ["1", "2", "4", "8", "16", "32", "128"])
    ax.set_xlabel("subspace rank"), ax.set_ylabel("IIA")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig_rank.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("experiments/recurrence_monitorability_v0/phase2"))
    parser.add_argument("--output", type=Path, default=Path("papers/rmp_looped_mechinterp/figures"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for make in (fig_n1, fig_transcoders, fig_e5, fig_schedule, fig_rank):
        make(args.results, args.output)
        print("wrote", make.__name__, flush=True)


if __name__ == "__main__":
    main()

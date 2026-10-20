"""Aggregate cells -> tables, statistics, plots.

    python -m crrbench.analyze --run results/full
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .stats import cluster_bootstrap_mean, holm, mean_ci_t, paired_test

REG_ORDER = ["final", "random_intermediate", "random_exit"]
REG_LABEL = {"final": "final-loop", "random_intermediate": "random intermediate loop", "random_exit": "randomized exit depth"}
REG_COLOR = {"final": "#2a78d6", "random_intermediate": "#eb6834", "random_exit": "#1baf7a"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
TARGET_KAPPA = (2, 3, 4, 6, 8, 12)   # pre-specified budgets, in fixed-depth-equivalent loops (FLOP-matched)

plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                     "axes.titlesize": 10, "legend.frameon": False, "lines.linewidth": 1.8})


def load(run):
    cells = []
    for f in sorted(glob.glob(os.path.join(run, "cells", "*.json"))):
        cells.append(json.load(open(f)))
    return cells


def nz(x):
    return np.array([np.nan if v is None else v for v in x], float)


def interp_at(xs, ys, x):
    """Linear interpolation of a frontier; NaN outside its observed range."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    o = np.argsort(xs)
    xs, ys = xs[o], ys[o]
    if x < xs[0] - 1e-9 or x > xs[-1] + 1e-9:
        return np.nan
    return float(np.interp(x, xs, ys))


def frontier_gain(cell, group, ref="fixed", targets=TARGET_KAPPA):
    """Mean over pre-specified budgets of (controller acc - baseline acc) at equal FLOP-equivalent loops."""
    g = cell["controller"][group]
    kap_c = [p["kappa"] for p in g["ctrl"]]
    acc_c = [p["acc"] for p in g["ctrl"]]
    fx_k = (np.array(g["fixed"]["cost"]) - g["readout_F"]) / g["loopF_mean"]
    fx_a = g["fixed"]["acc"]
    if ref == "conf":
        rk = [p["kappa"] for p in g["conf"]]
        ra = [p["acc"] for p in g["conf"]]
    else:
        rk, ra = fx_k, fx_a
    diffs = []
    for b in targets:
        c, r = interp_at(kap_c, acc_c, b), interp_at(rk, ra, b)
        if not np.isnan(c) and not np.isnan(r):
            diffs.append(c - r)
    return (float(np.mean(diffs)) if len(diffs) >= 3 else np.nan), len(diffs)


def band(ax, x, ys, color, label=None, ls="-"):
    ys = np.array(ys, float)
    m = np.nanmean(ys, 0)
    if ys.shape[0] > 1:
        se = np.nanstd(ys, 0, ddof=1) / np.sqrt(np.sum(~np.isnan(ys), 0).clip(1))
        ax.fill_between(x, m - 1.96 * se, m + 1.96 * se, color=color, alpha=0.18, lw=0)
    ax.plot(x, m, color=color, label=label, ls=ls)


def by_task_regime(cells):
    d = defaultdict(list)
    for c in cells:
        d[(c["task"], c["regime"])].append(c)
    return d


def fmt_ci(x):
    m, lo, hi = mean_ci_t(x)
    return "n/a" if np.isnan(m) else f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def pfmt(p):
    return "n/a" if p is None or np.isnan(p) else (f"{p:.4f}" if p >= 1e-4 else "<1e-4")


def run_analysis(run):
    cells = load(run)
    if not cells:
        raise SystemExit("no cells found")
    out = os.path.join(run, "analysis")
    os.makedirs(out, exist_ok=True)
    tasks = sorted({c["task"] for c in cells})
    regs = [r for r in REG_ORDER if any(c["regime"] == r for c in cells)]
    TR = by_task_regime(cells)
    cfg = cells[0]["cfg"]
    Kt, Ke = cfg["K_train"], cfg["K_eval"]
    md = []
    summ = {"n_cells": len(cells), "tasks": tasks, "regimes": regs,
            "seeds": sorted({c["seed"] for c in cells})}

    # ---------------- 1. compute / retain table ----------------------------------------
    md.append("## 1. Compute and retain (mean over seeds, 95% t-interval)\n")
    for grp in ("id", "extrap"):
        md.append(f"### Group `{grp}` (lengths {cfg['groups'][grp]})\n")
        md.append("| task | regime | seeds | solved ever | first-correct loop (median) | acc @ loop 8 | acc @ final loop | stable-solve | lost after solve | forget hazard / loop |")
        md.append("|---|---|---|---|---|---|---|---|---|---|")
        for t in tasks:
            for r in regs:
                cs = TR.get((t, r), [])
                if not cs:
                    continue
                g = lambda k: [c["compute_retain"][grp][k] for c in cs]
                a8 = [c["compute_retain"][grp]["acc_at_depths"]["8"] for c in cs]
                md.append(f"| {t} | {r} | {len(cs)} | {fmt_ci(g('solved_frac'))} | {fmt_ci(nz(g('kstar_median')))} | {fmt_ci(a8)} | "
                          f"{fmt_ci(g('final_acc'))} | {fmt_ci(g('stable_solve_frac'))} | {fmt_ci(g('lost_given_solved'))} | {fmt_ci(g('forget_hazard'))} |")
        md.append("")
    md.append("Majority-class accuracy by held-out length (chance reference): " +
              "; ".join(f"{t}: " + ", ".join(f"N={n}:{v:.2f}" for n, v in sorted(cells[[c['task'] for c in cells].index(t)]['majority'].items(), key=lambda kv: int(kv[0])))
                        for t in tasks) + "\n")

    # ---------------- figure: accuracy vs loop depth --------------------------------------
    for grp_set, fname in ((("id", "extrap"), "fig1_accuracy_vs_loop.png"),):
        fig, axes = plt.subplots(len(tasks), len(grp_set), figsize=(4.6 * len(grp_set), 2.7 * len(tasks)), squeeze=False, sharey=True)
        for i, t in enumerate(tasks):
            for j, grp in enumerate(grp_set):
                ax = axes[i, j]
                for r in regs:
                    cs = TR.get((t, r), [])
                    if cs:
                        band(ax, np.arange(1, Ke + 1), [c["compute_retain"][grp]["acc_by_loop"] for c in cs], REG_COLOR[r], REG_LABEL[r])
                maj = np.mean([cells[0]["majority"][str(n)] for n in cfg["groups"][grp]]) if str(cfg["groups"][grp][0]) in cells[0]["majority"] else None
                if maj:
                    ax.axhline(maj, color=INK2, lw=0.8, ls=":")
                ax.axvline(Kt, color=INK2, lw=0.8, ls="--")
                ax.set_title(f"{t} — {'trained lengths' if grp == 'id' else 'held-out longer lengths'} {cfg['groups'][grp]}")
                ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8, 16, 32]); ax.set_xticklabels([1, 2, 4, 8, 16, 32])
                ax.set_ylim(0, 1.02)
                if j == 0:
                    ax.set_ylabel("accuracy")
                if i == len(tasks) - 1:
                    ax.set_xlabel("loop depth (dashed = max training depth, dotted = majority class)")
        axes[0, 0].legend(loc="lower right", fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(out, fname), dpi=150); plt.close(fig)

    # ---------------- figure: retention curves ------------------------------------------------
    fig, axes = plt.subplots(len(tasks), 2, figsize=(9.2, 2.7 * len(tasks)), squeeze=False, sharey=True)
    for i, t in enumerate(tasks):
        for j, grp in enumerate(("id", "extrap")):
            ax = axes[i, j]
            for r in regs:
                cs = TR.get((t, r), [])
                if cs:
                    ys = [nz(c["compute_retain"][grp]["retention_from_kstar"])[:Ke - 1] for c in cs]
                    band(ax, np.arange(len(ys[0])), ys, REG_COLOR[r], REG_LABEL[r])
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{t} — {grp} lengths {cfg['groups'][grp]}")
            if j == 0:
                ax.set_ylabel("P(still correct | first correct at k*)")
            if i == len(tasks) - 1:
                ax.set_xlabel("extra loops after first correct prediction")
    axes[0, 0].legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_retention_curves.png"), dpi=150); plt.close(fig)

    fig, axes = plt.subplots(len(tasks), 2, figsize=(9.2, 2.7 * len(tasks)), squeeze=False, sharey=True)
    for i, t in enumerate(tasks):
        for j, k0 in enumerate((4, 8)):
            ax = axes[i, j]
            for r in regs:
                cs = TR.get((t, r), [])
                if cs:
                    ys = [nz(c["compute_retain"]["extrap"]["retention_from_depth"][str(k0)]) for c in cs]
                    band(ax, np.arange(len(ys[0])), ys, REG_COLOR[r], REG_LABEL[r])
            ax.set_ylim(0, 1.02); ax.set_title(f"{t} — held-out lengths, start at loop {k0}")
            if j == 0:
                ax.set_ylabel("P(correct at k+Δ | correct at k)")
            if i == len(tasks) - 1:
                ax.set_xlabel("Δ loops")
    axes[0, 0].legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig2b_retention_from_depth.png"), dpi=150); plt.close(fig)

    # ---------------- 2. reuse vs freeze delay ---------------------------------------------------
    delays = cfg["delays"]
    md.append("## 2. Reuse after delayed freezing (held-out lengths unless stated)\n")
    md.append("Δ = reuse(d) − reuse(0) on the same samples, averaged per seed; test = paired across seeds (sign-flip, exact); Holm over all rows of this table.\n")
    rows, pv = [], []
    for kind in ("probe", "native"):
        for t in tasks:
            for r in regs:
                cs = TR.get((t, r), [])
                if not cs:
                    continue
                for d in delays[1:]:
                    def val(c, d0):
                        v = c["reuse"][kind]["heldout"].get(str(d0))
                        if v is None:
                            return np.nan
                        return v[-1] if isinstance(v, list) else v
                    diffs = [val(c, d) - val(c, 0) for c in cs]
                    base = [val(c, 0) for c in cs]
                    tst = paired_test(diffs)
                    rows.append((kind, t, r, d, np.nanmean(base), tst))
                    pv.append(tst["signflip_p"])
    adj = holm(pv)
    md.append("| reuse kind | task | regime | delay d | reuse(0) | Δ(d−0) [95% CI] | sign-flip p | Holm p |")
    md.append("|---|---|---|---|---|---|---|---|")
    for (kind, t, r, d, b0, tst), pa in zip(rows, adj):
        md.append(f"| {kind} | {t} | {r} | {d} | {b0:.3f} | {tst['mean']:+.3f} [{tst['ci_lo']:+.3f}, {tst['ci_hi']:+.3f}] | {pfmt(tst['signflip_p'])} | {pfmt(pa)} |")
    md.append("")
    summ["reuse_tests"] = [{"kind": k, "task": t, "regime": r, "delay": d, "base": b0, **ts, "holm_p": pa}
                           for (k, t, r, d, b0, ts), pa in zip(rows, adj)]
    # pooled delay effect (d=8 vs 0), cluster bootstrap over task x regime
    for kind in ("probe", "native"):
        groups = []
        for t in tasks:
            for r in regs:
                cs = TR.get((t, r), [])
                v = []
                for c in cs:
                    f = lambda d0: (c["reuse"][kind]["heldout"].get(str(d0)) if c["reuse"][kind]["heldout"].get(str(d0)) is not None else np.nan)
                    a, b = f(delays[-1]), f(0)
                    if isinstance(a, list):
                        a, b = a[-1], b[-1]
                    v.append(a - b)
                if v:
                    groups.append(v)
        if groups:
            m, lo, hi = cluster_bootstrap_mean(groups)
            md.append(f"Pooled Δ reuse(d={delays[-1]}) − reuse(0), {kind}: {m:+.4f} [95% cluster-bootstrap CI {lo:+.4f}, {hi:+.4f}] over {len(groups)} task×regime clusters.\n")
            summ[f"pooled_reuse_delta_{kind}"] = [m, lo, hi]

    fig, axes = plt.subplots(len(tasks), 2, figsize=(9.2, 2.7 * len(tasks)), squeeze=False)
    for i, t in enumerate(tasks):
        for j, (kind, ttl) in enumerate((("probe", "probe resume (trained MLP on frozen state)"), ("native", "native resume (zero-shot core, M loops)"))):
            ax = axes[i, j]
            for r in regs:
                cs = TR.get((t, r), [])
                if not cs:
                    continue
                for grp, ls in (("heldout", "-"), ("id", "--")):
                    ys = []
                    for c in cs:
                        row = []
                        for d in delays:
                            v = c["reuse"][kind][grp].get(str(d))
                            row.append(np.nan if v is None else (v[-1] if isinstance(v, list) else v))
                        ys.append(row)
                    band(ax, np.arange(len(delays)), ys, REG_COLOR[r], REG_LABEL[r] if (grp == "heldout" and i == 0 and j == 0) else None, ls=ls)
            ax.set_xticks(range(len(delays))); ax.set_xticklabels(delays); ax.set_ylim(0, 1.02)
            ax.set_title(f"{t} — {ttl}", fontsize=8.5)
            if i == len(tasks) - 1:
                ax.set_xlabel("extra loops after first correct prediction before freezing")
            if j == 0:
                ax.set_ylabel("reuse accuracy (solid = held-out, dashed = trained lengths)")
    axes[0, 0].legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig3_reuse_vs_delay.png"), dpi=150); plt.close(fig)

    # ---------------- 3. controller vs static recurrence --------------------------------------------
    md.append("## 3. Controller vs fixed-depth recurrence at matched FLOPs\n")
    md.append(f"Endpoint per model = mean over the pre-specified budgets κ ∈ {list(TARGET_KAPPA)} (mean loop-equivalents per sample; controller overhead counted in FLOPs) "
              "of [controller accuracy − fixed-depth accuracy at the same mean FLOPs], taking the controller frontier (10 compute prices λ) and the fixed-depth curve by linear interpolation "
              "(a fractional depth = mixture of floor/ceil depth). A model needs ≥3 budgets inside the controller's attainable range to count. "
              "Unit of inference = independent training seed. Holm correction across all task×regime tests in a group.\n")
    prim = {}
    for grp in ("heldout", "id"):
        for ref in ("fixed", "conf"):
            md.append(f"### Group `{grp}` — controller vs {'fixed-depth' if ref == 'fixed' else 'confidence-threshold early exit'}\n")
            md.append("| task | regime | seeds used | mean gain [95% CI] | % seeds > 0 | t-test p | sign-flip p | Wilcoxon p | Holm p | dz |")
            md.append("|---|---|---|---|---|---|---|---|---|---|")
            tests, labels = [], []
            for t in tasks:
                for r in regs:
                    cs = TR.get((t, r), [])
                    if not cs:
                        continue
                    gains = [frontier_gain(c, grp, ref)[0] for c in cs]
                    tst = paired_test(gains)
                    tests.append(tst); labels.append((t, r))
            adj = holm([x["signflip_p"] for x in tests])
            for (t, r), tst, pa in zip(labels, tests, adj):
                md.append(f"| {t} | {r} | {tst['n']} | {tst['mean']:+.4f} [{tst['ci_lo']:+.4f}, {tst['ci_hi']:+.4f}] | {100 * tst['frac_pos']:.0f}% | "
                          f"{pfmt(tst['t_p'])} | {pfmt(tst['signflip_p'])} | {pfmt(tst['wilcoxon_p'])} | {pfmt(pa)} | {tst['dz']:+.2f} |")
            groups = [[frontier_gain(c, grp, ref)[0] for c in TR[(t, r)]] for (t, r) in labels]
            if groups:
                m, lo, hi = cluster_bootstrap_mean(groups)
                md.append(f"\nPooled over {len(groups)} task×regime clusters: gain = {m:+.4f} [95% cluster-bootstrap CI {lo:+.4f}, {hi:+.4f}].\n")
                prim[(grp, ref)] = {"pooled": [m, lo, hi], "tests": [{"task": t, "regime": r, **ts, "holm_p": pa} for (t, r), ts, pa in zip(labels, tests, adj)]}
    summ["controller_vs_baselines"] = {f"{g}|{ref}": v for (g, ref), v in prim.items()}

    # controller diagnostics table at fixed compute prices
    md.append("## 4. Controller diagnostics (held-out lengths; mean over seeds)\n")
    md.append("Calibration: ECE / Brier of the controller's own P(correct if halted now) = Q_halt vs the raw softmax max-probability, on the steps the controller visited. "
              "Premature = stopped before the first correct loop (among items solvable within the loop cap); lost = returned a wrong answer after having been correct; "
              "excess = loops used beyond the first-correct loop when the answer was right; futile = loops spent on items never solved.\n")
    for lam in (0.004, 0.016):
        md.append(f"### Compute price λ = {lam}\n")
        md.append("| task | regime | mean loops | acc | fixed acc (same FLOPs) | ECE Q_halt | ECE softmax | Brier Q_halt | Brier softmax | premature | lost after correct | excess loops | futile loops | frac CONT/PRES/HALT |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for t in tasks:
            for r in regs:
                cs = TR.get((t, r), [])
                if not cs:
                    continue
                pts = [[p for p in c["controller"]["heldout"]["ctrl"] if abs(p["lam"] - lam) < 1e-12][0] for c in cs]
                f = lambda k: np.nanmean(nz([p[k] for p in pts]))
                af = np.nanmean([p["act_frac"] for p in pts], 0)
                md.append(f"| {t} | {r} | {f('mean_loops'):.2f} | {f('acc'):.3f} | {f('matched_fixed_acc'):.3f} | {f('ece_qhalt'):.3f} | {f('ece_softmax'):.3f} | "
                          f"{f('brier_qhalt'):.3f} | {f('brier_softmax'):.3f} | {f('premature_rate'):.3f} | {f('lost_after_correct_rate'):.3f} | "
                          f"{f('excess_loops'):.2f} | {f('futile_loops'):.2f} | {af[0]:.2f}/{af[1]:.2f}/{af[2]:.2f} |")
        md.append("")

    # pooled calibration test: Q_halt ECE < softmax ECE?
    d_ece = []
    for c in cells:
        pts = [p for p in c["controller"]["heldout"]["ctrl"] if abs(p["lam"] - 0.004) < 1e-12]
        if pts and pts[0]["ece_qhalt"] is not None and pts[0]["ece_softmax"] is not None:
            d_ece.append(pts[0]["ece_softmax"] - pts[0]["ece_qhalt"])
    if d_ece:
        tst = paired_test(d_ece)
        md.append(f"Calibration gain (ECE softmax − ECE Q_halt, λ=0.004, held-out lengths) over all {tst['n']} models: {tst['mean']:+.4f} "
                  f"[{tst['ci_lo']:+.4f}, {tst['ci_hi']:+.4f}], sign-flip p={pfmt(tst['signflip_p'])}.\n")
        summ["calibration_gain"] = tst

    # ---------------- figure: accuracy vs FLOPs --------------------------------------------------------
    for grp in ("heldout", "id"):
        fig, axes = plt.subplots(len(tasks), len(regs), figsize=(3.7 * len(regs), 2.7 * len(tasks)), squeeze=False, sharey="row")
        for i, t in enumerate(tasks):
            for j, r in enumerate(regs):
                ax = axes[i, j]
                cs = TR.get((t, r), [])
                ax.set_title(f"{t} — {REG_LABEL[r]}", fontsize=8.5)
                if not cs:
                    continue
                fk = np.mean([(np.array(c["controller"][grp]["fixed"]["cost"]) - c["controller"][grp]["readout_F"]) / c["controller"][grp]["loopF_mean"] for c in cs], 0)
                fa = np.mean([c["controller"][grp]["fixed"]["acc"] for c in cs], 0)
                ax.plot(fk, fa, color=INK2, ls="--", label="fixed depth")
                ck = np.mean([[p["kappa"] for p in c["controller"][grp]["ctrl"]] for c in cs], 0)
                ca = np.mean([[p["acc"] for p in c["controller"][grp]["ctrl"]] for c in cs], 0)
                ax.plot(ck, ca, color="#2a78d6", marker="o", ms=4, label="controller")
                tk = np.mean([[p["kappa"] for p in c["controller"][grp]["conf"]] for c in cs], 0)
                ta = np.mean([[p["acc"] for p in c["controller"][grp]["conf"]] for c in cs], 0)
                ax.plot(tk, ta, color="#eb6834", marker="s", ms=4, label="confidence-threshold exit")
                ok = np.mean([c["controller"][grp]["oracle"]["kappa"] for c in cs]); oa = np.mean([c["controller"][grp]["oracle"]["acc"] for c in cs])
                ax.scatter([ok], [oa], color="#6250d6", marker="*", s=70, zorder=5, label="oracle halting")
                ax.set_xscale("log", base=2); ax.set_xlim(0.9, Ke * 1.1); ax.set_ylim(0, 1.02)
                if i == len(tasks) - 1:
                    ax.set_xlabel("mean inference FLOPs (fixed-depth loop-equivalents)")
                if j == 0:
                    ax.set_ylabel("accuracy")
        axes[0, 0].legend(fontsize=7, loc="lower right")
        fig.tight_layout(); fig.savefig(os.path.join(out, f"fig4_accuracy_vs_flops_{grp}.png"), dpi=150); plt.close(fig)

    # ---------------- figure: calibration reliability ---------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
    for ax, key, ttl, col in ((axes[0], "rel_qhalt", "controller Q_halt", "#2a78d6"), (axes[1], "rel_softmax", "softmax max-prob", "#eb6834")):
        xs, ys, ws = [], [], []
        for c in cells:
            for p in c["controller"]["heldout"]["ctrl"]:
                if p.get(key):
                    for (cf, tr, n) in p[key]:
                        xs.append(cf); ys.append(tr); ws.append(n)
        if xs:
            ws = np.array(ws, float)
            ax.scatter(xs, ys, s=4 + 40 * ws / ws.max(), color=col, alpha=0.25, lw=0)
            order = np.argsort(xs); xs_, ys_, ws_ = np.array(xs)[order], np.array(ys)[order], ws[order]
            edges = np.linspace(0, 1, 11); mid, val = [], []
            for a, b in zip(edges[:-1], edges[1:]):
                m = (xs_ >= a) & (xs_ <= b)
                if m.sum() > 0:
                    mid.append(np.average(xs_[m], weights=ws_[m])); val.append(np.average(ys_[m], weights=ws_[m]))
            ax.plot(mid, val, color=col, lw=2)
        ax.plot([0, 1], [0, 1], color=INK2, ls=":")
        ax.set_title(ttl); ax.set_xlabel("predicted P(correct)"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axes[0].set_ylabel("observed accuracy (held-out lengths)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig5_calibration.png"), dpi=150); plt.close(fig)

    # ---------------- figure: gain forest -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(tasks) * len(regs) + 1.4))
    ylabels, ypos = [], 0
    for (t, r) in [(t, r) for t in tasks for r in regs if TR.get((t, r))]:
        for ref, off, col in (("fixed", 0.12, "#2a78d6"), ("conf", -0.12, "#eb6834")):
            g = [frontier_gain(c, "heldout", ref)[0] for c in TR[(t, r)]]
            m, lo, hi = mean_ci_t(g)
            ax.errorbar([m], [ypos + off], xerr=[[m - lo], [hi - m]], fmt="o", ms=3.5, color=col, lw=1.2, capsize=2,
                        label=("vs fixed-depth" if ref == "fixed" else "vs confidence-threshold") if ypos == 0 else None)
        ylabels.append(f"{t} / {r}"); ypos += 1
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(ypos)); ax.set_yticklabels(ylabels, fontsize=7.5); ax.invert_yaxis()
    ax.set_xlabel("accuracy gain at equal FLOPs, held-out lengths (mean over seeds, 95% CI)")
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig6_controller_gain.png"), dpi=150); plt.close(fig)

    open(os.path.join(out, "tables.md"), "w", encoding="utf-8").write("\n".join(md))
    json.dump(summ, open(os.path.join(out, "summary.json"), "w", encoding="utf-8"), default=lambda o: None if o != o else str(o), indent=1)
    print("\n".join(md)[:6000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    run_analysis(ap.parse_args().run)

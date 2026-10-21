"""Aggregate runs/*/results.json into plots (reports/figs/*.png), tables (reports/tables.md) and a checkpoint
manifest index (reports/checkpoint_manifest.json).

  python scripts/make_report.py [--runs runs] [--out reports]
"""
import argparse
import glob
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs")
ap.add_argument("--out", default="reports")
a = ap.parse_args()
FIG = os.path.join(a.out, "figs")
os.makedirs(FIG, exist_ok=True)

R, M = {}, {}
for d in sorted(glob.glob(os.path.join(a.runs, "*"))):
    n = os.path.basename(d)
    if os.path.exists(os.path.join(d, "results.json")):
        R[n] = json.load(open(os.path.join(d, "results.json")))
        M[n] = json.load(open(os.path.join(d, "manifest.json"))) if os.path.exists(os.path.join(d, "manifest.json")) else {}

OI = {"A": "#000000", "B": "#7f7f7f", "C": "#D55E00", "C2": "#E69F00", "D": "#0072B2", "E": "#56B4E9", "F": "#009E73",
      "G": "#CC79A7"}
Ts = [1, 2, 4, 8, 16]


def S(run, T=None, key="text_kl"):
    try:
        r = R[run]["per_loop"][str(T)]
        return r["summary"][key]
    except KeyError:
        return float("nan")


def COST(run, T, key):
    try:
        return R[run]["per_loop"][str(T)][key]
    except KeyError:
        return float("nan")


def have(*names):
    return all(n in R for n in names)


def se_acc(p, n):
    return math.sqrt(max(p * (1 - p), 1e-9) / max(n, 1))


METRICS = [("text_kl", "teacher-logit KL on held-out text (nats)", True),
           ("mc_acc_norm", "knowledge: ARC-E/ARC-C/PIQA acc_norm", False),
           ("algo_id_direct", "algorithmic, in-distribution lengths, direct answer", False),
           ("algo_ood_direct", "algorithmic, LONGER than trained, direct answer", False)]
REF_LINES = [("teacher_tuned", None, "A teacher", OI["A"], "-"), ("B_pruned", None, "B LaCo-pruned", OI["B"], "--"),
             ("C_lora_matched", None, "C LoRA (params-matched)", OI["C"], "-."),
             ("C_lora_r32", None, "C LoRA r=32", OI["C2"], ":")]

# ---------------------------------------------------------------- fig 1 ----
if "D_main" in R:
    fig, axs = plt.subplots(1, 4, figsize=(18, 3.8))
    for ax, (key, lab, logy) in zip(axs, METRICS):
        for run, T, name, col, ls in REF_LINES:
            v = S(run, None, key)
            if not np.isnan(v):
                ax.axhline(v, color=col, ls=ls, lw=1.3, label=name)
        for run, name, col in (("D_main", "D recurrent (trained)", OI["D"]), ("E_naive_repeat", "naive repeat, no training", OI["E"])):
            ys = [S(run, T, key) for T in Ts]
            ok = [i for i, y in enumerate(ys) if not np.isnan(y)]
            if ok:
                ax.plot([Ts[i] for i in ok], [ys[i] for i in ok], "-o", color=col, label=name, ms=4)
        ax.axvspan(8.5, 18, color="#eeeeee", zorder=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks(Ts)
        ax.set_xticklabels([str(t) for t in Ts])
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("recurrent steps T at inference (shaded: beyond trained T)")
        ax.set_title(lab, fontsize=9)
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_loop_scaling.png"), dpi=140)
    plt.close(fig)

# ---------------------------------------------------------------- fig 2 ----
fam = [("teacher_tuned", None, "A teacher", "*", OI["A"]), ("B_pruned", None, "B (m=7)", "s", OI["B"]),
       ("B_pruned_f2", None, "B (m=6)", "s", OI["B"]), ("B_pruned_f4", None, "B (m=4)", "s", OI["B"]),
       ("C_lora_matched", None, "C (m=7)", "^", OI["C"]), ("C_lora_f2", None, "C (m=6)", "^", OI["C"]),
       ("C_lora_f4", None, "C (m=4)", "^", OI["C"])]
if "D_main" in R:
    fig, axs = plt.subplots(1, 4, figsize=(18, 3.8))
    for ax, (key, lab, logy) in zip(axs, METRICS):
        for run, T, name, mk, col in fam:
            if run in R:
                ax.scatter(COST(run, T, "flops_per_token_256") / 1e9, S(run, T, key), marker=mk, color=col, s=45, zorder=3)
        xs, ys = [], []
        for T in Ts:
            f, y = COST("D_main", T, "flops_per_token_256") / 1e9, S("D_main", T, key)
            if not np.isnan(y):
                xs.append(f); ys.append(y)
                ax.annotate(f"T={T}", (f, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.plot(xs, ys, "-o", color=OI["D"], ms=4, label="D recurrent, T=1..16")
        cx = [COST(r, None, "flops_per_token_256") / 1e9 for r in ("B_pruned", "B_pruned_f2", "B_pruned_f4", "teacher_tuned") if r in R]
        cy = [S(r, None, key) for r in ("C_lora_matched", "C_lora_f2", "C_lora_f4", "teacher_tuned") if r in R]
        cx = [COST(r, None, "flops_per_token_256") / 1e9 for r in ("C_lora_matched", "C_lora_f2", "C_lora_f4", "teacher_tuned") if r in R]
        if len(cx) > 1:
            o = np.argsort(cx)
            ax.plot(np.array(cx)[o], np.array(cy)[o], "-^", color=OI["C"], ms=4, label="C: keep more real layers (m=7,6,4,0) + LoRA")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("forward GFLOPs / token (ctx 256, incl. LM head)")
        ax.set_title(lab, fontsize=9)
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_flops_frontier.png"), dpi=140)
    plt.close(fig)

# ---------------------------------------------------------------- fig 3 ----
LV = {"fsm": [("8", "id"), ("16", "id"), ("32", "ood"), ("48", "ood")],
      "add": [("3", "id"), ("6", "id"), ("9", "ood"), ("12", "ood")],
      "mul": [("(2, 1)", "id"), ("(3, 2)", "id"), ("(4, 2)", "ood"), ("(5, 3)", "ood")],
      "chain": [("3", "id"), ("6", "id"), ("9", "ood"), ("12", "ood")]}
if "D_main" in R:
    cond = [("teacher_tuned", None, "A", OI["A"], "-"), ("B_pruned", None, "B", OI["B"], "--"),
            ("C_lora_matched", None, "C", OI["C"], "-."), ("D_main", 1, "D T=1", OI["E"], "-"), ("D_main", 8, "D T=8", OI["D"], "-"),
            ("D_main", 16, "D T=16 (extrap.)", OI["G"], ":")]
    for fmt in ("direct", "scratch"):
        fig, axs = plt.subplots(1, 4, figsize=(17, 3.5))
        for ax, f in zip(axs, LV):
            for run, T, name, col, ls in cond:
                if run not in R:
                    continue
                ys = []
                for lv, sp in LV[f]:
                    try:
                        ys.append(R[run]["per_loop"][str(T)]["sets"][f"{f}|{lv}|{sp}|{fmt}"]["exact"])
                    except KeyError:
                        ys.append(float("nan"))
                ax.plot(range(4), ys, ls, color=col, marker="o", ms=4, label=name)
            ax.axvline(1.5, color="k", lw=0.7, ls=":")
            ax.set_xticks(range(4))
            ax.set_xticklabels([l for l, _ in LV[f]], fontsize=8)
            ax.set_title(f"{f}  ({fmt})  | right of dotted line = longer than training", fontsize=8)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.3)
        axs[0].legend(fontsize=7)
        axs[0].set_ylabel("exact match")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"fig3_length_{fmt}.png"), dpi=140)
        plt.close(fig)

# ---------------------------------------------------------------- fig 4 ----
ABL = [("adapter rank", ["abl_rank2", "abl_rank4", "abl_rank8", "abl_ref", "abl_rank32"], ["r=2", "r=4", "r=8", "r=16", "r=32"]),
       ("trained loop count", ["abl_loops1", "abl_loops2", "abl_loops4", "abl_ref"], ["1", "2", "4", "8"]),
       ("norm / adapter sharing", ["abl_norm_frozen", "abl_norm_shared", "abl_ref", "abl_adapter_shared", "abl_adapter_none", "abl_nogate"],
        ["norm frozen", "norm shared", "norm per-loop", "adapter shared", "no adapter", "no gate"]),
       ("objective / supervision", ["abl_obj_ce", "abl_obj_kl", "abl_ref", "abl_obj_kl_hidfinal", "abl_obj_kl_hidsteps", "abl_scratch0", "abl_scratch1", "abl_tfixed"],
        ["CE only", "KL only", "KL+CE", "+hid final", "+hid steps", "no scratch", "all scratch", "T fixed=8"])]
if "abl_ref" in R:
    def bestT(run):
        ts = [t for t in (8, 4, 2, 1) if str(t) in R[run]["per_loop"]]
        return ts[0] if ts else None
    fig, axs = plt.subplots(len(ABL), len(METRICS), figsize=(17, 3.0 * len(ABL)))
    for i, (title, runs, labs) in enumerate(ABL):
        for j, (key, lab, logy) in enumerate(METRICS):
            ax = axs[i, j]
            vals, ok_l = [], []
            for r, l in zip(runs, labs):
                if r in R:
                    T = R[r]["meta"]["site"] and max(int(k) for k in R[r]["per_loop"] if k != "None" and R[r]["per_loop"][k].get("supported", True))
                    vals.append(S(r, T, key)); ok_l.append(l)
            ax.bar(range(len(vals)), vals, color=OI["D"], alpha=0.8)
            if "abl_C_ref" in R and not np.isnan(S("abl_C_ref", None, key)):
                ax.axhline(S("abl_C_ref", None, key), color=OI["C"], ls="--", label="C params-matched")
            if "B_pruned" in R and not np.isnan(S("B_pruned", None, key)):
                ax.axhline(S("B_pruned", None, key), color=OI["B"], ls=":", label="B")
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels(ok_l, rotation=35, ha="right", fontsize=7)
            if logy:
                ax.set_yscale("log")
            if i == 0:
                ax.set_title(lab, fontsize=8)
            if j == 0:
                ax.set_ylabel(title, fontsize=9)
            ax.grid(alpha=0.3, axis="y")
    axs[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_ablations.png"), dpi=140)
    plt.close(fig)

# ---------------------------------------------------------------- fig 5 ----
if "D_main" in R:
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.6))
    for run, name, col in (("D_main", "D recurrent", OI["D"]), ("E_naive_repeat", "naive repeat", OI["E"])):
        for ax, key in zip(axs[:2], ("latency_ms_b8_s256", "act_peak_mb_b8_s256")):
            ys = [COST(run, T, key) for T in Ts]
            ax.plot(Ts, ys, "-o", color=col, label=name, ms=4)
    for run, name, col, ls in (("teacher_tuned", "A teacher", OI["A"], "-"), ("B_pruned", "B pruned", OI["B"], "--"),
                               ("C_lora_matched", "C LoRA", OI["C"], "-.")):
        for ax, key in zip(axs[:2], ("latency_ms_b8_s256", "act_peak_mb_b8_s256")):
            v = COST(run, None, key)
            if not np.isnan(v):
                ax.axhline(v, color=col, ls=ls, label=name)
    for ax, t in zip(axs[:2], ("forward latency, batch 8 x 256 (ms; GPU was shared)", "activation peak memory (MB)")):
        ax.set_xscale("log", base=2); ax.set_xticks(Ts); ax.set_xticklabels([str(x) for x in Ts])
        ax.set_xlabel("recurrent steps T"); ax.set_title(t, fontsize=9); ax.grid(alpha=0.3)
    axs[0].legend(fontsize=7)
    names = [n for n in ("teacher_tuned", "C_lora_matched", "C_lora_r32", "D_main") if n in R]
    axs[2].bar(range(len(names)), [R[n]["meta"]["train_peak_mb"] for n in names], color=OI["D"])
    axs[2].set_xticks(range(len(names))); axs[2].set_xticklabels(names, rotation=20, fontsize=7)
    axs[2].set_title("training peak GPU memory (MB, grad-ckpt on)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_cost.png"), dpi=140)
    plt.close(fig)

# ---------------------------------------------------------------- fig 6 ----
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
for n, col in (("C_lora_matched", OI["C"]), ("D_main", OI["D"]), ("teacher_tuned", OI["A"]), ("abl_obj_kl_hidsteps", OI["F"])):
    f = os.path.join(a.runs, n, "train_log.jsonl")
    if os.path.exists(f):
        rows = [json.loads(l) for l in open(f)]
        for ax, k in zip(axs, ("text_kl", "task_ce")):
            xs = [r["step"] for r in rows if k in r]
            ys = [r[k] for r in rows if k in r]
            if xs:
                ax.plot(xs, ys, color=col, label=n)
for ax, t in zip(axs, ("train: KL(teacher||student) on text", "train: CE on task targets")):
    ax.set_title(t, fontsize=9); ax.set_xlabel("step"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig6_training.png"), dpi=140)
plt.close(fig)

# ---------------------------------------------------------------- fig 7 ----
def recovery(run, T, key, logy):
    A, B = S("teacher_tuned", None, key), S("B_pruned", None, key)
    X = S(run, T, key)
    if logy:   # KL: teacher is 0 -> recovery = 1 - KL/KL_B
        return 1 - X / B
    return (X - B) / (A - B) if abs(A - B) > 1e-9 else float("nan")

if have("teacher_tuned", "B_pruned", "D_main"):
    keys = [("mc_acc_norm", "knowledge (MC)", False), ("text_kl", "text KL", True),
            ("algo_id_direct", "algorithmic ID", False), ("algo_ood_direct", "algorithmic OOD-length", False),
            ("algo_id_scratch", "algo ID w/ scratchpad", False)]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    w = 0.12
    rows = [("C_lora_matched", None, "C matched", OI["C"])] + [("D_main", T, f"D T={T}", c) for T, c in
                                                              zip(Ts, ["#9ecae1", "#6baed6", "#3182bd", OI["D"], OI["G"]])]
    for i, (run, T, name, col) in enumerate(rows):
        ys = [recovery(run, T, k, lg) for k, _, lg in keys]
        ax.bar(np.arange(len(keys)) + i * w, ys, w, label=name, color=col)
    ax.axhline(0, color="k", lw=0.8); ax.axhline(1, color="k", lw=0.8, ls=":")
    ax.set_xticks(np.arange(len(keys)) + 2.5 * w); ax.set_xticklabels([l for _, l, _ in keys], fontsize=8)
    ax.set_ylabel("recovery  (X - B) / (A - B)")
    ax.set_title("Fraction of the pruning loss recovered (0 = LaCo-pruned B, 1 = teacher A)", fontsize=9)
    ax.legend(fontsize=7, ncol=3); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig7_recovery.png"), dpi=140)
    plt.close(fig)

# ---------------------------------------------------------------- tables ----
def fmt(v, p=3):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:.{p}f}" if abs(v) >= 1e-3 or v == 0 else f"{v:.1e}")


rows = []
def add(run, T, label):
    if run not in R:
        return
    r = R[run]["per_loop"].get(str(T))
    if not r:
        return
    s = r["summary"]
    rows.append([label, fmt(R[run]["meta"]["trainable_params"] / 1e6, 2), fmt(r["flops_per_token_256"] / 1e9, 2), str(r["layer_evals"]),
                 fmt(s["text_kl"], 4), fmt(s["text1024_kl"], 4), fmt(s.get("mc_acc_norm", float("nan"))),
                 fmt(s["algo_id_direct"]), fmt(s["algo_ood_direct"]), fmt(s["algo_id_scratch"]), fmt(s["algo_ood_scratch"]),
                 fmt(r.get("latency_ms_b8_s256", float("nan")), 1), fmt(r.get("act_peak_mb_b8_s256", float("nan")), 0)])
add("teacher_tuned", None, "A teacher (tuned)")
if "teacher_tuned" in R and "base_vs_tuned" in R["teacher_tuned"]:
    b = R["teacher_tuned"]["base_vs_tuned"]["summary"]
    rows.append(["A0 base (untuned) vs tuned", "0", "–", "28", fmt(b["text_kl"], 4), fmt(b["text1024_kl"], 4), fmt(b.get("mc_acc_norm", float("nan"))),
                 fmt(b["algo_id_direct"]), fmt(b["algo_ood_direct"]), fmt(b["algo_id_scratch"]), fmt(b["algo_ood_scratch"]), "–", "–"])
for run, lab in (("B_pruned", "B LaCo-pruned (m=7)"), ("C_lora_matched", "C LoRA, params-matched to D"), ("C_lora_r32", "C LoRA r=32"),
                 ("B_pruned_f2", "B LaCo m=6 (FLOPs of D T=2)"), ("C_lora_f2", "C LoRA m=6 (FLOPs of D T=2)"),
                 ("B_pruned_f4", "B LaCo m=4 (FLOPs of D T=4)"), ("C_lora_f4", "C LoRA m=4 (FLOPs of D T=4)")):
    add(run, None, lab)
for T in Ts:
    add("E_naive_repeat", T, f"E naive repeat T={T} (no training)")
for T in Ts:
    add("D_main", T, f"D recurrent T={T}" + (" (extrapolated)" if T > 8 else ""))
hdr = ["condition", "trainable (M)", "GFLOPs/tok@256", "layer evals", "text KL", "text KL@1024", "MC acc_norm", "algo ID direct",
       "algo LONG direct", "algo ID scratch", "algo LONG scratch", "latency ms", "act MB"]
md = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)] + ["| " + " | ".join(r) + " |" for r in rows]

# ablation table (at largest supported T)
arows = []
for title, runs, labs in ABL:
    for r, l in zip(runs, labs):
        if r in R:
            Ts_ = [int(k) for k in R[r]["per_loop"] if k != "None" and R[r]["per_loop"][k].get("supported", True)]
            T = max(Ts_) if Ts_ else None
            s = R[r]["per_loop"][str(T)]["summary"]
            arows.append([title, l, str(T), fmt(R[r]["meta"]["trainable_params"] / 1e6, 2), fmt(s["text_kl"], 4), fmt(s.get("mc_acc_norm", float("nan"))),
                          fmt(s["algo_id_direct"]), fmt(s["algo_ood_direct"]), fmt(s["algo_id_scratch"])])
if "abl_C_ref" in R:
    s = R["abl_C_ref"]["per_loop"]["None"]["summary"]
    arows.append(["reference", "C params-matched (ablation scale)", "-", fmt(R["abl_C_ref"]["meta"]["trainable_params"] / 1e6, 2), fmt(s["text_kl"], 4),
                  fmt(s.get("mc_acc_norm", float("nan"))), fmt(s["algo_id_direct"]), fmt(s["algo_ood_direct"]), fmt(s["algo_id_scratch"])])
ah = ["ablation", "setting", "T eval", "trainable (M)", "text KL", "MC acc_norm", "algo ID direct", "algo LONG direct", "algo ID scratch"]
amd = ["| " + " | ".join(ah) + " |", "|" + "---|" * len(ah)] + ["| " + " | ".join(r) + " |" for r in arows]

# per-family detail (direct, T=8 vs refs)
fr = []
for fmt_ in ("direct", "scratch"):
    for run, T, lab in (("teacher_tuned", None, "A"), ("B_pruned", None, "B"), ("C_lora_matched", None, "C"), ("D_main", 1, "D T=1"), ("D_main", 8, "D T=8"), ("D_main", 16, "D T=16")):
        if run in R and str(T) in R[run]["per_loop"]:
            s = R[run]["per_loop"][str(T)]["summary"]
            fr.append([fmt_, lab] + [fmt(s[f"{f}_{sp}_{fmt_}"]) for f in LV for sp in ("id", "ood")])
fh = ["format", "cond"] + [f"{f} {sp}" for f in LV for sp in ("ID", "LONG")]
fmd = ["| " + " | ".join(fh) + " |", "|" + "---|" * len(fh)] + ["| " + " | ".join(r) + " |" for r in fr]

with open(os.path.join(a.out, "tables.md"), "w") as f:
    f.write("## Main comparison\n\n" + "\n".join(md) + "\n\n## Ablations (D at largest supported T, ablation-scale training)\n\n" + "\n".join(amd) +
            "\n\n## Per task family (exact match)\n\n" + "\n".join(fmd) + "\n")

# checkpoint manifest index
idx = {}
for n, m in M.items():
    idx[n] = {k: m.get(k) for k in ("cond", "trainable_params", "teacher_dir", "teacher_sha256", "laco_manifest", "laco_manifest_sha256",
                                    "checkpoint_files", "timestamp", "env")}
    idx[n]["args"] = m.get("args")
json.dump(idx, open(os.path.join(a.out, "checkpoint_manifest.json"), "w"), indent=1)
print("runs:", len(R), "| wrote figs to", FIG, "| tables.md | checkpoint_manifest.json")

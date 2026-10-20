"""Aggregate every run's JSON into summary tables (markdown + JSON) and cross-run figures.

Runs are discovered from out/res/<tag>__analysis.json  (t3).  Other stages are joined by the same tag when present:
  __interventions (t4), __poscontrol (t4b), __attribution (t7), __reuse (t8), __controller (t5), __hrm_sae (t6).
Mini-LaCo replica: out/res/mini_s*__{compare,interventions,build}.json
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np

import recsae  # noqa: F401
from recsae import ROOT
from recsae.io import fig_path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(ROOT, "out", "res")


def J(tag, kind):
    p = os.path.join(RES, f"{tag}__{kind}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def fmt(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def ci(v, d=3):
    return f"{v[0]:+.{d}f} [{v[1]:+.{d}f}, {v[2]:+.{d}f}]"


def short(tag):
    return tag.replace("sib_", "").replace("mini_dfa_s", "MINI-LaCo recovered student, seed ").replace("_rec", "").replace("dfa_random_exit_K16_s0", "dfa8/random_exit/s0 (own)")


def main():
    tags = sorted(os.path.basename(p)[: -len("__analysis.json")] for p in glob.glob(os.path.join(RES, "*__analysis.json")))
    out = {"runs": {}}
    lines = []
    # ---------------- Table 1: behaviour + SAE fidelity + stability
    lines += ["## Table 1. Loop behaviour, SAE fidelity and feature stability (held-out splits)", "",
              "| run | K_train | peak acc (loop) ID | final acc ID | lost-answer frac ID | peak acc OOD | lost OOD | SAE NMSE@K_train ID | NMSE@K_eval ID | Jaccard(t,t+1)@K_train ID | label-matched null |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in tags:
        A = J(t, "analysis")
        Kt = A["K_train"]
        b, bo = A["behaviour"]["test_id"], A["behaviour"]["test_ood"]
        f = A["fidelity"]["all"]["test_id"]["nmse_own"]
        s = A["stability"]["test_id"]
        lines.append(f"| {short(t)} | {Kt} | {fmt(b['best_acc'])} ({b['best_loop']}) | {fmt(b['final_acc'])} | {fmt(b['taxonomy']['lost'])} | "
                     f"{fmt(bo['best_acc'])} | {fmt(bo['taxonomy']['lost'])} | {fmt(f[Kt-1], 4)} | {fmt(f[-1], 4)} | {fmt(s['jacc'][Kt-1])} | {fmt(s['jacc_null'][Kt-1])} |")
        out["runs"][t] = {"K_train": Kt}
    # ---------------- Table 2: predictive value (delta AUROC vs confidence baseline)
    lines += ["", "## Table 2. Predictive value of recurrent-state signals: AUROC gain over the confidence+loop-index baseline (held-out; 95% bootstrap CI over examples)", "",
              "| run | target | split | base AUROC | +SAE scalars | +state-dynamics (non-SAE) | +SAE sparse code | +raw state |", "|---|---|---|---|---|---|---|---|"]
    summary_pred = {}
    names_ = {"base+sae": "SAE scalars", "base+dyn": "state dynamics (non-SAE)", "base+code": "SAE sparse code", "base+raw": "raw state"}
    for t in tags:
        A = J(t, "analysis")
        for tgt in ("stay", "harm", "gain"):
            for sp in ("test_id", "test_ood"):
                p = A["prediction"][tgt][sp]["probes"]
                lines.append(f"| {short(t)} | {tgt} | {sp} | {fmt(p['base']['auc'])} | {ci(p['base+sae']['delta_vs_base'])} | {ci(p['base+dyn']['delta_vs_base'])} | {ci(p['base+code']['delta_vs_base'])} | {ci(p['base+raw']['delta_vs_base'])} |")
                for k in ("base+sae", "base+dyn", "base+code", "base+raw"):
                    summary_pred.setdefault((tgt, sp, k), []).append(p[k]["delta_vs_base"])
    out["prediction_delta_by_target_split_probe"] = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in summary_pred.items()}
    lines += ["", "### Table 2b. Cross-run tally of Table 2 (n = number of runs; 'sig+' = 95% CI above 0, 'sig-' = below 0)", "",
              "| target | split | probe | n runs | mean ΔAUROC | sig+ | sig- |", "|---|---|---|---|---|---|---|"]
    for (tgt, sp, k), v in sorted(summary_pred.items()):
        v = np.array(v)
        lines.append(f"| {tgt} | {sp} | {names_[k]} | {len(v)} | {v[:, 0].mean():+.3f} | {(v[:, 1] > 0).sum()} | {(v[:, 2] < 0).sum()} |")
    # ---------------- Table 3: event-feature replication
    lines += ["", "## Table 3. Do features selected on ana_train replicate on held-out data? (top-50 by |AUROC-0.5|; signed deviation vs matched-null p95)", "",
              "| run | event | split | held-out signed dev | null p95 | sign agreement |", "|---|---|---|---|---|---|"]
    for t in tags:
        A = J(t, "analysis")
        for ev in ("now", "harm", "gain"):
            for sp in ("test_id", "test_ood"):
                e = A["event_features"][sp][ev]
                lines.append(f"| {short(t)} | {ev} | {sp} | {fmt(e['top50_mean_test_signed_dev'], 4)} | {fmt(e['null_signed_dev_p95'], 4)} | {fmt(e['top50_sign_agreement'], 2)} |")
    # ---------------- Table 4: loss attribution
    lines += ["", "## Table 4. Why do correct answers disappear? Exact SAE attribution of the margin change (correct class vs winning rival)", "",
              "| run | split | n lost | share: supporting features switched off | share: rival features switched on | offsetting | SAE residual | rival-gain from rare-in-training features |", "|---|---|---|---|---|---|---|---|"]
    attr_rows = []
    for t in tags:
        A = J(t, "attr")
        if not A:
            continue
        for sp in ("test_id", "test_ood"):
            r = A["loss_attribution"].get(sp, {})
            if "share_support_lost" in r:
                lines.append(f"| {short(t)} | {sp} | {r['n_lost']} | {fmt(r['share_support_lost'], 2)} | {fmt(r['share_rival_gain'], 2)} | {fmt(r['share_offsetting_other'], 2)} | {fmt(r['share_sae_residual'], 2)} | {fmt(r['rival_gain_from_rare_train_features'], 2)} |")
                attr_rows.append((short(t), sp, r))
    # ---------------- Table 5: interventions (positive control)
    lines += ["", "## Table 5. Causal interventions on held-out inputs: ablating the m=8 active features that write the predicted class (paired Δ accuracy)", "",
              "| run | split | edited loop t0 | Δacc at t0: target | random features | random direction | Δacc at K=32: target | random features | random direction |", "|---|---|---|---|---|---|---|---|---|"]
    for t in tags:
        P = J(t, "poscontrol")
        if not P:
            continue
        for sp in ("test_id", "test_ood"):
            rows = P["rows"][sp]
            mmax = max(r["m"] for r in rows)
            for r in rows:
                if r["m"] != mmax:
                    continue
                lines.append(f"| {short(t)} | {sp} | {r['t0']} | {ci(r['target_d_at_t0'], 3)} | {ci(r['rand_feat_d_at_t0'], 3)} | {ci(r['rand_dir_d_at_t0'], 3)} | {ci(r['target_d_final'], 3)} | {ci(r['rand_feat_d_final'], 3)} | {ci(r['rand_dir_d_final'], 3)} |")
    tally = {}
    for t in tags:
        P = J(t, "poscontrol")
        if not P:
            continue
        for sp in ("test_id", "test_ood"):
            for r in P["rows"][sp]:
                if r["m"] != max(x["m"] for x in P["rows"][sp]):
                    continue
                for lab in ("at_t0", "final"):
                    d_ = tally.setdefault((sp, lab), [0, 0, 0, 0, 0])
                    tg, rf, rd = r[f"target_d_{lab}"], r[f"rand_feat_d_{lab}"], r[f"rand_dir_d_{lab}"]
                    d_[0] += 1
                    d_[1] += int(tg[2] < 0)                       # target significantly negative
                    d_[2] += int(tg[1] > 0)                       # target significantly positive
                    d_[3] += int(rf[1] > 0 or rf[2] < 0)          # random features significant either way
                    d_[4] += int(rd[1] > 0 or rd[2] < 0)
    lines += ["", "### Table 5b. Tally over all (run, t0) cells of Table 5 (cells with a 95% CI excluding 0)", "",
              "| split | readout | cells | target significantly harmful | target significantly helpful | random features significant | random direction significant |", "|---|---|---|---|---|---|---|"]
    for (sp, lab), d_ in sorted(tally.items()):
        lines.append(f"| {sp} | {lab} | {d_[0]} | {d_[1]} | {d_[2]} | {d_[3]} | {d_[4]} |")
    # ---------------- Table 6: reuse
    lines += ["", "## Table 6. Hidden-state reuse (native resume of a frozen state + 4 new symbols, 8 more loops)", "",
              "| run | split | resume acc (delay 0) | from-scratch acc (same total loops) | AUROC base | +SAE recon | +SAE code | +raw state | held-out feature replication (signed dev / null p95) |", "|---|---|---|---|---|---|---|---|---|"]
    for t in tags:
        R_ = J(t, "reuse")
        if not R_:
            continue
        for sp in ("test_id", "test_ood"):
            rt = R_[f"rate_{sp}"]["0"]
            pr = R_["prediction"][sp]["probes"]
            rp = R_["feature_replication"][sp]
            lines.append(f"| {short(t)} | {sp} | {fmt(rt['resume'])} | {fmt(rt['fresh_same_total_loops'])} | {fmt(pr['base']['auc'])} | {ci(pr['base+recon']['delta_vs_base'])} | {ci(pr['base+code']['delta_vs_base'])} | {ci(pr['base+raw']['delta_vs_base'])} | {fmt(rp['top30_signed_dev_test'], 4)} / {fmt(rp['null_p95'], 4)} |")
    # ---------------- Table 7: stopping policies
    lines += ["", "## Table 7. Stopping policies (held-out): Pareto-area (accuracy vs FLOP-charged mean loops, budgets 2..cmax) and paired differences", ""]
    for t in tags:
        C = J(t, "controller")
        if not C:
            continue
        for sp in ("test_id", "test_ood"):
            pol = C["policies"][sp]
            areas = {k: v["area"] for k, v in pol.items() if isinstance(v, dict) and "area" in v}
            lines.append(f"- **{short(t)} / {sp}**: " + ", ".join(f"{k} {v:.3f}" for k, v in areas.items()) + ".  Paired Δarea: " +
                         "; ".join(f"{k.replace('delta_area','')} {ci(v, 4)}" for k, v in pol.items() if k.startswith("delta_area")))
    lines += ["", "## Table 8. HRM-style Q-controller with SAE-derived inputs (metric M = mean advantage over budget-matched fixed depth, over price grid and seeds)", "",
              "| run | split | base | +sae | +sae_shuf (control) | +dyn (non-SAE control) | +sae+dyn | Δ(+sae − base) | Δ(+dyn − base) | Δ(+sae_shuf − base) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for t in tags:
        H = J(t, "hrm_sae")
        if not H:
            continue
        for sp in ("test_id", "test_ood"):
            p = H["per"][sp]
            M = p["M"]
            lines.append(f"| {short(t)} | {sp} | {fmt(M['base'], 4)} | {fmt(M['+sae'], 4)} | {fmt(M['+sae_shuf'], 4)} | {fmt(M['+dyn'], 4)} | {fmt(M['+sae+dyn'], 4)} | "
                         f"{ci(p['delta[+sae - base]'], 4)} | {ci(p['delta[+dyn - base]'], 4)} | {ci(p['delta[+sae_shuf - base]'], 4)} |")
    # ---------------- mini replica
    for p in sorted(glob.glob(os.path.join(RES, "mini_s*__compare.json"))):
        C = json.load(open(p))
        lines += ["", f"## Table M. Mini-LaCo replica seed {C['seed']}: do recurrent adapters rebuild the features of the pruned layers?", "",
                  "| split | variant | code cosine to teacher output | Jaccard | feature corr (span-written) | feature corr (other) | acc | top-1 agreement w/ teacher | CKA |", "|---|---|---|---|---|---|---|---|---|"]
        for sp in ("test_id", "test_ood"):
            r = C[sp]
            B = r["B_output_agreement"]
            T = C.get("T", 5)
            for v, t_ in (("laco", 1), ("rec0", T), ("rec", T)):
                b = B[v][t_ - 1]
                e = [x for x in r["E_repr_vs_function"] if x["variant"] == v and x["t"] == t_][0]
                lines.append(f"| {sp} | {v} (T={t_}) | {fmt(b['cos'])} | {fmt(b['jacc'])} | {fmt(r['C_span_written'][v]['written_corr'])} | {fmt(r['C_span_written'][v]['other_corr'])} | {fmt(e['acc'])} | {fmt(e['top1_agree_teacher'])} | {fmt(e['cka_to_teacher_out'])} |")
            lines.append(f"| {sp} | reference: null / adjacent teacher layer | {fmt(B['null_label_matched']['cos'])} / {fmt(B['adjacent_teacher_layer']['cos'])} | {fmt(B['null_label_matched']['jacc'])} / {fmt(B['adjacent_teacher_layer']['jacc'])} | | | | | |")
    for p in sorted(glob.glob(os.path.join(RES, "mini_s*__interv.json"))):
        C = json.load(open(p))
        lines += ["", f"### Table M2. Mini-LaCo replica seed {C['seed']}: ablating SAE_T 'span-written' features in the teacher (at H[a+span]) vs in the recovered student (loop T); paired Δ accuracy", "",
                  "| split | m | teacher: target | teacher: random feat | student: target | student: random feat | student: random dir |", "|---|---|---|---|---|---|---|"]
        for sp in ("id", "ood"):
            for r in C[sp]["cross_model"]:
                lines.append(f"| {sp} | {r['m']} | {ci(r['teacher_target_d'])} | {ci(r['teacher_rand_feat_d'])} | {ci(r['student_target_d'])} | {ci(r['student_rand_feat_d'])} | {ci(r['student_rand_dir_d'])} |")
    open(os.path.join(ROOT, "reports", "tables.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    json.dump(out, open(os.path.join(RES, "aggregate.json"), "w"), indent=1)
    # ---------------- cross-run figures
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
    for t in tags:
        A = J(t, "analysis")
        K = A["K_eval"]
        axs[0].plot(range(1, K + 1), A["behaviour"]["test_id"]["acc_by_loop"], label=short(t), lw=1.2)
        axs[1].plot(range(1, K), A["stability"]["test_id"]["jacc"], lw=1.2)
        axs[2].plot(range(1, K + 1), A["fidelity"]["trainK"]["test_id"]["nmse_own"], lw=1.2)
    axs[0].set_title("answer accuracy by loop (held-out, in-distribution lengths)"); axs[0].set_xlabel("loop"); axs[0].legend(fontsize=6)
    axs[1].set_title("active-feature-set Jaccard, loop t vs t+1"); axs[1].set_xlabel("loop t")
    axs[2].set_title("SAE (trained on loops <= K_train) NMSE by loop"); axs[2].set_yscale("log"); axs[2].set_xlabel("loop")
    plt.tight_layout(); plt.savefig(fig_path("all", "x1_overview"), dpi=130); plt.close()
    # forest plot of predictive deltas
    fig, axs = plt.subplots(2, 3, figsize=(15, 7), sharex=False)
    names = {"base+sae": "SAE scalars", "base+dyn": "state dynamics (no SAE)", "base+code": "SAE sparse code", "base+raw": "raw state"}
    for r_, sp in enumerate(("test_id", "test_ood")):
        for c_, tgt in enumerate(("stay", "harm", "gain")):
            ax = axs[r_, c_]
            for k_i, k in enumerate(names):
                v = np.array(summary_pred.get((tgt, sp, k), []))
                if len(v):
                    ax.scatter(v[:, 0], np.full(len(v), k_i) + np.linspace(-0.25, 0.25, len(v)), s=18)
                    for j, (m, lo, hi) in enumerate(v):
                        ax.plot([lo, hi], [k_i + np.linspace(-0.25, 0.25, len(v))[j]] * 2, c="gray", lw=0.6)
            ax.axvline(0, c="k", lw=0.6); ax.set_yticks(range(len(names))); ax.set_yticklabels(list(names.values()), fontsize=8)
            ax.set_title(f"{sp}: Δ AUROC vs baseline, target '{tgt}'", fontsize=9)
    plt.tight_layout(); plt.savefig(fig_path("all", "x2_predictive_gain"), dpi=130); plt.close()
    if attr_rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        labs = [f"{a}/{b[-2:]}" for a, b, _ in attr_rows]
        sl = [r["share_support_lost"] for _, _, r in attr_rows]
        rg = [r["share_rival_gain"] for _, _, r in attr_rows]
        x = np.arange(len(labs))
        ax.bar(x - 0.2, sl, 0.4, label="supporting features switched off"); ax.bar(x + 0.2, rg, 0.4, label="rival features switched on")
        ax.set_xticks(x); ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=7); ax.set_ylabel("share of margin loss"); ax.legend(fontsize=7)
        plt.tight_layout(); plt.savefig(fig_path("all", "x3_loss_attribution"), dpi=130); plt.close()
    print("wrote reports/tables.md", len(lines), "lines;", len(tags), "runs")


if __name__ == "__main__":
    main()

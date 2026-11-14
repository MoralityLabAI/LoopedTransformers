"""Combine completed seed131/seed149 receipts without pooling repeated goals.

Usage: python audit/rlt_compare.py <directory-containing-seed131-and-seed149>
"""
import json
import sys
from pathlib import Path


def main():
    root = Path(sys.argv[1])
    reports = {seed: json.loads((root / f"seed{seed}" / "RLT-pilot-receipts.json").read_text(encoding="utf-8"))
               for seed in (131, 149)}
    assert [len(reports[s]["arms"]) for s in (131, 149)] == [6, 2]
    for seed, r in reports.items():
        assert r["protocol"]["identity"]["seed"] == seed
        assert not r["lean_evaluation"]["failed_files"], "resolve failed Lean files before comparing seeds"
        assert r["lean_evaluation"]["splits"] == ["val"]
    assert len({r["lean_evaluation"]["states_sha256"] for r in reports.values()}) == 1
    cohorts = [{(s["file"], s["line"], s["col"]): (s["reached"], s["certifiable"])
                for s in r["lean_item_outcomes"]} for r in reports.values()]
    assert cohorts[0] == cohorts[1], "different evaluation coverage; do not compare rates silently"
    requested = len(cohorts[0])
    scored = sum(cert for reached, cert in cohorts[0].values())
    ties = all(p["wins"] == p["losses"] == 0
               for r in reports.values() for key in ("paired_close_at_8", "paired_close_at_8_encoder32")
               for name, p in r[key].items() if "_rlt_" in name)
    lines = ["# RLT: two-lane small-model comparison", "",
             "Eight XS Mathlib tactic models completed: loop/HRM × direct/RLT/matched damping at seed 131, plus a fresh HRM direct/RLT pair at seed 149. Each received 1,000 updates, batch 16, and training encoder K=8.", "",
             "RLT closed exactly the same goals as direct feedback in every tested pair at both inference budgets. No Lean completion edge was observed in this pilot." if ties else
             "The paired completion outcomes below are exploratory; this small validation cohort does not establish a general evaluation advantage.", "",
             f"Both seeds use the same {requested} requested validation goals and {scored} certifiable goals. This is one cohort, not {2 * scored} independent goals. One scored goal changes the rate by {100/max(1, scored):.2f} percentage points. No registered test split was scored.", "",
             "## Lean-certified completion", "",
             "close@8 means at least one of up to eight proposed tactics closes the goal, passes the Lean kernel, and uses only the permitted axioms. Model+automation interleaves model and automation tactics within the same eight-attempt budget.", "",
             "| Seed | Arm | Model K=8 | Model K=32 | Model+auto K=8 | Model+auto K=32 |",
             "|---|---|---:|---:|---:|---:|"]
    def count(r, policy):
        c = r["lean_policy_counts"][policy]
        return f"{c['close@8']}/{c['denominator']}"
    for seed, r in reports.items():
        for name in r["arms"]:
            lines.append(f"| {seed} | {name.removesuffix('_s'+str(seed))} | {count(r, name)} | {count(r, name+'_T32')} | {count(r, name+'+auto')} | {count(r, name+'_T32+auto')} |")
    lines += [""]
    for seed, r in reports.items():
        c = r["lean_policy_counts"]["human"]
        lines.append(f"Seed {seed}: automation alone {count(r, 'auto')}; human-source tactic check {c['close@1']}/{c['denominator']}; zero failed files.")
    if all(r["policy_top1_text_counts"][n] == {"simp": requested}
           for r in reports.values() for n in r["arms"]):
        lines += ["", "Every model's primary-budget first proposal is plain `simp` on all 24 requested goals. This tactic-prior behavior limits what the short pilot can tell us about learned proof reasoning."]
    lines += ["", "| Seed | RLT versus direct | K=8 wins/losses | K=32 wins/losses |",
              "|---|---|---:|---:|"]
    for seed, r in reports.items():
        for name, p in r["paired_close_at_8"].items():
            if "_rlt_" in name:
                d = r["paired_close_at_8_encoder32"][name]
                lines.append(f"| {seed} | {name.split('_')[0]} | {p['wins']}/{p['losses']} | {d['wins']}/{d['losses']} |")
    lines += ["", "The seed-131 matched-damping comparisons and every per-goal outcome are retained in the individual receipts. The second seed has no damping arm.", "",
              "## Byte loss and loop diagnostics", "",
              "FP32 bits per tactic byte on the same 128 validation states; lower is better. Each budget is a separate trajectory. Teacher-forced exact tactic match is reported separately from proof completion in the seed reports.", "",
              "| Seed | Arm | Bits/byte K=8 | Bits/byte K=32 |",
              "|---|---|---:|---:|"]
    for seed, r in reports.items():
        for name, arm in r["arms"].items():
            d = arm["diagnostics"]["depths"]
            lines.append(f"| {seed} | {name.removesuffix('_s'+str(seed))} | {d['8']['bits_per_byte']:.6f} | {d['32']['bits_per_byte']:.6f} |")
    deltas = {s: r["arms"][f"hrm_rlt_s{s}"]["diagnostics"]["depths"]["8"]["bits_per_byte"] -
                 r["arms"][f"hrm_direct_s{s}"]["diagnostics"]["depths"]["8"]["bits_per_byte"]
              for s, r in reports.items()}
    lines += ["", f"At the trained budget, HRM RLT minus direct bits/byte is {deltas[131]:+.6f} at seed 131 and {deltas[149]:+.6f} at seed 149; these small changes need a fresh cohort and more seeds."]
    if all(d["teacher_forced_exact"] == 0 for r in reports.values() for a in r["arms"].values()
           for d in a["diagnostics"]["depths"].values()):
        lines += ["", "All eight models have zero teacher-forced reference-tactic exact matches on the 128-state sample at every measured budget. This metric is distinct from the nonzero certified completions above."]
    lines += ["", "Raw state norms precede readout normalization. Noise enters the initially zero fast state with RMS 0.001 times valid-token input-embedding RMS. Gain includes fast and HRM slow-state differences and divides by the initial noise norm. Four fixed goals and one noise direction do not estimate a worst-case bound.", "",
              "| Seed | Arm | Final norm K=32 | Final update norm K=32 | Full gain K=8 | Full gain K=32 |",
              "|---|---|---:|---:|---:|---:|"]
    for seed, r in reports.items():
        for name, arm in r["arms"].items():
            d = arm["diagnostics"]["depths"]
            p, q = [d[str(k)]["profiles"]["0.001"][-1] for k in (8, 32)]
            lines.append(f"| {seed} | {name.removesuffix('_s'+str(seed))} | {q['hidden_norm_mean']:.4g} | {q['update_norm_mean']:.4g} | {p['full_state_perturbation_gain']:.4g} | {q['full_state_perturbation_gain']:.4g} |")
    lines += ["", "![Seed 131, including matched damping](seed131/RLT-loop-diagnostics.png)", "",
              "![Seed 149 replication](seed149/RLT-loop-diagnostics.png)", "",
              "The figures show K=32 trajectories throughout; their loop-8 points are not independently run K=8 endpoints. RLT's smaller raw state scale does not by itself establish contraction or better reasoning. At seed 131, matched damping tracks the loop encoder's RLT loss/norm behavior and gives lower HRM K=32 norm and gain than RLT. The paper's assumptions are not guaranteed in this hybrid.", "",
              "## Resources and verification", "",
              "| Lane | Arms | Training elapsed hours | Peak training commit GiB | Proposal+Lean minutes | Peak evaluation commit GiB |",
              "|---|---:|---:|---:|---:|---:|"]
    for seed, r in reports.items():
        t = r["training_resources"]
        lines.append(f"| {seed} | {len(r['arms'])} | {t['seconds']/3600:.2f} | {t['peak_commit_gib']:.2f} | {r['lean_evaluation']['seconds']/60:.1f} | {r['completion']['peak_commit_gib']:.2f} |")
    hours = sum(r["training_resources"]["seconds"] for r in reports.values()) / 3600
    maxdiff = max(arm["diagnostics"]["cpu_reference"]["max_abs_logit_difference"]
                  for r in reports.values() for arm in r["arms"].values())
    lines += ["", f"Summed training-lane elapsed time, including pauses/restarts: {hours:.2f} hours, below the four-hour limit. Seed-149 GPU training overlapped seed-131 CPU Lean checking. The shared host and different resource segments prevent a fair feedback-overhead timing comparison. The second lane resumed at checkpoint 200; its exact discarded unsaved-update count was not captured.", "",
              f"CUDA training used bf16; diagnostics/proposals used fp32 with TF32 off, and Lean checking used one CPU worker. Each GPU phase held its own 4 GiB lease and process-tree memory cap. All eight one-goal K=8 CPU/CUDA reference checks passed; largest logit difference {maxdiff:.3g}. These reference checks do not validate every K=32 internal state numerically.", "",
              "The feedback/checkpoint/cache checks passed (13 distinct tests). Receipt generation verified checkpoint SHA256, matched within-family parameter/configuration identities, exact completion counts, and common evaluation coverage. Loop models have 1,488,227 parameters; HRM models have 1,473,419; feedback adds no parameters or block calls within a family.", "",
              "## Limits and next decision", "",
              "This is a short, exploratory validation pilot: two HRM seeds, one loop seed, and 23 unique certifiable goals. Repeated evaluation on this cohort cannot establish a held-out advantage. It transfers only the feedback schedule, without the paper's time/step embeddings or uniform all-step endpoint loss. Compute-depth mismatch is not task-length extrapolation. Q5 F1/F8, its false-negative question, the graph HRM transfer and the 400M model were not evaluated here; the original Q5 threshold and 400M no-go remain unchanged.", "",
              "Before scaling this line of work, train a small baseline that adds certified completions beyond automation, then compare RLT with direct feedback and damping across fresh seeds and a fresh cohort. The present scale-control effect alone does not justify a larger run.", "",
              "## Reproducible receipts", ""]
    for seed, r in reports.items():
        lines.append(f"- Seed {seed}, experiment source `{r['source_commit']}`: [report](seed{seed}/RLT-pilot-report.md), [full diagnostics and provenance](seed{seed}/RLT-pilot-receipts.json), [raw Lean item results](seed{seed}/RLT-Lean-item-results.jsonl.gz).")
    lines += ["", "Regenerate each seed report with `python audit/rlt_report.py <pilot-root> <report-dir> --commit <experiment-source>`, plot with `python audit/rlt_plot.py <report-dir>` (matplotlib), then combine with `python audit/rlt_compare.py <reports-root>`.", ""]
    (root / "RLT-comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"requested_unique_goals": requested, "certifiable_unique_goals": scored,
                      "all_rlt_direct_completion_pairs_tied": ties, "training_hours": hours}))


if __name__ == "__main__":
    main()

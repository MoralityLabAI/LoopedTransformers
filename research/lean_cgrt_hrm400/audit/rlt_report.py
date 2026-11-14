"""Build checked receipts from a completed pilot, without loading torch.

python audit/rlt_report.py <pilot-root> <report-dir> --commit <experiment-source>
python audit/rlt_plot.py <report-dir>
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def completes(result):
    cert = (result or {}).get("cert") or {}
    return bool(result and result.get("ok") and result.get("goals_after") == 0 and cert.get("kernel")
                and set(cert.get("axioms", [])) <= {"propext", "Classical.choice", "Quot.sound"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--commit", required=True)
    a = ap.parse_args()
    manifest = read(a.root / "pilot.json")
    done = read(a.root / "complete.json")
    evaluation = read(a.root / "lean_eval" / "eval.json")
    assert evaluation["splits"] == ["val"]
    assert evaluation["depths"] == [8, 32]
    arms = {}
    for name in done["arms"]:
        run = a.root / name
        config, complete = read(run / "config.json"), read(run / "complete.json")
        best = read(run / "best.json")
        assert complete["step"] == manifest["identity"]["steps"] == best["step"]
        assert hashlib.sha256((run / "best.pt").read_bytes()).hexdigest() == best["sha256"]
        arms[name] = {"config": config, "training": complete, "diagnostics": read(run / "diagnostics.json")}
    families = [arch for arch in ("loop", "hrm") if any(n.startswith(arch + "_") for n in arms)]
    for arch in families:
        pair = [arms[n] for n in arms if n.startswith(arch + "_")]
        assert len({r["config"]["parameters"]["total"] for r in pair}) == 1
        identities = []
        for row in pair:
            identity = json.loads(json.dumps(row["config"]["identity"]))
            identity["model"].pop("feedback", None)
            identities.append(identity)
        assert all(x == identities[0] for x in identities)
    diagnostic_devices = {r["diagnostics"].get("device", "cpu") for r in arms.values()}
    assert len(diagnostic_devices) == 1
    device = next(iter(diagnostic_devices))
    references = [r["diagnostics"].get("cpu_reference") for r in arms.values()]
    with gzip.open(a.root / "lean_eval" / "states.jsonl.gz", "rt", encoding="utf-8") as f:
        states = [json.loads(line) for line in f]
    raw_results = (a.root / "lean_eval" / "states.jsonl.gz").read_bytes()
    reached = [s for s in states if s["reached"] and s.get("main_prop", True) and s.get("main_closed", True)]
    assert len(states) == evaluation["summary"]["all"]["states"]
    assert len(reached) == evaluation["summary"]["all"]["certifiable"]
    policy_counts = {}
    for policy in evaluation["policy_depths"]:
        assert policy in states[0]["policies"]
    for policy in states[0]["policies"]:
        counts = {f"close@{k}": sum(any(completes(s["results"].get(t))
                                      for t in s["policies"][policy][:k]) for s in reached)
                  for k in (1, 2, 4, 8)}
        counts["valid@1"] = sum(bool(s["policies"][policy]) and
                               bool((s["results"].get(s["policies"][policy][0]) or {}).get("ok"))
                               for s in reached)
        for metric, count in counts.items():
            assert round(count / max(1, len(reached)), 4) == evaluation["summary"]["all"][policy][metric]
        policy_counts[policy] = {"denominator": len(reached), **counts}
    item_outcomes = []
    for s in states:
        scored = bool(s["reached"] and s.get("main_prop", True) and s.get("main_closed", True))
        item_outcomes.append({**{k: s[k] for k in ("file", "line", "col", "reached")},
                              "certifiable": scored,
                              "close_at_8": {p: any(completes(s["results"].get(t)) for t in ts[:8])
                                             for p, ts in s["policies"].items()} if scored else None})
    paired, paired_deep, damping_contrasts = {}, {}, {}
    for arch in families:
        baseline = next(n for n in arms if n.startswith(arch + "_direct_"))
        def success(s, policy):
            return any(completes(s["results"].get(t)) for t in s["policies"][policy][:8])
        for name in arms:
            if name.startswith(arch + "_") and name != baseline:
                wins = sum(success(s, name) and not success(s, baseline) for s in reached)
                losses = sum(success(s, baseline) and not success(s, name) for s in reached)
                paired[name] = {"baseline": baseline, "wins": wins, "losses": losses, "states": len(reached)}
                deep_name, deep_baseline = name + "_T32", baseline + "_T32"
                wins = sum(success(s, deep_name) and not success(s, deep_baseline) for s in reached)
                losses = sum(success(s, deep_baseline) and not success(s, deep_name) for s in reached)
                paired_deep[name] = {"baseline": baseline, "wins": wins, "losses": losses, "states": len(reached)}
        rlt = next((n for n in arms if n.startswith(arch + "_rlt_")), None)
        damping = next((n for n in arms if n.startswith(arch + "_matched_damping_")), None)
        if rlt and damping:
            damping_contrasts[arch] = {}
            for budget, suffix in ((8, ""), (32, "_T32")):
                wins = sum(success(s, rlt + suffix) and not success(s, damping + suffix) for s in reached)
                losses = sum(success(s, damping + suffix) and not success(s, rlt + suffix) for s in reached)
                damping_contrasts[arch][str(budget)] = {"rlt": rlt, "damping": damping,
                                                       "wins": wins, "losses": losses,
                                                       "ties": len(reached) - wins - losses}
    resource_path = a.root / "resource_samples.jsonl"
    resources = [json.loads(l) for l in resource_path.read_text().splitlines()] if resource_path.exists() else []
    cpu_resource_path = a.root / "cpu_diagnostic_resources.jsonl"
    cpu_resources = [json.loads(l) for l in cpu_resource_path.read_text().splitlines()] if cpu_resource_path.exists() else []
    segments_path = a.root / "resource_segments.jsonl"
    segments = [json.loads(l) for l in segments_path.read_text().splitlines()] if segments_path.exists() else []
    controls_path = a.root / "control_events.jsonl"
    controls = [json.loads(l) for l in controls_path.read_text().splitlines()] if controls_path.exists() else []
    report = {"protocol": manifest, "source_commit": a.commit, "completion": done,
              "training_resources": read(a.root / "training_complete.json"), "resource_samples": resources,
              "cpu_diagnostic_resources": cpu_resources,
              "resource_segments": segments,
              "control_events": controls,
              "diagnostic_noise_reference": "Noise RMS is epsilon times valid-token input-embedding RMS; the unperturbed initial fast state is zero.",
              "arms": arms, "lean_evaluation": evaluation, "paired_close_at_8": paired,
              "lean_results_artifact": {"file": "RLT-Lean-item-results.jsonl.gz",
                                        "sha256": hashlib.sha256(raw_results).hexdigest()},
              "lean_policy_counts": policy_counts, "lean_item_outcomes": item_outcomes,
              "paired_close_at_8_encoder32": paired_deep,
              "paired_rlt_vs_damping": damping_contrasts,
              "limitations": ["One training seed; exploratory val evaluation, not a registered test result.",
                              "Teacher-forced exact match is not proof completion.",
                              "Changing inference budget is not task-length extrapolation.",
                              "This transfers the feedback schedule to a final-loss tactic model; it does not reproduce the paper's time conditioning or uniform all-step endpoint supervision.",
                              "Shared-host scheduling and checkpoint restarts affect wall time; per-arm wall_s records the final segment, not comparable feedback overhead.",
                              "The 400M no-go and original Q5 threshold remain unchanged."]}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "RLT-Lean-item-results.jsonl.gz").write_bytes(raw_results)
    (a.out / "RLT-pilot-receipts.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# RLT small-model pilot", "", f"Exploratory Mathlib validation comparison: seed {manifest['identity']['seed']}, {len(arms)} feedback arms.", "",
             f"Source commit: `{a.commit}`. All arms trained for {manifest['identity']['steps']:,} steps, batch {manifest['identity']['batch']}, K=8.", "",
             "## Lean-certified completion", "",
             f"Shared validation sample: {evaluation['summary']['all']['states']} requested, {evaluation['summary']['all']['reached']} reached, {len(reached)} certifiable; {len(evaluation['failed_files'])} failed files.", "",
             "| Arm | Parameters | close@1 | close@8 | model+automation close@8 | wins/losses vs direct |", "|---|---:|---:|---:|---:|---:|"]
    summary = evaluation["summary"]["all"]
    def count_text(policy, metric):
        c = policy_counts[policy]
        return f"{c[metric]}/{c['denominator']} ({summary[policy][metric]:.1%})"

    for name, row in arms.items():
        p = paired.get(name)
        pair = f"{p['wins']}/{p['losses']}" if p else "reference"
        lines.append(f"| {name} | {row['config']['parameters']['total']:,} | {count_text(name, 'close@1')} | {count_text(name, 'close@8')} | {count_text(name+'+auto', 'close@8')} | {pair} |")
    lines += ["", f"Automation alone close@8: {count_text('auto', 'close@8')}. Human-source tactic check close@1: {count_text('human', 'close@1')}.", "",
              "## Lean completion with extra inference loops", "",
              "Same validation goals and eight tactic attempts, with encoder K=8 versus K=32. K=32 uses additional recurrent computation.", "",
              "| Arm | close@8, K=8 | close@8, K=32 | Difference | K=32 wins/losses vs direct |", "|---|---:|---:|---:|---:|"]
    for name in arms:
        normal, deep = summary[name]["close@8"], summary[name + "_T32"]["close@8"]
        pair = paired_deep.get(name)
        pair_text = f"{pair['wins']}/{pair['losses']}" if pair else "reference"
        lines.append(f"| {name} | {count_text(name, 'close@8')} | {count_text(name+'_T32', 'close@8')} | {(deep-normal)*100:+.1f} pp | {pair_text} |")
    if damping_contrasts:
        lines += ["", "Paired RLT wins/losses against matched damping (model close@8):", ""]
        for arch, budgets in damping_contrasts.items():
            lines.append(f"- {arch}: " + "; ".join(f"K={k}: {p['wins']}/{p['losses']}" for k, p in budgets.items()) + ".")
    lines += ["",
              "## Inference-depth robustness", "", "Bits per tactic byte on the same 128 validation states; lower is better. These are separate trajectories at each requested budget.", "",
              "| Arm | K=4 | K=8 (trained) | K=16 | K=32 |", "|---|---:|---:|---:|---:|"]
    for name, row in arms.items():
        values = row["diagnostics"]["depths"]
        lines.append("| " + name + " | " + " | ".join(f"{values[str(k)]['bits_per_byte']:.4f}" for k in (4, 8, 16, 32)) + " |")
    lines += ["", "Teacher-forced tactic exact match on those same 128 states:", "",
              "| Arm | K=4 | K=8 | K=16 | K=32 |", "|---|---:|---:|---:|---:|"]
    for name, row in arms.items():
        values = row["diagnostics"]["depths"]
        lines.append("| " + name + " | " + " | ".join(f"{values[str(k)]['teacher_forced_exact']:.1%}" for k in (4, 8, 16, 32)) + " |")
    lines += ["", "## Per-loop diagnostics", "", "At K=32, valid-token norms are measured before readout normalization. Noise is added to the initially zero fast state with RMS 0.001 times the valid-token input-embedding RMS. Gains divide the resulting state difference by the initial noise norm; they are finite-direction measurements, not worst-case bounds.", "",
              "| Arm | Final mean state norm | Final mean update norm | Terminal fast gain | Terminal fast+slow gain |", "|---|---:|---:|---:|---:|"]
    for name, row in arms.items():
        p = row["diagnostics"]["depths"]["32"]["profiles"]["0.001"][-1]
        lines.append(f"| {name} | {p['hidden_norm_mean']:.4g} | {p['update_norm_mean']:.4g} | {p['perturbation_gain']:.4g} | {p['full_state_perturbation_gain']:.4g} |")
    lines += ["", "![Per-loop hidden norms and perturbation gains](RLT-loop-diagnostics.png)", "",
              "Every plotted curve uses the K=32 schedule throughout. Its loop-8 point is not the endpoint of an independently run K=8 trajectory.", "",
              "Full per-loop traces for budgets 4/8/16/32 and all three noise magnitudes are in [the receipts](RLT-pilot-receipts.json). [Raw Lean item results](RLT-Lean-item-results.jsonl.gz) retain proposals and kernel-check outcomes.", "",
              "## Limits", "", *["- " + x for x in report["limitations"]], "",
              "Small differences on this sample need replication across seeds and a fresh evaluation cohort. Smaller norms alone are not evidence of better reasoning.", ""]
    method = f"Diagnostic device: {device}, fp32. Lean kernel checking runs on CPU."
    if all(references):
        method += f" Largest CPU/CUDA reference logit difference: {max(r['max_abs_logit_difference'] for r in references):.3g}."
    lines.insert(4, method + "\n")
    (a.out / "RLT-pilot-report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"arms": len(arms), "certifiable": len(reached), "paired": paired,
                      "rlt_vs_damping": damping_contrasts, "counts": policy_counts}, indent=2))


if __name__ == "__main__":
    main()

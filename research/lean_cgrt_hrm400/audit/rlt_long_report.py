"""Check and report the completed longer RLT lanes, keeping fresh/calibration separate.

python audit/rlt_long_report.py <experiment-root> <output-dir> --source-commit <sha>
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from rlt_report import completes


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--source-commit", required=True)
    a = ap.parse_args()
    registration = read(a.root / "protocol.json")
    evaluation_done = read(a.root / "evaluation" / "evaluation_complete.json")
    assert evaluation_done["protocol_sha256"] == digest(a.root / "protocol.json")
    evaluation = read(a.root / "evaluation" / "lean_eval" / "eval.json")
    assert evaluation["depths"] == [8, 32]
    assert evaluation["splits"] == ["calibration", "fresh"]
    assert evaluation["states_file_sha256"] == registration["cohort_file_sha256"]
    assert evaluation["states_sha256"] == registration["cohort_states_sha256"]
    assert not evaluation["failed_files"], "resolve failed Lean files before claiming a complete comparison"
    raw = (a.root / "evaluation" / "lean_eval" / "states.jsonl.gz").read_bytes()
    states = [json.loads(l) for l in gzip.decompress(raw).decode("utf-8").splitlines()]
    assert len(states) == registration["fresh_requested"] + registration["calibration_requested"]
    arms, lanes = {}, {}
    for arch in ("loop", "hrm"):
        lane_root = a.root / arch
        lane = read(lane_root / "training_complete.json")
        normalized = []
        for path in map(Path, lane["selected_paths"]):
            name = path.name
            cfg, best = read(path / "config.json"), read(path / "best.json")
            assert digest(path / "best.pt") == best["sha256"]
            assert best["phase_step"] == lane["phase_step"] and best["step"] == lane["total_step"]
            ident = cfg["identity"]
            expected_hash = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]
            assert expected_hash == cfg["identity_hash"]
            parent = registration["parents"][name]
            assert ident["continuation"]["checkpoint_sha256"] == parent["sha256"]
            assert digest(Path(parent["path"]) / "latest.pt") == parent["sha256"]
            identity = json.loads(json.dumps(ident))
            identity["model"].pop("feedback", None)
            for field in ("checkpoint_sha256", "parent_identity_hash"):
                identity["continuation"].pop(field)
            normalized.append(identity)
            arm_root = lane_root / name
            metrics = [json.loads(l) for l in (arm_root / "metrics.jsonl").read_text().splitlines()]
            diag = read(path / "diagnostics.json")
            assert diag["checkpoint_step"] == best["step"] and diag["device"] == "cuda"
            assert diag["cpu_reference"]["budget"] == 8
            arms[name] = {"config": cfg, "selected_checkpoint": best,
                          "snapshot": read(path / "snapshot.json"), "diagnostics": diag,
                          "training_metrics": metrics,
                          "actual_latest": read(arm_root / "latest.json"),
                          "selected_phase_step": lane["phase_step"], "selected_total_step": lane["total_step"]}
        assert len(normalized) == 3 and all(x == normalized[0] for x in normalized)
        assert len({r["config"]["parameters"]["total"] for n, r in arms.items() if n.startswith(arch + "_")}) == 1
        lanes[arch] = {"training": lane, "manifest": read(lane_root / "lane.json")}
        for filename in ("resource_segments.jsonl", "control_events.jsonl", "diversity_probes.jsonl", "resource_samples.jsonl"):
            p = lane_root / filename
            lanes[arch][filename.removesuffix(".jsonl")] = [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
    assert len(arms) == 6
    counts, outcomes, paired = {}, [], {}
    for split in ("calibration", "fresh"):
        sub = [s for s in states if s["split"] == split]
        scored = [s for s in sub if s["reached"] and s.get("main_prop", True) and s.get("main_closed", True)]
        summary = evaluation["summary"][split]
        assert len(scored) == summary["certifiable"] and len(sub) == summary["states"]
        policies = sub[0]["policies"]
        counts[split] = {}
        for policy in policies:
            row = {f"close@{k}": sum(any(completes(s["results"].get(t)) for t in s["policies"][policy][:k])
                                     for s in scored) for k in (1, 2, 4, 8)}
            row["valid@1"] = sum(bool(s["policies"][policy]) and bool((s["results"].get(s["policies"][policy][0]) or {}).get("ok"))
                                 for s in scored)
            for metric, count in row.items():
                assert round(count / max(1, len(scored)), 4) == summary[policy][metric]
            closing = [t for s in scored for t in s["policies"][policy][:8] if completes(s["results"].get(t))]
            row.update(denominator=len(scored), requested=len(sub),
                       top1_texts=dict(Counter(s["policies"][policy][0] if s["policies"][policy] else "<empty>" for s in sub)),
                       closing_tactic_texts=dict(Counter(closing)),
                       non_simp_close_at_8=sum(any(t != "simp" and completes(s["results"].get(t))
                                                 for t in s["policies"][policy][:8]) for s in scored),
                       beyond_auto_at_8=sum(any(completes(s["results"].get(t)) for t in s["policies"][policy][:8])
                                            and not any(completes(s["results"].get(t)) for t in s["policies"]["auto"][:8]) for s in scored))
            counts[split][policy] = row
        paired[split] = {}
        for arch in ("loop", "hrm"):
            for budget, suffix in ((8, ""), (32, "_T32")):
                for left, right in (("rlt", "direct"), ("matched_damping", "direct"), ("rlt", "matched_damping")):
                    lp, rp = [f"{arch}_{f}_s131{suffix}" for f in (left, right)]
                    wins = losses = 0
                    for s in scored:
                        l, r = [any(completes(s["results"].get(t)) for t in s["policies"][p][:8]) for p in (lp, rp)]
                        wins += l and not r; losses += r and not l
                    paired[split][f"{arch}_K{budget}_{left}_vs_{right}"] = {"wins": wins, "losses": losses,
                                                                          "ties": len(scored) - wins - losses}
        for s in sub:
            scored_flag = s["reached"] and s.get("main_prop", True) and s.get("main_closed", True)
            outcomes.append({k: s[k] for k in ("file", "line", "col", "split", "reached")} |
                            {"certifiable": bool(scored_flag), "closed_at_8":
                             {p: any(completes(s["results"].get(t)) for t in ts[:8]) for p, ts in s["policies"].items()}
                             if scored_flag else None})
    receipt = {"protocol": {"identity": {"seed": registration["seed"], "batch": registration["batch"]}},
               "registration": registration, "source_commit": a.source_commit,
               "lanes": lanes, "arms": arms, "lean_evaluation": evaluation,
               "evaluation_resources": evaluation_done, "counts_by_split": counts,
               "paired_close_at_8": paired, "item_outcomes": outcomes,
               "raw_results_sha256": hashlib.sha256(raw).hexdigest()}
    execution_path = a.root / "execution.json"
    execution = read(execution_path) if execution_path.exists() else {}
    receipt["execution"] = execution
    power_path = a.root / "power_pauses.json"
    receipt["power_pauses"] = read(power_path) if power_path.exists() else []
    samples_path = a.root / "resource_samples.jsonl"
    receipt["resource_samples"] = [json.loads(l) for l in samples_path.read_text().splitlines()] if samples_path.exists() else []
    lane_seconds = sum(r["training"]["elapsed_seconds"] for r in lanes.values())
    preflight_seconds = execution.get("successful_preflight_seconds", registration["preflight"]["measured_successful_seconds"])
    failed_bound = execution.get("failed_probe_elapsed_upper_bound_seconds", 20 * 60 - preflight_seconds)
    assert lane_seconds + preflight_seconds + failed_bound <= 8 * 3600, "training budget exceeded"
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "RLT-long-receipts.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (a.out / "RLT-Lean-item-results.jsonl.gz").write_bytes(raw)
    lines = ["# RLT-02: longer models on fresh Lean goals", "",
             "Two parallel lanes continued loop-XS and HRM-XS, each with direct, RLT and matched damping. The comparison uses exact completed common milestones within each architecture. Partial later training is retained but excluded from these scores.", "",
             "Each trial proposes up to eight one-step tactics at a frozen Lean goal. close@k counts goals closed by at least one of the first k tactics, with a standalone kernel certificate using only permitted axioms. valid@1 counts an accepted first tactic, even when goals remain. Model+auto interleaves model and automation candidates within the same eight-attempt budget. These are goal completions rather than complete theorem-search rollouts.", "",
             "## Training received", "",
             "| Architecture | Parameters per arm | Total updates per arm | Training presentations per arm | Budgeted lane hours |",
             "|---|---:|---:|---:|---:|"]
    for arch, lane_row in lanes.items():
        t = lane_row["training"]
        params = arms[f"{arch}_direct_s131"]["config"]["parameters"]["total"]
        lines.append(f"| {arch} | {params:,} | {t['total_step']:,} | {t['trained_presentations_at_selected']:,} | {t['elapsed_seconds']/3600:.2f} |")
    lines += ["", f"The new phase used batch {registration['batch']}, activation checkpointing, retained AdamW moments and a common re-warmup/cosine recipe. The original checkpoint supplied 1,000 batch-16 updates. Architectures may have reached different milestones; feedback arms within an architecture received matched training.", ""]
    def frac(split, policy, metric="close@8"):
        row = counts[split][policy]
        return f"{row[metric]}/{row['denominator']}"
    for split, title in (("fresh", "Fresh file-held-out goals (primary)"), ("calibration", "Old calibration goals (readiness check)")):
        summary = evaluation["summary"][split]
        lines += ["## " + title, "",
                  f"Requested {summary['states']}, reached {summary['reached']}, certifiable {summary['certifiable']}; zero failed files. Automation close@8: {frac(split, 'auto')}. Human-source close@1: {frac(split, 'human', 'close@1')}.", "",
                  "| Arm | close@1 | close@8 | valid@1 | Goals closed by a non-simp tactic | Added solves beyond auto | Model+auto close@8 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for name in arms:
            row = counts[split][name]
            lines.append(f"| {name} | {frac(split, name, 'close@1')} | {frac(split, name)} | {frac(split, name, 'valid@1')} | {row['non_simp_close_at_8']} | {row['beyond_auto_at_8']} | {frac(split, name+'+auto')} |")
        lines += ["", "Non-simp counts include basic tactics such as `rfl`; added solves beyond automation are the stronger test of useful proof generation. These are counts of goals, not counts of tactic strings.", "",
                  "| Arm | First-proposal distribution | Certified non-simp closing texts (frequency) |", "|---|---|---|"]
        for name in arms:
            row = counts[split][name]
            closing = {t: n for t, n in row["closing_tactic_texts"].items() if t != "simp"}
            # Keep arbitrary generated tactic text out of Markdown table syntax.
            show = lambda x: json.dumps(x, ensure_ascii=False).replace("|", "&#124;").replace("\n", " ")
            lines.append(f"| {name} | {show(row['top1_texts'])} | {show(closing)} |")
        lines += [""]
    lines += ["## Paired fresh-goal differences", "", "Eight tactic attempts per model; wins/losses count discordant goals.", "",
              "| Comparison | Wins | Losses | Ties |", "|---|---:|---:|---:|"]
    for name, p in paired["fresh"].items():
        lines.append(f"| {name} | {p['wins']} | {p['losses']} | {p['ties']} |")
    lines += ["", "## Extra inference loops", "", "Changing encoder compute is distinct from task-length extrapolation.", "",
              "| Arm | Fresh close@8, K=8 | Fresh close@8, K=32 |", "|---|---:|---:|"]
    for name in arms:
        lines.append(f"| {name} | {frac('fresh', name)} | {frac('fresh', name+'_T32')} |")
    lines += ["", "## Loop diagnostics", "", "FP32 teacher-forced bits per byte on the same 128 validation states; this is separate from fresh Lean completion. Each K starts its own trajectory.", "",
              "| Arm | K=4 | K=8 | K=16 | K=32 |", "|---|---:|---:|---:|---:|"]
    for name, arm in arms.items():
        lines.append("| " + name + " | " + " | ".join(f"{arm['diagnostics']['depths'][str(k)]['bits_per_byte']:.5f}" for k in (4, 8, 16, 32)) + " |")
    lines += ["", "| Arm | Final norm K=32 | Final update norm K=32 | Full perturbation gain K=8 | Full perturbation gain K=32 |", "|---|---:|---:|---:|---:|"]
    for name, arm in arms.items():
        p, q = [arm["diagnostics"]["depths"][str(k)]["profiles"]["0.001"][-1] for k in (8, 32)]
        lines.append(f"| {name} | {q['hidden_norm_mean']:.4g} | {q['update_norm_mean']:.4g} | {p['full_state_perturbation_gain']:.4g} | {q['full_state_perturbation_gain']:.4g} |")
    lines += ["", "![Per-loop state and perturbation measurements](RLT-loop-diagnostics.png)", "",
              "Noise enters the initially zero fast state with RMS 0.001 times input-embedding RMS. Full gain includes HRM slow slots and divides by initial noise norm. Four fixed validation goals and finite random directions do not establish a worst-case bound. Norms precede readout normalization; lower scale alone does not demonstrate contraction. Plotted K=32 prefixes are not independent K=8 runs.", "",
              "## Provenance and limits", "",
              f"Summed budgeted lane time: {lane_seconds/3600:.2f} hours. Successful training-only preflights took {preflight_seconds/60:.2f} minutes; the failed launch is conservatively bounded by {failed_bound/60:.2f} minutes. Combined upper bound: {(lane_seconds+preflight_seconds+failed_bound)/3600:.2f} hours, within the eight-hour training cap. Confirmed external power-outage waiting is excluded; wall elapsed time, excluded intervals, other interruptions and discarded unsaved work are recorded separately.", "",
              f"Experiment source starts at `{a.source_commit}`; scheduling amendments and exact process segments are retained in the receipts. [Full receipts](RLT-long-receipts.json) include checkpoint/parent SHA256, recipe identities, selected versus latest steps, training curves, diversity probes, resource segments, CPU/CUDA references and every paired goal outcome. [Raw Lean results](RLT-Lean-item-results.jsonl.gz) preserve tactic proposals and kernel certificates.", "",
              "The fresh cohort was frozen before continuation training: 64 goals from 33 test_file files, excluding every encoded input in all training/validation packed rows, the old pilot files, and duplicate fresh inputs. The 24 old calibration goals are reported separately. Fresh results never selected weights. Human-source failures, if any, remain visible in the counts and must be considered when assessing the pipeline.", "",
              "This is one training seed per architecture, continued from the short pilot. Larger batches and longer training change the training recipe relative to that pilot, but are matched across feedback arms here. Shared-host scheduling and restarts prevent fair wall-time overhead comparisons. The paper's time/step embeddings and uniform all-step endpoint loss are absent. Q5 F1/F8, task-length extrapolation, graph HRM and 400M scaling remain unevaluated; their existing gates remain unchanged.", ""]
    (a.out / "RLT-long-report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"counts": {s: {n: counts[s][n]["close@8"] for n in arms} for s in counts},
                      "paired_fresh": paired["fresh"]}))


if __name__ == "__main__":
    main()

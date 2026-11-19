"""Read-only, standard-library audit of the recovered Deuterium records.

Run after artifacts.py has authenticated and extracted the archive. Historical
Python files are read as evidence; none is imported or executed. JSONL byte
streams are hashed while parsing, and every exported row retains its source.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import re
from typing import Any

RUNS = "sources/D/Research_Engine/runs"
ARENA = f"{RUNS}/metta_organelle_arena_paper_20260601T024147Z"
ORIGINAL = f"{RUNS}/metta_organelle_arena_full_20260531T180830Z"
WORDLE = f"{RUNS}/trm_wordle_vpd_hook_probe_20260603"
RECOVERED_REPO = "sources/D/projects/VDP"
DATASET = ("sources/D/VPDExperiments/VPD_2026-05-12_2026-07-15/"
           "raw/repo/artifacts/powered_int3_logic/gym_out/dataset")
SIGNALS = ("repair_preservation_rate_delta", "target_preservation_rate_delta",
           "logic_exact_on_commits_delta", "logic_cell_accuracy_on_commits_delta")


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Sources:
    def __init__(self, root: Path):
        self.root = root
        self.refs: dict[str, dict[str, Any]] = {}

    def register(self, path: str) -> dict[str, Any]:
        if path not in self.refs:
            p = self.root / path
            digest = hashlib.sha256()
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(chunk)
            self.refs[path] = {"path": path, "bytes": p.stat().st_size,
                               "sha256": digest.hexdigest()}
        return self.refs[path]

    def json(self, path: str) -> Any:
        self.register(path)
        return json.loads((self.root / path).read_text(encoding="utf-8-sig"))

    def jsonl(self, path: str):
        digest = hashlib.sha256()
        count = 0
        with (self.root / path).open("rb") as f:
            for line_number, raw in enumerate(f, 1):
                digest.update(raw)
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw.decode("utf-8-sig"))
                except (ValueError, UnicodeError) as exc:
                    raise ValueError(f"Malformed {path}:{line_number}") from exc
                count += 1
                yield line_number, row
        self.refs[path] = {"path": path, "bytes": (self.root / path).stat().st_size,
                           "sha256": digest.hexdigest(), "jsonl_rows": count}

    def rows(self, path: str) -> list[dict[str, Any]]:
        return [r for _, r in self.jsonl(path)]


def rank_key(row: dict[str, Any]) -> tuple[float, ...]:
    d = row["deltas"]
    return (max(float(d[k]) for k in SIGNALS),
            -float(d["false_commit_rate_delta"]),
            -float(d["false_reject_rate_delta"]), float(d["accuracy_delta"]),
            -float(row["mean_abs_logit_delta"]))


def guardrail(row: dict[str, Any]) -> bool:
    d = row["deltas"]
    return (float(row["scale"]) != 0 and float(row["mean_abs_logit_delta"]) > 0
            and float(d["accuracy_delta"]) >= -0.05
            and float(d["false_commit_rate_delta"]) <= 0
            and float(d["false_reject_rate_delta"]) <= 0.05
            and max(float(d[k]) for k in SIGNALS) >= 0)


def positive(row: dict[str, Any]) -> bool:
    return max(float(row["deltas"][k]) for k in (*SIGNALS, "accuracy_delta")) > 0


def graft_id(row: dict[str, Any], include_scale: bool = True) -> tuple[Any, ...]:
    key = (row["seed"], row["source_organelle_id"], row["target_organelle_id"],
           row["component_id"], row["component_source"], row["control_family"])
    return (*key, row["scale"]) if include_scale else key


def audit_arena(src: Sources, out: Path) -> dict[str, Any]:
    summary = src.json(f"{ARENA}/arena_summary.json")
    manifest = src.json(f"{ARENA}/run_manifest.json")
    registry = src.json(f"{ARENA}/organelle_registry.json")
    sweep = f"{ARENA}/graft_sweeps.jsonl"
    src.register(f"{RECOVERED_REPO}/param_decomp/experiments/metta_organelle_arena.py")
    counts: Counter[str] = Counter()
    by_seed: dict[int, Counter[str]] = defaultdict(Counter)
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    mismatches: Counter[str] = Counter()
    random_best: dict[tuple[Any, ...], tuple[tuple[float, ...], int, dict[str, Any]]] = {}
    candidates: list[tuple[int, dict[str, Any]]] = []
    recorded_accepted: list[tuple[int, dict[str, Any]]] = []
    for line, row in src.jsonl(sweep):
        seed = int(row["seed"])
        family = row["component_source"]
        g, p = guardrail(row), positive(row)
        counts["rows"] += 1
        by_seed[seed]["rows"] += 1
        source_counts[family]["rows"] += 1
        for key, computed in (("guardrail_pass", g), ("positive_signal", p),
                              ("stable_graft", g and not p)):
            if row.get(key) is not computed:
                mismatches[key] += 1
        if row.get("accepted"):
            recorded_accepted.append((line, row))
            counts["recorded_accepted"] += 1
            by_seed[seed]["recorded_accepted"] += 1
        if g and p:
            source_counts[family]["positive_guarded"] += 1
        if family == "vdp_or_class_row":
            counts["candidate_rows"] += 1
            by_seed[seed]["candidate_rows"] += 1
            if g and not p:
                counts["stable_candidate_rows"] += 1
                by_seed[seed]["stable_candidate_rows"] += 1
            if g and p:
                candidates.append((line, row))
        if family == "random_rank1" and g:
            group = (seed, row["target_organelle_id"])
            key = rank_key(row)
            if group not in random_best or key > random_best[group][0]:
                random_best[group] = (key, line, row)
    accepted = []
    for line, row in candidates:
        group = (int(row["seed"]), row["target_organelle_id"])
        best_key, best_line, best_row = random_best.get(group, ((-999.0,) * 5, 0, {}))
        beats = rank_key(row) > best_key
        if row.get("beats_random_control") is not beats:
            mismatches["beats_random_control_on_positive_guarded_candidates"] += 1
        if bool(row.get("accepted")) != beats:
            mismatches["accepted_on_positive_guarded_candidates"] += 1
        if beats:
            by_seed[int(row["seed"])]["recomputed_accepted"] += 1
            tie = rank_key(row)[:4] == best_key[:4]
            all_deltas_equal = row["deltas"] == best_row.get("deltas")
            counts["accepted_task_metrics_tie_random"] += int(tie)
            counts["accepted_task_metrics_strictly_beat_random"] += int(not tie)
            counts["accepted_all_recorded_task_deltas_equal"] += int(all_deltas_equal)
            accepted.append({"source": sweep, "line": line, "record": row,
                             "random_comparator_line": best_line,
                             "random_comparator_rank": list(best_key),
                             "random_comparator": best_row,
                             "task_metric_tie_random": tie,
                             "all_recorded_task_deltas_equal": all_deltas_equal})
    counts["recomputed_accepted"] = len(accepted)
    bad_accept = [line for line, row in recorded_accepted
                  if row["component_source"] != "vdp_or_class_row" or not guardrail(row) or not positive(row)]
    mismatches["accepted_without_source_guard_or_positive"] = len(bad_accept)
    preview = summary["best_grafts"]
    preview_ids = {graft_id(r) for r in preview if r.get("accepted")}
    preview_counts = Counter(int(r["seed"]) for r in preview if r.get("accepted"))
    missing_preview = [a for a in accepted if graft_id(a["record"]) not in preview_ids]
    seed_rows = []
    for seed in sorted(by_seed):
        c = by_seed[seed]
        seed_rows.append({"seed": seed, "graft_rows": c["rows"],
                          "candidate_rows": c["candidate_rows"],
                          "accepted": c["recomputed_accepted"],
                          "summary_preview_accepted": preview_counts[seed],
                          "stable_candidate_rows": c["stable_candidate_rows"]})
    write_csv(out / "replication_by_seed.csv", seed_rows)
    with (out / "accepted_grafts.jsonl").open("w") as f:
        for a in accepted:
            f.write(json.dumps(a, sort_keys=True, allow_nan=False) + "\n")
    original = src.json(f"{ORIGINAL}/arena_summary.json")
    return {
        "evidence_class": "independently_reconciled_archived_rows",
        "counts": dict(counts), "by_seed": seed_rows,
        "record_flags_mismatch_counts": dict(mismatches),
        "source_families": {k: dict(v) for k, v in sorted(source_counts.items())},
        "summary_reported_accepted": summary["accepted_count"],
        "summary_reported_graft_records": summary["graft_record_count"],
        "summary_reported_stable_grafts": summary["stable_graft_count"],
        "registry_entries": len(registry),
        "summary_preview_length": len(preview),
        "omitted_preview_source_lines": [a["line"] for a in missing_preview],
        "unique_accepted_seed_source_target_component": len({graft_id(a["record"], False) for a in accepted}),
        "accepted_component_types": dict(Counter(a["record"]["component_id"].split(":")[0] for a in accepted)),
        "accepted_target_gate_families": dict(Counter(a["record"]["target_gate_family"] for a in accepted)),
        "recorded_invocation": manifest["command"],
        "run_commit_authenticated": False,
        "original_full_arena": {
            "evidence_class": "recovered_summary_only",
            "summary_source": f"{ORIGINAL}/arena_summary.json",
            "accepted_count": original["accepted_count"],
            "graft_record_count": original["graft_record_count"],
            "organelle_count": original["organelle_count"],
            "stable_graft_count": original["stable_graft_count"],
            "sweep_present": (src.root / ORIGINAL / "graft_sweeps.jsonl").exists(),
            "missing_description": "Recovery package says full-arena sweep was not requested; absent here, not proved lost."},
        "limitations": [
            "Head SVD and class-row directions are labeled vdp_or_class_row; no learned adversarial decomposition in this arena.",
            "Acceptance permits five percentage points of accuracy loss and false-reject increase.",
            "beats_random_control is lexicographic and can be won solely by lower output perturbation at equal task metrics.",
            "Search scores and acceptance use the same scoring rows; no prospective fresh test or multiplicity correction is established.",
            "Scale variants, shared target rows and overlapping gate families make accepted rows dependent.",
            "Source-code agreement and count replay do not authenticate a historical execution commit or regenerate logits."]}


def sample_key(row: dict[str, Any], default_env: str = "", include_step: bool = False) -> str:
    key = f"{row.get('env_id', default_env)}:{row['trajectory_id']}"
    return f"{key}:step{row['step']}" if include_step else key


def overlap_analysis(cards: list[dict[str, Any]], env: str, include_step: bool = False) -> dict[str, Any]:
    sets = [{sample_key(r, env, include_step) for r in c["rows"]} for c in cards]
    pairwise = [{"left": cards[i]["name"], "right": cards[j]["name"],
                 "overlap_ids": len(sets[i] & sets[j])}
                for i, j in itertools.combinations(range(len(cards)), 2)]
    folds = []
    for i, card in enumerate(cards):
        others = set().union(*(s for j, s in enumerate(sets) if i != j))
        folds.append({"card": card["name"], "rows": len(card["rows"]),
                      "unique_ids": len(sets[i]), "shared_with_other_cards": len(sets[i] & others),
                      "unseen_in_other_cards": len(sets[i] - others),
                      "unseen_ids": sorted(sets[i] - others)})
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for card in cards:
        for row in card["rows"]:
            by_id[sample_key(row, env, include_step)].append(row)
    conflict_targets = [k for k, rs in by_id.items() if len({r["target_action"] for r in rs}) > 1]
    score_variants = [k for k, rs in by_id.items()
                      if len({json.dumps(r["scores"], sort_keys=True) for r in rs}) > 1]
    return {"identity": "env:trajectory_id:step" if include_step else "env:trajectory_id",
            "raw_rows": sum(len(c["rows"]) for c in cards), "unique_ids": len(by_id),
            "pairwise": pairwise, "folds": folds, "target_conflict_ids": conflict_targets,
            "score_variant_ids": score_variants}


def score_policy(rows: list[dict[str, Any]], policy: dict[str, Any], wordle: bool = False) -> dict[str, Any]:
    details = []
    for row in rows:
        scores = sorted(row["scores"], key=lambda s: float(s["logprob"]), reverse=True)
        base = scores[0]["action"]
        if wordle:
            active = bool(row.get("post_initial_gate"))
            action = policy["penalized_action"]
        else:
            active = (len(scores) >= 2 and base == policy["top_action"]
                      and scores[1]["action"] == policy["runner_up_action"]
                      and float(scores[0]["logprob"]) - float(scores[1]["logprob"]) <= policy["max_margin"])
            action = base
        prediction = max(scores, key=lambda s: float(s["logprob"]) -
                         (float(policy["penalty"]) if active and s["action"] == action else 0))["action"]
        details.append({"trajectory_id": row["trajectory_id"],
                        "baseline_hit": base == row["target_action"],
                        "hit": prediction == row["target_action"], "prediction": prediction,
                        "baseline_prediction": base, "target_action": row["target_action"],
                        "active": active})
    n = len(details)
    base_hits = sum(r["baseline_hit"] for r in details)
    hits = sum(r["hit"] for r in details)
    return {"n": n, "baseline_hits": base_hits, "policy_hits": hits,
            "baseline_accuracy": base_hits / n if n else None,
            "policy_accuracy": hits / n if n else None,
            "delta": (hits - base_hits) / n if n else None,
            "rescues": sum(not r["baseline_hit"] and r["hit"] for r in details),
            "damages": sum(r["baseline_hit"] and not r["hit"] for r in details),
            "details": details}


def audit_arc(src: Sources, out: Path) -> dict[str, Any]:
    cards = []
    for seed in (23, 37, 101, 151):
        path = f"{RUNS}/trm_choice_constrained_arc_challenge_32_seed{seed}_20260604/choice_constrained_sample_scores.jsonl"
        cards.append({"name": f"seed{seed}", "source": path, "rows": src.rows(path)})
    result = overlap_analysis(cards, "arc_challenge")
    write_csv(out / "arc_overlap.csv", result["pairwise"])
    policy = {"top_action": "D", "runner_up_action": "A", "max_margin": 1.0, "penalty": 1.0}
    unique = {}
    for c in cards:
        for r in c["rows"]:
            unique.setdefault(sample_key(r), r)
    result["retrospective_fixed_condition"] = policy
    result["deduplicated_condition_replay"] = score_policy(list(unique.values()), policy)
    for fold, card in zip(result["folds"], cards):
        unseen = set(fold["unseen_ids"])
        fold["condition_replay_all_rows"] = score_policy(card["rows"], policy)
        fold["condition_replay_unseen_rows"] = score_policy([r for r in card["rows"] if sample_key(r) in unseen], policy)
    result["sources"] = [c["source"] for c in cards]
    folds_path = f"{RUNS}/trm_gain_policy_condition_cv_arc_4seed_20260604/condition_cv_folds.jsonl"
    result["historical_cv_records"] = src.rows(folds_path)
    result["evidence_class"] = "independently_reconciled_cached_scores"
    result["limitations"] = [
        "Historical training deduplicates its own union; held-out score-file IDs are never removed from that union.",
        "Retrospective unseen-row slices are descriptive audits of a previously selected condition, not newly held-out experiments.",
        "A repeated ID can have different score bytes; first-occurrence deduplication follows the archived seed ordering.",
        "Cached constrained-answer reanalysis does not measure a model-internal parameter intervention."]
    return result


def windows_source(path: str) -> str:
    p = path.replace("\\", "/")
    if not re.match(r"^[A-Za-z]:/", p):
        raise ValueError(f"Unrecognized original path {path}")
    return "sources/" + p.replace(":", "", 1)


def audit_wordle(src: Sources, out: Path) -> dict[str, Any]:
    card_path = f"{WORDLE}/overnight_validation_32x2_20260604_retry_smallgpu/policy_validation_4_8_32_32/wordle_policy_validation_cards.jsonl"
    validation = src.rows(card_path)
    policy_path = f"{WORDLE}/policy_distill_post_initial_8sample/wordle_post_initial_anti_crane_policy.json"
    policy = src.json(policy_path)
    cards = []
    card_results = []
    for card in validation:
        path = windows_source(card["score_file"])
        rows = src.rows(path)
        name = path.split("/")[-2]
        cards.append({"name": name, "source": path, "rows": rows})
        replay = score_policy(rows, policy, wordle=True)
        by_id = {sample_key(r, "wordle", True): r for r in card["target_details"]}
        detail_mismatch = sum(any(r[k] != by_id[sample_key(original, "wordle", True)][k]
                                  for k in ("hit", "baseline_hit", "prediction"))
                              for r, original in zip(replay["details"], rows))
        controls = []
        for control in card["control_summaries"]:
            c = score_policy(rows, {**policy, "penalized_action": control["penalized_action"]}, wordle=True)
            controls.append({"action": control["penalized_action"],
                             "delta": c["delta"], "policy_hits": c["policy_hits"],
                             "damages": c["damages"]})
        card_results.append({"card": name, "source": path,
                             **{k: v for k, v in replay.items() if k != "details"},
                             "best_control_delta": max(c["delta"] for c in controls),
                             "historical_detail_mismatches": detail_mismatch,
                             "controls": controls})
    result = overlap_analysis(cards, "wordle", include_step=True)
    result["episode_overlap"] = overlap_analysis(cards, "wordle")
    result["episode_overlap"]["target_changes_across_steps_ids"] = result["episode_overlap"].pop("target_conflict_ids")
    write_csv(out / "wordle_overlap.csv", result["pairwise"])
    result["cards"] = card_results
    result["policy"] = policy
    unique = {}
    for c in cards:
        for r in c["rows"]:
            unique.setdefault(sample_key(r, "wordle", True), r)
    result["deduplicated_replay"] = score_policy(list(unique.values()), policy, wordle=True)
    result["evidence_class"] = "independently_reconciled_cached_live_score_records"
    result["mean_card_delta"] = sum(c["delta"] for c in card_results) / len(card_results)
    result["pooled_row_delta"] = sum(c["policy_hits"] - c["baseline_hits"] for c in card_results) / result["raw_rows"]
    result["limitations"] = [
        "The four- and eight-row cards contributed to policy discovery; card seed changes do not establish a disjoint independent test.",
        "Distinct trajectory IDs provide an observable overlap check; missing source puzzle provenance can conceal common underlying Wordle episodes.",
        "Constrained candidate likelihood policies do not establish free-form Wordle success or a learned VPD weight edit."]
    return result


def audit_adapters(src: Sources) -> dict[str, Any]:
    train = src.rows(f"{DATASET}/train.jsonl")
    evaluation = src.rows(f"{DATASET}/eval.jsonl")
    split = src.json(f"{DATASET}/manifest.json")
    comparisons = {}
    funcs = {"row_id": lambda r: r["id"], "exact_text": lambda r: r["text"],
             "task_prefix_before_MODEL_ACTION": lambda r: r["text"].split("\nMODEL_ACTION:", 1)[0]}
    for name, key in funcs.items():
        tr = {key(r) for r in train}
        ev = {key(r) for r in evaluation}
        comparisons[name] = {"train_unique": len(tr), "eval_unique": len(ev),
                             "shared_unique": len(tr & ev),
                             "eval_rows_with_train_identity": sum(key(r) in tr for r in evaluation),
                             "eval_rows_without_train_identity": sum(key(r) not in tr for r in evaluation)}
    runs = []
    for folder in ("powered_int3_vpd_gated_adapter", "powered_int3_vpd_gated_adapter_realvpd",
                   "powered_int3_vpd_gated_adapter_validated", "powered_int3_vpd_gated_adapter_validated_zero_step",
                   "powered_int3_vpd_gated_adapter_smoke"):
        path = f"{RUNS}/trm_param_decomp/{folder}/vpd_gated_adapter_results.json"
        value = src.json(path)
        runs.append({"run": folder, "source": path,
                     "train_rows_reported": value.get("train_rows"),
                     "validation_rows_reported": value.get("validation_rows"),
                     "eval_rows_reported": value.get("eval_rows"), "steps_reported": value.get("steps"),
                     "selection_source": value.get("selection_source", "unspecified_in_result"),
                     "selection_method": value.get("selection_method", "unspecified_in_result"),
                     "arms": [{"name": a["name"], "component_count": a["component_count"],
                               "eval": a["eval"], "coefficients": a["coefficients"],
                               "initial_coefficients": {c["component_id"]: c.get("initial_coeff", "unrecorded") for c in a["components"]}}
                              for a in value["arms"]]})
    script = f"{RECOVERED_REPO}/scripts/run_powered_int3_vpd_gated_adapter.py"
    src.register(script)
    return {"evidence_class": "reconciled_result_json_and_dataset_identity_audit",
            "train_rows": len(train), "eval_rows": len(evaluation), "overlap": comparisons,
            "dataset_manifest": split, "runs": runs, "source_code": script,
            "limitations": [
                "Exact historical invocations and selected train/validation/eval IDs are missing; script defaults are not execution evidence.",
                "Reported bucket accuracy evaluates a critic's reward bucket, not end-to-end logic-task solution accuracy.",
                "The static adapter arm uses SVD directions; realvpd and validated runs refer to learned component checkpoints.",
                "Validated selected components start at signed nonzero coefficients; random components start at zero, confounding equal-count comparisons.",
                "Random component count matches, while edit norms, direction rank and independent random replication are unauthenticated.",
                "Task-prefix overlap measures exact shared task context before MODEL_ACTION; no semantic equivalence inference is required."]}


def audit_governance(src: Sources) -> dict[str, Any]:
    report = f"{RECOVERED_REPO}/reports/trm_governance_report.md"
    code = f"{RECOVERED_REPO}/scripts/trm_governance_test.py"
    src.register(report)
    src.register(code)
    text = (src.root / report).read_text(encoding="utf-8-sig")
    baseline = re.search(r"Baseline accuracy: target ([\d.]+), nontarget ([\d.]+)\.", text)
    if baseline is None:
        raise ValueError("Governance baseline not parseable")
    arms = [{"arm": "baseline", "target_accuracy": float(baseline[1]),
             "nontarget_accuracy": float(baseline[2])}]
    for line in text.splitlines():
        parts = [x.strip() for x in line.strip("|").split("|")]
        if len(parts) == 3 and re.fullmatch(r"\d+\.\d+", parts[1]):
            arms.append({"arm": parts[0], "target_accuracy": float(parts[1]),
                         "nontarget_accuracy": float(parts[2])})
    return {"evidence_class": "historical_report_only_with_source_audit", "report": report,
            "source_code": code, "arms": arms,
            "limitations": [
                "Exact invocation, stdout, selected rows, dirty-diff checkpoint and projection coefficients were not recovered.",
                "Available script trains the dirty patch on the same target slice it reports and ranks cleanup using both reported slices.",
                "Projection subtracts individual nonorthogonal components without a Gram inverse; overlapping components can overremove the patch.",
                "The code's rank-4 diff and top-8 SVD removal defaults erase the whole patch; actual invocation is unknown.",
                "precision_trm_mcq is a feedforward TF-IDF classifier, so this result does not establish recursive TRM governance."]}


def audit_roundout(src: Sources, out: Path) -> dict[str, Any]:
    result = {"evidence_class": "historical_reported_collation",
              "limitations": ["The recovered CSV claim labels are historical author judgments; this audit does not endorse them.",
                              "Collation values summarize earlier runs and deterministic estimates; source-row recounts are identified separately."]}
    for filename, key in (("paper_metric_table.csv", "metrics"),
                          ("paper_policy_comparison.csv", "policies"),
                          ("paper_experiment_manifest.csv", "experiments"),
                          ("paper_claim_ledger.csv", "claims")):
        path = f"{RUNS}/vpd_trm_paper_roundout_20260604/{filename}"
        src.register(path)
        with (src.root / path).open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        result[key] = {"source": path, "rows": rows}
        write_csv(out / f"roundout_{key}.csv", rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    src = Sources(args.records_root)
    src.register("MISSING_FILES.md")
    src.register("MANIFEST.csv")
    src.register(f"{RECOVERED_REPO}/papers/VPD_TRM_Feature_Steering/vpd_trm_feature_steering_paper.md")
    result = {"schema_version": 1,
              "method": "Retrospective audit; source bytes preserved; no historical execution implied.",
              "arc": audit_arc(src, args.out_dir), "wordle": audit_wordle(src, args.out_dir),
              "adapters": audit_adapters(src), "governance": audit_governance(src),
              "historical_roundout": audit_roundout(src, args.out_dir)}
    dump(args.out_dir / "audit_partial.json", result)
    print("Small-record audits completed; streaming replication sweep.", flush=True)
    result["replication"] = audit_arena(src, args.out_dir)
    result["sources"] = sorted(src.refs.values(), key=lambda x: x["path"])
    dump(args.out_dir / "audit.json", result)
    (args.out_dir / "audit_partial.json").unlink()
    metrics = []
    for name, section in (("arc_condition", result["arc"]["deduplicated_condition_replay"]),
                          ("wordle_policy", result["wordle"]["deduplicated_replay"])):
        metrics.append({"family": name, "n": section["n"], "baseline_hits": section["baseline_hits"],
                        "policy_hits": section["policy_hits"], "delta": section["delta"],
                        "rescues": section["rescues"], "damages": section["damages"]})
    write_csv(args.out_dir / "historical_metrics.csv", metrics)
    print(json.dumps({"replication": result["replication"]["counts"],
                      "arc_unique": result["arc"]["unique_ids"],
                      "wordle_unique": result["wordle"]["unique_ids"]}, sort_keys=True))


if __name__ == "__main__":
    main()

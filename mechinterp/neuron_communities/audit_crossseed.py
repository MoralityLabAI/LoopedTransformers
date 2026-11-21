"""NEC-02B: read-only six-checkpoint cross-seed community causal audit.

Development-only retrospective study on the same 48 context groups.
Every selected candidate, negative contrast and paired logit stays visible.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from mechinterp.neuron_communities.audit_hosted import (
    SOURCE_SHA, read_json, raw_sha, verify_raw_context_traces,
    verify_registered_partition,
)
from research.pef07.pef07 import SEEDS, TASKS

PARENT_RUN = 44788623483
PARENT_ARTIFACT = 14655785897


def exact_count(fraction: float, count: int = 768) -> int:
    value = float(fraction) * count
    if not math.isfinite(value):
        raise AssertionError("nonfinite effect cannot represent a case count")
    rounded = round(value)
    if abs(rounded-value) > 1e-6:
        raise AssertionError("score differs from an integer heldout count")
    return rounded


def candidate_record(row: dict) -> dict:
    cond = row["condition_summaries"]
    names = ("ablate", "sham", "retain", "hubs", "blank")
    if set(cond) != set(names):
        raise AssertionError("incomplete causal family")
    errors = {name:exact_count(cond[name]["correctness_drop"]["mean"])
              for name in names}
    nec = row["necessity_excess_kl_over_frequency_matched_sham"]
    suf = row["sufficiency_kl_gain_over_hubs_only"]
    if nec["contexts"] != 48 or suf["contexts"] != 48:
        raise AssertionError("wrong source-context count")
    return {
        "id":row["id"],"nodes":row["nodes"],"visits":row["visits"],
        "n_nodes":len(row["nodes"]),
        "errors_after_targeted_ablation":errors["ablate"],
        "errors_after_frequency_matched_sham":errors["sham"],
        "errors_after_blank_MLP":errors["blank"],
        "errors_after_hubs_only":errors["hubs"],
        "errors_after_hubs_plus_community":errors["retain"],
        "conditional_rescued_predictions":errors["hubs"]-errors["retain"],
        "necessity_excess_KL_vs_sham":nec,
        "conditional_KL_gain_vs_hubs":suf,
    }


def audit(root: Path) -> dict:
    source = read_json(root/"pef07"/"source_split.json")
    verify_registered_partition(source)
    ledger = read_json(root/"pef07"/"sealed_training_ledger.json")
    scores = read_json(root/"pef07"/"summary.json")
    if not ledger["all_six_trunks_sealed_before_scoring"]:
        raise AssertionError("PEF-07 was not source-frozen")
    if ledger["source_sha256"] != scores["source_sha256"] != SOURCE_SHA:
        raise AssertionError("source manifest mismatch")
    checkpoints = {(r["seed"],r["task"]):r for r in ledger["checkpoints"]}
    expected = {(s,t) for s in SEEDS for t in TASKS}
    if set(checkpoints) != expected or len(ledger["checkpoints"]) != 6:
        raise AssertionError("six trained checkpoints required")
    for rec in checkpoints.values():
        if raw_sha(root/"pef07"/rec["file"]) != rec["checkpoint_sha256"]:
            raise AssertionError("frozen checkpoint bytes differ")
    models = []
    for seed in SEEDS:
        for task in TASKS:
            rec = checkpoints[(seed,task)]
            row_score = next(x for x in scores["models"]
                             if (x["seed"],x["task"]) == (seed,task))
            if row_score["trained_h12_native_head"]["correct"] != 768:
                raise AssertionError("pilot precondition clean 768/768 unmet")
            folder = root/"analysis"/f"seed{seed}_{task}"
            d = read_json(folder/"discovery.json")
            t = read_json(folder/"heldout_causal.json")
            roster_sha = raw_sha(folder/"discovery.json")
            if (folder/"discovery.sha256").read_text().strip() != roster_sha:
                raise AssertionError("candidate roster mutated")
            if t["candidate_roster_sha256"] != roster_sha:
                raise AssertionError("causal evaluation used different roster")
            identity = {"provenance":"pef07-frozen-trained",
                        "model_sha256":rec["frozen_weights_sha256"],
                        "checkpoint_sha256":rec["checkpoint_sha256"],
                        "training_steps":450}
            if t["identity"] != identity or d["identity"] != identity:
                raise AssertionError("frozen identity mismatch")
            if (d["seed"], d["task"]) != (seed,task):
                raise AssertionError("wrong task or seed")
            if (d["source_split"],d["context_groups_used"],d["top_k_per_visit"]) != ("train_only",160,2):
                raise AssertionError("source discovery was altered")
            if (t["context_count"],t["random_smoke"]) != (48,False):
                raise AssertionError("wrong heldout population")
            if d["source_sha256"] != t["source_sha256"] != SOURCE_SHA:
                raise AssertionError("different source across models")
            if [c["id"] for c in d["candidates"]] != [c["id"] for c in t["tests"]]:
                raise AssertionError("discovered candidate roster differs")
            raw = verify_raw_context_traces(root,folder,d,t)
            candidates = [candidate_record(c) for c in t["tests"]]
            rescues=[c["conditional_rescued_predictions"] for c in candidates]
            models.append({
                "seed":seed, "task":task,
                "checkpoint_sha256":rec["checkpoint_sha256"],
                "discovery_sha256":roster_sha,
                "raw_logit_sha256":raw["raw_context_trace_sha256"],
                "source_graph_edges":len(d["graph"]["edges"]),
                "source_hubs":len(d["graph"]["hubs"]),
                "candidate_count":len(candidates),
                "independent_context_groups":48,
                "candidate_context_rows":raw["raw_context_records"],
                "targeted_ablation_error_count_summed_across_candidates":
                    sum(c["errors_after_targeted_ablation"] for c in candidates),
                "largest_conditional_rescue_selected_posthoc":
                    max(rescues) if rescues else None,
                "smallest_conditional_rescue_selected_posthoc":
                    min(rescues) if rescues else None,
                "candidates":candidates,
            })
    report={
        "schema":"NEC-02B/six-checkpoint-crossseed-exploratory-v1",
        "evidence_class":"posthoc-development-only",
        "source_run_id":PARENT_RUN,
        "source_artifact_id":PARENT_ARTIFACT,
        "source_manifest_sha256":SOURCE_SHA,
        "registered_pef07_outcomes_unchanged":True,
        "six_checkpoint_files_reused_no_training":True,
        "models":models,
        "total_discovered_communities":sum(x["candidate_count"] for x in models),
        "total_candidate_context_rows":sum(x["candidate_context_rows"] for x in models),
        "independent_contexts_per_model":48,
        "counterfactuals_per_context":16,
        "same_heldout_contexts_reused_across_all_models":True,
        "familywise_significance_inferred":False,
        "independent_task_generalization_inferred":False,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    (root/"nec02b_six_seed_audit.json").write_text(
        json.dumps(report,sort_keys=True,indent=2)+"\n")
    return report


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    args=ap.parse_args(argv)
    report=audit(args.root)
    print(json.dumps({"models":len(report["models"]),
                      "candidates":report["total_discovered_communities"],
                      "candidate_context_rows":report["total_candidate_context_rows"],
                      "gate":report["FSI_01_gate"]},sort_keys=True))


if __name__=="__main__":
    main()

"""NEC-02 audit: independent hosted PEF-07 training + exploratory NEC-01.

The 2026-11-15 locally trained checkpoints are a separate historical
evidence class. This runner trains a fresh hosted set and records actual
checkpoint byte hashes; equality with historical bytes is diagnostic only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ORIGINAL_LOCAL_BYTES = {
    (103, "coalition"): "dc0cecc46b52904476d8e3ede3d5d9d1e2b0d9c8061d6e9b72222679e58e33d8",
    (103, "threshold"): "7290b63158af42302a777ae47487009e8afbb401511ec8d78cae8dbd5fed33b6",
    (131, "coalition"): "ac6ee6819096957dc0f3c7f28f7d9eb9e67e9a2f68b9141f1fee99640a06685e",
    (131, "threshold"): "f8f2f3cbb822ed7d501af92ebe8836db0f49a81e4005efd4acce281e506eb2ed",
    (167, "coalition"): "ca637fdb461f0ebae6bfb255c5c4b283039f472434266f165bbafd6736157fdb",
    (167, "threshold"): "232e586c8c307ead403ff36641eda4aaf537c43efc3b1b31362862bfb474e412",
}
# Two distinct serialized manifest digests exist; preserve both.
# Direct byte comparison needs the original November 15 manifest file.
HISTORICAL_LOCAL_MANIFEST_SHA = "849d039e47dec472b0b4835d24928e9a547e262bf84fd7dfed16741df9911ac1"
SOURCE_SHA = "de4e0263864ee70c2bf94ca95008ace10fce1323d1a51b4c8829ec26e40bbd9b"


def verify_registered_partition(src: dict) -> None:
    """Reconstruct PEF-07 context membership from the preregistration.

    This is independent of the PEF-07 implementation and checks all 256
    eight-bit context assignments and exact 16-cell quartet closure.
    """
    counts = {"train": 160, "probe": 48, "heldout": 48}
    contexts = [tuple((i >> bit) & 1 for bit in range(8)) for i in range(256)]
    contexts.sort(key=lambda c: (
        hashlib.sha256(b"pef07-v1-quartet:" + bytes(c)).digest(), bytes(c)))
    assignment = {}
    offset = 0
    for partition, n in counts.items():
        for context in contexts[offset:offset+n]:
            assignment["".join(str(bit) for bit in context)] = partition
        offset += n
    if src["context_count_by_split"] != counts:
        raise AssertionError("PEF-07 context cohort counts differ")
    if src["context_assignment"] != assignment:
        raise AssertionError("PEF-07 preregistered context membership differs")
    if src["contexts_in_sha256_order"] != [
            "".join(str(bit) for bit in c) for c in contexts]:
        raise AssertionError("PEF-07 context hash ordering differs")
    if src["history_count_by_split"] != {"train": 2560, "probe": 768, "heldout": 768}:
        raise AssertionError("PEF-07 quartet closure count differs")
    if src.get("source_contexts_are_disjoint") is not True:
        raise AssertionError("PEF-07 source closure not asserted")
    complete = dict(src)
    digest = complete.pop("sha256")
    calculation = hashlib.sha256((json.dumps(complete, sort_keys=True,
                      separators=(",", ":")) + "\n").encode()).hexdigest()
    if digest != calculation or digest != SOURCE_SHA:
        raise AssertionError("PEF-07 hosted canonical manifest digest mismatch")



def raw_sha(file: Path) -> str:
    h = hashlib.sha256()
    with file.open("rb") as src:
        for chunk in iter(lambda: src.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object: {path}")
    return obj


def audit_training(root: Path):
    src = read_json(root/"pef07/source_split.json")
    ledger = read_json(root/"pef07/sealed_training_ledger.json")
    score = read_json(root/"pef07/summary.json")
    verify_registered_partition(src)
    assert src["sha256"] == SOURCE_SHA
    assert ledger["source_sha256"] == score["source_sha256"] == SOURCE_SHA
    assert ledger["all_six_trunks_sealed_before_scoring"]
    assert score["all_models_sealed_before_any_probe_or_test"]
    assert score["full_registered_run"]
    assert score["seed_order"] == [103, 131, 167]
    assert score["task_order"] == ["coalition", "threshold"]
    assert score["steps"] == 450
    assert src["history_count_by_split"] == {"train": 2560, "probe": 768, "heldout": 768}
    assert len(ledger["checkpoints"]) == len(score["models"]) == 6
    assert ledger["onednn_max_cpu_isa"] == "AVX2"
    receipts = []
    for rec in ledger["checkpoints"]:
        key = (rec["seed"], rec["task"])
        if key not in ORIGINAL_LOCAL_BYTES:
            raise AssertionError("unknown PEF-07 seed/task")
        actual_sha = raw_sha(root/"pef07"/rec["file"])
        assert actual_sha == rec["checkpoint_sha256"]
        assert rec["steps"] == 450
        assert rec["source_sha256"] == SOURCE_SHA
        result = next(x for x in score["models"]
                      if (x["seed"], x["task"]) == key)
        assert result["checkpoint_sha256"] == actual_sha
        assert result["all_original_weights_frozen_before_probe"]
        assert result["trained_h12_native_head"]["total"] == 768
        receipts.append({
            "seed": key[0], "task": key[1],
            "new_hosted_checkpoint_sha256": actual_sha,
            "historical_2026_11_15_checkpoint_sha256": ORIGINAL_LOCAL_BYTES[key],
            "same_bytes_as_historical_local_checkpoint":
                actual_sha == ORIGINAL_LOCAL_BYTES[key],
            "new_hosted_native_h12_correct_of_768": result["trained_h12_native_head"]["correct"],
            "new_hosted_fitted_h14_correct_of_768": result["frozen_h14_downstream_ridge"]["correct"],
        })
    return score, receipts


def audit_exploration(root: Path, task: str, ledger: dict) -> dict:
    folder = root/"analysis"/f"seed103_{task}"
    frozen_bytes = (folder/"discovery.json").read_bytes()
    if raw_sha(folder/"discovery.json") != (folder/"discovery.sha256").read_text().strip():
        raise AssertionError("discovery roster was changed")
    discovery = json.loads(frozen_bytes)
    test = read_json(folder/"heldout_causal.json")
    assert discovery["schema"] == test["schema"]
    assert discovery["identity"]["provenance"] == "pef07-frozen-trained"
    assert test["identity"] == discovery["identity"]
    assert discovery["identity"]["checkpoint_sha256"] == next(
        x["checkpoint_sha256"] for x in ledger["checkpoints"]
        if x["seed"] == 103 and x["task"] == task)
    assert discovery["source_sha256"] == test["source_sha256"] == SOURCE_SHA
    assert discovery["seed"] == 103 and discovery["task"] == task
    assert discovery["source_split"] == "train_only"
    assert discovery["context_groups_used"] == 160
    assert discovery["counterfactuals_per_context"] == 16
    assert discovery["top_k_per_visit"] == 2
    assert test["context_count"] == 48 and test["counterfactuals_per_context"] == 16
    assert test["random_smoke"] is False
    assert test["candidate_roster_sha256"] == raw_sha(folder/"discovery.json")
    ids = [x["id"] for x in discovery["candidates"]]
    assert ids == [x["id"] for x in test["tests"]]
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicated candidate")
    for row in test["tests"]:
        assert set(row["condition_summaries"]) == {
            "ablate", "sham", "retain", "hubs", "blank"}
        for summary in row["condition_summaries"].values():
            for endpoint in ("kl", "correctness_drop", "top_probability_drop"):
                assert summary[endpoint]["contexts"] == 48
        assert row["necessity_excess_kl_over_frequency_matched_sham"]["contexts"] == 48
        assert row["sufficiency_kl_gain_over_hubs_only"]["contexts"] == 48
    return {"task": task, "trained_model": True, "seed": 103,
            "discovery_sha256": test["candidate_roster_sha256"],
            "candidate_count": len(ids),
            "graph_edges": len(discovery["graph"]["edges"]),
            "graph_hubs": len(discovery["graph"]["hubs"]),
            "candidates": [{
                "id": row["id"],
                "node_count": len(row["nodes"]),
                "visits": row["visits"],
                "ablation_excess_KL_over_sham":
                    row["necessity_excess_kl_over_frequency_matched_sham"],
                "retention_gain_KL_over_hubs":
                    row["sufficiency_kl_gain_over_hubs_only"],
                "ablation_correctness_drop":
                    row["condition_summaries"]["ablate"]["correctness_drop"],
                "sham_correctness_drop":
                    row["condition_summaries"]["sham"]["correctness_drop"]}
                for row in test["tests"]]}


def audit(root: Path):
    score, receipts = audit_training(root)
    ledger = read_json(root/"pef07/sealed_training_ledger.json")
    arms = [audit_exploration(root, t, ledger) for t in ("coalition", "threshold")]
    report = {
        "schema": "NEC-02/hosted-trained-pef07-20261121/v1",
        "evidence_class": "exploratory-causal-pilot-retrained-CPU",
        "primary_2026_11_15_pef07_results_are_unchanged": True,
        "historical_checkpoint_comparison_is_descriptive": True,
        "training": {"torch": ledger.get("cpu_torch"), "processor": ledger.get("processor"),
                     "cpu_capability": ledger.get("cpu_capability"),
                     "pef07_primary_decision_on_this_new_run": score["primary_decision"],
                     "pef07_diagnostic_decision_on_this_new_run": score["diagnostic_decision"],
                     "checkpoints": receipts},
        "source_sha256": SOURCE_SHA,
        "historical_local_report_manifest_sha256": HISTORICAL_LOCAL_MANIFEST_SHA,
        "hosted_manifest_sha256": SOURCE_SHA,
        "historical_and_hosted_manifest_digests_differ":
            HISTORICAL_LOCAL_MANIFEST_SHA != SOURCE_SHA,
        "hosted_source_partition_reconstructed_from_preregistration": True,
        "original_2026_11_15_manifest_bytes_available_for_direct_diff": False,
        "analysis": arms,
        "inference": {
            "independent_unit": "heldout context quartet per fixed trained checkpoint",
            "independent_contexts_per_model": 48,
            "observations_per_context": 16,
            "trained_model_seeds_analyzed": 1,
            "tasks_analyzed": ["coalition", "threshold"],
            "comparisons_include_all_discovered_candidates": True,
            "familywise_error_protected": False,
            "calibrated_mechanistic_sham_floor": None,
            "minimum_meaningful_behavioral_effect_registered": False,
            "FSI_01_gate": "REDESIGN_OR_PILOT_ONLY",
            "claim": "Engineering-complete exploratory neuron interventions; descriptive effects only",
            "next": "Calibrate floor with independent dev worlds; freeze Holm family and meaningful correctness delta; compare native units with source-locked SAE interventions; use additional trained seeds for confirmation."
        }}
    out = root/"nec02_audit.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args(argv)
    result = audit(args.root)
    print(json.dumps({"status": result["evidence_class"],
                      "pef07_decision": result["training"]["pef07_primary_decision_on_this_new_run"],
                      "arms": [(x["task"], x["candidate_count"]) for x in result["analysis"]]}, sort_keys=True))


if __name__ == "__main__":
    main()

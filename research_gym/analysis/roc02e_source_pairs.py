"""ROC-02E calibration-only legal rule/cue counterfactuals and dose matcher.

Preregistration: configs/roc02e_rule_vs_cue_causal_dev_v1.json.
This module reads or generates ONLY the already public calibration split.
The sealed-test generator is deliberately never imported or invoked as a split.
Model interventions require the separately frozen ROC-02C observer replay.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

try:
    from research_gym.analysis.roc02_delayed_rule_source import (
        build_split, canonical_hash, encode_event, rule_table, symbolic_execute
    )
except ModuleNotFoundError:
    from roc02_delayed_rule_source import (
        build_split, canonical_hash, encode_event, rule_table, symbolic_execute
    )

CALIBRATION_SPLIT_SHA256 = "a9a8190aee2b4c2111afd8ceaf59422b55b483f7e54182795d986cc8739b818c"
ALLOWED_FAMILIES = (4, 5)
RADII = (0.05, 0.10, 0.20)
EXPECTED_COUNT = 256


def canonical_jsonline(record: dict) -> bytes:
    return (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def calibration_source() -> list[dict]:
    """Regenerate the already published 256-row dev split; never touch test."""
    original = build_split("calibration")
    if len(original) != EXPECTED_COUNT:
        raise AssertionError("calibration count mismatch")
    contents = b"".join(canonical_jsonline(x) for x in original)
    if hashlib.sha256(contents).hexdigest() != CALIBRATION_SPLIT_SHA256:
        raise AssertionError("frozen calibration split SHA mismatch")
    if any(x["family"] not in ALLOWED_FAMILIES for x in original):
        raise AssertionError("an unexpected source family was loaded")
    return original


def _checked_original(record: dict) -> tuple[int, int]:
    family = record["family"]
    if family not in ALLOWED_FAMILIES:
        raise ValueError("ROC02E accepts calibration rule families 4 and 5 only")
    events = record["events"]
    if not events or events[0].get("op") != "bind" or events[0].get("src") != "A":
        raise ValueError("the first event must bind source A")
    if events[-1].get("op") != "query" or events[-1].get("src") != "A":
        raise ValueError("the last event must query source A")
    if events[0]["table"] != rule_table(family):
        raise AssertionError("original A table disagrees with source family")
    if record["id"] != canonical_hash(record):
        raise AssertionError("original case hash mismatch")
    y = symbolic_execute(events)
    if y != {"status": "ok", "label": record["label"]}:
        raise AssertionError("original record label disagrees with oracle")
    return family, int(record["label"])


def make_counterfactual(record: dict, kind: str) -> dict:
    """Legal semantic edits with the same, oracle-verified target label."""
    family, y = _checked_original(record)
    edit = copy.deepcopy(record)
    if kind == "rule":
        other = 5 if family == 4 else 4
        edit["family"] = other
        edit["events"][0]["table"] = rule_table(other)
    elif kind == "cue":
        query = edit["events"][-1]
        query["value"] = (int(query["value"]) + 4) % 8
    else:
        raise ValueError("edit kind must be rule or cue")
    answer = symbolic_execute(edit["events"])
    if answer["status"] != "ok" or answer["label"] != (y + 4) % 8:
        raise AssertionError("the proposed source edit has wrong oracle target")
    edit["label"] = answer["label"]
    edit["id"] = canonical_hash(edit)
    if edit["id"] == record["id"]:
        raise AssertionError("edit leaves identical input hash")
    if len(edit["events"]) != len(record["events"]) or edit["delay"] != record["delay"]:
        raise AssertionError("edit changed the sequence length or delay")
    diffs = [i for i, (a, b) in enumerate(zip(record["events"], edit["events"])) if a != b]
    expected_position = 0 if kind == "rule" else len(record["events"]) - 1
    if diffs != [expected_position]:
        raise AssertionError("edit changed more than its intended event")
    return edit


def raw_feature_l2(before: dict, after: dict) -> float:
    if len(before["events"]) != len(after["events"]):
        raise ValueError("features require identical event length")
    return math.sqrt(sum(
        (a - b) ** 2
        for e0, e1 in zip(before["events"], after["events"])
        for a, b in zip(encode_event(e0), encode_event(e1))
    ))


def make_pair(record: dict) -> tuple[dict, dict, dict]:
    family, label = _checked_original(record)
    rule = make_counterfactual(record, "rule")
    cue = make_counterfactual(record, "cue")
    if rule["label"] != cue["label"] or rule["label"] == label:
        raise AssertionError("paired counterfactuals have different targets")
    rule_l2 = raw_feature_l2(record, rule)
    cue_l2 = raw_feature_l2(record, cue)
    if abs(rule_l2 - 4.0) > 1e-12 or abs(cue_l2 - math.sqrt(2.0)) > 1e-12:
        raise AssertionError("unexpected feature-space causal-edit dose")
    receipt = {
        "original_id": record["id"], "rule_id": rule["id"], "cue_id": cue["id"],
        "original_family": family, "rule_family": rule["family"],
        "original_label": label, "counterfactual_label": rule["label"],
        "delay": record["delay"], "n_events": len(record["events"]),
        "rule_feature_l2": rule_l2, "cue_feature_l2": cue_l2,
        "n_distractor_events": sum(e["op"] == "distractor" for e in record["events"]),
        "calibration_only": True,
    }
    return rule, cue, receipt


def source_pair_audit() -> tuple[list[dict], dict]:
    original = calibration_source()
    pairs = [make_pair(case)[2] for case in original]
    ids = [row["original_id"] for row in pairs]
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate calibration IDs")
    families = Counter(row["original_family"] for row in pairs)
    if families != {4: 117, 5: 139}:
        raise AssertionError("calibration source-family count mismatch")
    if len(pairs) != EXPECTED_COUNT:
        raise AssertionError("wrong number of paired records")
    audit = {
        "study": "ROC-02E-v1-source-pair-audit",
        "scope": "calibration-only; no original test input or label read",
        "prereg": "configs/roc02e_rule_vs_cue_causal_dev_v1.json",
        "original_calibration_sha256": CALIBRATION_SPLIT_SHA256,
        "paired_cases": len(pairs),
        "family_counts": {"4": families[4], "5": families[5]},
        "both_oracle_targets_equal": True,
        "all_target_labels_change_by_four_modulo_eight": True,
        "rule_feature_l2": 4.0,
        "cue_feature_l2": math.sqrt(2.0),
        "sealed_test_evaluated": False,
    }
    return pairs, audit


def interpolate_to_midpoint_radius(model, x0, x1, mask, midpoint_fn, radius,
                                   grid=64, bisections=24):
    """Find first radius crossing along soft input interpolation.

    midpoint_fn(model,x,mask) must return a [1,128] torch Tensor.
    The full discrete input x1 is legal. Intermediate feature vectors are
    continuous off-manifold controls without independently valid rule labels.
    This function makes no prediction or claim about those soft-input labels.
    """
    import torch
    if radius not in RADII or grid != 64 or bisections != 24:
        raise ValueError("dose matcher outside frozen ROC02E protocol")
    if x0.shape != x1.shape or x0.shape[0] != 1 or mask.shape[0] != 1:
        raise ValueError("dose matcher expects paired single-case tensors")
    with torch.no_grad():
        base = midpoint_fn(model, x0, mask).double()
        if base.shape != (1, 128):
            raise ValueError("invalid model midpoint shape")
        delta = x1 - x0

        def state_at(t):
            return midpoint_fn(model, x0 + t * delta, mask).double()

        full = state_at(1.0)
        full_radius = float(torch.linalg.vector_norm(full - base))
        if full_radius < radius:
            return {"available": False, "reason": "full legal edit below radius",
                    "full_legal_midpoint_l2": full_radius}

        left = 0.0
        right = None
        prev_t = 0.0
        for j in range(1, grid + 1):
            t = j / grid
            if float(torch.linalg.vector_norm(state_at(t) - base)) >= radius:
                left, right = prev_t, t
                break
            prev_t = t
        if right is None:
            return {"available": False, "reason": "no first radius crossing",
                    "full_legal_midpoint_l2": full_radius}

        for _ in range(bisections):
            t = (left + right) / 2.0
            if float(torch.linalg.vector_norm(state_at(t) - base)) >= radius:
                right = t
            else:
                left = t
        mid = state_at(right)
        actual = float(torch.linalg.vector_norm(mid - base))
        if abs(actual - radius) > 2e-5:
            raise AssertionError("matched radius outside 2e-5 tolerance")
        return {
            "available": True, "t": right, "requested_midpoint_l2": radius,
            "actual_midpoint_l2": actual, "full_legal_midpoint_l2": full_radius,
            "soft_input_has_symbolic_label": False, "midpoint": mid,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    pairs, audit = source_pair_audit()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "roc02e_source_pairs.jsonl"
    output.write_bytes(b"".join(canonical_jsonline(x) for x in pairs))
    audit["pairs_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    (args.out_dir / "roc02e_source_pair_audit.json").write_text(
        json.dumps(audit, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()

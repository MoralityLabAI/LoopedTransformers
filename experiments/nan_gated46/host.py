"""Independent native qualification and paired next-pass scoring for GATED46."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import fmean

from experiments.nan_comp14.parser import label
from experiments.nan_delay39.host_gate import assess
from experiments.nan_ranksite13.resources import below_normal
from experiments.nan_readout34.runtime import TARGETS
from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import sha

from .gated import ARMS, SIGNS
from .source import CONFIG, HERE, bound_rows, digest


def qualify(records, gold, cfg):
    result = assess(records, gold, cfg)
    text = {row["id"]: row["text"] for row in records}
    phased = defaultdict(list)
    for row in gold:
        if row["arm"] == "focused":
            phased[row["root"]].append(label(text[row["id"]]) == row["answer"])
    return result | {"qualified_focused_roots": sorted(root for root, match in phased.items() if all(match)),
                     "native_records_sha256": digest(records)}


def score_world(result, recipient, donor, cfg):
    status = result["status"]
    cases = result.get("cases", {})
    if status != "closed":
        if result.get("phase2_edit_count", 0) != 0 or any("future" in case for case in cases.values()):
            raise ValueError("An unqualified world recorded a second-pass edit")
        if status not in {"native_root_unqualified", "direction_unqualified", "phase1_gate_unqualified",
                          "matched_dose_unqualified"}:
            raise ValueError("Unknown GATED46 row status")
        return None
    if result["phase2_edit_count"] != 4 or set(cases) != {f"{arm}/{sign}" for arm in ARMS for sign in SIGNS}:
        raise ValueError("Gated world is missing a matched signed arm")
    if not math.isfinite(result["donor_cosine"]) or result["donor_cosine"] < cfg["minimum_donor_cosine"]:
        raise ValueError("Gated world has no qualified donor alignment")
    if label(result["clean"]["text"]) != recipient:
        raise ValueError("Clean recipient does not close relative to native qualification")
    requested, reached = [], []
    for case in cases.values():
        p = case["phase1_gate"]
        desired, actual = case["requested_l2"], case["realized_l2"]
        fields = (desired, actual, case["direction_cosine"], p["max_abs_logit_delta"], p["margin_delta_abs"])
        if (not all(math.isfinite(v) for v in fields) or not case["eligible_dose"] or desired <= 0 or
                abs(actual / desired - 1) > cfg["dose_relative_tolerance"]
                or case["direction_cosine"] < cfg["minimum_direction_cosine"]
                or not p["passed"] or not p["greedy_same"] or p["max_abs_logit_delta"] > cfg["maximum_head_delta"]
                or p["margin_delta_abs"] > cfg["maximum_margin_delta"]
                or not math.isfinite(case["future"]["margin"])):
            raise ValueError("Claimed native edit violates the frozen gate")
        requested.append(desired)
        reached.append(actual)
    if len(set(requested)) != 1 or max(reached) - min(reached) > cfg["max_contrast_dose_gap_fraction"] * requested[0]:
        raise ValueError("Four-arm physical doses were not matched")

    def scaled(arm):
        a, b = (cases[f"{arm}/{sign}"] for sign in SIGNS)
        return requested[0] * (a["future"]["margin"] - b["future"]["margin"]) / (
            a["realized_l2"] + b["realized_l2"])

    treated = label(cases[ARMS[0] + "/positive"]["future"]["text"])
    sham = label(cases[ARMS[1] + "/positive"]["future"]["text"])
    return {"sham_adjusted_margin": scaled(ARMS[0]) - scaled(ARMS[1]),
            "treatment_recipient_correct": treated == recipient, "sham_recipient_correct": sham == recipient,
            "treatment_donor_flip": treated == donor, "sham_donor_flip": sham == donor}


def collect(work, gold, cfg):
    manifest = HERE / "SOURCE_MANIFEST.json"
    terminal = json.loads((work / "terminal.json").read_bytes())
    backend = terminal["backend"]
    if (terminal["source_manifest_sha256"] != sha(manifest) or
            terminal["target_before"] != TARGETS[backend] or terminal["target_after"] != TARGETS[backend]
            or not terminal["quant_unchanged"] or not terminal["checkpoint_verified_after"]
            or terminal["qualification_sha256"] != sha(work / "qualification.json")):
        raise ValueError("Worker terminal integrity check failed")
    records = json.loads((work / "native.json").read_bytes())
    gate = qualify(records, gold, cfg)
    if json.loads((work / "qualification.json").read_bytes()) != gate:
        raise ValueError("Native qualification could not be independently reproduced")
    focused = {(row["root"], row["world"]): row for row in gold if row["arm"] == "focused"}
    source = {row["id"]: row for row in focused.values()}
    rows = {}
    if terminal["status"] == "closed" and gate["passed"]:
        journal = work / "closed.jsonl"
        if sha(journal) != terminal["closed_journal_sha256"]:
            raise ValueError("Gated journal was altered after terminal closure")
        for i, line in enumerate(journal.read_bytes().splitlines(keepends=True)):
            binding = json.loads(line)
            path = work / f"episode_{i:04d}.json"
            if not line.endswith(b"\n") or binding != {"file": path.name, "sha256": sha(path)}:
                raise ValueError("GATED46 episode digest or order changed")
            item = json.loads(path.read_bytes())
            item_id = item["source_id"]
            if item_id not in source or item_id in rows:
                raise ValueError("Unknown or repeated focused source")
            row = source[item_id]
            if (row["root"] in gate["qualified_focused_roots"]) == (item["status"] == "native_root_unqualified"):
                raise ValueError("Independent native root eligibility changed")
            rows[item_id] = item
        if set(rows) != set(source) or terminal["records"] != len(rows):
            raise ValueError("Gated journal coverage is incomplete")
    elif (terminal["status"] != "native_gate_failed" or gate["passed"] or
          terminal["records"] != 0 or (work / "closed.jsonl").exists()):
        raise ValueError("Only a closed model or explicit native qualification refusal can be scored")

    counts, per_root = Counter(), defaultdict(list)
    for row_id, row in source.items():
        if row_id not in rows:
            counts["native_gate_failed"] += 1
            continue
        donor = focused[row["root"], "B" if row["world"] == "A" else "A"]["answer"]
        measured = score_world(rows[row_id], row["answer"], donor, cfg)
        counts[rows[row_id]["status"]] += 1
        if measured is not None:
            per_root[row["root"]].append(measured)
    worlds = [world for group in per_root.values() for world in group]
    root_margin = [fmean(m["sham_adjusted_margin"] for m in group) for group in per_root.values() if len(group) == 2]
    minimum = cfg["minimum_meaningful_delayed_margin_logit"]
    return {
        "backend": backend, "terminal_status": terminal["status"], "registered_roots": cfg["source_roots"],
        "registered_worlds": len(source), "native_qualified_roots": len(gate["qualified_focused_roots"]),
        "complete_matched_worlds": len(worlds), "complete_matched_both_world_roots": len(root_margin),
        "status_counts": counts,
        "complete_root_macro_sham_adjusted_margin": fmean(root_margin) if root_margin else None,
        "complete_sham_adjusted_worlds_at_positive_effect_floor": sum(
            row["sham_adjusted_margin"] >= minimum for row in worlds),
        "effect_floor_logit": minimum,
        "exact_endpoint_counts": {k: sum(world[k] for world in worlds) for k in (
            "treatment_recipient_correct", "sham_recipient_correct", "treatment_donor_flip", "sham_donor_flip")},
        "formal_comparisons": 0,
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source-dir", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    action = cli.add_mutually_exclusive_group(required=True)
    action.add_argument("--native-records", type=Path)
    action.add_argument("--artifact-dir", type=Path)
    args = cli.parse_args()
    below_normal()
    cfg = json.loads(CONFIG.read_bytes())
    gold = bound_rows(args.source_dir, "gold")
    result = (qualify(json.loads(args.native_records.read_bytes()), gold, cfg) if args.native_records
              else collect(args.artifact_dir, gold, cfg))
    write_new(args.output, result)


if __name__ == "__main__":
    main()

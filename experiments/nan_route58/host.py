"""Independent qualification, terminal custody, and signed delayed ROUTE58 scoring."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import fmean

from experiments.nan_gated46.host import qualify as old_qualify
from experiments.nan_readout34.runtime import TARGETS
from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import sha
from experiments.nan_ranksite13.resources import below_normal

from .custody import REPO, RUNTIME, SOURCE_CHECK, bound_source, settings


SIGNS = ("positive", "negative")
ARMS = ("contrast", "sham")
KEYS = {f"{arm}/{sign}" for arm in ARMS for sign in SIGNS}


def native_gate(folder, records, cfg):
    gold = bound_source(folder, "gold")
    return old_qualify(records, gold, cfg)


def preflight_receipt(value, vocabulary, max_position):
    depths = value["depths"]
    if set(depths) != {"1", "2"}:
        raise ValueError("Both physical depths must close preflight")
    for depth in depths.values():
        if depth["tested_max_position"] != max_position or depth["KV_bytes_at_max_position"] <= 0:
            raise ValueError("Preflight capacity differs from the source")
        if set(depth["checks"]) != {"actual_CAL", "padded_CAL"}:
            raise ValueError("Both preflight pair layouts are required")
        for pair in depth["checks"].values():
            if len(pair["prefill"]) != 2 or len(pair["increment"]) != 2:
                raise ValueError("Preflight omitted a paired row")
            for cell in [*pair["prefill"], *pair["increment"], pair["pruned_increment"]]:
                if cell["greedy_tokens_equal"] is not True or cell["vocabulary_rows_compared"] != vocabulary:
                    raise ValueError("Incomplete full-vocabulary B2 parity")


def finite(*values):
    return all(type(v) in (float, int) and math.isfinite(v) for v in values)


def score_world(result, expected_winner, donor, round_cfg, dose):
    """Recompute the endpoint from verified four-arm finite records."""
    if result["status"] != "closed":
        if (result["status"] not in {"native_gate_failed", "native_root_unqualified",
                                    "physical_unqualified", "head_unqualified"}
                or result["edited_pass2_runs"] != 0 or "outcomes" in result or "phase1_profile" in result):
            raise ValueError("An unqualified world recorded edited outcomes")
        return None
    cases = result["cases"]
    arms, before, after = cases["arm_dose"], result["phase1_profile"], result["outcomes"]
    if result["edited_pass2_runs"] != 4 or any(set(packet) != KEYS for packet in (arms, before, after)):
        raise ValueError("Missing a signed four-arm record")
    if not cases["physical_gate"] or result["baseline"]["winner"] != expected_winner:
        raise ValueError("Unqualified native or physical baseline")
    baseline = result["baseline"]
    if (not finite(baseline["bare_margin"], *baseline["phase2_bpd_read"]) or len(baseline["phase2_bpd_read"]) != 8
            or baseline["winner"] == donor):
        raise ValueError("Invalid native score or BPD read")
    residual_norm = result["phase1_residual_l2"]
    if not finite(residual_norm) or residual_norm <= 0:
        raise ValueError("A finite native residual norm is required")
    registered_radius = residual_norm * dose["phase1_dose_fraction"]
    requested, actual = [], []
    for key, arm in arms.items():
        pre, post = before[key], after[key]
        radius, reached = arm["requested_l2"], arm["realized_l2"]
        cosine, target = arm["direction_cosine"], arm["cosine_with_original_contrast"]
        gate = arm["live_phase1"]
        if (not finite(radius, reached, cosine, target, pre["rms_logit_drift"], pre["signed_bare_margin_delta"],
                       gate["max_abs_logit_delta"], gate["margin_delta_abs"], post["bare_margin"])
                or not finite(*post["phase2_bpd_read"]) or len(post["phase2_bpd_read"]) != 8
                or type(post["winner"]) is not int or not 0 <= post["winner"] < result["vocabulary_size"]
                or arm["eligible"] is not True or radius <= 0 or reached <= 0
                or not math.isclose(radius, registered_radius, rel_tol=1e-12)
                or abs(reached / radius - 1.) > dose["dose_relative_tolerance"]
                or not dose["minimum_direction_cosine"] <= cosine <= 1. + 1e-12
                or not -1. - 1e-12 <= target <= 1. + 1e-12
                or (key.startswith("contrast/") and target < round_cfg["minimum_reached_contrast_cosine"])
                or (key.startswith("sham/") and abs(target) > round_cfg["maximum_reached_sham_cosine"])
                or gate["passed"] is not True or gate["greedy_same"] is not True
                or not 0 <= pre["rms_logit_drift"] <= gate["max_abs_logit_delta"] + 1e-12
                or gate["max_abs_logit_delta"] > dose["maximum_head_delta"]
                or gate["margin_delta_abs"] != abs(pre["signed_bare_margin_delta"])
                or gate["margin_delta_abs"] > dose["maximum_margin_delta"]):
            raise ValueError("A claimed arm violates a finite gate or profile identity")
        requested.append(radius)
        actual.append(reached)
    separation = cases["distance_in_nominal_radii"]
    if (len(set(requested)) != 1 or max(actual)-min(actual) > dose["max_contrast_dose_gap_fraction"] * requested[0]
            or set(separation) != set(SIGNS) or not finite(*separation.values())
            or min(separation.values()) < round_cfg["minimum_contrast_sham_distance_in_nominal_radii"]):
        raise ValueError("Four realized arms were not matched and distinct")

    def signed(packet, field, arm):
        plus, minus = (f"{arm}/{sign}" for sign in SIGNS)
        scale = requested[0] / (arms[plus]["realized_l2"] + arms[minus]["realized_l2"])
        return scale * (packet[plus][field] - packet[minus][field])

    second = signed(after, "bare_margin", "contrast") - signed(after, "bare_margin", "sham")
    direct = signed(before, "signed_bare_margin_delta", "contrast") - signed(before, "signed_bare_margin_delta", "sham")
    peak_gap = max(abs(before[f"sham/{s}"]["rms_logit_drift"] - before[f"contrast/{s}"]["rms_logit_drift"])
                   for s in SIGNS)
    return {"signed_pass2_margin_contrast_minus_sham": second, "signed_pass1_margin_contrast_minus_sham": direct,
            "max_signed_rms_gap": peak_gap, "active_positive_bare_donor_flip": after["contrast/positive"]["winner"] == donor,
            "sham_positive_bare_donor_flip": after["sham/positive"]["winner"] == donor}


def audit(folder, work, manifest_sha, cfg, round_cfg, dose):
    record = json.loads((work / "terminal.json").read_bytes())
    if (record["runtime_sha256"] != manifest_sha or record["backend"] != cfg["backend"]
            or record["target_before"] != TARGETS[cfg["backend"]] or record["target_after"] != record["target_before"]
            or record["quant_unchanged"] is not True or record["checkpoint_verified_after"] is not True
            or record["status"] != "completed" or sha(work / "preflight.json") != record["preflight_sha256"]
            or sha(work / "native.json") != record["native_sha256"]):
        raise ValueError("Missing registered immutable model and checkpoint closure")
    preflight_receipt(json.loads((work / "preflight.json").read_bytes()), record["vocabulary_size"],
                      json.loads(SOURCE_CHECK.read_bytes())["max_prompt_length"] + 2)
    native = json.loads((work / "native.json").read_bytes())
    gate = native_gate(folder, native, cfg)
    gate_file = work / "qualification.json"
    if sha(gate_file) != record["qualification_sha256"] or json.loads(gate_file.read_bytes()) != gate:
        raise ValueError("Live and independent competence decisions differ")
    gold = {(r["root"], r["world"]): r for r in bound_source(folder, "gold") if r["arm"] == "focused"}
    labels = bound_source(folder, "tokens")["bare_label_token_ids"]
    native_tokens = {r["id"]: r["token_id"] for r in native}
    journal = work / "closed.jsonl"
    if sha(journal) != record["closed_journal_sha256"]:
        raise ValueError("The signed-world closure journal changed")
    lines = journal.read_bytes().splitlines(keepends=True)
    if len(lines) != 2 * cfg["source_roots"] or record["records"] != len(lines) or any(not x.endswith(b"\n") for x in lines):
        raise ValueError("The signed-world roster or journal is incomplete")
    roots, unknown = defaultdict(list), defaultdict(int)
    for i, (line, ((root, world), meta)) in enumerate(zip(lines, sorted(gold.items()), strict=True)):
        path = work / f"episode_{i:03d}.json"
        if json.loads(line) != {"file": path.name, "sha256": sha(path)}:
            raise ValueError("An immutable world record differs from its journal")
        row = json.loads(path.read_bytes())
        if row["source_id"] != meta["id"]:
            raise ValueError("The record is attached to the wrong registered world")
        foil = gold[root, "B" if world == "A" else "A"]
        expected_status = (
            "native_gate_failed" if not gate["passed"] else
            "native_root_unqualified" if root not in gate["qualified_focused_roots"] else None)
        if (expected_status and row["status"] != expected_status) or (expected_status is None and row["status"].startswith("native_")):
            raise ValueError("Continuation disagrees with independent native root qualification")
        if row["status"] == "closed" and row["vocabulary_size"] != record["vocabulary_size"]:
            raise ValueError("Edited vocabulary rows do not match worker scope")
        score = score_world(row, native_tokens[meta["id"]], labels[foil["answer"]], round_cfg, dose)
        roots[root].append(score)
        if score is None:
            unknown[row["status"]] += 1
    complete = [scores for scores in roots.values() if len(scores) == 2 and all(x is not None for x in scores)]
    root_means = [{key: fmean(row[key] for row in pair) for key in pair[0]} for pair in complete]
    means = {key: fmean(row[key] for row in root_means) for key in root_means[0]} if root_means else {}
    floor = cfg["minimum_meaningful_delayed_margin_logit"]
    return {"experiment": cfg["experiment"], "runtime_sha256": manifest_sha, "worker_terminal_sha256": sha(work / "terminal.json"),
            "roots_registered": len(roots), "roots_with_both_worlds_scored": len(complete),
            "worlds_scored": sum(x is not None for pair in roots.values() for x in pair),
            "refusals": dict(unknown), "native_gate_passed": gate["passed"],
            "root_mean": means, "engineering_margin_floor": floor,
            "complete_roots_at_or_above_floor": sum(r["signed_pass2_margin_contrast_minus_sham"] >= floor for r in root_means),
            "complete_roots_at_or_below_negative_floor": sum(r["signed_pass2_margin_contrast_minus_sham"] <= -floor for r in root_means),
            "formal_comparisons": 0, "future_sham_variation_floor_calibrated": False, "sealed_test_accessed": False}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source-dir", type=Path, required=True)
    cli.add_argument("--native-records", type=Path)
    cli.add_argument("--work", type=Path)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if (args.native_records is None) == (args.work is None):
        cli.error("Supply exactly one of --native-records or --work")
    below_normal()
    cfg, _, round_cfg, dose = settings()
    if args.native_records is not None:
        value = native_gate(args.source_dir, json.loads(args.native_records.read_bytes()), cfg)
    else:
        freeze = json.loads(RUNTIME.read_bytes())
        for path, expected in freeze["files_sha256"].items():
            if sha(REPO / path) != expected:
                raise ValueError("The frozen host runtime changed")
        value = audit(args.source_dir, args.work, sha(RUNTIME), cfg, round_cfg, dose)
    write_new(args.output, value)


if __name__ == "__main__":
    main()

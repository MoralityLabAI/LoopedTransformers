"""Qualify the larger native roster and bind the matched-sham evidence."""

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import json
from math import isfinite
from pathlib import Path
import re
import sys

from experiments.nan_comp14.parser import label
from experiments.nan_delay39 import pilot
from experiments.nan_delay39.contrasts import qualified_pair
from experiments.nan_delay39.host_gate import assess
from experiments.nan_delay39.source import digest
from experiments.nan_measuredwire93.run import audit_slopes
from experiments.nan_phase35.run import lease
from experiments.nan_readout34 import runtime
from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import sha, tokenizer
from experiments.nan_readpath37.run import bound

from .source import EXPERIMENT, config, read_source


BUDGET = {"minimum_start_available_RAM_bytes": 5 * 2**30, "minimum_available_RAM_bytes": 4 * 2**30,
          "maximum_private_bytes": 384 * 2**20, "maximum_seconds": 180}
SHAMS = ("span_sham", "ambient_sham")


def qualify(data, records):
    cfg = config()
    return assess(records, data["gold"], cfg) | {"experiment": EXPERIMENT, "native_record_digest": digest(records)}


def summarize(episodes, data, decoder, cfg):
    gold = {(p["root"], p["world"]): p for p in data["gold"] if p["arm"] == "focused"}
    by_id = {p["id"]: p for p in gold.values()}
    known = defaultdict(dict)
    coverage, behavior = Counter(), Counter()
    for item in episodes:
        record = by_id[item["source_id"]]
        root, world = record["root"], record["world"]
        counterpart = gold[root, "B" if world == "A" else "A"]
        a, b = [ord(value["answer"]) - ord("A") for value in (record, counterpart)]
        observed = {"shams": {}}
        if world in known[root] or a == b:
            raise ValueError("Repeated or invalid focused world")
        known[root][world] = observed
        base = item["baseline"]
        cases = item["cases"]
        expected = {tag: qualified_pair(cases["twin_donor"], cases[tag], cfg) for tag in SHAMS}
        if item["donor_sham_pairs"] != expected:
            raise ValueError("Matched donor-sham flags disagree with reached doses")
        for response in (base, *(case["response"] for case in cases.values() if "response" in case)):
            if (len(response["phase2_label_logits"]) != 16 or not all(map(isfinite, response["phase2_label_logits"]))
                    or len(response["top_token"]) != 2 or type(response["top_token"][1]) is not int):
                raise ValueError("Native delayed response does not close")
            if response["phase2_decoded"] != pilot.literal(decoder, response["top_token"][1]):
                raise ValueError("Decoded native decision differs from the pinned tokenizer")
        base_logits = base["phase2_label_logits"]
        observed["clean_correct"] = label(base["phase2_decoded"]) == record["answer"]
        if cases["twin_donor"]["eligible"]:
            correct = label(cases["twin_donor"]["response"]["phase2_decoded"]) == record["answer"]
            behavior["repair" if correct and not observed["clean_correct"] else
                     "regression" if observed["clean_correct"] and not correct else "tie"] += 1
        for tag in SHAMS:
            if expected[tag]:
                edit = cases[tag]["response"]["phase2_label_logits"]
                observed["shams"][tag] = abs((edit[a] - edit[b]) - (base_logits[a] - base_logits[b]))
                coverage[tag] += 1
    samples = [max(v for row in worlds.values() for v in row["shams"].values())
               for worlds in known.values() if set(worlds) == {"A", "B"}
               and all(set(r["shams"]) == set(SHAMS) for r in worlds.values())]
    enough = len(samples) >= cfg["minimum_matched_sham_roots"]
    return {"registered_roots": len(gold)//2, "closed_focused_worlds": len(episodes),
            "matched_sham_worlds": dict(coverage), "complete_two_sham_roots": len(samples),
            "sham_calibration_class": "conditional_tail_ready" if enough else "insufficient_complete_roots",
            "complete_root_observed_sham_maximum": max(samples) if samples else None,
            "candidate_sham_bound_95_percentile": max(samples) if enough else None,
            "nominal_coverage_if_independent_complete_sampling": float(1 - Fraction(19, 20)**len(samples))
            if enough else None, "donor_vs_clean_world_counts": dict(behavior),
            "formal_comparisons": 0, "sealed_test_accessed": False}, dict(known)


def collect(directory, data, decoder, writes, commit, frozen, check, np):
    t = directory / "terminal.json"
    terminal_hash = sha(t)
    terminal = json.loads(bound(t, terminal_hash))
    ci = job_admission(directory, terminal["admission_sha256"], frozen, commit)
    backend = terminal["backend"]
    if (backend != ci["backend"] or terminal["published_commit"] != commit or terminal["runtime_sha256"] != frozen
            or terminal["experiment"] != EXPERIMENT or terminal["target_before"] != runtime.TARGETS[backend]
            or terminal["target_after"] != terminal["target_before"] or terminal["quant_unchanged"] is not True
            or terminal["checkpoint_verified_after"] is not True):
        raise ValueError("Native packed terminal differs from the registered condition")
    proof = json.loads(bound(directory / "preflight.json", terminal["preflight_sha256"]))
    if (proof["packed_parity_worlds"] != 2 or proof["packed_audited_edits"] < 28
            or proof["status"] != "passed"):
        raise ValueError("Native admission lost full packed parity or capacity")
    raw = json.loads(bound(directory / "native.json", terminal["native_sha256"]))
    verdict = qualify(data, raw)
    if verdict != json.loads(bound(directory / "qualification.json", terminal["qualification_sha256"])):
        raise ValueError("Independent native competence differs from the archived gate")
    cfg = config()
    roots = sorted({p["root"] for p in data["public"]})
    focus = {(p["root"], p["world"]): p for p in data["public"] if p["arm"] == "focused"}
    source_ids = [focus[root, w]["id"] for root in roots for w in ("A", "B")]
    if terminal["status"] == "native_gate_failed":
        if verdict["passed"] or terminal["closed_focused_worlds"] or (directory / "closed.jsonl").exists():
            raise ValueError("A failed native condition exposes edited outcomes")
        result, private = summarize([], data, decoder, cfg)
        return result | {"terminal_sha256": terminal_hash, "status": terminal["status"]}, private
    if terminal["status"] != "closed" or not verdict["passed"] or terminal["closed_focused_worlds"] != len(source_ids):
        raise ValueError("A qualified terminal did not close the registered roster")
    journal = bound(directory / "closed.jsonl", terminal["closed_journal_sha256"]).splitlines()
    if len(journal) != len(source_ids):
        raise ValueError("The closed native journal missed focused sources")
    episodes, work = [], Counter()
    scales = np.linalg.norm(writes, axis=1).tolist()
    for i, (line, identity) in enumerate(zip(journal, source_ids, strict=True)):
        check()
        entry = json.loads(line)
        file = f"episode_{i:04d}.json"
        if entry["file"] != file:
            raise ValueError("The edited journal order changed")
        item = json.loads(bound(directory / file, entry["sha256"]))
        if (item["source_id"] != identity or item["prompt_sha256"] != data["tokens"]["row_bindings"][identity]["sha256"]
                or item["self_closes"] is not True):
            raise ValueError("The edited response did not close its focused source")
        items = item["work"]
        scheduled = i//2 % cfg["parity_audit_root_stride"] == 0
        qualified = sum(case["eligible"] for case in item["cases"].values())
        if (items["qualified_edited_rows"] != qualified
                or items["audited_edited_rows"] != (qualified if scheduled else 0)
                or items["full_B2_prefills"] != 2 + items["audited_edited_rows"]
                or items["phase1_primes"] != 1
                or items["physical_B2_pass_traversals"] != (2 * items["full_B2_prefills"] + 1
                                                            + items["packed_phase2_branches"])):
            raise ValueError("Packed traversal or periodic parity accounting differs")
        audit_slopes(item, writes, scales, cfg, list(range(len(writes))), np)
        work.update(items)
        episodes.append(item)
    outcome, private = summarize(episodes, data, decoder, cfg)
    timing = {k: terminal[k] for k in ("baseline_and_host_gate_wall_seconds", "focused_measurement_wall_seconds")}
    if not all(isfinite(v) and v >= 0 for v in timing.values()):
        raise ValueError("Native terminal timing is missing or invalid")
    return outcome | {"terminal_sha256": terminal_hash, "status": terminal["status"],
                      "measured_work": dict(work), "timing": timing}, private


def job_admission(directory, expected, frozen, commit):
    record = json.loads(bound(directory / "admission.json", expected))
    if (record["runtime_sha256"] != frozen or record["published_commit"] != commit
            or record["experiment"] != EXPERIMENT or re.fullmatch(r"[0-9a-f]{40}", commit) is None):
        raise ValueError("The native worker has no matching published admission")
    return record


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source-dir", type=Path, required=True)
    cli.add_argument("--snapshot", type=Path, required=True)
    cli.add_argument("--work", type=Path, required=True)
    cli.add_argument("--qualification", action="store_true")
    args = cli.parse_args()
    from .lane import verified
    frozen = verified()
    if "torch" in sys.modules:
        raise ValueError("Independent packed host imported Torch")
    with lease(BUDGET) as (check, _samples):
        data, pins = read_source(args.source_dir, ("public", "gold", "tokens"))
        if args.qualification:
            a = json.loads((args.work / "admission.json").read_bytes())
            job_admission(args.work, sha(args.work / "admission.json"), frozen, a["published_commit"])
            rows = json.loads((args.work / "native.json").read_bytes())
            write_new(args.work / "qualification.json", qualify(data, rows))
        else:
            import numpy as np

            tok, verified_pins = tokenizer(args.snapshot)
            if verified_pins != pins["tokenizer_pins"]:
                raise ValueError("Native host tokenizer differs from the frozen source")
            terminal = json.loads((args.work / "terminal.json").read_bytes())
            _, writes = pilot.get_bank(terminal["backend"], np)
            result, private = collect(args.work, data, tok.tok, writes, terminal["published_commit"], frozen, check, np)
            file = args.work / "private_outcomes.json"
            write_new(file, private)
            write_new(args.work / "aggregate.json", result | {"experiment": EXPERIMENT, "backend": terminal["backend"],
                       "runtime_sha256": frozen, "private_outcomes_sha256": sha(file)})
    assert "torch" not in sys.modules


if __name__ == "__main__":
    main()

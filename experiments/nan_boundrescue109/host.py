"""Qualify full native baselines and independently reduce closed reader-rescue roots."""

import argparse
from collections import Counter
import json
from math import isclose, isfinite
from pathlib import Path
from statistics import fmean
import sys

from experiments.nan_bankdose99.geometry import CURRENT_DRIFT_LIMIT
from experiments.nan_chordsham106.geometry import MAX_ABS_COSINE, MAX_ENDPOINT_RADIUS_ERROR, MAX_RMS_GAP
from experiments.nan_delay39.source import digest
from experiments.nan_phase35.run import lease
from experiments.nan_readout34 import runtime
from experiments.nan_readout34.tokenizer_check import sha
from experiments.nan_readpath37.run import bound, write_new
from experiments.nan_vocab82.heads import partition_sha
from experiments.nan_vocab83.custody import HOST_BUDGET, qualification, vocabulary
from experiments.nan_writer77.host import reading

from . import roster
from .core import ARMS, SIGNS, WORLDS


CFG = roster.CONFIG | {"minimum_clean_both_world_roots": 12, "minimum_literal_copy_both_world_roots": 16}
MODE = "pretrained_cuda"
ENDPOINTS = ("edited", "k_rescue")
REFUSE = ("geometry_unqualified", "mixed_duplicate_unqualified", "head_unqualified")


def host_inputs(folder):
    proof = json.loads(roster.RECEIPT.read_bytes())
    if (proof["manifest_sha256"] != sha(roster.MANIFEST) or json.loads(roster.MANIFEST.read_bytes()) != roster.registration()):
        raise ValueError("The host roster freeze differs")
    source = {name: json.loads((folder / f"{name}.json").read_bytes()) for name in ("public", "gold", "tokens")}
    if any(digest(value) != proof["payload_sha256"][name] for name, value in source.items()):
        raise ValueError("A frozen source payload changed")
    return source, proof


def qualify(source, rows, baseline_sha, classes, tok, admission):
    result = qualification(source, rows, baseline_sha, classes, tok, CFG)
    gold = {(r["root"], r["world"]): r["answer"] for r in source["gold"] if r["arm"] == "focused"}
    bare = source["tokens"]["bare_label_token_ids"]
    pairs = {}
    if result["gate"]["passed"]:
        for root in result["gate"]["qualified_focused_roots"]:
            pairs[root] = [bare[gold[root, world]] for world in WORLDS]
            if len(set(pairs[root])) != 2:
                raise ValueError("The oracle did not produce distinct paired labels")
    return result | {"experiment": roster.EXPERIMENT, "runtime_sha256": admission["runtime_sha256"],
                     "backend": admission["backend"], "mode": admission["mode"], "answer_token_pairs": pairs}


def finite_range(value, maximum):
    return type(value) in (int, float) and isfinite(value) and 0 <= value <= maximum


def check_plan(plan, record):
    state = record["status"]
    expected = "qualified" if state == "closed" else state
    if plan["status"] != expected or any(plan[k] != record[k] for k in ("root", "geometry", "signed")):
        raise ValueError("A root record differs from the pre-continuation plan")
    if state != "closed":
        if record.get("edited_pass2_runs") != 0 or "worlds" in record or "head_observations" in record:
            raise ValueError("A refused root contains edited continuations")
        return
    if set(plan["signed"]) != set(WORLDS) or set(plan["geometry"]) != set(WORLDS):
        raise ValueError("A closed plan lacks both worlds")
    for world in WORLDS:
        by_sign = plan["signed"][world]
        geometry = plan["geometry"][world]
        if (geometry["parent"]["status"] != "measured" or set(by_sign) != {"1", "-1"}
                or set(geometry["signed"]) != {"1", "-1"}):
            raise ValueError("A closed plan lacks the attained parent and both signs")
        for sign, signed in by_sign.items():
            audit = geometry["signed"][sign]
            sham = audit["sham"]
            if (audit["centered"]["differential"]["qualified"] is not True or sham["status"] != "matched"
                    or not finite_range(sham["reached_abs_chord_cosine"], MAX_ABS_COSINE)
                    or any(not finite_range(sham[k], MAX_ENDPOINT_RADIUS_ERROR)
                           for k in ("chord_relative_error", "max_endpoint_radius_relative_error"))):
                raise ValueError("The reached sham misses its physical geometry")
            if (signed["passed"] is not True or len(signed["arm_gates"]) != len(ARMS)
                    or any(g["passed"] is not True or g["greedy_same"] is not True
                           or not finite_range(g["max_abs_logit_delta"], CURRENT_DRIFT_LIMIT)
                           or not finite_range(g["margin_delta_abs"], CURRENT_DRIFT_LIMIT) for g in signed["arm_gates"])):
                raise ValueError("An edited endpoint did not clear the unrestricted first-pass gate")
            for key in ("absolute_rms_gaps", "excluded_rms_gaps"):
                if len(signed[key]) != 2 or any(not finite_range(v, MAX_RMS_GAP) for v in signed[key]):
                    raise ValueError("An edited chord missed one of the two RMS gates")
            if any(len(signed[k]) != 2 or any(not finite_range(v, float("inf")) for v in signed[k])
                   for k in ("active_rms", "sham_rms")):
                raise ValueError("A planned RMS value is missing or nonfinite")
            gaps = [abs(a-b) for a, b in zip(signed["active_rms"], signed["sham_rms"], strict=True)]
            if gaps != signed["absolute_rms_gaps"]:
                raise ValueError("The sealed full-vocabulary RMS gaps do not close")


def same(a, b):
    return isclose(a, b, rel_tol=0., abs_tol=1e-6)


def margin(packet, own, other):
    return packet["labels"][own] - packet["labels"][other]


def recompute(root, clean, gold, bare, tok, vocab):
    """Reduce recorded full-head winners and bare logits without the native worker."""
    if (root.get("edited_pass2_runs") != 32 or len(root["head_observations"]) != 32
            or set(root["worlds"]) != set(WORLDS)):
        raise ValueError("The closed root lacks the complete edited and rescued head series")
    reads = iter(root["head_observations"])
    contrasts, outcomes, evidence = {}, Counter(), {}
    for i, world in enumerate(WORLDS):
        own, other = gold[root["root"], world]["answer"], gold[root["root"], WORLDS[1-i]]["answer"]
        before = clean[root["source_ids"][i]]["phases"]
        evidence[world] = {}
        for sign in map(str, SIGNS):
            signed = root["worlds"][world][sign]
            if set(signed["arms"]) != set(ARMS):
                raise ValueError("A root is missing a matched arm")
            effects, records = {}, {}
            for arm in ARMS:
                native = signed["arms"][arm]
                if native["zeroed_k_delta"] is not True or native["k_rescue"]["reached_projection"]["k"]["l2_change"] != 0:
                    raise ValueError("The rescue did not deliver clean K")
                effects[arm], records[arm] = {}, {}
                for condition in ENDPOINTS:
                    obs = next(reads)
                    if obs["rescued"] is not (condition == "k_rescue") or len(obs["phases"]) != 2:
                        raise ValueError("An edited phase or rescue occurs out of order")
                    scored = [reading(row, own, bare, tok, vocab) for row in obs["phases"]]
                    delta = [margin(row, own, other)-margin(b, own, other) for row, b in zip(obs["phases"], before, strict=True)]
                    if not same(delta[1], native[condition]["answer_margin_change"]):
                        raise ValueError("Recorded candidate logits disagree with the worker margin")
                    if ((obs["phases"][1]["winner"] != before[1]["winner"]) != native[condition]["greedy_token_changed"]
                            or (obs["phases"][1]["winner"] == bare[own]) != native[condition]["reached_greedy_correct"]):
                        raise ValueError("The raw unrestricted winner differs from the worker outcome")
                    effects[arm][condition] = delta
                    records[arm][condition] = obs["phases"][0], scored
                if records[arm]["edited"][0] != records[arm]["k_rescue"][0]:
                    raise ValueError("K restoration changed the first-pass head receipt")
            def chord(condition, phase):
                t1, t2, s1, s2 = (effects[arm][condition][phase] for arm in ARMS)
                return ((t2-t1) - (s2-s1))/2

            first, future, restored = chord("edited", 0), chord("edited", 1), chord("k_rescue", 1)
            contrast = {"first_pass": first, "second_pass": future, "second_pass_with_clean_k": restored,
                        "delayed": future-first, "delayed_with_clean_k": restored-first,
                        "k_rescue_removed": future-restored}
            if (not same(first, root["signed"][world][sign]["paired_label_contrast"])
                    or any(not same(v, signed["contrast"][k]) for k, v in contrast.items())):
                raise ValueError("The independent signed two-pass contrast differs")
            contrasts[world, sign] = contrast
            a = records["active_t2"]["edited"][1][1]
            b = records["active_t1"]["edited"][1][1]
            rescued = records["active_t2"]["k_rescue"][1][1]
            outcomes["active_t2_repairs_vs_t1"] += a and not b
            outcomes["active_t2_regressions_vs_t1"] += b and not a
            outcomes["active_t2_rescue_repairs"] += rescued and not a
            outcomes["active_t2_rescue_regressions"] += a and not rescued
            evidence[world][sign] = {"paired_margin": contrast, "endpoint_grammar_correct": {
                arm: {mode: records[arm][mode][1] for mode in ENDPOINTS} for arm in ARMS}}
    macro = {k: sum(contrasts[w, "1"][k]-contrasts[w, "-1"][k] for w in WORLDS)/4
             for k in ("first_pass", "second_pass", "delayed", "delayed_with_clean_k", "k_rescue_removed")}
    if any(not same(v, root["root_signed_contrast"][k]) for k, v in macro.items()):
        raise ValueError("The independent root macro differs")
    return {"root": root["root"], "signed_macro": macro, "worlds": evidence}, outcomes


def collect(work, source, classes, tok, frozen, backend, *, mode=MODE):
    end_path = work / "terminal.json"
    end_sha = sha(end_path)
    end = json.loads(bound(end_path, end_sha))
    expected = (roster.EXPERIMENT, "completed", frozen, backend, mode)
    if tuple(end[k] for k in ("experiment", "status", "runtime_sha256", "backend", "mode")) != expected:
        raise ValueError("Terminal does not certify the admitted execution mode")
    target = runtime.TARGETS[backend]
    if (end["target_before"] != target or end["target_after"] != target
            or end["checkpoint_verified_after"] is not True or end["quant_before"] != end["quant_after"]
            or end["partition_sha256"] != partition_sha(classes)):
        raise ValueError("Checkpoint or vocabulary identity changed")
    admission = json.loads(bound(work/"admission.json", end["admission_sha256"]))
    if (any(admission[k] != end[k] for k in ("experiment", "runtime_sha256", "backend", "mode", "published_commit"))
            or len(end["published_commit"]) != 40 or any(c not in "0123456789abcdef" for c in end["published_commit"])):
        raise ValueError("Terminal does not bind the admitted publication")
    preflight = json.loads(bound(work/"preflight.json", end["preflight_sha256"]))
    position = json.loads(roster.RECEIPT.read_bytes())["max_position"]
    if (preflight["native_heads_exact"] is not True or preflight["head_placement_exact"] is not True
            or set(preflight["depths"]) != {"1", "2"}
            or any(d["tested_max_position"] != position for d in preflight["depths"].values())):
        raise ValueError("Terminal does not bind the B2 capacity preflight")
    baseline = json.loads(bound(work/"baseline.json", end["baseline_sha256"]))
    gate = qualify(source, baseline, end["baseline_sha256"], classes, tok, admission)
    if gate != json.loads(bound(work/"qualification.json", end["qualification_sha256"])):
        raise ValueError("Independent competence gate differs from the closed receipt")
    journal = bound(work/"closed.jsonl", end["journal_sha256"]).splitlines()
    roots = sorted({p["root"] for p in source["public"]})
    if len(journal) != len(roots) or end["root_records"] != len(roots) or 2*end["baseline_batches"] != len(baseline):
        raise ValueError("The worker did not close its full roster")
    clean = {r["id"]: r for r in baseline}
    gold = {(r["root"],r["world"]): r for r in source["gold"] if r["arm"] == "focused"}
    statuses, counts, private = Counter(), Counter(), []
    for i,(root,line) in enumerate(zip(roots, journal, strict=True)):
        entry = json.loads(line)
        name = f"root_{i:03d}.json"
        if entry["file"] != name:
            raise ValueError("Wrong root order in the terminal journal")
        record = json.loads(bound(work/name, entry["sha256"]))
        if record["root"] != root or record["source_ids"] != [gold[root,w]["id"] for w in WORLDS]:
            raise ValueError("Closed root does not bind its A/B inputs")
        state = record["status"]
        allowed = ("native_gate_failed",) if not gate["gate"]["passed"] else (
            ("closed", *REFUSE) if root in gate["answer_token_pairs"] else ("native_root_unqualified",))
        if state not in allowed:
            raise ValueError("Closed root contradicts baseline competence")
        statuses[state] += 1
        if state in REFUSE or state == "closed":
            plan = json.loads(bound(work/f"plan_{i:03d}.json", record["plan_sha256"]))
            check_plan(plan, record)
        elif record.get("edited_pass2_runs",0) != 0 or "plan_sha256" in record or "worlds" in record:
            raise ValueError("A competence refusal contains edited outcomes")
        if state == "closed":
            paired, ce = recompute(record, clean, gold, source["tokens"]["bare_label_token_ids"], tok, len(classes))
            private.append(paired)
            counts.update(ce)
    macro = ({k: fmean(v["signed_macro"][k] for v in private) for k in private[0]["signed_macro"]} if private else {})
    return {"experiment": roster.EXPERIMENT, "backend": backend, "mode": mode, "runtime_sha256": frozen,
            "terminal_sha256": end_sha, "registered_roots": len(roots), "closed_roots": len(private),
            "status_counts": dict(statuses), "answer_events_by_signed_world": dict(counts),
            "closed_root_macro": macro, "evidence": "development_calibration", "formal_comparisons": 0,
            "sealed_test_accessed": False}, private


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "source-dir", "work"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--backend", choices=("nf4","bf16"), required=True)
    parser.add_argument("--qualification", action="store_true")
    args = parser.parse_args()
    from .lane import verified

    frozen = verified()
    if "torch" in sys.modules or args.snapshot.name != roster.CONFIG["checkpoint_revision"]:
        raise RuntimeError("Host scorer requires the pinned tokenizer without Torch")
    with lease(HOST_BUDGET) as (check, samples):
        source, pins = host_inputs(args.source_dir)
        tok, classes = vocabulary(args.snapshot, check, pins["tokenizer_pins"])
        if args.qualification:
            admission = json.loads((args.work/"admission.json").read_bytes())
            if (admission["runtime_sha256"], admission["backend"], admission["mode"]) != (frozen,args.backend,MODE):
                raise ValueError("Qualification failed to bind the admitted run")
            baseline = args.work/"baseline.json"
            pin = sha(baseline)
            rows = json.loads(bound(baseline,pin))
            write_new(args.work/"qualification.json", qualify(source,rows,pin,classes,tok,admission))
            return
        results, private = collect(args.work, source, classes, tok, frozen, args.backend)
    write_new(args.work/"private_outcomes.json", private)
    write_new(args.work/"aggregate.json", results | {
        "private_outcomes_sha256": sha(args.work/"private_outcomes.json"),
        "max_sampled_private_bytes": max(s["private_bytes"] for s in samples),
        "min_sampled_available_RAM_bytes": min(s["available_physical_bytes"] for s in samples),
    })


if __name__ == "__main__":
    main()

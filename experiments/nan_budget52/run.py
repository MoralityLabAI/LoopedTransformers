"""Replay fixed response budgets using only independently audited STREAM41 outcomes."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

from experiments.nan_gym05.common import prompt_ids
from experiments.nan_phase35.run import lease
from experiments.nan_readout34 import runtime, source, worlds
from experiments.nan_readout34.runtime_audit import bounds, paired
from experiments.nan_readout34.runtime_plan import CAP, context_bindings, schedule, write_new
from experiments.nan_readout34.tokenizer_check import id_hash, settings, sha, tokenizer
from experiments.nan_stream41.pipeline import collect, fixed_jobs
from experiments.nan_stream41.run import ARTIFACT, MANIFEST, MAXIMUM_POSITION, STREAM36, verify


TOKEN_CAPS = (8, 16, 32, 48, 64, CAP)
DECODE_VISIT_CAPS = (16, 32, 64, 94)
LIMITS = {
    "minimum_start_available_RAM_bytes": 5 * 2**30,
    "minimum_available_RAM_bytes": 4 * 2**30,
    "maximum_private_bytes": 768 * 2**20,
    "maximum_seconds": 300,
}


def prefixes(receipt, cap, passes):
    if receipt is None:
        return None
    grade, record = receipt
    emitted = len(record["generated_ids"])
    tokens = min(cap, emitted)
    # First-stop audit already ruled out a parseable reply at every earlier prefix.
    closed = emitted <= cap
    return {
        "exact": grade["joint_exact"] if closed else 0,
        "format_valid": grade["format_valid"] if closed else 0,
        "final_room": grade["fields_correct"]["final_room"] if closed else 0,
        "tokens": tokens,
        "decode_visits": passes * (tokens - 1),
        "unpadded_visits": passes * (record["prompt_tokens"] + tokens - 1),
    }


def workload(row):
    return (row["split"], row["query_type"], row["event_count"])


def field(observation, name):
    return None if observation is None else observation[name]


def twins(values):
    if set(values) != {"A", "B"}:
        raise ValueError("Incomplete registered counterfactual roster")
    return 0 if 0 in values.values() else None if None in values.values() else 1


def depth_cell(rows, verified, canonical_tokens, caps):
    outcomes = [
        [prefixes(verified.get(slot.ordinal), cap, slot.passes) for slot, cap in zip(pair, caps)]
        for pair in rows
    ]
    complete = [pair for pair in outcomes if None not in pair]
    root_values = [defaultdict(dict), defaultdict(dict)]
    for slots, pair in zip(rows, outcomes):
        row = slots[0].source
        for i, item in enumerate(pair):
            root_values[i][row["root"]][row["world"]] = field(item, "exact")
    both_worlds = [[twins(values) for values in roots.values()] for roots in root_values]
    return {
        "token_caps_by_depth": list(caps),
        "canonical_reply_fits_by_depth": [
            sum(canonical_tokens[pair[0].source["id"]] <= cap for pair in rows) for cap in caps
        ],
        "exact_by_depth": [bounds([field(item, "exact") for item in depth]) for depth in zip(*outcomes)],
        "exact_one_to_two": paired([tuple(field(item, "exact") for item in pair) for pair in outcomes]),
        "format_one_to_two": paired([tuple(field(item, "format_valid") for item in pair) for pair in outcomes]),
        "both_worlds_by_depth": [bounds(values) for values in both_worlds],
        "both_worlds_one_to_two": paired(list(zip(*both_worlds))),
        "complete_pair_cost": {
            "pairs": len(complete),
            **{name: [sum(pair[i][name] for pair in complete) for i in range(2)]
               if complete else None for name in ("tokens", "decode_visits", "unpadded_visits")},
        },
    }


def evidence_cell(views, verified, cap):
    results = [
        [prefixes(verified.get(slot.ordinal), cap, slot.passes) for slot in (final, packet)]
        for final, packet in views
    ]
    return {
        "output_token_cap": cap,
        "path_and_checkpoints_verifiable": bounds([field(packet, "exact") for _, packet in results]),
        "final_only_to_packet_final": paired([
            (field(final, "exact"), field(packet, "final_room")) for final, packet in results
        ]),
        "packet_final_to_verified_packet": paired([
            (field(packet, "final_room"), field(packet, "exact")) for _, packet in results
        ]),
    }


def summarize(slots, data, canonical_tokens):
    indexed = {(slot.source["id"], slot.passes): slot for slot in slots}
    depth_groups = defaultdict(list)
    view_groups = defaultdict(lambda: defaultdict(dict))
    for shallow in (slot for slot in slots if slot.passes == 1):
        row = shallow.source
        readout = row["schema"]["readout"]
        matched = (shallow, indexed[row["id"], 2])
        depth_groups[(*workload(row), row["arm"], readout)].append(matched)
        if row["arm"] in {"compute", "copy"}:
            for slot in matched:
                view_groups[(*workload(row), row["arm"], slot.passes)][row["root"], row["world"]][readout] = slot

    budget_caps = [
        *(("output_tokens", cap, (cap, cap)) for cap in TOKEN_CAPS),
        *(("decode_phase_visits", cap, (1 + cap, 1 + cap // 2)) for cap in DECODE_VISIT_CAPS),
    ]
    result = {}
    for backend, verified in data.items():
        depth = [
            {"stratum": list(key), "budget_kind": kind, "budget": cap,
             **depth_cell(rows, verified, canonical_tokens, caps)}
            for key, rows in sorted(depth_groups.items()) for kind, cap, caps in budget_caps
        ]
        evidence = [
            {"stratum": list(key), **evidence_cell(
                [(view["final"], view["packet"]) for view in cases.values()], verified, cap)}
            for key, cases in sorted(view_groups.items()) for cap in TOKEN_CAPS
        ]
        result[backend] = {"depth": depth, "answer_and_packet": evidence}
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--snapshot", type=Path, required=True)
    cli.add_argument("--tokenizer-receipt", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    verify()
    settings()
    with lease(LIMITS) as (check, samples):
        receipt = context_bindings(args.tokenizer_receipt)
        snapshot = args.snapshot.resolve()
        if snapshot.name != source.CONFIG["checkpoint_revision"]:
            raise ValueError("Tokenizer snapshot revision mismatch")
        tok, pins = tokenizer(snapshot)
        if pins != receipt["tokenizer_files_sha256"]:
            raise ValueError("Tokenizer binding changed")
        rows = source.build()
        slots = schedule(rows)
        bindings = receipt["contexts"]
        jobs = fixed_jobs(slots, bindings, json.loads((STREAM36 / "SCHEDULING_VERIFICATION.json").read_bytes()))
        canonical_tokens = {}
        canonical_maxima = Counter()
        for row in rows:
            ids = prompt_ids(tok, snapshot, row["prompt"])
            if bindings[row["id"]] != {"tokens": len(ids), "ids_sha256": id_hash(ids)}:
                raise ValueError("Independent prompt token replay differs")
            readout = row["schema"]["readout"]
            canonical = worlds.canonical(row["label"], readout)
            canonical_tokens[row["id"]] = len(tok.tok.encode(canonical, add_special_tokens=False))
            canonical_maxima[readout] = max(canonical_maxima[readout], canonical_tokens[row["id"]])
            check()
        if tok.tok.eos_token_id != receipt["eos_token_id"]:
            raise ValueError("Tokenizer EOS binding changed")
        if canonical_maxima != receipt["max_answer_tokens_by_readout"]:
            raise ValueError("Canonical answer budgets differ from pinned verification")

        manifest_sha = sha(MANIFEST)
        data, states = {}, {}
        for backend, target in runtime.TARGETS.items():
            data[backend], _, states[backend] = collect(
                ARTIFACT / backend, jobs, bindings, tok, backend=backend,
                manifest_sha=manifest_sha, expected_target=target, maximum_position=MAXIMUM_POSITION,
            )
            check()
        result = summarize(slots, data, canonical_tokens)
        if any(name.split(".")[0] == "torch" for name in sys.modules):
            raise RuntimeError("Budget replay imported Torch")
        check()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_new(args.output, {
            "experiment": "NAN-BUDGET-52",
            "analysis_sha256": sha(__file__),
            "stream41_manifest_sha256": manifest_sha,
            "private_tokenizer_receipt_sha256": sha(args.tokenizer_receipt),
            "evidence": "development_pilot",
            "formal_comparisons": 0,
            "sealed_test_accessed": False,
            "source_roots": {"CAL": 3, "DEV": 9},
            "backend_states": states,
            "verified_slots": {backend: len(verified) for backend, verified in data.items()},
            "stratum_keys": {
                "depth": ["split", "query_type", "event_count", "arm", "readout"],
                "answer_and_packet": ["split", "query_type", "event_count", "arm", "passes"],
            },
            "sampled_resources": {
                "max_private_bytes": max(item["private_bytes"] for item in samples),
                "max_working_set_bytes": max(item["working_set_bytes"] for item in samples),
                "min_available_RAM_bytes": min(item["available_physical_bytes"] for item in samples),
            },
            "outcomes": result,
        })


if __name__ == "__main__":
    main()

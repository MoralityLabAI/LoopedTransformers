"""Freeze the outcome-blind lookup roster and exercise tokenizer and runtime contracts."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys

from experiments.nan_gym05.common import REVISION, prompt_ids
from experiments.nan_phase35.run import lease
from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import TOKENIZER_PIN, id_hash, sha, tokenizer

from . import oracle, runner, source

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BUDGET = {
    "minimum_start_available_RAM_bytes": 5 * 2**30,
    "minimum_available_RAM_bytes": 4 * 2**30,
    "maximum_private_bytes": 768 * 2**20,
    "maximum_seconds": 180,
}
SOURCE_FILES = (".gitattributes", "PROTOCOL.md", "__init__.py", "source.py", "oracle.py",
                "runner.py", "verify.py")
SUPPORT = (
    "experiments/nan_gym05/common.py", "experiments/nan_phase35/run.py",
    "experiments/nan_ram28/resources.py", "experiments/nan_ranksite13/resources.py",
    "experiments/nan_readout34/depth_engine.py", "experiments/nan_readout34/runtime_audit.py",
    "experiments/nan_readout34/runtime_plan.py", "experiments/nan_readout34/source.py",
    "experiments/nan_readout34/tokenizer_check.py", "mechinterp/nanbeige_bpd_gym/chunked_cache.py",
)


def token_audit(rows, snapshot, check):
    if snapshot.name != REVISION:
        raise ValueError("Lookup fixture requires the pinned Nanbeige snapshot")
    tokenizer_wrapper, pins = tokenizer(snapshot)
    tok = tokenizer_wrapper.tok

    def decode(ids):
        return tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    bindings = {}
    prompts = {}
    replies = {}
    twin_groups = defaultdict(list)
    hop_groups = defaultdict(set)
    maxima = defaultdict(int)
    for row in rows:
        check()
        ids = prompt_ids(tokenizer_wrapper, snapshot, row["prompt"])
        text = source.canonical(row["path"], row["readout"])
        answer = tok.encode(text, add_special_tokens=False)
        assert 0 < len(answer) <= runner.CAP and len(ids) + runner.CAP <= 4608
        assert type(tok.eos_token_id) is int and tok.eos_token_id not in answer and decode(answer) == text
        maxima[row["readout"]] = max(maxima[row["readout"]], len(answer))
        key = row["id"]
        prompts[key] = ids
        replies[key] = answer
        bindings[key] = {"tokens": len(ids), "ids_sha256": id_hash(ids)}
        if row["arm"] != "copy":
            group = (row["root"], row["table_rows"], row["readout"], row["arm"])
            twin_groups[(*group, row["hops"])].append(bindings[key])
            hop_groups[(*group, row["world"])].add(len(ids))
    for key, pair in twin_groups.items():
        assert len(pair) == 2 and pair[0]["tokens"] == pair[1]["tokens"], key
        if key[3] == "erased":
            assert pair[0] == pair[1]
    assert all(len(lengths) == 1 for lengths in hop_groups.values()), hop_groups
    assert {name: sha(snapshot / name) for name in pins} == pins
    assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
    return {
        "tokenizer_hashes": pins,
        "binding_sha256": source.digest(bindings),
        "equal_length_twin_pairs": len(twin_groups),
        "equal_length_hop_series": len(hop_groups),
        "max_prompt_tokens": max(map(len, prompts.values())),
        "max_reply_tokens_by_readout": dict(maxima),
        "generated_token_cap": runner.CAP,
        "eos_token_id": tok.eos_token_id,
    }, bindings, prompts, replies, decode


class ScriptedEngine:
    """Expose the same token/phase contracts as the matched-depth cache engine."""
    def __init__(self, vocab):
        self.vocabulary_size = vocab
        self.script = ()

    def reset(self, depth):
        assert depth in source.DEPTHS
        self.sequence = None

    def first(self, prompt):
        assert prompt and self.sequence is None
        self.sequence = iter(self.script)
        return next(self.sequence)

    def next(self, token):
        assert type(token) is int
        return next(self.sequence)


def exercise(slots, bindings, prompts, replies, decode, eos, check):
    assert len(slots) == 960 and {slot.ordinal for slot in slots} == set(range(len(slots)))
    assert sum(slot.row["split"] == "DEV" for slot in slots) == 720
    for left, right in zip(slots[::2], slots[1::2]):
        assert (left.passes, right.passes) == source.DEPTHS and left.row is right.row
    vocab = max(eos, *(max(ids) for ids in replies.values())) + 1
    engine = ScriptedEngine(vocab)
    grades = {}
    for slot in slots:
        key = slot.row["id"]
        engine.script = replies[key]
        result = runner.generate(engine, slot, prompts[key], bindings[key],
                                 decode=decode, eos=eos, guard=check)
        grades[slot.ordinal] = runner.audit_episode(
            slot, bindings[key], result, decode=decode, eos=eos, vocab=vocab)
        assert grades[slot.ordinal]["joint_exact"]
        if slot.ordinal == 0:
            invalid = {**result, "generated_ids": result["generated_ids"] + [eos]}
            try:
                runner.audit_episode(slot, bindings[key], invalid, decode=decode, eos=eos, vocab=vocab)
            except ValueError as exc:
                assert "after its first stop" in str(exc)
            else:
                raise AssertionError("A reply with a late stop passed the audit")
    # Script one real repair, one regression and one missing closure in the otherwise perfect fixture.
    a, b = slots[:2], slots[10:12]  # A/B for the first compute/direct cell.
    assert a[0].row["world"] == "A" and b[0].row["world"] == "B"
    assert a[0].row["readout"] == b[0].row["readout"] == "direct"
    assert a[0].row["path"][-1] != b[0].row["path"][-1]
    for slot, other in ((a[0], b[0]), (b[1], a[1])):
        key = slot.row["id"]
        engine.script = replies[other.row["id"]]
        result = runner.generate(engine, slot, prompts[key], bindings[key],
                                 decode=decode, eos=eos, guard=check)
        grades[slot.ordinal] = runner.audit_episode(
            slot, bindings[key], result, decode=decode, eos=eos, vocab=vocab)
        assert result["stop"] == "complete" and not grades[slot.ordinal]["joint_exact"]
    grades.pop(slots[-1].ordinal)
    reports = runner.summarize(slots, grades)
    assert len(reports) == 60
    metric = next(row["metrics"]["joint_exact"] for row in reports if
                  (row["split"], row["table_rows"], row["hops"], row["readout"], row["arm"])
                  == ("CAL", 24, 1, "direct", "compute"))
    assert metric["world_pairs"] == {"registered_pairs": 4, "repair": 1, "regression": 1,
                                     "both_correct": 2, "both_incorrect": 0, "unknown": 0}
    assert metric["both_world_root_pairs"]["both_incorrect"] == 1
    assert sum(row["metrics"]["joint_exact"]["both_world_root_pairs"]["unknown"]
               for row in reports) == 1
    trace = next(slot for slot in slots if slot.row["readout"] == "trace")
    correct = source.canonical(trace.row["path"], "trace")
    wrong_trace = "PATH=99;" + correct.split(";", 1)[1]
    payload = oracle.grade(trace.row, wrong_trace)
    assert payload == {"format_valid": True, "final_exact": True, "trace_exact": False,
                       "joint_exact": False}
    return {"synthetic_slots_passed": len(slots), "wrong_complete_replies_replayed": 2,
            "late_stops_rejected": 1, "unknown_root_pairs_retained": 1, "aggregate_cells": len(reports)}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--snapshot", required=True, type=Path)
    cli.add_argument("--receipt", required=True, type=Path, help="Write an exclusive aggregate receipt")
    args = cli.parse_args()
    if not __debug__:
        cli.error("enable Python assertions")
    with lease(BUDGET) as (check, samples):
        rows = source.build()
        roster = source.audit(rows, check)
        tokens, bindings, prompts, replies, decode = token_audit(rows, args.snapshot, check)
        fixture = exercise(runner.schedule(rows), bindings, prompts, replies, decode,
                           tokens["eos_token_id"], check)
    support = {name: sha(REPO / name) for name in SUPPORT}
    support[TOKENIZER_PIN.relative_to(REPO).as_posix()] = sha(TOKENIZER_PIN)
    result = {
        "experiment": "NAN-SERIAL-62",
        "status": "source_and_cpu_fixture_verified",
        "revision": REVISION,
        "source_sha256": {name: sha(HERE / name) for name in SOURCE_FILES},
        "support_sha256": support,
        "roster": roster, "tokenizer": tokens, "fixture": fixture,
        "resources": {
            **BUDGET,
            "maximum_sampled_private_bytes": max(s["private_bytes"] for s in samples),
            "minimum_sampled_available_ram_bytes": min(s["available_physical_bytes"] for s in samples),
        },
        "model_calls": 0, "inference_allowed": False, "formal_inferential_tests": 0,
    }
    write_new(args.receipt, result)
    print(result["status"], result["roster"]["prompts"], sha(args.receipt))


if __name__ == "__main__":
    main()

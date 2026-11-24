"""Freeze source bytes and commission the relabeled B2 schedule and host scoring."""
import argparse
import json
from pathlib import Path
import sys

from experiments.nan_gated46.source import inputs
from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import digest as sha, write_new
from experiments.nan_readout34.tokenizer_check import TOKENIZER_PIN, tokenizer
from experiments.nan_route58.custody import bound_source

from . import audit, plan, source

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MANIFEST = HERE / "SOURCE_MANIFEST.json"
REVISION = "451ed48c3273ecef7ea8faaa43c31ce529763bb1"
BUDGET = {"minimum_start_available_RAM_bytes": 5 * 2**30, "minimum_available_RAM_bytes": 4 * 2**30,
          "maximum_private_bytes": 768 * 2**20, "maximum_seconds": 160}


def seal():
    files = [HERE / name for name in (".gitattributes", "__init__.py", "PROTOCOL.md",
                                     "source.py", "audit.py", "plan.py", "verify.py")]
    files += [TOKENIZER_PIN] + [REPO / name for name in (
        "experiments/nan_comp14/sources.py", "experiments/nan_comp14/fixtures/public.jsonl",
        "experiments/nan_comp14/parser.py", "experiments/nan_phase35/run.py",
        "experiments/nan_gated46/source.py", "experiments/nan_readout34/source.py",
        "experiments/nan_gym05/common.py", "experiments/nan_readout34/runtime_audit.py",
        "experiments/nan_readout34/tokenizer_check.py", "experiments/nan_route58/SOURCE_CHECK.json",
        "reports/nanbeige_bpd_gym/nan_attnspan67_label_holdout_results.json")]
    return {"experiment": "NAN-RELABEL-68", "checkpoint_revision": REVISION,
            "frozen_source_and_host_only": True, "resources": BUDGET,
            "files_sha256": {p.relative_to(REPO).as_posix(): sha(p) for p in files}}


def earlier_prompts():
    previous = [json.loads(line) for line in
                (REPO / "experiments/nan_comp14/fixtures/public.jsonl").read_bytes().splitlines()]
    previous.extend(bound_source(REPO / "runs/nan_route58/source", "public"))
    return {x["prompt"] for x in previous if x["arm"] == "focused"}


def verify_tokens(snapshot, public, gold, check):
    tok, pins = tokenizer(snapshot)
    bindings = inputs(tok, public, snapshot, check)
    counts = {r["id"]: bindings["row_bindings"][r["id"]]["tokens"] for r in public}
    lengths_by_shape = {}
    for r in public:
        shape = r["arm"]
        lengths_by_shape.setdefault(shape, set()).add(counts[r["id"]])
    for row in gold:
        token = bindings["bare_label_token_ids"][row["answer"]]
        assert tok.tok.decode([token], skip_special_tokens=False, clean_up_tokenization_spaces=False) == row["answer"]
    sizes = {name: sorted(values) for name, values in lengths_by_shape.items()}
    return bindings, pins, sizes


def test_score(public, gold):
    pairs = plan.relabel_maps(public)
    replies = {(r["id"], p): r["answer"] for r in gold for p in plan.DEPTHS}
    focus = [r for r in gold if r["split"] == "development" and r["hops"] == 1 and r["arm"] == "focused"]
    roots = sorted({r["root"] for r in focus})
    subset = {(r["root"], r["encoding"], r["world"]): r for r in focus}
    root0, root1, root2, root3 = roots[:4]
    row0, row1, row2 = (subset[r, "original", "A"] for r in (root0, root1, root2))
    replies[row0["id"], 1] = "Invalid narrative"
    replies[row1["id"], 2] = "Invalid narrative"
    del replies[row2["id"], 2]
    # Consistent wrong answer under a full relabeling counts as equivariant and
    # incorrect. This distinguishes monitoring invariance from solving the task.
    sem_a, sem_b = (subset[root3, enc, "A"] for enc in source.ENCODINGS)
    wrong = next(c for c in source.NODES if c != sem_a["answer"])
    for phase in plan.DEPTHS:
        replies[sem_a["id"], phase] = wrong
        replies[sem_b["id"], phase] = pairs[root3][wrong]
    scores = plan.score(public, gold, replies)
    assert len(scores) == len(source.SPLITS)*len(source.HOPS)*len(source.ARMS)
    cell, = [r for r in scores if (r["split"], r["hops"], r["arm"]) == ("development", 1, "focused")]
    expected = {"registered_pairs": 16, "repair": 1, "regression": 1,
                "both_correct": 12, "both_incorrect": 1, "unknown": 1}
    assert cell["phase1_to_phase2"]["all_worlds_both_encodings_correct"] == expected
    assert cell["per_phase"]["1"]["semantic_prediction_equivariant"] == {
        "registered": 32, "correct": 31, "incorrect": 1, "unknown": 0}
    assert cell["per_phase"]["2"]["semantic_prediction_equivariant"] == {
        "registered": 32, "correct": 30, "incorrect": 1, "unknown": 1}
    return {"fixture_cells": len(scores), "root_repair": 1, "root_regression": 1,
            "unknown_root_retained": 1, "wrong_equivariant_pair_retained": 1}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--snapshot", type=Path)
    cli.add_argument("--out", type=Path)
    args = cli.parse_args()
    if not __debug__:
        cli.error("Enable assertions for source verification")
    if args.freeze:
        write_new(MANIFEST, seal())
        return
    if not args.snapshot or not args.out or json.loads(MANIFEST.read_bytes()) != seal():
        cli.error("Require the frozen manifest, snapshot and output")
    assert args.snapshot.name == REVISION
    with lease(BUDGET) as (check, samples):
        public, gold = source.build()
        earlier = earlier_prompts()
        structure = audit.check(public, gold, earlier, check)
        tokens, pins, lengths = verify_tokens(args.snapshot.resolve(), public, gold, check)
        jobs = plan.pair_roster(public, tokens["row_bindings"])
        assert len(jobs) == structure["paired_world_batches_per_backend"] == 288
        assert [item["ordinal"] for item in jobs] == list(range(288))
        assert all(len(shape) == 1 for shape in lengths.values())
        scores = test_score(public, gold)
        check()
        assert not any(k == "torch" or k.startswith("torch.") for k in sys.modules)
    args.out.mkdir(parents=True, exist_ok=False)
    payloads = {"public": public, "gold": gold, "token_bindings": tokens, "b2_roster": jobs}
    for name, data in payloads.items():
        write_new(args.out / (name + ".json"), data)
    result = {"experiment": "NAN-RELABEL-68", "status": "source_tokenizer_and_host_verified",
              "structure": structure, "fixture": scores, "exact_prompt_counts_by_arm": lengths,
              "named_prior_prompts_checked": len(earlier), "tokenizer_pins": pins, "model_calls": 0,
              "sealed_test_accessed": False, "formal_comparisons": 0, "gpu_allowed": False,
              "manifest_sha256": sha(MANIFEST),
              "payload_sha256": {name: source.digest(value) for name,value in payloads.items()},
              "max_sampled_private_bytes": max(x["private_bytes"] for x in samples),
              "min_sampled_available_RAM_bytes": min(x["available_physical_bytes"] for x in samples)}
    write_new(args.out / "verification.json", result)
    print(result["status"], structure["prompts"], lengths, flush=True)


if __name__ == "__main__":
    main()

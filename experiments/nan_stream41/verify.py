"""Validate the real CPU cache path and synthetic whole-batch closure."""
import argparse
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

from experiments.nan_phase35.run import lease
from experiments.nan_readout34 import source, worlds
from experiments.nan_readout34.runtime_plan import context_bindings, schedule, write_new
from experiments.nan_readout34.tokenizer_check import id_hash, sha
from experiments.nan_stream36.stream import Job

from .pipeline import collect, commit, execute, fixed_jobs, verify_batch
from .run import benchmark, MANIFEST, STREAM36, verify


def rejection(operation, message):
    try:
        operation()
    except ValueError as error:
        if message not in str(error):
            raise
    else:
        raise AssertionError(f"Expected refusal: {message}")


def journal_fixture(work, slots):
    selected = tuple(slot for slot in slots if slot.passes == 2 and slot.source["schema"]["readout"] == "final")[:2]
    job = Job(selected, 99)
    inputs = [[1, 2, 3], [3, 4]]
    prompts = {slot.source["id"]: ids for slot, ids in zip(selected, inputs)}
    bindings = {key: {"tokens": len(ids), "ids_sha256": id_hash(ids)} for key, ids in prompts.items()}
    correct = [worlds.canonical(slot.source["label"], "final") for slot in selected]
    events = []

    def decode(tokens, **_):
        return {(5,): correct[0], (6,): "unfinished", (6, 7): correct[1]}[tuple(tokens)]

    class Fake:
        vocabulary_size = 20

        def reset(self, depth):
            events.append(("reset", depth))

        def prefill(self, values):
            assert values == inputs
            return [5, 6]

        def advance(self, survivors, tokens):
            assert survivors == [1] and tokens == [6]
            return [7]

    packet, = tuple(execute(Fake(), [job], prompts, bindings, decode=decode,
                            eos=0, guard=lambda _t, b, p: events.append((b, p)), synchronize=lambda: None))
    assert events == [("reset", 2), (2, 3), (1, 4), (1, 4), ("reset", 2)]
    assert packet["batch_rows_per_step"] == [2, 1]
    path = work / "fixture"
    path.mkdir()
    commit(path, packet, 0)
    preflight = path / "preflight.json"
    write_new(preflight, {"depths": {d: {"tested_max_position": 99} for d in ("1", "2")}})
    terminal = {"status": "completed", "failure": None, "backend": "nf4", "runtime_manifest_sha256": "lock",
                "target_before": "target", "target_after": "target", "quant_before": "q", "quant_after": "q",
                "registered_batches": 1, "closed_batches": 1, "closed_slots": 2, "checkpoint_verified_after": True,
                "preflight_passed": True, "preflight_sha256": sha(preflight),
                "closed_journal_sha256": sha(path / "closed.jsonl"), "vocabulary_size": 20}
    write_new(path / "terminal.json", terminal)
    tok = SimpleNamespace(tok=SimpleNamespace(eos_token_id=0, decode=decode))
    collect_fixture = lambda location: collect(location, [job], bindings, tok, backend="nf4", manifest_sha="lock",
                                                expected_target="target", maximum_position=99)
    grades, stats, state = collect_fixture(path)
    assert state == "completed" and [grade["joint_exact"] for grade, _ in grades.values()] == [1, 1]
    assert stats["2"]["full_head_rows"] == 3 and stats["2"]["physical_phase_token_visits"] == 14
    assert stats["2"]["wall_seconds"] == packet["batch_wall_seconds"]
    altered = copy.deepcopy(packet)
    altered["batch_rows_per_step"] = [2, 2]
    rejection(lambda: verify_batch(altered, 0, job, bindings, decode=decode, eos=0, vocab=20), "row occupancy")
    altered = copy.deepcopy(packet)
    altered["episodes"].reverse()
    rejection(lambda: verify_batch(altered, 0, job, bindings, decode=decode, eos=0, vocab=20), "frozen slot")
    rejection(lambda: commit(path, packet, 1), "out of order")
    closed_file = path / "batch_0000.json"
    closed_file.write_bytes(closed_file.read_bytes() + b" ")
    rejection(lambda: collect_fixture(path), "chronological custody")
    refusal = work / "refused"
    refusal.mkdir()
    write_new(refusal / "terminal.json", terminal | {"status": "partial", "failure": {"type": "FakeDevice"},
              "closed_batches": 0, "closed_slots": 0, "preflight_passed": False, "preflight_sha256": None,
              "closed_journal_sha256": None})
    assert collect_fixture(refusal) == ({}, {}, "device_preflight_failed")
    assert collect_fixture(work / "never_started") == ({}, {}, "no_integrity_terminal")
    return {"independently_graded_responses": 2, "phase_token_visits": 14, "head_rows": [2, 1],
            "changed_packets_rejected": 4, "refused_device_verified_responses": 0}


def cpu_cache(check):
    import torch
    from experiments.nan_stream36.batch_engine import BatchedDepthEngine
    from experiments.nan_stream36.verify_device_preflight import verify as prior_verify
    from mechinterp.nanbeige_bpd_gym.streamed_vocab_head import StreamedVocabularyHead
    from mechinterp.nanbeige_bpd_gym.tests.test_chunked_cache import Loop

    torch.manual_seed(41)
    preflight, guards = prior_verify(check)
    loop = Loop()
    head = StreamedVocabularyHead(torch.randn(64, 4, dtype=torch.float64), loop.device, 7)
    engine = BatchedDepthEngine(loop, head, chunk=3)

    def budget(_began, batch, position):
        assert 1 <= batch <= 2 and position <= 8
        check()

    try:
        timings = benchmark(engine, [[1, 3, 7, 9], [2, 5, 10, 4]], budget, lambda: None)
    finally:
        engine.close()
        head.close()
    assert all(len(item["comparisons"]) == 10 and item["solo_total_over_pair_wall"] > 0
               for item in timings.values())
    assert not torch.cuda.is_initialized()
    return {"preflight_depths": sorted(preflight), "preflight_guard_calls": guards,
            "fixed_token_benchmark_comparisons": {d: len(item["comparisons"]) for d, item in timings.items()},
            "cuda_initialized": False, "pretrained_forwards": 0}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--tokenizer-receipt", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    verify()
    cfg = {"minimum_start_available_RAM_bytes": 5 * 2**30, "minimum_available_RAM_bytes": 4 * 2**30,
           "maximum_private_bytes": 768 * 2**20, "maximum_seconds": 120}
    with lease(cfg) as (check, samples):
        receipt = context_bindings(args.tokenizer_receipt)
        slots = schedule(source.build())
        jobs = fixed_jobs(slots, receipt["contexts"],
                          json.loads((STREAM36 / "SCHEDULING_VERIFICATION.json").read_bytes()))
        args.output.mkdir(parents=True, exist_ok=False)
        journal = journal_fixture(args.output, slots)
        os.environ.update(USE_TORCH="1", USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1")
        device = cpu_cache(check)
    write_new(args.output / "verification.json", {
        "status": "passed", "runtime_manifest_sha256": sha(MANIFEST),
        "slots": len(slots), "B2_batches": len(jobs), "journal_fixture": journal, "tiny_cpu_fixture": device,
        "max_sampled_private_bytes": max(item["private_bytes"] for item in samples),
        "max_sampled_working_set_bytes": max(item["working_set_bytes"] for item in samples),
        "min_sampled_available_RAM_bytes": min(item["available_physical_bytes"] for item in samples),
    })


if __name__ == "__main__":
    main()

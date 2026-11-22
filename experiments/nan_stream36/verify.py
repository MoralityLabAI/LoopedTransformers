"""Verify the frozen prompt roster, packing invariants and two asynchronous stops."""
import argparse
from collections import Counter
from pathlib import Path

from experiments.nan_ranksite13.resources import below_normal
from experiments.nan_readout34 import source, worlds
from experiments.nan_readout34.runtime_audit import audit_episode
from experiments.nan_readout34.runtime_plan import context_bindings, schedule, write_new
from experiments.nan_readout34.tokenizer_check import id_hash, sha

from .stream import Job, generate, pack


def fixture(slots):
    capable = [slot for slot in slots if slot.passes == 2 and slot.source["schema"]["readout"] == "final"]
    pair = tuple(capable[:2])
    inputs = [[1, 2, 3], [3, 4]]
    bindings = {slot.source["id"]: {"tokens": len(ids), "ids_sha256": id_hash(ids)}
                for slot, ids in zip(pair, inputs)}
    work = Job(pair, 99)
    answers = [worlds.canonical(slot.source["label"], "final") for slot in pair]
    events = []

    class Tiny:
        vocabulary_size = 20

        def reset(self, depth):
            events.append(("reset", depth))

        def prefill(self, prompts):
            assert prompts == inputs
            events.append(("prefill", 2))
            return [5, 6]

        def advance(self, survivors, tokens):
            assert survivors == [1] and tokens == [6]
            events.append(("advance", 1))
            return [7]

    def decode(tokens):
        return {(5,): answers[0], (6,): "unfinished", (6, 7): answers[1]}[tuple(tokens)]

    guards = []
    result = generate(Tiny(), work, inputs, bindings, decode=decode, eos_token=0,
                      guard=lambda _started, batch, position: guards.append((batch, position)))
    assert result["batch_rows_per_step"] == [2, 1] and result["padded_prefill_tokens"] == 1
    assert guards == [(2, 3), (1, 4), (1, 4)]
    assert events == [("reset", 2), ("prefill", 2), ("advance", 1), ("reset", 2)]
    for slot, episode in zip(pair, result["episodes"]):
        checked = audit_episode(episode, slot, bindings[slot.source["id"]], decode, 0, 20)
        assert checked["joint_exact"] == 1
    return {"simulated_cache_rows": [2, 1], "max_physical_canvas_in_fixture": 4,
            "separately_verified_fixture_replies": 2}


def main():
    if not __debug__:
        raise RuntimeError("Validate the stream plan with assertions enabled")
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--receipt", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    below_normal()
    source_file = Path(__file__).resolve().parent
    receipt = context_bindings(args.receipt)
    slots = schedule(source.build())
    jobs = pack(slots, receipt["contexts"])
    assert len(slots) == 672 and sum(len(job.slots) for job in jobs) == len(slots)
    sizes = Counter(len(job.slots) for job in jobs)
    pads = sum(sum(job.admitted_max_position - 96 - receipt["contexts"][slot.source["id"]]["tokens"]
                   for slot in job.slots) for job in jobs)
    counts = Counter((job.slots[0].passes, len(job.slots)) for job in jobs)
    proof = {
        "scope": "READOUT34 streaming software plan, GPU and real B2 preflight pending",
        "source_manifest_sha256": receipt["source_manifest_sha256"],
        "readout_runtime_manifest_sha256": sha(source_file.parent / "nan_readout34/RUNTIME_MANIFEST.json"),
        "fixed_slots_per_precision": len(slots),
        "scheduled_prefills_per_precision": len(jobs),
        "scheduled_B1": sizes[1],
        "scheduled_B2": sizes[2],
        "slots_at_B2_per_precision": 2 * sizes[2],
        "maximum_padded_position_plus_full_reply_cap": max(job.admitted_max_position for job in jobs),
        "total_right_padding_tokens_in_all_prefill_canvases": pads,
        "jobs_by_depth_and_batch_size": [{"passes": d, "batch_size": b, "jobs": n}
                                         for (d, b), n in sorted(counts.items())],
        "schedule_sha256": source.digest([[job.admitted_max_position, [slot.ordinal for slot in job.slots]] for job in jobs]),
        "implementation_sha256": {name: sha(source_file / name) for name in ("stream.py", "verify.py")},
        "fixture": fixture(slots),
        "new_model_calls": 0,
    }
    write_new(args.output, proof)
    print(proof, flush=True)


if __name__ == "__main__":
    main()

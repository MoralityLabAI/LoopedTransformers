"""Batch commit and independent verification for the frozen READOUT34 roster."""
from collections import Counter
import json
import math
from pathlib import Path
from time import monotonic

from experiments.nan_readout34 import source
from experiments.nan_readout34.runtime_audit import audit_episode, decode_with
from experiments.nan_readout34.runtime_plan import CAP, write_new
from experiments.nan_readout34.tokenizer_check import sha
from experiments.nan_stream36.stream import generate, pack


def fixed_jobs(slots, bindings, reference):
    jobs = pack(slots, bindings)
    roster = [[job.admitted_max_position, [slot.ordinal for slot in job.slots]] for job in jobs]
    if (source.digest(roster) != reference["schedule_sha256"] or len(jobs) != 336
            or Counter(job.slots[0].passes for job in jobs) != {1: 120, 2: 216}
            or any(len(job.slots) != 2 for job in jobs)
            or max(job.admitted_max_position for job in jobs) != 3463
            or any(len({bindings[slot.source["id"]]["tokens"] for slot in job.slots}) != 1
                   for job in jobs)):
        raise ValueError("B2 workload differs from the verified STREAM36 roster")
    return jobs


def append_new(path, value):
    with path.open("a", encoding="utf-8", newline="\n") as journal:
        journal.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def filename(index):
    return f"batch_{index:04d}.json"


def execute(engine, jobs, prompts, bindings, *, decode, eos, guard, synchronize):
    for i, job in enumerate(jobs):
        inputs = [prompts[slot.source["id"]] for slot in job.slots]
        synchronize()
        begin = monotonic()
        result = generate(engine, job, inputs, bindings, decode=decode, eos_token=eos, guard=guard)
        synchronize()
        yield {"index": i, "passes": job.slots[0].passes, "batch_wall_seconds": monotonic() - begin, **result}


def commit(work, record, count):
    if record["index"] != count or len(record["episodes"]) != 2:
        raise ValueError("Batch closure is out of order or missing a physical row")
    path = work / filename(count)
    write_new(path, record)
    append_new(work / "closed.jsonl", {"file": path.name, "sha256": sha(path)})


def verify_batch(record, index, job, bindings, *, decode, eos, vocab):
    canvas = job.admitted_max_position - CAP
    episodes = record["episodes"]
    elapsed = record["batch_wall_seconds"]
    lengths = [bindings[slot.source["id"]]["tokens"] for slot in job.slots]
    if ((record["index"], record["passes"], record["initial_canvas"], record["padded_prefill_tokens"])
            != (index, job.slots[0].passes, canvas, len(lengths) * canvas - sum(lengths))
            or len(episodes) != len(job.slots) or isinstance(elapsed, bool)
            or not math.isfinite(elapsed) or elapsed < 0):
        raise ValueError("Batch timing, shape or slot membership differs from the frozen plan")
    audited = {}
    for slot, episode in zip(job.slots, episodes):
        if episode["wall_seconds"] > elapsed:
            raise ValueError("A row stopped after its batch closed")
        score = audit_episode(episode, slot, bindings[slot.source["id"]], decode, eos, vocab)
        audited[slot.ordinal] = (score, episode)
    emitted = [len(episode["generated_ids"]) for episode in episodes]
    rowsteps = [sum(step < length for length in emitted) for step in range(max(emitted))]
    if record["batch_rows_per_step"] != rowsteps:
        raise ValueError("Physical row occupancy disagrees with independently verified stops")
    stats = {"wall_seconds": elapsed, "batches": 1, "responses": len(episodes),
             "full_head_rows": sum(rowsteps), "B2_head_calls": rowsteps.count(2),
             "head_calls": len(rowsteps), "generated_tokens": sum(emitted),
             "physical_phase_token_visits": job.slots[0].passes * (len(lengths) * canvas + sum(rowsteps[1:]))}
    return audited, stats


def collect(work, jobs, bindings, tok, *, backend, manifest_sha, expected_target, maximum_position):
    terminal_path = work / "terminal.json"
    if not terminal_path.is_file():
        return {}, {}, "no_integrity_terminal"
    terminal = json.loads(terminal_path.read_bytes())
    if ((terminal["backend"], terminal["runtime_manifest_sha256"]) != (backend, manifest_sha)
            or terminal["registered_batches"] != len(jobs)
            or not 0 <= terminal["closed_batches"] <= len(jobs)
            or terminal["status"] not in {"completed", "partial"}
            or (terminal["status"] == "completed") != (terminal["failure"] is None)
            or (terminal["status"] == "completed" and terminal["closed_batches"] != len(jobs))
            or terminal["closed_slots"] != 2 * terminal["closed_batches"]
            or terminal["target_before"] != terminal["target_after"]
            or terminal["target_after"] != expected_target
            or terminal["quant_before"] != terminal["quant_after"]
            or terminal["checkpoint_verified_after"] is not True):
        raise ValueError("GPU terminal does not attest to the frozen execution contract")
    if terminal["preflight_passed"] is not True:
        if terminal["closed_batches"] or terminal["preflight_sha256"] is not None:
            raise ValueError("Scored batches preceded CUDA admission")
        return {}, {}, "device_preflight_failed"
    proof_path = work / "preflight.json"
    if sha(proof_path) != terminal["preflight_sha256"]:
        raise ValueError("CUDA preflight bytes changed after closure")
    proof = json.loads(proof_path.read_bytes())
    if (set(proof["depths"]) != {"1", "2"} or any(
            block["tested_max_position"] != maximum_position for block in proof["depths"].values())):
        raise ValueError("CUDA gate did not cover both scheduled maximum positions")

    journal = work / "closed.jsonl"
    if terminal["closed_batches"] == 0:
        if journal.exists() or terminal["closed_journal_sha256"] is not None:
            raise ValueError("Unexpected journal for an empty attempt")
        return {}, {}, terminal["status"]
    if sha(journal) != terminal["closed_journal_sha256"]:
        raise ValueError("Batch journal hash changed after terminal closure")
    lines = journal.read_bytes().splitlines(keepends=True)
    if len(lines) != terminal["closed_batches"]:
        raise ValueError("Batch journal count disagrees with the terminal")
    data, performance = {}, {}
    decode = decode_with(tok)
    for index, line in enumerate(lines):
        if not line.endswith(b"\n"):
            raise ValueError("Incomplete batch journal row")
        published = json.loads(line)
        path = work / filename(index)
        if published != {"file": path.name, "sha256": sha(path)}:
            raise ValueError("Closed batch bytes or chronological custody changed")
        audited, stats = verify_batch(json.loads(path.read_bytes()), index, jobs[index], bindings, decode=decode,
                                      eos=tok.tok.eos_token_id, vocab=terminal["vocabulary_size"])
        if data.keys() & audited.keys():
            raise ValueError("Repeated scored slot")
        data.update(audited)
        depth = str(jobs[index].slots[0].passes)
        totals = performance.setdefault(depth, Counter())
        totals.update(stats)
    for totals in performance.values():
        seconds = totals["wall_seconds"]
        totals["responses_per_batch_wall_second"] = totals["responses"] / seconds if seconds else None
        totals["generated_tokens_per_batch_wall_second"] = totals["generated_tokens"] / seconds if seconds else None
    return data, performance, terminal["status"]

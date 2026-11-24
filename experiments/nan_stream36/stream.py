"""Shape-matched single-checkpoint READOUT34 packing and genuine B2 stop logic."""
from collections import defaultdict
from dataclasses import dataclass
import time

from experiments.nan_readout34.runtime_plan import CAP, stop_reason
from experiments.nan_readout34.tokenizer_check import id_hash


CAPACITY = 4608


@dataclass(frozen=True)
class Job:
    slots: tuple
    admitted_max_position: int


def pack(slots, bindings):
    """Pair like workload shapes using only prospectively bound prompt lengths."""
    groups = defaultdict(list)
    for slot in slots:
        row = slot.source
        shape = (slot.passes, row["split"], row["event_count"], row["arm"], row["schema"]["readout"])
        groups[shape].append(slot)
    jobs = []
    for group in groups.values():
        ordered = sorted(group, key=lambda slot: (bindings[slot.source["id"]]["tokens"], slot.ordinal))
        for index in range(0, len(ordered), 2):
            batch = tuple(sorted(ordered[index:index + 2], key=lambda slot: slot.ordinal))
            capacity = max(bindings[slot.source["id"]]["tokens"] for slot in batch) + CAP
            if capacity > CAPACITY:
                raise ValueError("The group exceeds the registered shape ceiling")
            jobs.append(Job(batch, capacity))
    jobs.sort(key=lambda job: job.slots[0].ordinal)
    if sorted(slot.ordinal for job in jobs for slot in job.slots) != list(range(len(slots))):
        raise ValueError("The complete fixed schedule did not close exactly once")
    return jobs


def generate(engine, job, inputs, bindings, *, decode, eos_token, guard):
    """Keep one live canvas. Prune closed rows without compacting the time axis."""
    slots = job.slots
    if not 1 <= len(slots) == len(inputs) <= 2 or len({slot.passes for slot in slots}) != 1:
        raise ValueError("A genuine B1/B2 capture requires a shared physical depth")
    depth = slots[0].passes
    for slot, ids in zip(slots, inputs):
        binding = bindings[slot.source["id"]]
        if len(ids) != binding["tokens"] or id_hash(ids) != binding["ids_sha256"]:
            raise ValueError("Prompt differs from its receipt")
    canvas = max(map(len, inputs))
    if canvas + CAP != job.admitted_max_position or job.admitted_max_position > CAPACITY:
        raise ValueError("Unexpected padded B2 canvas")
    engine.reset(depth)
    start = time.monotonic()
    histories = [[] for _ in slots]
    active = list(range(len(slots)))
    previous = []
    completed = {}
    batch_rows_per_step = []
    try:
        for step in range(CAP):
            guard(start, len(active), canvas + step)
            if step == 0:
                tokens = engine.prefill(inputs)
            else:
                keep = [previous.index(index) for index in active]
                tokens = engine.advance(keep, [histories[index][-1] for index in active])
            if len(tokens) != len(active):
                raise ValueError("Model returned the wrong number of physical rows")
            batch_rows_per_step.append(len(active))
            previous = active
            active = []
            for index, token in zip(previous, tokens):
                if type(token) is not int or not 0 <= token < engine.vocabulary_size:
                    raise ValueError("Model returned an invalid vocabulary token")
                slot = slots[index]
                ids = histories[index]
                ids.append(token)
                eos = token == eos_token
                output = decode(ids[:-1] if eos else ids)
                reason = stop_reason(output, slot.source["schema"], eos, step + 1)
                if reason is None:
                    active.append(index)
                    continue
                prompt = inputs[index]
                completed[index] = {
                    "ordinal": slot.ordinal,
                    "source_id": slot.source["id"],
                    "passes": depth,
                    "prompt_tokens": len(prompt),
                    "prompt_ids_sha256": bindings[slot.source["id"]]["ids_sha256"],
                    "generated_ids": ids,
                    "text": output,
                    "stop_reason": reason,
                    "phase_token_visits": depth * (len(prompt) + len(ids) - 1),
                    "wall_seconds": time.monotonic() - start,
                }
            if not active:
                break
        if len(completed) != len(slots):
            raise RuntimeError("A scheduled B2 response failed to close")
        guard(start, len(previous), canvas + len(batch_rows_per_step) - 1)
        return {
            "initial_canvas": canvas,
            "padded_prefill_tokens": len(slots) * canvas - sum(map(len, inputs)),
            "batch_rows_per_step": batch_rows_per_step,
            "episodes": [completed[i] for i in range(len(slots))],
        }
    finally:
        engine.reset(depth)

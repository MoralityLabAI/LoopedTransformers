"""Run matched-depth lookup slots and replay their emitted answers."""
from collections import defaultdict
from dataclasses import dataclass
from itertools import product

from experiments.nan_readout34.runtime_audit import paired
from experiments.nan_readout34.tokenizer_check import id_hash

from . import oracle, source

CAP = 64
CHAR_CAP = 256


@dataclass(frozen=True)
class Slot:
    ordinal: int
    passes: int
    row: dict


def schedule(rows):
    return tuple(Slot(i, depth, row)
                 for i, (row, depth) in enumerate(product(rows, source.DEPTHS)))


def stop_reason(text, row, eos, count):
    if eos:
        return "eos"
    if oracle.parse(text, row["readout"], row["hops"]) is not None:
        return "complete"
    if len(text) > CHAR_CAP:
        return "character_cap"
    if count == CAP:
        return "token_cap"
    return None


def generate(engine, slot, prompt, binding, *, decode, eos, guard):
    """Use the READOUT34 DepthEngine interface with independent lookup stops."""
    if binding != {"tokens": len(prompt), "ids_sha256": id_hash(prompt)}:
        raise ValueError("Lookup prompt does not match its fixed token binding")
    row = slot.row
    engine.reset(slot.passes)
    reply = []
    try:
        for count in range(1, CAP + 1):
            guard()
            token = engine.first(prompt) if count == 1 else engine.next(reply[-1])
            if type(token) is not int or not 0 <= token < engine.vocabulary_size:
                raise ValueError("Invalid full-vocabulary token")
            reply.append(token)
            text = decode(reply[:-1] if token == eos else reply)
            reason = stop_reason(text, row, token == eos, count)
            if reason:
                guard()
                return {
                    "source_id": row["id"], "ordinal": slot.ordinal, "passes": slot.passes,
                    "prompt": binding, "generated_ids": reply, "text": text, "stop": reason,
                    "phase_token_visits": slot.passes * (len(prompt) + count - 1),
                }
    finally:
        engine.reset(slot.passes)
    raise RuntimeError("Finite generation failed to close")


def audit_episode(slot, binding, episode, *, decode, eos, vocab):
    if (episode["source_id"], episode["ordinal"], episode["passes"], episode["prompt"]) != (
            slot.row["id"], slot.ordinal, slot.passes, binding):
        raise ValueError("Lookup episode does not belong to this slot")
    tokens = episode["generated_ids"]
    if not 1 <= len(tokens) <= CAP or any(type(t) is not int or not 0 <= t < vocab for t in tokens):
        raise ValueError("Invalid lookup reply token sequence")
    for count, token in enumerate(tokens, 1):
        text = decode(tokens[:count - int(token == eos)])
        reason = stop_reason(text, slot.row, token == eos, count)
        if reason and count < len(tokens):
            raise ValueError("Lookup episode continued after its first stop")
    if (reason is None or (text, reason) != (episode["text"], episode["stop"])
            or episode["phase_token_visits"] != slot.passes * (binding["tokens"] + len(tokens) - 1)):
        raise ValueError("Lookup reply closure does not replay")
    return {
        **oracle.grade(slot.row, text),
        "generated_tokens": len(tokens),
        "phase_token_visits": episode["phase_token_visits"],
    }


def both_worlds(values):
    if len(values) != 2:
        raise ValueError("A lookup root cell must contain both worlds")
    if None in values:
        return None
    return all(values)


def summarize(slots, scores):
    """Accept per-slot grades only after episode and terminal custody has been verified."""
    by_cell = defaultdict(dict)
    expected = {slot.ordinal for slot in slots}
    if set(scores) - expected:
        raise ValueError("Unknown lookup slot in score receipt")
    for slot in slots:
        row = slot.row
        cell = (row["split"], row["table_rows"], row["hops"], row["readout"], row["arm"])
        by_cell[cell][row["root"], row["world"], slot.passes] = scores.get(slot.ordinal)
    output = []
    for cell, values in sorted(by_cell.items()):
        split, size, hops, readout, arm = cell
        roots = sorted({root for root, _, _ in values})
        metrics = {}
        measures = ("format_valid", "final_exact", "joint_exact")
        if readout == "trace":
            measures += ("trace_exact",)
        for measure in measures:
            def get(root, world, depth):
                grade = values[root, world, depth]
                return None if grade is None else grade[measure]

            metrics[measure] = {
                "world_pairs": paired([tuple(get(root, world, d) for d in source.DEPTHS)
                                       for root, world in product(roots, ("A", "B"))]),
                "both_world_root_pairs": paired([
                    tuple(both_worlds([get(root, world, d) for world in ("A", "B")])
                          for d in source.DEPTHS)
                    for root in roots
                ]),
            }
        cost = {}
        for depth in source.DEPTHS:
            known = [item for (_, _, d), item in values.items() if d == depth and item is not None]
            cost[str(depth)] = {
                "verified_replies": len(known),
                "generated_tokens_sum": sum(r["generated_tokens"] for r in known),
                "phase_token_visits_sum": sum(r["phase_token_visits"] for r in known),
            }
        output.append({"split": split, "table_rows": size, "hops": hops, "readout": readout,
                       "arm": arm, "metrics": metrics, "cost": cost})
    return output

"""Independently verify terminal provenance, reply stops and root-level counts."""
from collections import defaultdict
import json
import math

from . import oracle, source
from .runtime_plan import CAP, CHAR_CAP, schedule
from .tokenizer_check import sha


def audit_episode(result, slot, binding, decode, eos_token, vocab):
    row = slot.source
    if (result["ordinal"], result["source_id"], result["passes"],
            result["prompt_tokens"], result["prompt_ids_sha256"]) != (
                slot.ordinal, row["id"], slot.passes, binding["tokens"], binding["ids_sha256"]):
        raise ValueError("Episode does not belong to its frozen slot")
    tokens = result["generated_ids"]
    if not 1 <= len(tokens) <= CAP or any(type(t) is not int or not 0 <= t < vocab for t in tokens):
        raise ValueError("Invalid bounded token history")
    elapsed = result["wall_seconds"]
    if (result["phase_token_visits"] != slot.passes * (binding["tokens"] + len(tokens) - 1)
            or isinstance(elapsed, bool) or not math.isfinite(elapsed) or elapsed < 0):
        raise ValueError("Episode compute or time receipt changed")
    schema = row["schema"]
    for n in range(1, len(tokens) + 1):
        eos = tokens[n - 1] == eos_token
        text = decode(tokens[:n - 1] if eos else tokens[:n])
        if eos:
            reason = "eos"
        elif oracle.packet(text, schema) is not None:
            reason = "complete_packet"
        elif len(text) > CHAR_CAP:
            reason = "character_cap"
        elif n == CAP:
            reason = "token_cap"
        else:
            continue
        if n != len(tokens) or reason != result["stop_reason"] or text != result["text"]:
            raise ValueError("Response does not close at its first independently verified stop")
        return oracle.grade(text, schema, row["label"])
    raise ValueError("No valid independent terminal stop")


def bounds(values):
    return {"registered": len(values), "verified": sum(v is not None for v in values),
            "correct": values.count(1), "incorrect": values.count(0), "unknown": values.count(None),
            "correct_count_bounds": [values.count(1), values.count(1) + values.count(None)]}


def paired(values):
    result = {"registered_pairs": len(values), "repair": 0, "regression": 0,
              "both_correct": 0, "both_incorrect": 0, "unknown": 0}
    for a, b in values:
        if a is None or b is None:
            result["unknown"] += 1
        else:
            result[{(0, 1): "repair", (1, 0): "regression",
                    (1, 1): "both_correct", (0, 0): "both_incorrect"}[a, b]] += 1
    return result


def metrics(row, score):
    kind = row["schema"]["readout"]
    names = ["format_valid", "joint_exact"]
    fields = {
        "final": ["final_room"], "single_checkpoint": ["single_room"],
        "checkpoint_array": ["checkpoint_rooms"], "packet": ["checkpoint_rooms", "path", "final_room"],
    }[kind]
    if score is None:
        names += fields
        if "checkpoint_rooms" in fields:
            names += [f"checkpoint_{i}" for i in (1, 2, 3)]
        return dict.fromkeys(names)
    values = {key: score[key] for key in names} | score["fields_correct"]
    if score["checkpoint_items_correct"] is not None:
        values.update({f"checkpoint_{i}": value for i, value in enumerate(
            score["checkpoint_items_correct"], 1)})
    return values


def summarize(slots, data):
    strata = defaultdict(list)
    roots = defaultdict(list)
    ratios = defaultdict(list)
    raw_pairs = {}
    for backend, verified in data.items():
        for slot in slots:
            row = slot.source
            key = (row["split"], row["query_type"], row["event_count"], row["arm"], row["schema"]["readout"])
            cell = (backend, slot.passes, *key)
            receipt = verified.get(slot.ordinal)
            score, record = (None, None) if receipt is None else receipt
            raw_pairs[backend, row["id"], slot.passes] = metrics(row, score)
            if record:
                ratios[cell].append((len(record["generated_ids"]), record["phase_token_visits"],
                                     record["wall_seconds"], record["stop_reason"]))
            for metric, value in metrics(row, score).items():
                strata[cell, metric].append(value)
                roots[cell, metric, row["root"]].append(value)
    both_worlds = defaultdict(list)
    for (cell, metric, _), values in roots.items():
        assert len(values) == 2
        both_worlds[cell, metric].append(0 if 0 in values else None if None in values else 1)
    depth_pairs = defaultdict(list)
    precision_pairs = defaultdict(list)
    for slot in slots:
        row = slot.source
        key = (row["split"], row["query_type"], row["event_count"], row["arm"], row["schema"]["readout"])
        if slot.passes == 1:
            for backend in data:
                a = raw_pairs[backend, row["id"], 1]
                b = raw_pairs[backend, row["id"], 2]
                for metric in a:
                    depth_pairs[(backend, *key), metric].append((a[metric], b[metric]))
        if set(data) == {"nf4", "bf16"}:
            a = raw_pairs["nf4", row["id"], slot.passes]
            b = raw_pairs["bf16", row["id"], slot.passes]
            for metric in a:
                precision_pairs[(slot.passes, *key), metric].append((a[metric], b[metric]))
    return {
        "outcomes": [{"cell": list(cell), "metric": metric, "worlds": bounds(values),
                      "both_worlds_by_root": bounds(both_worlds[cell, metric])}
                     for (cell, metric), values in sorted(strata.items())],
        "one_to_two_pass": [{"cell": list(cell), "metric": metric, **paired(values)}
                            for (cell, metric), values in sorted(depth_pairs.items())],
        "nf4_to_bf16": [{"cell": list(cell), "metric": metric, **paired(values)}
                        for (cell, metric), values in sorted(precision_pairs.items())],
        "verified_compute": [{"cell": list(cell), "rows": len(values),
                              "generated_tokens": sum(v[0] for v in values),
                              "phase_token_visits": sum(v[1] for v in values),
                              "wall_seconds": sum(v[2] for v in values),
                              "stops": {reason: sum(v[3] == reason for v in values)
                                        for reason in ("eos", "complete_packet", "character_cap", "token_cap")}}
                             for cell, values in sorted(ratios.items())],
    }


def decode_with(tok):
    return lambda ids: tok.tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def collect(path, selected, *, bindings, tok, manifest_sha, expected_target, backend):
    terminal_path = path / "terminal.json"
    if not terminal_path.is_file():
        return {}
    terminal = json.loads(terminal_path.read_bytes())
    if (terminal.get("status") not in {"completed", "partial"}
            or terminal.get("backend") != backend or terminal.get("runtime_manifest_sha256") != manifest_sha
            or terminal.get("target_before") != expected_target or terminal.get("target_after") != expected_target
            or terminal.get("quant_before") != terminal.get("quant_after")
            or terminal.get("checkpoint_verified_after") is not True or terminal.get("preflight_passed") is not True):
        return {}
    records = {}
    slots = {slot.file: slot for slot in selected}
    journal = path / "closed.jsonl"
    if not journal.is_file():
        return records
    if sha(journal) != terminal.get("closed_journal_sha256"):
        raise ValueError("Closed journal differs from the terminal binding")
    for line in journal.read_bytes().splitlines(keepends=True):
        if not line.endswith(b"\n"):
            raise ValueError("The terminal journal contains an incomplete line")
        item = json.loads(line)
        slot = slots[item["file"]]
        if slot.ordinal in records:
            raise ValueError("Repeated verified episode")
        episode = path / slot.file
        if sha(episode) != item["sha256"]:
            raise ValueError("The closed reply differs from its journal")
        output = json.loads(episode.read_bytes())
        score = audit_episode(output, slot, bindings[slot.source["id"]], decode_with(tok),
                              tok.tok.eos_token_id, terminal["vocabulary_size"])
        records[slot.ordinal] = (score, output)
    if len(records) != terminal["closed_slots"]:
        raise ValueError("Closed row count disagrees with terminal receipt")
    return records

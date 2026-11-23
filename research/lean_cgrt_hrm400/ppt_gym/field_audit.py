"""LSA-02D: public evidence retention in the *actual* 384-byte PPT input.

No shortest-proof label, generator hint, private oracle, live evaluator, or
checkpoint is used to build a representation. The complete teacher manifest
is authenticated before reading its public training packets. This audit
compares whole-field byte-span survival in canonical raw JSON against the
typed encoder's declared public fact/rule entries. It does not use string
substring guesses about truncated JSON.
"""
from __future__ import annotations

from collections import defaultdict
import argparse
import json
from pathlib import Path

from .data import canonical, sha256_file
from .native_train import _load_teacher
from .typed_state import _formula, _public, _rule, _type_entries, encode_public_packet


SCHEMA = "lsa02d.public_field_retention.v1"


def _canonical_bytes_and_spans(value):
    """Map nested JSON field paths to their *exact serialized byte spans*."""
    parts = []
    spans = {}
    offset = 0

    def emit(fragment: str):
        nonlocal offset
        data = fragment.encode("utf-8")
        parts.append(data)
        offset += len(data)

    def walk(obj, path):
        start = offset
        if isinstance(obj, dict):
            if any(not isinstance(key, str) for key in obj):
                raise ValueError("public canonical state has non-string object key")
            emit("{")
            for index, key in enumerate(sorted(obj)):
                if index:
                    emit(",")
                emit(json.dumps(key, ensure_ascii=False, separators=(",", ":")))
                emit(":")
                walk(obj[key], path + (key,))
            emit("}")
        elif isinstance(obj, list):
            emit("[")
            for index, item in enumerate(obj):
                if index:
                    emit(",")
                walk(item, path + (index,))
            emit("]")
        else:
            emit(canonical(obj))
        spans[path] = (start, offset)

    walk(value, ())
    return b"".join(parts), spans


def _fully_visible(spans, path: tuple, start_of_tail: int) -> bool:
    span = spans.get(path)
    return span is not None and span[0] >= start_of_tail


def _entry(name: str, premises: tuple[str, ...], head: str) -> str:
    return f"{name}:" + ("+".join(premises) + ">" if premises else "") + f"{head};"


def inspect_public_frame(frame: dict, max_bytes: int = 384) -> dict:
    """Assess complete public fields after the exact ByteEncoder tail crop."""
    if type(max_bytes) is not int or not 32 <= max_bytes <= 384:
        raise ValueError("outside model byte budget")
    packet = frame["goal"]
    actions = frame["candidates"]
    parsed = json.loads(packet)
    data, spans = _canonical_bytes_and_spans(parsed)
    if data != packet.encode("utf-8"):
        raise ValueError("teacher goal packet differs from canonical public JSON")
    goals, budget, failed = _public(packet)
    typed = encode_public_packet(packet, actions, view="typed", max_bytes=max_bytes)
    if len(typed.encode("utf-8")) > max_bytes:
        raise ValueError("typed representation exceeded model byte budget")
    tail_start = max(0, len(data) - max_bytes)

    front = goals[0]
    entries = _type_entries(front)
    typed_parts = {name: _entry(name, pre, head) for name, pre, head in entries}
    # Drop the fixed B/G/F header. Each remaining ';'-terminated entry
    # retains a complete name:type relation, including the first candidate.
    typed_set = {
        part + ";" for part in typed.rsplit("|", 1)[-1].split(";") if part
    }
    names = [hyp["name"] for hyp in front["hypotheses"]]
    hyp_paths = {
        name: ("state", "goals", 0, "hypotheses", index)
        for index, name in enumerate(names)
    }
    apply_names = [
        action[6:] for action in actions if action.startswith("apply ")
    ]
    if any(name not in typed_parts for name in apply_names):
        raise ValueError("candidate action not present as public type declaration")
    typed_candidate = sum(typed_parts[name] in typed_set for name in apply_names)
    raw_candidate = sum(
        _fully_visible(spans, hyp_paths[name], tail_start) for name in apply_names
    )
    background = [name for name, _, _ in entries if name not in apply_names]
    background_typed = sum(typed_parts[name] in typed_set for name in background)
    background_raw = sum(
        _fully_visible(spans, hyp_paths[name], tail_start) for name in background
    )
    pp = ("state", "goals", 0, "pp")
    result = {
        "root": frame["root"],
        "lineage": frame["lineage"],
        "frame_step": frame["step"],
        "source_bytes": len(data),
        "raw_visible_bytes": min(len(data), max_bytes),
        "typed_visible_bytes": len(typed.encode("utf-8")),
        "raw_full_current_goal": int(_fully_visible(
            spans, ("state", "goals", 0, "target"), tail_start,
        )),
        "typed_full_current_goal": int(
            f"G:{','.join(_formula(g['target']) for g in goals)}|" in typed
        ),
        "raw_full_budget": int(_fully_visible(
            spans, ("checks_remaining",), tail_start,
        )),
        "typed_full_budget": int(typed.startswith(f"B{budget}|")),
        "raw_full_pretty": int(_fully_visible(spans, pp, tail_start)),
        "typed_full_pretty": 0,
        "raw_full_failed_choices": int(_fully_visible(
            spans, ("failed_choices",), tail_start,
        )),
        "typed_full_failed_choices": int(
            not failed or "F:" + ",".join(
                a[6:] if a.startswith("apply ") else a for a in failed
            ) + "|" in typed
        ),
        "raw_complete_candidate_types": raw_candidate,
        "typed_complete_candidate_types": typed_candidate,
        "candidate_types": len(apply_names),
        "raw_complete_background_types": background_raw,
        "typed_complete_background_types": background_typed,
        "background_types": len(background),
        "raw_complete_pending_goal_types": sum(
            _fully_visible(spans, ("state", "goals", i, "target"), tail_start)
            for i in range(1, len(goals))
        ),
        "typed_complete_pending_goal_types": len(goals) - 1,
        "pending_goals": len(goals) - 1,
    }
    if not result["typed_full_current_goal"] or not result["typed_full_budget"]:
        raise ValueError("typed representation lost the current public target or budget")
    return result


def _root_weighted(rows: list[dict], field: str) -> float:
    groups = defaultdict(list)
    for item in rows:
        groups[item["lineage"]].append(item[field])
    return sum(sum(v) / len(v) for v in groups.values()) / len(groups)


def analyze(teacher: Path, out: Path, *, max_bytes: int = 384) -> dict:
    if out.exists():
        raise ValueError("output directory must be fresh")
    frames, meta = _load_teacher(teacher)
    if not meta["backend"]["actual_lean_process"]:
        raise ValueError("field retention requires native Lean compiled teacher")
    rows = [inspect_public_frame(row, max_bytes=max_bytes) for row in frames]
    counts = {
        "frames": len(rows),
        "lineages": len({r["lineage"] for r in rows}),
        "source_bytes_mean": sum(r["source_bytes"] for r in rows) / len(rows),
        "visible_raw_bytes_mean": sum(r["raw_visible_bytes"] for r in rows) / len(rows),
        "visible_typed_bytes_mean": sum(r["typed_visible_bytes"] for r in rows) / len(rows),
    }
    for name in (
        "raw_full_current_goal", "typed_full_current_goal",
        "raw_full_budget", "typed_full_budget",
        "raw_full_pretty", "typed_full_pretty",
        "raw_full_failed_choices", "typed_full_failed_choices",
        "raw_complete_candidate_types", "typed_complete_candidate_types",
        "candidate_types", "raw_complete_background_types",
        "typed_complete_background_types", "background_types",
        "raw_complete_pending_goal_types", "typed_complete_pending_goal_types",
        "pending_goals",
    ):
        counts[name] = sum(row[name] for row in rows)
        counts[name + "_root_mean"] = _root_weighted(rows, name)
    source_sha = sha256_file(teacher / "manifest.json")
    report = {
        "schema": SCHEMA,
        "source_teacher_manifest_sha256": source_sha,
        "source_lineages": meta["independent_lineages"],
        "source_frames": meta["train_examples"],
        "model_max_bytes": max_bytes,
        "method": "exact canonical JSON byte-span survival versus full typed public entries",
        "counts": counts,
        "train_labels_consulted_for_field_reconstruction": False,
        "model_uses_last_raw_bytes": True,
        "scientific_eligible": False,
        "no_lean_calls": True,
        "locked_test_opened": False,
        "scope": "read-only public representation custody audit on exposed training sources",
    }
    out.mkdir(parents=True)
    with (out / "public_field_frames.jsonl").open("x", encoding="utf-8") as writer:
        for row in rows:
            writer.write(canonical(row) + "\n")
    report["frames_sha256"] = sha256_file(out / "public_field_frames.jsonl")
    (out / "manifest.json").write_text(canonical(report) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-bytes", type=int, default=384)
    args = p.parse_args(argv)
    m = analyze(args.teacher, args.out, max_bytes=args.max_bytes)
    print(canonical(m))
    return m


if __name__ == "__main__":
    main()

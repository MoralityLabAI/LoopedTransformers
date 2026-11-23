"""LBF-05 public-only 384-byte source-fidelity audit.

Reads only independently certified *training* teacher roots. No label or proof
reference is consulted when producing metrics or choosing a representation.
All comparisons see exactly the ByteEncoder-visible tail of each input.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

LCGRT = Path(__file__).resolve().parents[1] / "lean_cgrt_hrm400"
if str(LCGRT) not in sys.path:
    sys.path.insert(0, str(LCGRT))

from ppt_gym.data import canonical, sha256_file
from ppt_gym.native_train import _load_teacher
from ppt_gym.typed_state import (_formula, _type_entries, _public,
                                  encode_public_packet)


VIEWS = ("raw", "typed", "hybrid")
SCHEMA = "lbf05.public_window_audit.v1"
ATOM = re.compile(r"P[0-9]+")
HYP = re.compile(r"(?:h|r)[0-9]+")
ALLOWED_FRAME_FIELDS = {
    "root", "lineage", "goal", "candidates", "checks_remaining", "labels",
    "schema", "family", "step", "evidence", "public_state_sha256",
    "chosen_reference_tactic", "reference_proof_size", "label_semantics",
}


def _visible(text: str, budget: int) -> str:
    return text.encode("utf-8")[-budget:].decode("utf-8", errors="ignore")


def _mentions(name: str, view: str) -> bool:
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])", view) is not None


def coverage(public: str, candidates: list[str], max_bytes: int = 384) -> dict:
    """Label-free metrics. 'token' means lexical presence, never causal utility."""
    goals, remaining, failed = _public(public)
    entries = _type_entries(goals[0])
    first = goals[0]
    target_atoms = set(ATOM.findall(_formula(first["target"])))
    candidate_names = [c[6:] if c.startswith("apply ") else c for c in candidates]
    named = {name: (pre, head) for name, pre, head in entries}
    candidate_rule_symbols = set()
    for name in candidate_names:
        if name in named:
            pre, head = named[name]
            candidate_rule_symbols.update(ATOM.findall(head))
            for item in pre:
                candidate_rule_symbols.update(ATOM.findall(item))
    pretty = re.sub(r"\s+", " ", first.get("pp", "")).strip()
    pp_atoms = set(ATOM.findall(pretty))
    records = {}
    for view in VIEWS:
        result = encode_public_packet(public, candidates, view=view, max_bytes=max_bytes)
        window = _visible(result, max_bytes)
        # Input metadata counts do not depend on proof labels.
        records[view] = {
            "source_bytes": len(result.encode("utf-8")),
            "visible_bytes": len(window.encode("utf-8")),
            "over_budget": len(result.encode("utf-8")) > max_bytes,
            "candidate_name_recall": (sum(_mentions(x, window) for x in candidate_names)
                                      / len(candidate_names)),
            "goal_atom_recall": (sum(_mentions(x, window) for x in target_atoms)
                                 / len(target_atoms) if target_atoms else 1.0),
            "candidate_rule_atom_recall": (sum(_mentions(x, window) for x in candidate_rule_symbols)
                                           / len(candidate_rule_symbols)
                                           if candidate_rule_symbols else 1.0),
            "pretty_atom_recall": (sum(_mentions(x, window) for x in pp_atoms)
                                   / len(pp_atoms) if pp_atoms else None),
            "has_public_pretty_channel": "P:" in window if view == "hybrid" else bool(pretty),
            "window_sha256": hashlib.sha256(window.encode("utf-8")).hexdigest(),
        }
    return records


def audit(teacher: Path, out: Path, *, max_bytes: int = 384) -> dict:
    if out.exists():
        raise ValueError("audit destination must be fresh")
    if max_bytes != 384:
        raise ValueError("LBF-05 preregisters the 384-byte native ByteEncoder limit")
    frames, manifest = _load_teacher(teacher)
    if (manifest["split"] != "train"
        or manifest["backend"]["actual_lean_process"] is not True
        or manifest["source_compiler_passed"] is not True
        or manifest["query_roots_exposed_to_fit"] != 0
        or manifest["independent_lineages"] < 1):
        raise ValueError("only fully certified train-only native teacher permitted")
    # The importer already checks file hashes; the audit uses only public fields.
    records = []
    seen_lineage = set()
    for frame in frames:
        if not set(frame).issubset(ALLOWED_FRAME_FIELDS):
            raise ValueError("unexpected extra field in public/source teacher row")
        for key in ("lineage", "goal", "candidates", "root"):
            if key not in frame:
                raise ValueError("missing public field: " + key)
        if not isinstance(frame["candidates"], list):
            raise ValueError("public candidate menu must be a list")
        seen_lineage.add(frame["lineage"])
        records.append({
            "root_id": frame["root"], "lineage": frame["lineage"],
            "public_state_sha256": frame.get("public_state_sha256"),
            "public_packet_sha256": hashlib.sha256(frame["goal"].encode()).hexdigest(),
            "n_candidates": len(frame["candidates"]),
            "windows": coverage(frame["goal"], frame["candidates"], max_bytes),
        })
    if len(seen_lineage) != manifest["independent_lineages"]:
        raise ValueError("independent source lineages disagree with manifest")
    by_root = defaultdict(list)
    for row in records:
        by_root[row["lineage"]].append(row)
    aggregations = {}
    measures = ("source_bytes", "visible_bytes", "candidate_name_recall",
                "goal_atom_recall", "candidate_rule_atom_recall")
    for view in VIEWS:
        sums = {}
        for measure in measures:
            sums[measure] = sum(
                sum(x["windows"][view][measure] for x in by_root[root]) / len(by_root[root])
                for root in by_root
            ) / len(by_root)
        sums["fraction_packets_over_budget"] = sum(
            int(r["windows"][view]["over_budget"]) for r in records
        ) / len(records)
        sums["pretty_channel_present_count"] = sum(
            int(r["windows"][view]["has_public_pretty_channel"]) for r in records)
        aggregations[view] = sums
    out.mkdir(parents=True)
    lines = out / "source_window_metrics.jsonl"
    lines.write_text("".join(canonical(x) + "\n" for x in records), encoding="utf-8")
    result = {
        "schema": SCHEMA, "teacher_manifest_sha256": sha256_file(teacher / "manifest.json"),
        "training_frames": len(records), "independent_lineages": len(by_root),
        "source_native_compiled": True, "source_split": "train",
        "byte_limit": max_bytes, "views": list(VIEWS), "aggregates_root_weighted": aggregations,
        "per_frame_metrics_sha256": sha256_file(lines),
        "labels_exported": 0, "verifier_outcomes_accessed_for_view_selection": False,
        "locked_test_opened": False, "scientific_eligible": False,
        "interpretation": "lexical public-token retention, not structural proof validity or policy accuracy",
    }
    (out / "manifest.json").write_text(canonical(result) + "\n", encoding="utf-8")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--max-bytes", default=384, type=int)
    args = p.parse_args()
    print(canonical(audit(args.teacher, args.out, max_bytes=args.max_bytes)))


if __name__ == "__main__":
    main()

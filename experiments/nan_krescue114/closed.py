"""Audit a closed grouped BF16 terminal and apply the registered K-rescue gate."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

from .score import COMPLETE, WORLDS, bind_parent_orientation, gate, margin_footprint


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REGISTRATION = HERE / "REGISTRATION.json"


def checked_bytes(path, expected):
    payload = path.read_bytes()
    if sha256(payload).hexdigest() != expected:
        raise ValueError(f"frozen custody digest changed for {path.name}")
    return payload


def lock():
    data = REGISTRATION.read_bytes()
    registration = json.loads(data)
    for relative, expected in registration["parent_sha256"].items():
        checked_bytes(REPO / relative, expected)
    return registration, sha256(data).hexdigest()


def collect(work, source, classes, tok, frozen, *, fixture=False):
    from experiments.nan_boundrescue109 import host as serial_host
    from experiments.nan_boundrescue109.verify_runtime import FIXTURE
    from experiments.nan_rescuegpu111 import host as grouped_host

    registration, reg_sha = lock()
    cfg = registration["source"]
    if frozen != registration["parent_sha256"]["experiments/nan_rescuegpu111/RUNTIME_MANIFEST.json"]:
        raise ValueError("the grouped BF16 runtime differs from the preregistered implementation")
    if (serial_host.CFG["minimum_clean_both_world_roots"] != registration["outcome"]["minimum_complete_roots"]
            or serial_host.CFG["minimum_literal_copy_both_world_roots"] != registration["outcome"]["min_copy_roots"]):
        raise ValueError("the parent competence thresholds differ from registration")
    if {p["split"] for p in source["public"]} != {cfg["split"]}:
        raise ValueError("the root roster is outside fresh development")

    # Terminal is a closed, write-once record. The parent closes all source,
    # resource, stage ordering, head, geometry and full root identities.
    mode = FIXTURE if fixture else serial_host.MODE
    parent, audited = grouped_host.collect(work, source, classes, tok, frozen, mode=mode)
    terminal = json.loads(checked_bytes(work / "terminal.json", parent["terminal_sha256"]))
    journal = checked_bytes(work / "closed.jsonl", terminal["journal_sha256"]).splitlines()
    gold = {(g["root"], g["world"]): g["answer"] for g in source["gold"] if g["arm"] == "focused"}
    roots = sorted({r for r, _ in gold})
    verified = {r["root"]: r["signed_macro"] for r in audited}
    records = []
    for i, (root, line) in enumerate(zip(roots, journal, strict=True)):
        entry = json.loads(line)
        name = f"root_{i:03d}.json"
        if entry["file"] != name:
            raise ValueError("root journal order changed during the audit")
        native = json.loads(checked_bytes(work / name, entry["sha256"]))
        status = native["status"]
        if native["root"] != root or ((status == COMPLETE) != (root in verified)):
            raise ValueError("root or closure identity changed during the audit")
        row = {"root": root, "status": status}
        if status == COMPLETE:
            own, peer = [gold[root, w] for w in WORLDS]
            result = margin_footprint(native["head_observations"], own, peer)
            bind_parent_orientation(result, verified[root])
            row.update(result)
        records.append(row)
    criteria = registration["outcome"] | {"registered_roots": cfg["roots"]}
    decision = gate(records, roots, criteria)
    if fixture:
        decision["decision"] = "CPU_FIXTURE_ONLY"
    return {
        "experiment": "NAN-KRESCUE114", "registration_sha256": reg_sha,
        "runtime_sha256": frozen, "audited_terminal_sha256": parent["terminal_sha256"],
        "endpoint": criteria["primary"], "backend": "bf16", "source_split": cfg["split"],
        "evidence": "cpu_fixture" if fixture else "fresh_native_development",
        "status_counts": parent["status_counts"], "formal_comparisons": 0,
        "sealed_test_accessed": False, "decision": decision,
    }, records


def main():
    from experiments.nan_boundrescue109 import host as serial_host
    from experiments.nan_phase35.run import lease
    from experiments.nan_readpath37.run import write_new
    from experiments.nan_rescuegpu111.lane import verified
    from experiments.nan_vocab83.custody import HOST_BUDGET, vocabulary

    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("snapshot", "source-dir", "terminal-dir", "out"):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args()
    settings, _ = lock()
    if (args.snapshot.name != settings["source"]["checkpoint"] or not __debug__
            or "torch" in sys.modules or not (args.terminal_dir / "terminal.json").is_file()):
        parser.error("use the pinned tokenizer and a closed grouped BF16 terminal")
    output = args.out.resolve()
    if output.exists() or any(output.is_relative_to(x.resolve()) for x in (args.snapshot, args.source_dir, args.terminal_dir)):
        parser.error("use a new output directory outside the immutable inputs")
    with lease(HOST_BUDGET) as (check, _):
        source, pins = serial_host.host_inputs(args.source_dir)
        tok, classes = vocabulary(args.snapshot, check, pins["tokenizer_pins"])
        summary, private = collect(args.terminal_dir, source, classes, tok, verified())
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "private_roots.json", private)
    write_new(output / "aggregate.json", summary)


if __name__ == "__main__":
    main()

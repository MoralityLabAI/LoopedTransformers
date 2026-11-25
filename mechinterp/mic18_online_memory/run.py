"""Exhaustive neural-register replay. Pure standard library, integer arithmetic."""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path

import finite_memory
from register import MemoryProgram, pack, word_bits

HERE = Path(__file__).resolve().parent


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_lock() -> dict:
    lock = json.loads((HERE / "SOURCE_LOCK.json").read_text())
    for filename, expected in lock["files"].items():
        actual = sha(HERE / filename)
        if actual != expected:
            raise ValueError(f"source hash mismatch: {filename}")
    return lock


def controller_from_program(program: MemoryProgram) -> dict:
    width = program.width
    capacity = 2**width if program.mode == "register" else 2
    states = [word_bits(s, width) if program.mode == "register" else (s,) + (0,)*(width-1)
              for s in range(capacity)]
    encode = pack if program.mode == "register" else lambda state: state[0]
    return {
        "schema": "MIC-18/controller/v1", "name": "exact_relu_" + program.mode,
        "task": program.task, "width": width, "capacity": capacity, "initial_state": 0,
        "transitions": [[[encode(program.update(state, bit, phase)) for bit in (0, 1)]
                         for state in states] for phase in range(width)],
        "outputs": [[program.query(state, q) for q in range(width if program.task == "query" else 1)]
                    for state in states],
    }


def exhaustive(records: list | None = None) -> dict:
    protocol = json.loads((HERE / "PROTOCOL.json").read_text())
    lock = verify_lock()
    rows = []
    digest = hashlib.sha256()
    def record(row: dict) -> None:
        digest.update(canonical(row) + b"\n")
        if records is not None:
            records.append(row)
    totals = {"matrix_transition_checks": 0, "matrix_readout_checks": 0,
              "query_scores": 0, "parity_words": 0, "finite_controller_replays": 0}
    for width in protocol["widths"]:
        by_mode = {}
        signatures = set()
        for mode in ("register", "parity", "last_bit"):
            program = MemoryProgram.build(width, mode)
            transition_checks = 0
            readout_checks = 0
            for word in range(2**width):
                state = word_bits(word, width)
                for phase in range(width):
                    for incoming in (0, 1):
                        expected = list(state) if mode == "register" else [0]*width
                        expected[phase if mode == "register" else 0] = (
                            state[0] ^ incoming if mode == "parity" else incoming)
                        actual = program.update(state, incoming, phase)
                        if actual != tuple(expected):
                            raise AssertionError((width, mode, state, phase, incoming, actual, expected))
                        transition_checks += 1
                for query in range(width):
                    expected = (state[query] if mode == "register" else
                                state[0] if (mode == "last_bit" and query == width-1)
                                or (mode == "parity" and width == 1) else 0)
                    if program.query(state, query) != expected:
                        raise AssertionError("query matrix mismatch")
                    readout_checks += 1
            correct = 0
            for word in range(2**width):
                observations = word_bits(word, width)
                state = (0,)*width
                path = [state]
                for phase, bit in enumerate(observations):
                    state = program.update(state, bit, phase)
                    path.append(state)
                predictions = tuple(program.query(state, query) for query in range(width))
                correct += sum(a == b for a, b in zip(predictions, observations))
                record({"width": width, "mode": mode, "word": word,
                        "states": path, "predictions": predictions,
                        "targets": observations})
                if mode == "register":
                    signatures.add(predictions)
            core = finite_memory.evaluate(controller_from_program(program))
            if (core["score"], core["total"]) != (correct, width * 2**width):
                raise AssertionError("shared finite-controller score mismatch")
            by_mode[mode] = {"correct": correct, "total": width * 2**width,
                             "fraction": core["fraction"], "resources": program.resource_account(),
                             "matrix_transition_checks": transition_checks,
                             "matrix_readout_checks": readout_checks,
                             "reachable_state_counts": core["reachable_state_counts"]}
            totals["matrix_transition_checks"] += transition_checks
            totals["matrix_readout_checks"] += readout_checks
            totals["query_scores"] += width * 2**width
            totals["finite_controller_replays"] += 1
        padded = MemoryProgram.build(width, "parity", "parity")
        compact = MemoryProgram.build(1, "parity", "parity")
        parity_correct = 0
        for word in range(2**width):
            observations = word_bits(word, width)
            state = (0,)
            for bit in observations:
                state = compact.update(state, bit, 0)
            target = sum(observations) % 2
            answers = (compact.query(state), padded.query(padded.commit(observations)))
            if answers != (target, target):
                raise AssertionError("parity control failed")
            parity_correct += 1
            record({"width": width, "word": word,
                    "parity_target": target, "compact_and_padded": answers})
        parity_core = finite_memory.evaluate(controller_from_program(padded))
        if (parity_core["score"], parity_core["total"]) != (parity_correct, 2**width):
            raise AssertionError("parity controller bridge failed")
        totals["finite_controller_replays"] += 1
        totals["parity_words"] += parity_correct
        if len(signatures) != 2**width:
            raise AssertionError("future-response signatures collapsed")
        rows.append({"width": width, "query_modes": by_mode,
                     "parity": {"correct": parity_correct, "total": 2**width,
                                "compact_dynamic_bits": 1,
                                "compact_update_relu_evaluations": 2,
                                "compact_episode_relu_evaluations": 2*width+1,
                                "padded_resources": padded.resource_account()},
                     "distinct_future_query_signatures": len(signatures),
                     "zero_error_query_state_lower_bound": 2**width,
                     "zero_error_query_bit_lower_bound": width})
    return {"schema": "MIC-18/neural-register-results/v1",
            "evidence_class": "exact finite domain", "status": "PASS_EXACT_CONSTRUCTED_CELL",
            "training_updates": 0, "trained_checkpoints": 0,
            "source_files": lock["files"], "provenance": lock["provenance"],
            "counts": totals, "raw_replay_sha256": digest.hexdigest(), "by_width": rows,
            "population_gate": "NO_TRAINED_ROOTS_OR_SHAM_CALIBRATION",
            "arithmetic_scope": "exact integer matrices over enumerated binary states"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    records = []
    result = exhaustive(records)
    raw = b"".join(canonical(row) + b"\n" for row in records)
    receipt = HERE / "RESULTS.json"
    archive = HERE / "paths.jsonl.gz.b64"
    if args.check:
        if json.loads(receipt.read_text()) != result:
            raise SystemExit("stored neural receipt differs from exhaustive replay")
        if gzip.decompress(base64.b64decode(archive.read_bytes())) != raw:
            raise SystemExit("stored neural paths differ from exhaustive replay")
    if args.write:
        receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        archive.write_bytes(base64.b64encode(gzip.compress(raw, compresslevel=9, mtime=0)) + b"\n")
    print(json.dumps({"status": result["status"], "counts": result["counts"],
                      "raw_replay_sha256": result["raw_replay_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()

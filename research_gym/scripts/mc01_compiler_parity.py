"""Compile MC-01 with Circom and independently check exported R1CS/WTNS.

Run with an already installed, checksum-pinned compiler. No setup or ZK proving.
The compiler disables witness-time asserts so false witnesses reach the R1CS audit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import subprocess

from research_gym.analysis.mechanistic_certificate import BN254, Statement
from research_gym.scripts.mechanistic_certificate import export_relation, file_hash


def read_wtns(path: Path) -> tuple[int, list[int]]:
    raw = path.read_bytes()
    if len(raw) < 12 or raw[:4] != b"wtns":
        raise ValueError("invalid WTNS magic")
    version, count = struct.unpack_from("<II", raw, 4)
    if version != 2 or count != 2:
        raise ValueError("expected version-2, two-section WTNS")
    sections = {}
    offset = 12
    for _ in range(count):
        if offset + 12 > len(raw):
            raise ValueError("truncated section header")
        kind, length = struct.unpack_from("<IQ", raw, offset)
        offset += 12
        if kind in sections or offset + length > len(raw):
            raise ValueError("duplicate or truncated section")
        sections[kind] = raw[offset:offset + length]
        offset += length
    if offset != len(raw) or set(sections) != {1, 2}:
        raise ValueError("trailing bytes or unknown sections")
    header = sections[1]
    if len(header) < 4:
        raise ValueError("short WTNS field header")
    size = struct.unpack_from("<I", header)[0]
    if size != 32 or len(header) != size + 8:
        raise ValueError("expected 32-byte field")
    prime = int.from_bytes(header[4:4 + size], "little")
    wires = struct.unpack_from("<I", header, 4 + size)[0]
    payload = sections[2]
    if prime != BN254 or len(payload) != size * wires:
        raise ValueError("WTNS field/length mismatch")
    values = [int.from_bytes(payload[i:i + size], "little") for i in range(0, len(payload), size)]
    if not values or values[0] != 1 or any(value >= prime for value in values):
        raise ValueError("noncanonical WTNS values")
    return prime, values


def satisfies(constraints, values, prime):
    def linear(expression):
        return sum(int(coefficient) * values[int(index)] for index, coefficient in expression.items()) % prime
    return all((linear(a) * linear(b) - linear(c)) % prime == 0 for a, b, c in constraints)


def run(circom: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    statement = Statement((2, 2, 1), 0, 3, 2)
    export_relation(statement, (0, 2, 1, 2), out)
    version = subprocess.check_output([str(circom), "--version"], text=True).strip()
    command = [str(circom), str(out / "trace.circom"), "--r1cs", "--wasm", "--sym", "--json",
               "--O0", "--sanity_check", "0", "-o", str(out)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    (out / "compile.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"Circom compilation failed:\n{result.stdout}\n{result.stderr}")
    compiled = json.loads((out / "trace_constraints.json").read_text())["constraints"]
    symbols = {}
    for line in (out / "trace.sym").read_text().splitlines():
        _, wire, _, name = line.split(",", 3)
        if int(wire) < 1 or name in symbols:
            raise ValueError("unexpected O0 symbol layout")
        symbols[name] = int(wire)
    template = json.loads((out / "witness.json").read_text())
    cases = [("valid", template, True)]
    for key, array in template.items():
        for i in range(len(array)):
            mutated = {name: values.copy() for name, values in template.items()}
            mutated[key][i] += 1
            cases.append((f"{key}_{i}", mutated, False))
    rows = []
    for name, data, expected in cases:
        source = out / "current_input.json"
        witness = out / "current.wtns"
        source.write_text(json.dumps(data))
        generated = subprocess.run(["node", str(out / "trace_js/generate_witness.js"),
                        str(out / "trace_js/trace.wasm"), str(source), str(witness)],
                       capture_output=True, text=True, timeout=30)
        if generated.returncode:
            message = f"Witness generation failed for {name}:\n{generated.stdout}\n{generated.stderr}"
            (out / "witness_error.log").write_text(message)
            raise RuntimeError(message)
        prime, values = read_wtns(witness)
        for key, array in data.items():
            for i, value in enumerate(array):
                if values[symbols[f"main.{key}[{i}]"]] != value:
                    raise ValueError("compiled witness differs from input at symbol mapping")
        accepted = satisfies(compiled, values, prime)
        if accepted != expected:
            raise RuntimeError(f"compiled constraint mismatch for {name}")
        rows.append({"case": name, "expected": expected, "accepted": accepted})
    receipt = {"status": "COMPILED_RELATION_PARITY", "compiler_version": version,
               "compiler_sha256": file_hash(circom), "compile_command": command,
               "compiled_constraints": len(compiled), "compiled_wires": len(values),
               "cases": rows, "cases_checked": len(rows),
               "false_witnesses_rejected": sum(not row["accepted"] for row in rows),
               "proof_generated": False, "trusted_setup": False,
               "cryptographic_source_binding": False,
               "artifact_sha256": {name: file_hash(out / name) for name in (
                   "trace.circom", "trace.r1cs", "trace_constraints.json", "trace.sym", "trace_js/trace.wasm")}}
    (out / "compiler_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circom", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.circom.resolve(), args.out.resolve())

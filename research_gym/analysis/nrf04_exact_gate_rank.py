"""NRF-04 independent exact binary32 full-domain gate-closure verifier.

Scale IEEE754 float32 parameters by 2^149, giving exact integers.
Their modular ranks over F_65521 and F_65519 certify rational/real rank
without floating SVD, tolerance, rounding, or checkpoint weight changes.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

from research_gym.analysis.nrf03_checkpoint_probe import (
    manifest_entries, load_pinned_weights,
)

PRIMES = (65521, 65519)


def float32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def dyadic_integer_mod(bits: int, prime: int) -> int:
    exponent = (bits >> 23) & 255
    if exponent == 255:
        raise ValueError("nonfinite IEEE binary32 parameter")
    if (bits & 0x7fffffff) == 0:
        return 0
    mantissa = bits & 0x7fffff
    if exponent:
        mantissa += 1 << 23
        amount = mantissa * pow(2, exponent - 1, prime) % prime
    else:
        amount = mantissa % prime
    return (-amount) % prime if bits >> 31 else amount


def finite_field_det(bits: list[list[int]], prime: int) -> dict:
    n = len(bits)
    if n < 1 or any(len(row) != n for row in bits):
        raise ValueError("square matrix of float32 bit patterns required")
    a = [[dyadic_integer_mod(x, prime) for x in row] for row in bits]
    swaps, determinant, rank = 0, 1, 0
    pivots = []
    for col in range(n):
        pick = next((i for i in range(rank, n) if a[i][col]), None)
        if pick is None:
            continue
        if pick != rank:
            a[rank], a[pick] = a[pick], a[rank]
            swaps += 1
        pivot = a[rank][col]
        pivots.append(pivot)
        determinant = determinant * pivot % prime
        inverse = pow(pivot, prime - 2, prime)
        for i in range(rank + 1, n):
            factor = a[i][col] * inverse % prime
            a[i][col] = 0
            if factor:
                for j in range(col + 1, n):
                    a[i][j] = (a[i][j] - factor * a[rank][j]) % prime
        rank += 1
    if swaps % 2:
        determinant = (-determinant) % prime
    return {
        "prime": prime, "rank": rank,
        "determinant_mod_prime": determinant if rank == n else 0,
        "pivot_count": len(pivots),
        "pivots_sha_proxy": pivots[:6],
    }


def certificate(w1, w2):
    if w1.ndim != 2 or w1.shape != w2.shape or w1.shape[0] != w1.shape[1]:
        raise ValueError("equal square weight matrices required")
    if w1.dtype != w2.dtype or str(w1.dtype) != "torch.float32":
        raise ValueError("IEEE binary32 matrices required")
    bits = [[float32_bits(v) for v in row] for row in w1.tolist()]
    zero_count = sum(
        (float32_bits(v) & 0x7fffffff) == 0
        for row in w2.tolist() for v in row
    )
    fields = [finite_field_det(bits, p) for p in PRIMES]
    full = any(field["rank"] == len(bits) for field in fields)
    return {
        "modular_certificates": fields,
        "w2_zero_entries": zero_count,
        "exact_gate_closure_dimension": (
            len(bits) if full and zero_count == 0 else None
        ),
    }


def verify_manifest(checkpoint_dir: Path, events: Path, reference: Path):
    expected = json.loads(reference.read_text(encoding="utf-8"))
    entries = manifest_entries(checkpoint_dir, events)
    by_id = {row["id"]: row for row in expected["rows"]}
    if len(entries) != 12 or len(by_id) != 12:
        raise ValueError("twelve sealed artifacts required")
    verified = []
    for entry in entries:
        cell = f"R{entry['rounds']}_S{entry['seed']}_E{entry['exposure']}"
        ref = by_id[cell]
        if ref["checkpoint_sha256"] != entry["sha256"]:
            raise ValueError(f"checkpoint SHA-256 mismatch: {cell}")
        w1, _, w2, _ = load_pinned_weights(entry)
        value = certificate(w1, w2)
        for field, other in zip(
            value["modular_certificates"], ref["modular_certificates"]
        ):
            for key in ("prime", "rank", "determinant_mod_prime", "pivot_count"):
                if field[key] != other[key]:
                    raise ValueError(f"modular rank disagreement: {cell}")
        if (value["w2_zero_entries"] != ref["w2_zero_entries"]
                or value["exact_gate_closure_dimension"]
                    != ref["exact_gate_closure_dimension"]):
            raise ValueError(f"gate-closure disagreement: {cell}")
        verified.append(cell)
    return {
        "study_id": "NRF-04-exact-gate-rank-v1",
        "status": "verified", "verified_count": len(verified),
        "ids": verified,
    }


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=(
        root / "experiments/loop_schedule_kappa_surface_v1/checkpoints"))
    parser.add_argument("--events", type=Path, default=(
        root / "experiments/loop_schedule_kappa_surface_v1/surface_events.jsonl"))
    parser.add_argument("--reference", type=Path, default=(
        root / "experiments/numerical_range_fixed_point_v1/nrf04_exact_gate_closure.json"))
    args = parser.parse_args()
    print(json.dumps(
        verify_manifest(args.checkpoint_dir, args.events, args.reference),
        indent=2))


if __name__ == "__main__":
    main()

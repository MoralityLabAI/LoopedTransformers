"""NRF-04: exact verifier of learned decoder gate-closure pairings.

Uses recorded binary64 basis coefficients and SHA-pinned float32 W2.
After scaling by 2^(1074+149), every pairing is an exact integer; its
nonzero residue modulo either prime certifies nonzero real pairing.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from research_gym.analysis.nrf03_checkpoint_probe import manifest_entries, load_pinned_weights
from research_gym.analysis.nrf04_exact_gate_rank import (
    PRIMES, dyadic_integer_mod, float32_bits, verify_manifest,
)


def float64_integer_mod(hex_bits: str, prime: int) -> int:
    if len(hex_bits) != 16 or any(c not in "0123456789abcdef" for c in hex_bits):
        raise ValueError("canonical 16-character binary64 pattern required")
    bits = int(hex_bits, 16)
    sign, exponent = bits >> 63, (bits >> 52) & 2047
    fraction = bits & ((1 << 52) - 1)
    if exponent == 2047:
        raise ValueError("nonfinite binary64 basis coefficient")
    if (bits & ((1 << 63) - 1)) == 0:
        return 0
    mantissa = fraction + ((1 << 52) if exponent else 0)
    amount = mantissa * pow(2, max(0, exponent-1), prime) % prime
    return (-amount) % prime if sign else amount


def pairing_residues(vector_bits: list[str], w2, prime: int) -> list[int]:
    n = len(vector_bits)
    if not n or w2.shape != (n, n):
        raise ValueError("basis and W2 dimension mismatch")
    u = [float64_integer_mod(bit, prime) for bit in vector_bits]
    w = [[dyadic_integer_mod(float32_bits(x), prime) for x in row]
         for row in w2.tolist()]
    return [sum(u[i]*w[i][j] for i in range(n)) % prime for j in range(n)]


def verify_learned(checkpoints: Path, events: Path,
                   rank_reference: Path, learned_reference: Path) -> dict:
    base = verify_manifest(checkpoints, events, rank_reference)
    if base["verified_count"] != 12:
        raise ValueError("missing full rank evidence")
    result = json.loads(learned_reference.read_text(encoding="utf-8"))
    records = {x["id"]: x for x in result["rows"]}
    entries = manifest_entries(checkpoints, events)
    if len(records) != 12:
        raise ValueError("exactly twelve learned vectors required")
    checked = []
    for entry in entries:
        key = f"R{entry['rounds']}_S{entry['seed']}_E{entry['exposure']}"
        row = records[key]
        if row["sha256"] != entry["sha256"]:
            raise ValueError("checkpoint digest mismatch")
        _, _, w2, _ = load_pinned_weights(entry)
        bits = row["first_task_supervised_basis_vector_float64_hex"]
        if len(bits) != 128:
            raise ValueError("expected 128D vector")
        for prime in PRIMES:
            residues = pairing_residues(bits, w2, prime)
            zeros = sum(x == 0 for x in residues)
            if zeros != row[f"zero_mod_{prime}"]:
                raise ValueError(f"zero-count mismatch {key}")
            if residues[:8] != row["first_eight_pairing_residues"][str(prime)]:
                raise ValueError(f"pairing mismatch {key}")
            if zeros:
                raise ValueError(f"unwitnessed gate {key}")
        if row["exact_supervised_projection_gate_closure_dimension"] != 128:
            raise ValueError("full closure not recorded")
        checked.append(key)
    return {"study_id": "NRF-04-learned-observable-gate-closure-v1",
            "status": "verified", "verified_count": len(checked), "ids": checked}


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=(
        root / "experiments/loop_schedule_kappa_surface_v1/checkpoints"))
    parser.add_argument("--events", type=Path, default=(
        root / "experiments/loop_schedule_kappa_surface_v1/surface_events.jsonl"))
    parser.add_argument("--rank-reference", type=Path, default=(
        root / "experiments/numerical_range_fixed_point_v1/nrf04_exact_gate_closure.json"))
    parser.add_argument("--learned-reference", type=Path, default=(
        root / "experiments/numerical_range_fixed_point_v1/nrf04_learned_projection_exact_closure.json"))
    args = parser.parse_args()
    print(json.dumps(verify_learned(
        args.checkpoint_dir, args.events,
        args.rank_reference, args.learned_reference), indent=2))


if __name__ == "__main__":
    main()

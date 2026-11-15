"""NRF-05: modular proof that every proper linear observation leaks globally.

Uses the already pinned native checkpoint loader. This module verifies
rank(W1), rank(W2), and complete support of W1*W2 using the exact integers
2**149 times every stored IEEE binary32 weight. A modular nonzero witness
is a one-way mathematical certificate over the reals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_gym.analysis.nrf03_checkpoint_probe import (
    manifest_entries, load_pinned_weights,
)
from research_gym.analysis.nrf04_exact_gate_rank import (
    float32_bits, dyadic_integer_mod, finite_field_det,
)

PRIME = 67_108_859  # Prime 2**26 - 5; p*p < 2**53 for exact JS crosschecks.
CROSS_PRIME = 65_521


def modular_rows(weight, prime: int = PRIME) -> list[list[int]]:
    if (weight.ndim != 2 or weight.shape[0] != weight.shape[1]
            or str(weight.dtype) != "torch.float32"):
        raise ValueError("square IEEE binary32 weights required")
    return [
        [dyadic_integer_mod(float32_bits(x), prime) for x in row]
        for row in weight.tolist()
    ]


def product_support(a: list[list[int]], b: list[list[int]], prime: int = PRIME):
    """Enumerate every residue of (2**149 W1)(2**149 W2)."""
    n = len(a)
    if not n or len(b) != n or any(len(row) != n for row in (*a, *b)):
        raise ValueError("equal nonempty square matrices required")
    zeros = []
    first_four = []
    for i in range(n):
        for j in range(n):
            value = 0
            for k in range(n):
                value = (value + a[i][k] * b[k][j]) % prime
            if value == 0:
                zeros.append((i, j))
            if i == 0 and j < 4:
                first_four.append(value)
    return {
        "zero_count": len(zeros),
        "zero_locations": zeros[:12],
        "first_row_four_residues": first_four,
        "checked_entries": n * n,
    }


def certify_matrix_pair(w1, w2):
    a = modular_rows(w1)
    b = modular_rows(w2)
    p_a = finite_field_det([
        [float32_bits(x) for x in row] for row in w1.tolist()
    ], PRIME)
    p_b = finite_field_det([
        [float32_bits(x) for x in row] for row in w2.tolist()
    ], PRIME)
    q_b = finite_field_det([
        [float32_bits(x) for x in row] for row in w2.tolist()
    ], CROSS_PRIME)
    support = product_support(a, b)
    n = len(a)
    return {
        "prime": PRIME,
        "w1": {"rank": p_a["rank"], "determinant": p_a["determinant_mod_prime"]},
        "w2": {"rank": p_b["rank"], "determinant": p_b["determinant_mod_prime"]},
        "w2_crossprime": {"prime": CROSS_PRIME, "rank": q_b["rank"],
                          "determinant": q_b["determinant_mod_prime"]},
        "product_support": support,
        "every_nonzero_linear_observation_collapses": (
            p_a["rank"] == n and p_b["rank"] == n and support["zero_count"] == 0
        ),
    }


def verify_frozen_receipt(checkpoints: Path, events: Path, reference: Path) -> dict:
    expected = json.loads(reference.read_text(encoding="utf-8"))
    entries = manifest_entries(checkpoints, events)
    by_name = {r["filename"]: r for r in expected["rows"]}
    if len(entries) != 12 or len(by_name) != 12:
        raise ValueError("twelve original checkpoint receipts required")
    compared = []
    for entry in entries:
        ref = by_name.get(entry["name"])
        if ref is None or ref["checkpoint_sha256"] != entry["sha256"]:
            raise ValueError("source SHA-256 mismatches NRF-05 receipt")
        w1, _, w2, _ = load_pinned_weights(entry)
        new = certify_matrix_pair(w1, w2)
        if new["w1"] != ref["w1"] or new["w2"] != ref["w2"]:
            raise ValueError(f"modular determinant mismatch: {entry['name']}")
        if new["w2_crossprime"] != ref["w2_crossprime"]:
            raise ValueError(f"cross-prime determinant mismatch: {entry['name']}")
        if new["product_support"] != ref["product_support"]:
            raise ValueError(f"matrix-product support mismatch: {entry['name']}")
        if not new["every_nonzero_linear_observation_collapses"]:
            raise ValueError(f"no-go hypothesis fails: {entry['name']}")
        compared.append(entry["name"])
    return {
        "study": "NRF-05-full-linear-gate-collapse-v1",
        "status": "verified",
        "checkpoint_count": len(compared),
        "all_w1_w2_invertible": True,
        "all_product_entries_nonzero": True,
        "names": compared,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=(
        root / "experiments/loop_schedule_kappa_surface_v1/checkpoints"))
    parser.add_argument("--events", type=Path, default=(
        root / "experiments/loop_schedule_kappa_surface_v1/surface_events.jsonl"))
    parser.add_argument("--reference", type=Path, default=(
        root / "experiments/numerical_range_fixed_point_v1/nrf05_full_linear_collapse.json"))
    args = parser.parse_args()
    print(json.dumps(verify_frozen_receipt(
        args.checkpoint_dir, args.events, args.reference), indent=2))


if __name__ == "__main__":
    main()

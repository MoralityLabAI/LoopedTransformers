"""COL-OBS-01: exact finite readout ceilings for recurrent observers.

No model is fitted. Every reported ceiling uses all target labels in the
specified finite population and is an oracle bound for that representation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import gzip
import json
from math import gcd
from pathlib import Path

try:
    from .collatz_core import analyze, branch_history, modular_labels
except ImportError:
    from collatz_core import analyze, branch_history, modular_labels

DEPTHS = (0, 1, 2, 4, 5, 8, 16)
FAMILIES = ("quotient_repeat", "branch_prefix", "branch_exponent_prefix",
            "branch_exponent_bag")
SCHEMA = "col-observer-gym-v1"


def ceiling(labels: dict[int, int], observations: dict[int, object]) -> dict:
    if set(labels) != set(observations):
        raise ValueError("one observation is required for every labeled state")
    cells: dict[object, Counter] = defaultdict(Counter)
    for n, target in labels.items():
        cells[observations[n]][target] += 1
    correct = sum(max(c.values()) for c in cells.values())
    cross_basin_pairs = sum((sum(c.values()) ** 2 - sum(v * v for v in c.values())) // 2
                            for c in cells.values())
    return {"correct": correct, "total": len(labels), "cells": len(cells),
            "ambiguous_cells": sum(len(c) > 1 for c in cells.values()),
            "cross_basin_pairs": cross_basin_pairs,
            "accuracy_fraction": str(Fraction(correct, len(labels)))}


def observer(core, quotient: dict[int, int], histories: dict[int, tuple],
             family: str, depth: int) -> dict[int, object]:
    result = {}
    for n in core.states:
        q = quotient[n % core.d]
        prefix = histories[n][:depth]
        if family == "quotient_repeat":
            observation = (q,) * (depth + 1)
        elif family == "branch_prefix":
            observation = (q, tuple(u for u, _ in prefix))
        elif family == "branch_exponent_prefix":
            observation = (q, prefix)
        elif family == "branch_exponent_bag":
            observation = (q, tuple(sorted(prefix)))
        else:
            raise ValueError("unknown observer family")
        result[n] = observation
    return result


def matched_shuffle(core, quotient: dict[int, int], observation: dict[int, object],
                    seed: int) -> dict[int, object]:
    """Randomize observation assignments within q, preserving cell occupancies.

    This is an exact deterministic hash permutation, not an iid null draw or
    an intervention on a trained model. Every assignment has the same q and
    cell-size multiset as its matched observer.
    """
    groups = defaultdict(list)
    for n in core.states:
        groups[quotient[n % core.d]].append(n)
    result = {}
    for q, states in groups.items():
        shuffled = sorted(states, key=lambda n: hashlib.sha256(
            f"col-observer-sham-v1:{seed}:{core.d}:{n}".encode()).digest())
        for n, donor in zip(states, shuffled):
            result[n] = observation[donor]
    return result


def recover_seed(d: int, history: tuple[tuple[int, int], ...]) -> int | None:
    """Recover the seed from affine-prefix congruence once the modulus exceeds d.

    An explicit replay checks branch choices and exact valuations. Below the
    sufficient uniqueness threshold, return None even if another decoder could
    exploit additional branch constraints.
    """
    A, u, B = 0, 0, 0
    for triple, a in history:
        if triple not in (0, 1) or type(a) is not int or a < 2:
            raise ValueError("each emission needs a binary branch and at least two halvings")
        c = 3 if triple else 1
        B = c * B + (1 << A)
        A += a
        u += triple
    modulus = 1 << (A + 1)
    if modulus <= d:
        return None
    # The final accelerated state is odd, so the numerator equals 2**A modulo 2**(A+1).
    n = (((1 << A) - B * d) * pow(pow(3, u, modulus), -1, modulus)) % modulus
    if n <= 0 or n > d or n % 2 != 1:
        return None
    return n if branch_history(n, d, len(history)) == history else None


def first_collision(core, q: dict[int, int]) -> dict | None:
    representatives = {}
    for n in core.states:
        key = q[n % core.d]
        if key in representatives:
            previous = representatives[key]
            if core.basin[n] != core.basin[previous]:
                return {"left": previous, "right": n, "quotient": key,
                        "left_basin": core.basin[previous], "right_basin": core.basin[n]}
        else:
            representatives[key] = n
    return None


def run_parameter(d: int) -> dict:
    core = analyze(d)
    quotient = modular_labels(d)
    histories = {n: branch_history(n, d, max(DEPTHS)) for n in core.states}
    rows = []
    for family in FAMILIES:
        for depth in DEPTHS:
            observations = observer(core, quotient, histories, family, depth)
            row = {"family": family, "depth": depth,
                   "emitted_pairs_per_case": depth,
                   "ceiling": ceiling(core.basin, observations)}
            # Keep all four fixed controls, including ties.
            row["matched_shuffles"] = [
                {"seed": seed, **ceiling(core.basin, matched_shuffle(
                    core, quotient, observations, seed))}
                for seed in (11, 29, 47, 71)
            ]
            rows.append(row)
    return {"parameter": d, "states": len(core.states), "cycles": len(core.cycles),
            "quotient_classes_on_core": len({quotient[n % d] for n in core.states}),
            "gcd_ceiling": ceiling(core.basin, {n: gcd(n, d) for n in core.states}),
            "full_state_ceiling": ceiling(core.basin, {n: n for n in core.states}),
            "first_quotient_collision": first_collision(core, quotient),
            "seed_recovery_by_depth": [{"depth": depth,
                "correct": sum(recover_seed(d, histories[n][:depth]) == n for n in core.states),
                "total": len(core.states)} for depth in DEPTHS], "observers": rows}


def summarize(results: list[dict]) -> list[dict]:
    summary = []
    for family in FAMILIES:
        for depth in DEPTHS:
            rows = [next(row for row in result["observers"]
                         if row["family"] == family and row["depth"] == depth)
                    for result in results]
            macro = sum((Fraction(row["ceiling"]["correct"], row["ceiling"]["total"])
                         for row in rows), Fraction()) / len(rows)
            correct = sum(row["ceiling"]["correct"] for row in rows)
            total = sum(row["ceiling"]["total"] for row in rows)
            summary.append({"family": family, "depth": depth,
                            "correct": correct, "total": total,
                            "micro_fraction": str(Fraction(correct, total)),
                            "macro_fraction": str(macro),
                            "perfect_parameters": sum(row["ceiling"]["correct"] == row["ceiling"]["total"]
                                                      for row in rows),
                            "sham_correct_by_seed": [sum(row["matched_shuffles"][i]["correct"]
                                                         for row in rows) for i in range(4)]})
    return summary


def run(max_d: int = 1025, anchor: int = 2021) -> dict:
    if type(max_d) is not int or max_d < 1 or max_d > 10001:
        raise ValueError("max_d must be in [1,10001]")
    parameters = [d for d in range(1, max_d + 1, 4) if gcd(d, 6) == 1]
    results = [run_parameter(d) for d in parameters]
    return {"schema": SCHEMA, "evidence_class": "exact finite domain",
            "selection_exposure": "exploratory; all finite targets used by ceiling decoders",
            "sampling_unit_for_future_neural_work": "parameter family, with checkpoint seed nested",
            "population": "uniform over positive odd n <= d within each parameter",
            "parameters": parameters, "n_units": len(parameters),
            "n_cases": sum(row["states"] for row in results),
            "core_sha256": hashlib.sha256(Path(__file__).with_name("collatz_core.py").read_bytes()).hexdigest(),
            "summary": summarize(results), "results": results,
            "known_anchor": {"selection": "previously inspected stress case, outside sweep",
                             "result": run_parameter(anchor)}}


def export_cases(path: Path, parameters: list[int], depth: int = 16) -> int:
    """Export observer inputs and evaluation-only labels for adapter development."""
    count = 0
    with path.open("w") as handle:
        for d in parameters:
            core = analyze(d)
            q = modular_labels(d)
            for n in core.states:
                row = {"case_id": f"d{d}:n{n}", "parameter_id": d,
                       "inputs": {"initial_state": n, "quotient": q[n % d],
                                  "branch_exponent_emissions": branch_history(n, d, depth)},
                       "evaluation_only": {"basin_minimum": core.basin[n]}}
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-d", type=int, default=1025)
    parser.add_argument("--anchor", type=int, default=2021)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check", type=Path, help="replay and compare an existing receipt")
    parser.add_argument("--summary", type=Path, help="write a readable aggregate receipt")
    parser.add_argument("--export", type=Path, help="optional public development JSONL")
    args = parser.parse_args()
    receipt = run(args.max_d, args.anchor)
    if args.check:
        data = args.check.read_bytes()
        if args.check.suffix == ".gz":
            data = gzip.decompress(data)
        if receipt != json.loads(data):
            raise SystemExit("receipt replay differs")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        data = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
        args.out.write_bytes(gzip.compress(data, mtime=0) if args.out.suffix == ".gz" else data)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        brief = {key: value for key, value in receipt.items() if key != "results"}
        args.summary.write_text(json.dumps(brief, indent=2, sort_keys=True) + "\n")
    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        export_cases(args.export, receipt["parameters"])
    print(json.dumps({"n_units": receipt["n_units"], "n_cases": receipt["n_cases"],
                      "receipt_replay": "passed" if args.check else "generated"}))


if __name__ == "__main__":
    main()

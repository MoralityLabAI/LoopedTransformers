"""NRF-05: post-hoc Pareto crossover of frozen NRF-04 decoder measurements.

Every random arm is evaluated as a complete (task MSE, squared leakage)
pair. The median is applied *after* forming each arm's total objective.
Uses exact fractions from the decimal JSON values for all breakpoints.
This is diagnostic analysis of already exposed evaluation data, and may
only motivate a separately registered future data split.
"""
from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path


def loss_line(arm: dict) -> tuple[Fraction, Fraction]:
    e = Fraction(str(arm["heldout_mse"]))
    leakage = Fraction(str(arm["local_fiber_leakage"]))
    if e < 0 or leakage < 0:
        raise ValueError("nonnegative measures required")
    return e, leakage * leakage


def median(lines: list[tuple[Fraction, Fraction]], lam: Fraction) -> Fraction:
    if not lines or len(lines) % 2:
        raise ValueError("nonempty even-length random control set required")
    values = sorted(e + lam * leak_sq for e, leak_sq in lines)
    n = len(values)
    return (values[n // 2 - 1] + values[n // 2]) / 2


def first_crossover(supervised: dict, random_controls: list[dict]):
    """Smallest lam>=0 such that J_supervised >= median(J_random).

    Random-line order can change as lambda varies. Partition at every
    pairwise intersection, then solve the piecewise-affine difference
    exactly; a median of standalone E and L does not define a comparator.
    """
    if len(random_controls) != 8:
        raise ValueError("eight frozen controls required")
    sv = loss_line(supervised)
    lines = [loss_line(arm) for arm in random_controls]
    difference = lambda lam: (
        sv[0] + lam * sv[1] - median(lines, lam)
    )
    knots = {Fraction(0)}
    for i, (e1, s1) in enumerate(lines):
        for e2, s2 in lines[i + 1:]:
            if s1 == s2:
                continue
            cut = (e2 - e1) / (s1 - s2)
            if cut > 0:
                knots.add(cut)
    knots = sorted(knots)
    prev = knots[0]
    prev_val = difference(prev)
    if prev_val >= 0:
        return prev
    for nxt in knots[1:]:
        next_val = difference(nxt)
        if next_val >= 0:
            return prev + (nxt - prev) * (-prev_val) / (next_val - prev_val)
        prev, prev_val = nxt, next_val
    slope = difference(prev + 1) - prev_val
    return prev - prev_val / slope if slope > 0 else None


def analyze_records(files: list[Path]):
    rows = []
    for path in files:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("study_id") != "NRF-04-semantic-fiber-decoder-v1":
            raise ValueError("unexpected study source")
        for row in receipt["rows"]:
            arm = row["arms"]
            controls = [arm[f"random_{k}"] for k in range(8)]
            cross = first_crossover(arm["supervised"], controls)
            if cross is None:
                raise ValueError(f"no positive crossover: {row['id']}")
            rows.append({
                "id": row["id"], "rounds": row["rounds"],
                "seed": row["seed"], "exposure": row["exposure"],
                "crossover_lambda": float(cross),
                "observed_supervised_better_at_zero": (
                    float(loss_line(arm["supervised"])[0])
                    < float(median([loss_line(c) for c in controls], Fraction(0)))
                ),
            })
    if len(rows) != 12 or len({r["id"] for r in rows}) != 12:
        raise ValueError("expected twelve distinct original cells")
    rows.sort(key=lambda v: (v["rounds"], v["seed"], v["exposure"]))
    return {
        "study_id": "NRF-05-posthoc-Pareto-crossover-v1",
        "rows": rows,
        "claim_boundary": (
            "This is a post-hoc objective sensitivity analysis on previously "
            "exposed test data. Lambda is a unit-dependent diagnostic; any "
            "candidate training intervention needs new, sealed evaluation data."
        ),
    }


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r16", type=Path, default=(
        root / "experiments/numerical_range_fixed_point_v1/nrf04_r16_connector_replay.json"))
    parser.add_argument("--r32", type=Path, default=(
        root / "experiments/numerical_range_fixed_point_v1/nrf04_r32_connector_replay.json"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = analyze_records([args.r16, args.r32])
    payload = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()

"""Dependency-free exact arithmetic checker for MOLT-16 cut feasibility.

Inputs are exact dyadics encoded as canonical Python float.hex strings.
The mathematical model fixes those values as real numbers. Neither the
floating-point execution nor the origin of the input data is certified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from fractions import Fraction as F
from pathlib import Path

SCHEMA = "molt17-cut-problems-v1"
BITS = 80
SLACK = F(1, 10**8)
ZERO = F(0)
ONE = F(1)


def dyadic(value: str) -> F:
    if not isinstance(value, str) or len(value) > 32:
        raise ValueError("canonical finite binary64 hex required")
    x = float.fromhex(value)
    if not math.isfinite(x) or x.hex() != value:
        raise ValueError("noncanonical or nonfinite binary64")
    return F.from_float(x)


def vector(values, n):
    if not isinstance(values, list) or len(values) != n:
        raise ValueError("vector dimension mismatch")
    return tuple(dyadic(x) for x in values)


def matrix(values, n, m):
    if not isinstance(values, list) or len(values) != n:
        raise ValueError("matrix dimension mismatch")
    return tuple(vector(row, m) for row in values)


def dot(a, b):
    return sum((x * y for x, y in zip(a, b, strict=True)), ZERO)


def norm2(a):
    return dot(a, a)


def mv(a, x):
    return tuple(dot(row, x) for row in a)


def transpose(a):
    return tuple(zip(*a, strict=True))


def center(x):
    mean = sum(x, ZERO) / len(x)
    return tuple(v - mean for v in x)


def inverse(a):
    """Exact Gauss-Jordan inverse. Fail on genuinely dependent columns."""
    n = len(a)
    rows = [list(row) + [F(i == j) for j in range(n)] for i, row in enumerate(a)]
    for j in range(n):
        pivot = next((i for i in range(j, n) if rows[i][j]), None)
        if pivot is None:
            raise ValueError("singular support Gram matrix")
        rows[j], rows[pivot] = rows[pivot], rows[j]
        p = rows[j][j]
        rows[j] = [v / p for v in rows[j]]
        for i in range(n):
            if i != j and rows[i][j]:
                v = rows[i][j]
                rows[i] = [x - v * y for x, y in zip(rows[i], rows[j], strict=True)]
    return tuple(tuple(row[n:]) for row in rows)


def gram_inverse(columns):
    return inverse(tuple(tuple(dot(x, y) for y in columns) for x in columns))


def projected_norm2(x, columns, inv):
    rhs = mv(columns, x)
    return dot(rhs, mv(inv, rhs))


def sqrt_bounds(x: F, bits: int = BITS) -> tuple[F, F]:
    """Outward dyadic square-root enclosure using integer arithmetic only."""
    if x < 0 or not 1 <= bits <= 256:
        raise ValueError("invalid square-root enclosure")
    denominator = 1 << bits
    k = math.isqrt((x.numerator << (2 * bits)) // x.denominator)
    lo = F(k, denominator)
    hi = lo if lo * lo == x else F(k + 1, denominator)
    if not lo * lo <= x <= hi * hi:
        raise ArithmeticError("square-root enclosure failed")
    return lo, hi


def fraction_text(x):
    return f"{x.numerator}/{x.denominator}"


class Chart:
    def __init__(self, q):
        self.q = q
        self.columns = transpose(q)
        self.inverse = gram_inverse(self.columns)
        self.centered_columns = tuple(center(c) for c in self.columns)
        try:
            self.centered_inverse = gram_inverse(self.centered_columns)
        except ValueError:
            # Full centered ambient space is a conservative superset.
            self.centered_inverse = None

    def envelope(self, z, radius, eps):
        """Exact denominator bounds for ||d|| <= R and d in range(Q).

        No orthonormality assumption is made about the binary64 chart.
        Centering is a contraction, so ||C d|| <= ||d||.
        """
        total = norm2(z)
        parallel = (projected_norm2(z, self.centered_columns, self.centered_inverse)
                    if self.centered_inverse is not None else total)
        perpendicular = total - parallel
        if min(parallel, perpendicular) < 0:
            raise ArithmeticError("invalid exact orthogonal decomposition")
        pl, pu = sqrt_bounds(parallel)
        low2 = (perpendicular + max(ZERO, pl - radius)**2) / len(z) + eps
        high2 = (perpendicular + (pu + radius)**2) / len(z) + eps
        return sqrt_bounds(low2)[0], sqrt_bounds(high2)[1]

    def support_squared(self, linear):
        # Supremum of linear dot d over physical ball in range(Q) is
        # radius * sqrt(||Proj_range(Q) linear||^2).
        result = projected_norm2(linear, self.columns, self.inverse)
        if result < 0:
            raise ArithmeticError("negative exact support norm")
        return result


class Readout:
    def __init__(self, weight, gamma, bias, eps):
        self.weight, self.gamma, self.bias, self.eps = weight, gamma, bias, eps
        if eps <= 0:
            raise ValueError("positive LayerNorm epsilon required")
        self.n = len(gamma)
        self.m = len(weight)

    def contrasts(self, winner):
        if type(winner) is not int or not 0 <= winner < self.m:
            raise ValueError("invalid target class")
        result = []
        for j, row in enumerate(self.weight):
            if j != winner:
                w = tuple(a - b for a, b in zip(self.weight[winner], row, strict=True))
                result.append((tuple(a * b for a, b in zip(w, self.gamma, strict=True)),
                               dot(w, self.bias)))
        return result


def check_case(readout, chart, h, radius, winner, dual, coeff, raw):
    """Check exact primal and dual witnesses at a common physical dose.

    The tested cap is the binary64 radius plus 1e-8, matching the
    previous pilot's numerical allowance. A primal is a strict legal
    winner, not a certificate about a future suffix or a gold label.
    """
    cap = radius + SLACK
    if radius < 0 or norm2(raw) > cap**2:
        raise ValueError("raw write outside declared physical budget")
    if any(x < 0 for x in dual) or len(dual) != readout.m - 1:
        raise ValueError("nonnegative dual weights required")
    z = center(h)
    rows = readout.contrasts(winner)
    low, high = chart.envelope(z, cap, readout.eps)
    if not 0 < low <= high:
        raise ArithmeticError("invalid denominator bounds")
    # Necessary RHS for each rival. The weighted inequality is computed
    # in ambient coordinates, retaining the exact physical norm constraint.
    lhs = sum((l * (min(-b * low, -b * high) - dot(a, z))
               for l, (a, b) in zip(dual, rows, strict=True)), ZERO)
    g = center(tuple(sum((l * a[i] for l, (a, _) in zip(dual, rows, strict=True)), ZERO)
                     for i in range(readout.n)))
    support2 = chart.support_squared(g)
    separation2 = lhs**2 - cap**2 * support2
    obstructed = lhs > 0 and separation2 > 0
    # Reconstruct the proposed write using the exact stored chart and
    # coefficient values; no rounded matrix product is trusted.
    d = mv(chart.q, coeff)
    dose2 = norm2(d)
    reached = center(tuple(x + y for x, y in zip(h, d, strict=True)))
    dl, du = sqrt_bounds(norm2(reached) / len(h) + readout.eps)
    lower = min(dot(a, reached) + min(b * dl, b * du) for a, b in rows)
    primal = dose2 <= cap**2 and lower > 0
    if obstructed and primal:
        raise ArithmeticError("exact primal and dual contradiction")
    return {
        "status": "exact_primal" if primal else "exact_obstruction" if obstructed else "unresolved",
        "exact_primal": primal,
        "exact_obstruction": obstructed,
        "radius_with_slack": fraction_text(cap),
        "necessary_lhs": fraction_text(lhs),
        "support_squared": fraction_text(support2),
        "dual_squared_separation": fraction_text(separation2),
        "primal_dose_squared": fraction_text(dose2),
        "primal_margin_numerator_lower": fraction_text(lower),
        "denominator_bounds": [fraction_text(low), fraction_text(high)],
    }


def verify(bundle):
    if bundle.get("schema") != SCHEMA:
        raise ValueError("unexpected problem schema")
    n, m, k = (bundle[name] for name in ("width", "legal_classes", "rank"))
    if (any(type(x) is not int for x in (n, m, k))
            or not 2 <= m <= 32 or not 1 <= k <= n <= 128 or k > 32):
        raise ValueError("unsupported exact problem dimensions")
    readout = Readout(matrix(bundle["weight"], m, n), vector(bundle["gamma"], n),
                      vector(bundle["bias"], n), dyadic(bundle["epsilon"]))
    if set(bundle["charts"]) != {"native", "observer", "random"}:
        raise ValueError("incomplete chart roster")
    charts = {name: Chart(matrix(q, n, k)) for name, q in bundle["charts"].items()}
    records = bundle["records"]
    if not isinstance(records, list) or not 1 <= len(records) <= 384:
        raise ValueError("invalid problem roster")
    results = []
    seen = set()
    for row in records:
        key = (row["root"], row["kind"], row["chart"])
        if (key in seen or row["kind"] not in {"same", "changed"}
                or row["chart"] not in charts or not isinstance(row["root"], str)):
            raise ValueError("invalid or duplicate case identity")
        seen.add(key)
        result = check_case(readout, charts[row["chart"]], vector(row["h"], n),
                            dyadic(row["radius"]), row["winner"], vector(row["dual"], m - 1),
                            vector(row["coeff"], k), vector(row["raw"], n))
        results.append({"root": row["root"], "kind": row["kind"], "chart": row["chart"],
                        "parent_status": row["parent_status"], **result})
    roots = {r["root"] for r in records}
    if seen != {(r, kind, chart) for r in roots for kind in ("same", "changed") for chart in charts}:
        raise ValueError("incomplete paired chart roster")
    summary = []
    for chart in charts:
        for kind in ("changed", "same"):
            group = [r for r in results if r["chart"] == chart and r["kind"] == kind]
            summary.append({"chart": chart, "kind": kind, "rows": len(group),
                            **{name: sum(r["status"] == name for r in group)
                               for name in ("exact_primal", "exact_obstruction", "unresolved")}})
    return {"schema": "molt17-exact-cut-check-v1", "status": "exact_rational_model_checked",
            "world_roots": len(roots), "summary": summary, "rows": results,
            "radius_slack": fraction_text(SLACK), "sqrt_fraction_bits": BITS,
            "input_origin_verified": False, "native_float_execution_certified": False,
            "future_suffix_certified": False, "promotion_authorized": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--problems", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.problems.stat().st_size > 32 * 1024 * 1024:
        p.error("problem bundle exceeds fixture size cap")
    raw = args.problems.read_bytes()
    result = verify(json.loads(raw))
    result["problems_sha256"] = hashlib.sha256(raw).hexdigest()
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, sort_keys=True))


if __name__ == "__main__":
    main()

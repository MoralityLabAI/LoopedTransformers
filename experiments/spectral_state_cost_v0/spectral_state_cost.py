"""SPC-01: exact phase geometry and sparsu Fourier recovery for loop probes.

The Fourier transform is O(n**2) on purpose: this is a small, deterministic
reference oracle, independent of FFT implementations and model checkpoints.
No OpenAI paper's claimed algorithm or state lower bound is assumed.
"""

from __future__ import annotations

import argparse
import cmath
import itertools
import json
import math
from typing import Sequence


def dft(signal: Sequence[complex]) -> list[complex]:
    n = len(signal)
    if n == 0:
        raise ValueError("empty input")
    return [sum(complex(x) * cmath.exp(-2j * math.pi * f * t / n)
                for t, x in enumerate(signal)) for f in range(n)]


def idft(spectrum: Sequence[complex]) -> list[complex]:
    n = len(spectrum)
    if n == 0:
        raise ValueError("empty spectrum")
    return [sum(complex(a) * cmath.exp(2j * math.pi * f * t / n)
                for f, a in enumerate(spectrum)) / n for t in range(n)]


def validate_phases(n: int, d: int, offsets: Sequence[int]) -> tuple[int, tuple[int, ...]]:
    if not isinstance(n, int) or not isinstance(d, int) or n < 1 or d < 1 or n % d:
        raise ValueError("n and d must be positive integers with d dividing n")
    phases = tuple(offsets)
    if any(not isinstance(s, int) or not 0 <= s < d for s in phases):
        raise ValueError("phase offsets must lie in [0,d)")
    if len(phases) != len(set(phases)):
        raise ValueError("duplicate phase offset")
    return n // d, phases


def sample_phases(signal: Sequence[complex], d: int,
                  offsets: Sequence[int]) -> list[list[complex]]:
    m, phases = validate_phases(len(signal), d, offsets)
    return [[complex(signal[s + d * k]) for k in range(m)] for s in phases]


def alias_formula(spectrum: Sequence[complex], d: int,
                  offset: int, j: int) -> complex:
    n = len(spectrum)
    m, phases = validate_phases(n, d, [offset])
    if not 0 <= j < m:
        raise ValueError("alias frequency is outside [0,n/d)")
    s = phases[0]
    return sum(cmath.exp(2j * math.pi * (j + r * m) * s / n)
               * complex(spectrum[j + r * m]) for r in range(d)) / d


def measurement_matrix(n: int, d: int,
                       offsets: Sequence[int]) -> list[list[complex]]:
    m, phases = validate_phases(n, d, offsets)
    return [[complex(int(i == s + d * k)) for i in range(n)]
            for s in phases for k in range(m)]


def matrix_rank(rows: Sequence[Sequence[complex]], tol: float = 1e-9) -> int:
    if not rows:
        return 0
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("ragged matrix")
    a = [[complex(x) for x in row] for row in rows]
    pivot_row = 0
    for col in range(width):
        if pivot_row == len(a):
            break
        best = max(range(pivot_row, len(a)), key=lambda i: abs(a[i][col]))
        if abs(a[best][col]) < tol:
            continue
        a[pivot_row], a[best] = a[best], a[pivot_row]
        pivot = a[pivot_row][col]
        for i in range(pivot_row + 1, len(a)):
            factor = a[i][col] / pivot
            for j in range(col, width):
                a[i][j] -= factor * a[pivot_row][j]
        pivot_row += 1
    return pivot_row


def rank_budget(n: int, d: int, offsets: Sequence[int]) -> dict[str, int]:
    m, phases = validate_phases(n, d, offsets)
    return {"observed_rank": m * len(phases),
            "blind_dimension": n - m * len(phases),
            "full_dimension": n}


def block_hankel(outputs: Sequence[Sequence[complex]]) -> list[list[complex]]:
    if not outputs or not outputs[0]:
        raise ValueError("need at least one nonempty output period")
    m = len(outputs[0])
    if any(len(seq) != m for seq in outputs):
        raise ValueError("all sampled channels need a common period")
    return [[complex(seq[(i + j) % m]) for j in range(m)]
            for seq in outputs for i in range(m)]


def spectral_state_order(outputs: Sequence[Sequence[complex]],
                         tol: float = 1e-8) -> int:
    if not outputs or not outputs[0]:
        raise ValueError("need at least one output period")
    periods = [dft(seq) for seq in outputs]
    m = len(periods[0])
    if any(len(seq) != m for seq in periods):
        raise ValueError("inconsistent periods")
    return sum(any(abs(spec[j]) > tol for spec in periods) for j in range(m))


def solve_square(a: Sequence[Sequence[complex]], b: Sequence[complex],
                 tol: float = 1e-11) -> list[complex]:
    n = len(b)
    if len(a) != n or any(len(row) != n for row in a):
        raise ValueError("square solver needs n by n matrix")
    mat = [[complex(x) for x in row] + [complex(y)] for row, y in zip(a, b)]
    for col in range(n):
        i = max(range(col, n), key=lambda k: abs(mat[k][col]))
        if abs(mat[i][col]) < tol:
            raise ValueError("singular support")
        mat[col], mat[i] = mat[i], mat[col]
        z = mat[col][col]
        for j in range(col, n + 1):
            mat[col][j] /= z
        for k in range(n):
            if k != col:
                z = mat[k][col]
                for j in range(col, n + 1):
                    mat[k][j] -= z * mat[col][j]
    return [mat[i][-1] for i in range(n)]


def sparse_recover(n: int, d: int, offsets: Sequence[int],
                   outputs: Sequence[Sequence[complex]],
                   k: int, tol: float = 1e-8) -> list[complex]:
    """Recover unknown <=k Fourier modes per alias class from phases 0..r-1.

    Uses exhaustive support enumeration, intended for d <= 8 and CPU tests.
    It is a reference identifiability checker, not a scalable sparse FFT.
    """
    m, phases = validate_phases(n, d, offsets)
    if not isinstance(k, int) or k < 1 or k > d:
        raise ValueError("require 1 <= k <= d")
    if phases != tuple(range(len(phases))):
        raise ValueError("sparse recovery requires consecutive offsets starting at zero")
    if len(phases) < min(2 * k, d):
        raise ValueError("insufficient phases for uniform k-sparsu recovery")
    if len(outputs) != len(phases) or any(len(y) != m for y in outputs):
        raise ValueError("observation shape does not match phase schedule")
    seen = [dft(y) for y in outputs]
    recovered = [0j] * n
    for j in range(m):
        b = [d * seen[s][j] * cmath.exp(-2j * math.pi * j * s / n)
             for s in phases]
        if max(abs(x) for x in b) < tol:
            continue
        found = False
        for size in range(1, k + 1):
            for support in itertools.combinations(range(d), size):
                a = [[cmath.exp(2j * math.pi * rr * s / d)
                      for rr in support] for s in range(size)]
                try:
                    amps = solve_square(a, b[:size])
                except ValueError:
                    continue
                if any(abs(v) < tol for v in amps):
                    continue
                residual = max(abs(sum(cmath.exp(2j * math.pi * rr * s / d) * v
                                       for rr, v in zip(support, amps)) - b[s])
                               for s in phases)
                if residual <= tol * (1 + max(abs(v) for v in b)):
                    for rr, amp in zip(support, amps):
                        recovered[j + rr * m] = amp
                    found = True
                    break
            if found:
                break
        if not found:
            raise ValueError(f"alias group {j} violates k-sparsu/noiseless assumptions")
    return recovered


def demo() -> dict[str, float | int | str]:
    n, d, k = 24, 6, 2
    offsets = list(range(min(2 * k, d)))
    spectrum = [0j] * n
    spectrum[0] = 0.5
    spectrum[2] = 1.2 - 0.4j
    spectrum[22] = spectrum[2].conjugate()
    spectrum[5] = -0.7 + 0.1j
    spectrum[19] = spectrum[5].conjugate()
    signal = idft(spectrum)
    observed = sample_phases(signal, d, offsets)
    recovered = sparse_recover(n, d, offsets, observed, k)
    invisible = [0j] * n
    invisible[5] = 1
    budget = rank_budget(n, d, offsets)
    return {"campaign": "SPC-01", "n": n, "stride": d,
            "max_modes_per_alias": k, "phase_count": len(offsets),
            "measurement_rank": budget["observed_rank"],
            "measurement_nullity": budget["blind_dimension"],
            "true_spectral_state_order": spectral_state_order([signal]),
            "probed_joint_state_order": spectral_state_order(observed),
            "max_fourier_recovery_error": round(max(abs(a - b) for a, b in
                                                   zip(spectrum, recovered)), 12),
            "invisible_true_state_order": spectral_state_order([invisible]),
            "invisible_probed_state_order": spectral_state_order(
                sample_phases(invisible, d, offsets))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="write JSON diagnostics")
    _ = parser.parse_args()
    print(json.dumps(demo(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

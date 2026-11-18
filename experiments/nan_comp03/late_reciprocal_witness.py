"""Exact source-only witness for a possible NAN-COMP-04 late-evidence protocol.

Pure integer series arithmetic. This constructs no experiment cohort and
does not read or regenerate any protected test. Under f=1+a*x^m+c*x^(m+2)
with m odd and >=3, N(f) and 1/f agree through degree 2m-1. Their
first two differences are a^2 at degree 2m and 2ac at degree 2m+2.
"""
from __future__ import annotations


def power_series_reciprocal(coeff: list[int]) -> list[int]:
    """Truncated formal inverse with a0=1, exact coefficient arithmetic."""
    if not coeff or coeff[0] != 1:
        raise ValueError("integer unit-constant series required")
    out = [1]
    for n in range(1, len(coeff)):
        out.append(-sum(coeff[k] * out[n-k] for k in range(1, n+1)))
    return out


def source(m: int, a: int, c: int) -> list[int]:
    if type(m) is not int or m < 3 or m % 2 != 1:
        raise ValueError("odd m >=3")
    if type(a) is not int or type(c) is not int or a == 0 or c == 0:
        raise ValueError("nonzero integer a,c")
    f = [0] * (2*m+3)
    f[0] = 1
    f[m] = a
    f[m+2] = c
    return f


def four_operators(f: list[int]) -> dict[str, list[int]]:
    size = len(f)
    negate = [a * (-1 if i % 2 else 1) for i,a in enumerate(f)]
    square_substitution = [f[i//2] if i % 2 == 0 else 0 for i in range(size)]
    return {
        "I": list(f),
        "N": negate,
        "S": square_substitution,
        "R": power_series_reciprocal(f),
    }


def witness(m: int, a: int, c: int) -> dict:
    f = source(m,a,c)
    data = four_operators(f)
    n,r = data["N"],data["R"]
    if n[:2*m] != r[:2*m]:
        raise AssertionError("N/R collision through 2m-1 failed")
    if r[2*m] - n[2*m] != a*a:
        raise AssertionError("R minus N at 2m must equal a^2")
    if r[2*m+2] - n[2*m+2] != 2*a*c:
        raise AssertionError("R minus N at 2m+2 must equal 2ac")
    if len({tuple(values[:2*m+1]) for values in data.values()}) != 4:
        raise AssertionError("four operators must be unique by degree 2m")
    return {
        "m":m,"a":a,"c":c,
        "prefix_N_equals_R_through_degree":2*m-1,
        "delta_degree_2m":a*a,
        "delta_degree_2m_plus_2":2*a*c,
        "four_operators_distinct_by_degree":2*m,
        "full_series":data,
        "sealed_test_accessed":False,
    }


if __name__ == "__main__":
    for m in (3,5):
        for a in (-2,-1,1,2):
            for c in (-2,-1,1,2):
                x=witness(m,a,c)
                print(m,a,c,x["delta_degree_2m"],x["delta_degree_2m_plus_2"])

"""Compute an exact necessary nominal dose for a signed route contrast."""
from collections import defaultdict
from fractions import Fraction
import math

from experiments.nan_lattice53.run import next_neighbors


def minimum_step(weights, steps, fraction):
    bins = defaultdict(Fraction)
    for weight, step in zip(weights, steps, strict=True):
        if weight:
            bins[step] += weight
    if not bins:
        raise ValueError("Zero route contrast")
    need = sum(bins.values()) * fraction
    covered = Fraction(0)
    for step in sorted(bins):
        covered += bins[step]
        if covered >= need:
            return step
    raise RuntimeError("Complete positive support did not meet the registered threshold")


def signed_nominal_floor(h, target, cosine, tolerance, np):
    before, after = next_neighbors(h.astype(np.float32), np)
    steps = np.where(target > 0, after.astype(np.float64) - h, h - before.astype(np.float64))
    if not np.isfinite(steps).all() or float(steps.min()) <= 0:
        raise ValueError("Nonfinite or nonpositive adjacent BF16 step")
    squared = [Fraction(v)**2 for v in target.tolist()]
    cut = minimum_step(squared, steps.tolist(), Fraction(str(cosine))**2)
    radius_squared = sum(Fraction(v)**2 for v in h.tolist())
    return Fraction(cut)**2 / (radius_squared * (1 + Fraction(str(tolerance)))**2)


def proof_data(value):
    return {"nominal_dose_lower_bound": math.sqrt(float(value)),
            "exact_squared_nominal_bound": str(value)}


def verify(np):
    f = Fraction
    assert minimum_step([f(1), f(4), f(4)], [.01, .02, .02], f(1, 4)) == .02
    assert minimum_step([f(0), f(1)], [1, 2], f(1, 4)) == 2
    h = np.array([1., 1.])
    v = np.array([2., 0.])
    positive = signed_nominal_floor(h, v, .5, .03, np)
    negative = signed_nominal_floor(h, -v, .5, .03, np)
    assert positive == f(1, 128)**2 / (2*f("1.03")**2)
    assert positive == 4 * negative


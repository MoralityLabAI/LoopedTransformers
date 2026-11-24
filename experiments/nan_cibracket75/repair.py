"""Repair a discrete BF16 norm jump without weakening physical gates."""
from math import sqrt

from experiments.nan_carry63.algebra import round_bf16
from experiments.nan_cidose74.geometry import COSINE, DOSE, DOSE_TOLERANCE
from experiments.nan_delay39.instrument import Patch, prepare
from experiments.nan_dose38.run import MAX_MULTIPLE, STEPS, displacement


def compensate(base, direction, np):
    original = prepare(base, direction, DOSE, DOSE_TOLERANCE, COSINE, np)
    if original.eligible:
        return original, "original_ray", 0
    target = original.requested_l2
    length = float(np.linalg.norm(direction))
    if not target or not length:
        return original, "unqualified", 0

    def reached(scale):
        return displacement(base, unit[None], np.array([scale]), np)[0]

    unit = direction / length
    low, high = 0., MAX_MULTIPLE * target
    if np.linalg.norm(reached(high)) < target:
        return original, "unqualified", 0
    for _ in range(STEPS):
        middle = (low + high)/2
        if np.linalg.norm(reached(middle)) < target:
            low = middle
        else:
            high = middle
    lower, upper = reached(low), reached(high)
    fixed = lower != upper
    count = int(fixed.sum())
    complement = ~fixed
    base_rms = float(np.linalg.norm(base[complement]))
    candidates = []
    if count and base_rms:
        for endpoint in (lower, upper):
            locked = endpoint[fixed]
            remaining = target**2 - float(locked @ locked)
            if remaining <= 0:
                continue
            patch = prepare(base[complement], unit[complement], sqrt(remaining)/base_rms,
                            DOSE_TOLERANCE, 0., np)
            delta = endpoint.copy()
            delta[complement] = patch.after - patch.before
            actual = float(np.linalg.norm(delta))
            if not actual:
                continue
            cosine = float(delta @ unit / actual)
            error = abs(actual/target - 1)
            after = base + delta
            np.testing.assert_array_equal(after, round_bf16(after, np))
            if error <= DOSE_TOLERANCE and cosine >= COSINE:
                candidates.append((error, -cosine, Patch(base, after, target, actual, cosine, True)))
    if candidates:
        return min(candidates, key=lambda c: c[:2])[2], "compensated", count
    return original, "unqualified", count


def verify(np):
    base = np.array([13., .25, .25, .25])
    direction = np.array([1., .1, .1, .1])
    initial = prepare(base, direction, DOSE, DOSE_TOLERANCE, COSINE, np)
    assert not initial.eligible and initial.cosine >= COSINE
    repaired, mode, jumps = compensate(base, direction, np)
    assert repaired.eligible and mode == "compensated" and jumps == 1
    assert abs(repaired.realized_l2 / repaired.requested_l2 - 1) <= DOSE_TOLERANCE
    assert repaired.cosine >= COSINE
    assert compensate(base, np.zeros_like(direction), np)[1] == "unqualified"
    return {"initial_relative_dose_error": abs(initial.realized_l2 / initial.requested_l2 - 1),
            "compensated_relative_dose_error": abs(repaired.realized_l2 / repaired.requested_l2 - 1),
            "compensated_cosine": repaired.cosine}

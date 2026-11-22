"""Endpoint readout accounting under a fixed BF16 head."""
from fractions import Fraction
from itertools import permutations
from math import comb

from experiments.nan_lattice53.run import next_neighbors
from experiments.nan_readpath37.run import bf16

GROUPS = ("in_span", "off_span", "effective_carry")


def basis(rows, np):
    if np.linalg.matrix_rank(rows) != rows.shape[0]:
        raise ValueError("Write span does not have the registered rank")
    q, _ = np.linalg.qr(rows.T)
    np.testing.assert_allclose(q.T @ q, np.eye(rows.shape[0]), rtol=0, atol=1e-12)
    return q


def round_bf16(value, np):
    """Round the float64 coalition once onto the BF16 lattice."""
    mid = bf16(value, np)
    lo, hi = next_neighbors(mid, np)
    if not np.isfinite([lo, mid, hi]).all():
        raise ValueError("Coalition leaves the finite BF16 neighborhood")
    # The preliminary FP32 cast can introduce a false BF16 midpoint.
    # At a genuine tie it already supplies the ties-to-even result.
    return np.where(abs(value - lo) < abs(value - mid), lo,
                    np.where(abs(value - hi) < abs(value - mid), hi, mid))


def shapley(margins):
    """Use rational arithmetic on the eight rounded-head margins."""
    if len(margins) != 8:
        raise ValueError("Three groups require eight coalition values")
    scores = tuple(Fraction(float(value)) for value in margins)
    result = []
    for group in (1, 2, 4):
        result.append(sum((scores[mask | group] - scores[mask]) /
                          (3 * comb(2, mask.bit_count()))
                          for mask in range(8) if not (mask & group)))
    if sum(result) != scores[-1] - scores[0]:
        raise ArithmeticError("Exact rounded-head attribution did not close")
    return dict(zip(GROUPS, map(float, result), strict=True))


def carry_radius(h, np):
    """Conservative radius for a native BF16 add, including a possible FP32 intermediate."""
    lo, hi = (x.astype(np.float64) for x in next_neighbors(h.astype(np.float32), np))
    if not np.isfinite([lo, hi]).all():
        raise ValueError("Native BF16 state has a nonfinite successor or predecessor")
    bf16_radius = np.maximum(h - lo, hi - h) / 2
    f32_radius = np.maximum(abs(lo), abs(hi)) * 2**-24 + 2**-150
    return bf16_radius + f32_radius


def continuous(h, direction, eps, np):
    h_a, h_b = h
    alpha_a, alpha_b = 1 / np.sqrt(np.mean(h*h, axis=1) + float(eps))
    numerator_a, numerator_b = h @ direction
    mean_alpha = (alpha_a + alpha_b) / 2
    rms_term = float((numerator_a + numerator_b) / 2 * (alpha_a - alpha_b))
    swing = float(numerator_a * alpha_a - numerator_b * alpha_b)
    assert h_a.shape == h_b.shape == direction.shape
    return mean_alpha * direction, rms_term, swing


def margins(head, states, tokens, np):
    logits = head.logits(states, tokens).astype(np.float64)
    return logits[:, 0] - logits[:, 1]


def account(head, h, delta_y, projector, tokens, scaled_direction, observed, np):
    """Three groups close to the archived A/B contrast, then use the pinned head."""
    h_a, h_b = h
    in_span = (delta_y @ projector) @ projector.T
    off_span = delta_y - in_span
    carry = (h_a - h_b) - delta_y
    parts = (in_span, off_span, carry)
    # The two native endpoints are known exactly. Anchor both to avoid
    # turning a cancellation at a native zero into a nonzero BF16 value.
    np.testing.assert_allclose(h_b + sum(parts), h_a, rtol=0, atol=1e-12)
    coalitions = [h_b, h_b + in_span, h_b + off_span, h_b + delta_y,
                  h_a - delta_y, h_a - off_span, h_a - in_span, h_a]
    states = round_bf16(np.asarray(coalitions), np)
    if not np.array_equal(states[[7, 0]], h):
        raise ValueError("Coalition endpoints differ from the native BF16 states")
    results = margins(head, states, tokens, np)
    if (results[-1] - results[0]) != observed:
        raise ArithmeticError("Rounded-head coalition score differs from the native replay")
    return (dict(zip(GROUPS, (float(v @ scaled_direction) for v in parts), strict=True)),
            shapley(results), in_span, results)


def self_test(np):
    three = 3
    synthetic = [3 * bool(mask & 1) - 2 * bool(mask & 2) +
                 7 * bool(mask & 4) + 6 * ((mask & three) == three) for mask in range(8)]
    direct = shapley(synthetic)
    by_order = [Fraction(0)] * 3
    for order in permutations(range(3)):
        seen = 0
        for i in order:
            new = seen | (1 << i)
            by_order[i] += Fraction(synthetic[new] - synthetic[seen], 6)
            seen = new
    assert list(direct.values()) == by_order == [6, 1, 7]
    midpoint = 1 + 2**-8
    sample = np.array([midpoint - 2**-29, midpoint, midpoint + 2**-29])
    np.testing.assert_array_equal(round_bf16(sample, np), [1, 1, 1 + 2**-7])
    np.testing.assert_array_equal(round_bf16(-sample, np), [-1, -1, -1 - 2**-7])
    h = np.array([[4.0, 3.0], [1.0, 2.0]])
    direction = np.array([0.7, -1.3])
    scaled, rms, swing = continuous(h, direction, 1e-5, np)
    np.testing.assert_allclose((h[0] - h[1]) @ scaled + rms, swing, atol=1e-15, rtol=0)
    assert carry_radius(np.array([1.0]), np)[0] >= 2**-8

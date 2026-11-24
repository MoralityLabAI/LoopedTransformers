"""Independently enumerate the BF16 stencil and check the differential gate."""

from fractions import Fraction
from itertools import product
from math import isclose
from struct import pack, unpack

from experiments.nan_readpath37.run import bf16
from experiments.nan_shellrepair90.geometry import fixture as verify_repair

from .rounding import adjacent, assess, coordinate_obstruction, joint_round


def fmt(bit):
    return unpack("<f", pack("<I", bit << 16))[0]


def stencil(center):
    code = unpack("<I", pack("<f", center))[0] >> 16
    values = [fmt(code), fmt(code - 1), fmt(code + 1)]
    return [Fraction(v) for v in values]


def fixture(cfg, np):
    checks = verify_repair(np)
    centers = np.asarray([0., -0., 1., -1., 2., -2., fmt(1), fmt(0x8001), fmt(0x7F7F), fmt(0xFF7F)])
    tiny = fmt(1)
    down, up = adjacent(centers, np)
    assert down.tolist() == [-tiny, -tiny, .99609375, -1.0078125, 1.9921875, -2.015625, 0.,
                              -2*tiny, fmt(0x7F7E), fmt(0xFF7F)]
    assert up.tolist() == [tiny, tiny, 1.0078125, -.99609375, 2.015625, -1.9921875, 2*tiny, -0.,
                            fmt(0x7F7F), fmt(0xFF7E)]
    goals = np.asarray([-1., -.5, .5, 1.], dtype=np.float64)
    offsets = np.asarray([-5, -3, -1, 0, 1, 3, 5], dtype=np.float64) / 1024
    base, left_offset, right_offset = np.asarray(list(product(goals, offsets, offsets))).T
    left, right = base + left_offset, base + right_offset
    want = right - left
    (a, b), lower = joint_round(left, right, want, np)
    expected_sq = Fraction(0)
    for i, (x, y) in enumerate(zip(left, right, strict=True)):
        ends = [stencil(float(bf16(np.array([v]), np)[0])) for v in (x, y)]
        x, y = map(Fraction, (x, y))
        options = [((q - p - (y - x))**2, (p - x)**2 + (q - y)**2) for p, q in product(*ends)]
        best = min(options)
        actual = Fraction(a[i]), Fraction(b[i])
        assert ((actual[1] - actual[0] - (y - x))**2, (actual[0] - x)**2 + (actual[1] - y)**2) == best
        expected_sq += best[0]
    assert isclose(lower**2, float(expected_sq), rel_tol=1e-12)
    before = np.array([2., 2., 2.])
    ideal = [np.array([.125, 0., 0.]), np.array([.125, .015625, 0.])]
    rows = [before + delta for delta in ideal]
    good = assess(before, rows, ideal, .125, cfg, np, lower_bound=0.)
    assert good["qualified"] and not good["stencil_certified_incompatible"]
    unresolved = [ideal[0], np.array([.125, 2**-16, 0.])]
    found, bound = joint_round(before + unresolved[0], before + unresolved[1], unresolved[1] - unresolved[0], np)
    aliased = assess(before, found, unresolved, .125, cfg, np, lower_bound=bound)
    assert aliased["dose_qualified"] and aliased["exact_state_alias"] and aliased["stencil_certified_incompatible"]
    assert not aliased["qualified"]
    swapped = assess(before, rows[::-1], ideal, .125, cfg, np)
    assert swapped["dose_qualified"] and swapped["differential"]["cosine"] == -1 and not swapped["qualified"]
    axis = np.array([1., 0.])
    witness = coordinate_obstruction(np.array([2., 1.]), axis, .01, cfg, np)
    assert witness["coordinate"] == 0 and 2. < witness["disallowed_interval"][0]
    assert witness["disallowed_interval"][1] < witness["next_bf16_at_or_above_lower"] == 2.015625
    assert coordinate_obstruction(np.array([2., 1.]), -axis, .01, cfg, np) is not None
    assert coordinate_obstruction(np.array([2., 1.]), axis, .015625, cfg, np) is None
    return checks | {"fraction_stencil_coordinates_exact": len(left), "adjacent_boundary_coordinates": len(centers),
                     "exact_signed_pair_admitted": True, "dose_matched_alias_rejected": True,
                     "dose_matched_sign_reversal_rejected": True, "global_coordinate_obstruction_fixture": True}

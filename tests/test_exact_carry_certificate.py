from fractions import Fraction as F

import pytest

from lsa.exact_carry_certificate import Chart, Readout, check_case, dyadic, sqrt_bounds


def test_outward_square_roots_and_hex():
    for x in (F(0), F(2), F(1, 7), F(1, 10**100), F(49, 64)):
        l, u = sqrt_bounds(x)
        assert 0 <= l <= u
        assert l*l <= x <= u*u
        assert u-l <= F(1, 2**80)
    assert dyadic(float(0.1).hex()) == F.from_float(0.1)
    with pytest.raises(ValueError):
        dyadic("nan")


def test_physical_norm_nonorthonormal_chart():
    readout = Readout(((F(1), F(0), F(0)), (F(-1), F(0), F(0))),
                      (F(1),)*3, (F(0),)*3, F(1, 100))
    q = Chart(((F(2),), (F(0),), (F(0),)))
    h = (F(-2), F(1), F(1))
    small = check_case(readout, q, h, F(1), 0, (F(1),), (F(1, 2),),
                       (F(1), F(0), F(0)))
    assert small["status"] == "exact_obstruction"
    large = check_case(readout, q, h, F(4), 0, (F(1),), (F(2),),
                       (F(1), F(0), F(0)))
    assert large["status"] == "exact_primal"
    # A coefficient of 1 is not a physically legal write at radius 1.
    out = check_case(readout, q, h, F(1), 0, (F(0),), (F(1),),
                     (F(1), F(0), F(0)))
    assert out["status"] == "unresolved"
    with pytest.raises(ValueError):
        check_case(readout, q, h, F(1), 0, (F(-1),), (F(0),),
                   (F(1), F(0), F(0)))


def test_constant_direction_and_bias():
    # The support is the mean direction, so the centered write is zero.
    q = Chart(((F(1),), (F(1),), (F(1),)))
    assert q.centered_inverse is None
    w = ((F(1), F(0), F(0)), (F(0), F(0), F(0)))
    h = (F(-2), F(1), F(1))
    for bias, expected in ((F(0), "exact_obstruction"), (F(4), "exact_primal")):
        r = Readout(w, (F(1),)*3, (bias, F(0), F(0)), F(1, 100))
        out = check_case(r, q, h, F(10), 0, (F(1),), (F(0),), (F(0),)*3)
        assert out["status"] == expected


def test_strict_positive_margin_with_bias():
    q = Chart(((F(1),), (F(0),), (F(0),)))
    readout = Readout(((F(1), F(0), F(0)), (F(0), F(0), F(0))),
                      (F(1),)*3, (F(-4), F(0), F(0)), F(1, 100))
    r = check_case(readout, q, (F(2), F(-1), F(-1)), F(0), 0,
                   (F(1),), (F(0),), (F(0),)*3)
    assert r["status"] == "exact_obstruction"

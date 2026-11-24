"""Check the carrier rule, spherical chord and CAL selection independently."""

from experiments.nan_bankgap100.verify import fixture as previous_fixture

from .geometry import attained, measure, select, sphere_pair


def fixture(cfg, np):
    checked = previous_fixture(cfg, np)
    a, b = [np.asarray(x, dtype=np.float64) / np.sqrt(1.01) for x in ((1., -.1, 0., 0.), (1., .1, 0., 0.))]
    (a1, b1), shift1 = sphere_pair(a, b, 1, np)
    np.testing.assert_allclose([a1, b1], [a, b], atol=1e-15, rtol=0)
    assert shift1 <= 1e-15
    (a4, b4), shift4 = sphere_pair(a, b, 4, np)
    np.testing.assert_allclose([np.linalg.norm(a4), np.linalg.norm(b4)], [1., 1.], atol=1e-15, rtol=0)
    np.testing.assert_allclose(b4 - a4, 4*(b - a), atol=1e-15, rtol=0)
    assert shift4 > 0. and sphere_pair(a, b, 32, np) is None and sphere_pair(a, a, 1, np) is None

    before = np.asarray([13., .25, .25, 0.])
    axis = np.asarray([1., 0., 0., 0.])
    states, nominal, fraction = attained(before, axis, .008, cfg, np)
    assert nominal == .125 and fraction != .008
    assert states["+"][0] == 13.125 and states["-"][0] == 12.875
    clean = np.asarray([0., 0., 0., 128.])
    carrier = attained(clean, axis, .01, cfg, np)
    record = measure(clean, {"svd": axis, "t1_fit": a, "t2_fit": b}, carrier, 1, cfg, np)
    assert record["status"] == "measured" and all(x["joint"]["qualified"] for x in record["signs"].values())

    def candidate(gain, radius, roots, seed):
        return {"gain": gain, "median_reached_fraction": radius, "joint": {"faithful_roots": roots},
                "seed_fraction": seed}
    no = candidate(1, .005, 5, .005)
    preferred = candidate(2, .01, 6, .0125)
    rows = [no, candidate(4, .005, 8, .005), candidate(2, .02, 7, .015), preferred, candidate(2, .01, 8, .015)]
    lock = select(rows)
    assert lock == {"gain": 2, "seed_fraction": .0125, "cal_faithful_roots": 6, "cal_median_reached_fraction": .01}
    assert select([no]) is None
    return checked | {
        "gain_one_recovers_intended_unit_writers": True, "same_dose_amplified_signed_chord_verified": True,
        "overlarge_and_zero_chords_refused": True, "reached_radius_differs_from_seed": True,
        "both_signed_fit_pairs_pass_lattice_fixture": True, "deterministic_cal_lock_and_refusal": True,
    }

"""Construct a reachable carrier and evaluate a same-radius signed bank chord."""

from statistics import median

from experiments.nan_bankgap100.rounding import assess, joint_round, vector_metrics
from experiments.nan_readpath37.run import bf16


SEED_FRACTIONS = (.005, .0075, .01, .0125, .015, .0175, .02, .0225, .025, .03, .035, .04)
GAIN_LADDER = (1, 2, 4, 8, 16, 32)
REACHED_FRACTIONS = (.004, .042)
MIN_CAL = 6
SIGNS = (("+", 1), ("-", -1))


def unit_directions(directions, np):
    use = ("svd", "t1_fit", "t2_fit")
    norms = {name: float(np.linalg.norm(directions[name])) for name in use}
    if not all(norm > 0 and np.isfinite(norm) for norm in norms.values()):
        return None
    return {name: directions[name] / norms[name] for name in use}


def attained(before, svd, seed, cfg, np):
    radius = float(np.linalg.norm(before))
    if not radius:
        return None
    signed = [(name, sign*svd) for name, sign in SIGNS]
    states = {name: bf16(before + radius*seed*direction, np).astype(np.float64) for name, direction in signed}
    norms = [float(np.linalg.norm(state - before)) for state in states.values()]
    nominal = sum(norms)/2
    fraction = nominal/radius
    if not REACHED_FRACTIONS[0] <= fraction <= REACHED_FRACTIONS[1]:
        return None
    if abs(norms[0] - norms[1]) > cfg["max_contrast_dose_gap_fraction"]*nominal:
        return None
    for name, direction in signed:
        if not vector_metrics(states[name] - before, direction, nominal, cfg, np)["qualified"]:
            return None
    return states, nominal, fraction


def sphere_pair(a, b, gain, np):
    midpoint, chord = (a + b)/2, (b - a)*gain
    mid_norm = float(np.linalg.norm(midpoint))
    squared = float(chord @ chord) / 4
    if not mid_norm or not 0 < squared < 1:
        return None
    center = midpoint * (np.sqrt(1 - squared) / mid_norm)
    return (center - chord/2, center + chord/2), float(np.linalg.norm(center - midpoint))


def measure(before, dirs, carrier, gain, cfg, np):
    if dirs is None or carrier is None:
        return {"status": "carrier_unqualified"}
    stretched = sphere_pair(dirs["t1_fit"], dirs["t2_fit"], gain, np)
    if stretched is None:
        return {"status": "zero_or_oversized_chord"}
    (a, b), mid_shift = stretched
    states, nominal, fraction = carrier
    signs = {}
    for name, sign in SIGNS:
        ideals = [sign*nominal*v for v in (a, b)]
        centers = [states[name] + d - sign*nominal*dirs["svd"] for d in ideals]
        nearest = [bf16(x, np).astype(np.float64) for x in centers]
        gap = ideals[1] - ideals[0]
        joint, bound = joint_round(*centers, gap, np)
        signs[name] = {"nearest": assess(before, nearest, ideals, nominal, cfg, np),
                       "joint": assess(before, joint, ideals, nominal, cfg, np, lower_bound=bound)}
    return {"status": "measured", "reached_fraction": fraction, "ideal_midpoint_shift_over_dose": mid_shift,
            "signs": signs}


def select(rows):
    eligible = [r for r in rows if r["joint"]["faithful_roots"] >= MIN_CAL]
    if not eligible:
        return None
    chosen = min(eligible, key=lambda r: (r["gain"], r["median_reached_fraction"], r["seed_fraction"]))
    return {"gain": chosen["gain"], "seed_fraction": chosen["seed_fraction"],
            "cal_faithful_roots": chosen["joint"]["faithful_roots"],
            "cal_median_reached_fraction": chosen["median_reached_fraction"]}


def median_or_none(values):
    complete = [v for v in values if v is not None]
    return median(complete) if complete else None

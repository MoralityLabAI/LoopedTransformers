"""Use a shared carrier and audit the geometry of the actual pair difference."""

from itertools import product
from math import sqrt

from experiments.nan_delay39.instrument import prepare
from experiments.nan_readpath37.run import bf16
from experiments.nan_shellrepair90.geometry import repaired_candidates


GAINS = (1, 2, 4, 8)
FULL = ("svd", "t1_fit", "t2_fit")
METHODS = ("original_rays", "independent_repair", *(f"{kind}_x{gain}"
        for gain in GAINS for kind in ("nearest", "joint_stencil")))
MAX_BF16 = float.fromhex("0x1.fep+127")


def adjacent(x, np):
    """Return the nearest finite lower and higher BF16 values, including across zero."""
    bits = x.astype(np.float32).view(np.uint32)
    mag = bits & np.uint32(0x7FFFFFFF)
    sign = bits & np.uint32(0x80000000)
    step = np.uint32(0x10000)
    shorter = np.maximum(mag, step) - step
    longer = np.minimum(mag + step, np.uint32(0x7F7F0000))
    lower = np.where(sign != 0, sign | longer, shorter)
    upper = np.where(sign != 0, sign | shorter, longer)
    lower = np.where(mag == 0, np.uint32(0x80010000), lower)
    upper = np.where(mag == 0, step, upper)
    return tuple(v.astype(np.uint32).view(np.float32).astype(np.float64) for v in (lower, upper))


def joint_round(left, right, ideal_gap, np):
    """Minimize differential error over the product stencil at each coordinate."""
    centers = tuple(bf16(v, np).astype(np.float64) for v in (left, right))
    if not all(np.isfinite(v).all() for v in centers):
        raise ValueError("Bank comparison endpoints exceed finite BF16")
    options = [(c, *adjacent(c, np)) for c in centers]
    best_error = np.full(left.shape, np.inf)
    best_center_distance = best_error.copy()
    best_a, best_b = centers
    for a, b in product(*options):
        error = ((b - a) - ideal_gap)**2
        distance = (a - left)**2 + (b - right)**2
        take = (error < best_error) | ((error == best_error) & (distance < best_center_distance))
        best_a = np.where(take, a, best_a)
        best_b = np.where(take, b, best_b)
        best_error = np.where(take, error, best_error)
        best_center_distance = np.where(take, distance, best_center_distance)
    return (best_a, best_b), sqrt(float(best_error.sum()))


def coordinate_obstruction(before, ideal, target, cfg, np):
    """Return a coordinate that rules out every BF16 dose-and-cosine match."""
    u = np.clip(ideal / np.linalg.norm(ideal), -1., 1.)
    c = cfg["minimum_direction_cosine"]
    tol = cfg["dose_relative_tolerance"]
    span = np.sqrt((1 - c**2) * (1 - u**2))
    low = np.where(u <= -c, -1., c*u - span)
    high = np.where(u >= c, 1., c*u + span)
    limits = np.stack([np.stack((low*r, high*r)) for r in (1 - tol, 1 + tol)])
    slack = 128 * np.finfo(np.float64).eps * np.maximum(np.maximum(abs(before), target), 1.)
    a = before + target*limits.min(axis=(0, 1)) - slack
    b = before + target*limits.max(axis=(0, 1)) + slack
    first = bf16(np.clip(a, -MAX_BF16, MAX_BF16), np).astype(np.float64)
    _, above = adjacent(first, np)
    possible = np.where(first < a, above, first)
    excluded = np.flatnonzero((a > MAX_BF16) | (possible > b))
    if not len(excluded):
        return None
    i = excluded[0]
    return {"coordinate": int(i), "disallowed_interval": [float(a[i]), float(b[i])],
            "next_bf16_at_or_above_lower": None if a[i] > MAX_BF16 else float(possible[i])}


def reached(before, direction, cfg, np):
    ray = prepare(before, direction, cfg["phase1_dose_fraction"], cfg["dose_relative_tolerance"],
                  cfg["minimum_direction_cosine"], np)
    if ray.eligible:
        return ray, "original"
    if ray.cosine < cfg["minimum_direction_cosine"]:
        return ray, "unqualified"
    repairs, _ = repaired_candidates(before, direction, ray, np)
    return (repairs[0], "complement_fill") if repairs else (ray, "unqualified")


def vector_metrics(actual, ideal, nominal, cfg, np):
    length = float(np.linalg.norm(actual))
    alignment = float(np.clip(actual @ ideal / (length * np.linalg.norm(ideal)), -1., 1.)) if length else None
    err = abs(length / nominal - 1.)
    return {"length": length, "cosine": alignment, "relative_dose_error": err,
            "qualified": alignment is not None and alignment >= cfg["minimum_direction_cosine"]
            and err <= cfg["dose_relative_tolerance"]}


def assess(before, pair, ideals, target, cfg, np, *, inputs_eligible=True, lower_bound=None):
    actual = [p - before for p in pair]
    writes = [vector_metrics(a, d, target, cfg, np) for a, d in zip(actual, ideals, strict=True)]
    spread = abs(writes[0]["length"] - writes[1]["length"]) / target
    dose_ok = inputs_eligible and spread <= cfg["max_contrast_dose_gap_fraction"] and all(w["qualified"] for w in writes)
    ideal_gap = ideals[1] - ideals[0]
    reached_gap = pair[1] - pair[0]
    gap_length = float(np.linalg.norm(ideal_gap))
    gap = vector_metrics(reached_gap, ideal_gap, gap_length, cfg, np)
    lower_relative = lower_bound / gap_length if lower_bound is not None else None
    normalized_error = float(np.linalg.norm(reached_gap - ideal_gap) / gap_length)
    midpoint_error = float(np.linalg.norm((actual[0] + actual[1] - ideals[0] - ideals[1])/2) / target)
    if lower_relative is not None and abs(lower_relative - normalized_error) > 1e-10:
        raise ValueError("The attained differential disagrees with its coordinate minima")
    r = 1 + cfg["dose_relative_tolerance"]
    max_error = sqrt((r - 1)**2 + 2*r*(1 - cfg["minimum_direction_cosine"]))
    return {"writes": writes, "dose_spread_fraction": spread, "dose_qualified": dose_ok, "differential": gap,
            "differential_relative_error": normalized_error, "midpoint_error_relative_to_nominal": midpoint_error,
            "nominal": target, "exact_state_alias": np.array_equal(*pair),
            "stencil_certified_incompatible": lower_relative is not None and lower_relative > max_error + 1e-10,
            "qualified": bool(dose_ok and gap["qualified"])}


def compare(before, directions, cfg, np):
    before = before.astype(np.float64)
    norms = {name: float(np.linalg.norm(directions[name])) for name in FULL}
    if not all(n > 0 for n in norms.values()):
        return {"status": "zero_full_fit_direction"}
    target = float(np.linalg.norm(before) * cfg["phase1_dose_fraction"])
    if target <= 0:
        return {"status": "zero_native_norm"}
    unit = {name: directions[name] * (target / norms[name]) for name in FULL}
    if np.linalg.norm(unit["t2_fit"] - unit["t1_fit"]) == 0:
        return {"status": "zero_ideal_differential"}
    result = {}
    for sign in (1, -1):
        ideals = [sign * unit[name] for name in FULL[1:]]
        intended = {name: sign * directions[name] for name in FULL}
        rays = [prepare(before, intended[name], cfg["phase1_dose_fraction"], cfg["dose_relative_tolerance"],
                        cfg["minimum_direction_cosine"], np) for name in FULL[1:]]
        repairs = {name: reached(before, intended[name], cfg, np) for name in FULL}
        fixed = [repairs[n][0] for n in FULL[1:]]
        modes = {key: assess(before, [p.after for p in arms], ideals, target, cfg, np,
                            inputs_eligible=all(p.eligible for p in arms))
                 for key, arms in (("original_rays", rays), ("independent_repair", fixed))}
        anchor, carrier_mode = repairs["svd"]
        if anchor.eligible:
            midpoint, gap = (ideals[0] + ideals[1]) / 2, ideals[1] - ideals[0]
            for gain in GAINS:
                aimed = [midpoint - gain*gap/2, midpoint + gain*gap/2]
                endpoints = [anchor.after + d - sign*unit["svd"] for d in aimed]
                nearest = [bf16(x, np).astype(np.float64) for x in endpoints]
                jointly, bound = joint_round(*endpoints, aimed[1] - aimed[0], np)
                modes[f"nearest_x{gain}"] = assess(before, nearest, aimed, target, cfg, np)
                modes[f"joint_stencil_x{gain}"] = assess(before, jointly, aimed, target, cfg, np, lower_bound=bound)
        obstruction = {name: coordinate_obstruction(before, sign*unit[name], target, cfg, np) for name in FULL}
        if any(obstruction[name] is not None and repairs[name][0].eligible for name in FULL):
            raise ValueError("A qualified repair contradicted a BF16 coordinate obstruction")
        result["+" if sign > 0 else "-"] = {"carrier_mode": carrier_mode, "methods": modes,
                                             "coordinate_obstructions": obstruction}
    return {"status": "measured_geometry", "signs": result}

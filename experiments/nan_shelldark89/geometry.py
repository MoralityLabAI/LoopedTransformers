"""Generalize the frozen ROUTEMATCH86 anchored shell to an explicit dose."""

from hashlib import sha256

from experiments.nan_delay39.instrument import prepare
from experiments.nan_lattice53.run import exact_infeasible, screen
from experiments.nan_readpath37.run import bf16
from experiments.nan_routematch86.geometry import (
    ARMS, COSINE, DOSE, DOSE_TOLERANCE, GRID, MAX_TANGENT_COSINE,
    anchored_tangent, joint, match_options, span_orthogonal_seed,
)


def shell_options(native, axis, direction, aim, pivot, rest, anchor, nominal, grid, np):
    extent = (1 + DOSE_TOLERANCE) * nominal
    start = np.searchsorted(grid, native[pivot] - extent)
    stop = np.searchsorted(grid, native[pivot] + extent, side="right")
    remainder = axis.copy()
    remainder[pivot] = 0.
    axis_share = float(np.linalg.norm(remainder))
    corrections, scales = np.asarray(GRID).T
    choices = []
    for value in grid[start:stop]:
        pivot_change = float(value - native[pivot])
        radial = (anchor[2] + corrections * nominal - axis[pivot] * pivot_change) / axis_share
        remaining_square = anchor[1] ** 2 - pivot_change ** 2 - radial ** 2
        possible = remaining_square > 0
        proposals = native + radial[possible, None] * rest
        proposals += (scales[possible] * np.sqrt(remaining_square[possible]))[:, None] * direction
        proposals[:, pivot] = value
        reached = bf16(proposals, np).astype(np.float64)
        delta = reached - native
        mass = np.linalg.norm(delta, axis=1)
        goals = proposals - native if aim is None else aim
        numerator = np.sum(delta * goals, axis=1) if aim is None else delta @ aim
        denominator = mass * np.linalg.norm(goals, axis=1) if aim is None else mass
        cosines = np.divide(numerator, denominator, out=np.zeros_like(mass), where=denominator > 0)
        eligible = (abs(mass / nominal - 1) <= DOSE_TOLERANCE) & (cosines >= COSINE)
        radials = delta @ axis
        norms = np.linalg.norm(reached, axis=1)
        for i in np.flatnonzero(eligible):
            choices.append((reached[i], float(mass[i]), float(radials[i]), float(norms[i]),
                            float(cosines[i]), int(np.flatnonzero(possible)[i])))
    return choices


def reached_shell(native, axis, tangents, seed, fraction, grid, np):
    radius = float(np.linalg.norm(native))
    nominal = fraction * radius
    learned = prepare(native, tangents[0], fraction, DOSE_TOLERANCE, COSINE, np)
    if not learned.eligible:
        return {}, {"status": "learned_tangent_unqualified", "ray_cosine": learned.cosine,
                    "ray_dose_relative_error": abs(learned.realized_l2 / nominal - 1)}
    delta = learned.after - native
    radial = float(delta @ axis)
    anchor = (learned.after, learned.realized_l2, radial, float(np.linalg.norm(learned.after)),
              learned.cosine, 0)
    pivot = int(np.argmax(abs(native)))
    rest = axis.copy()
    rest[pivot] = 0.
    rest_length = np.linalg.norm(rest)
    if rest_length <= 1e-10:
        return {}, {"status": "degenerate_remainder"}
    rest /= rest_length
    local = tangents.copy()
    local[:, pivot] = 0.
    local -= (local @ rest)[:, None] * rest
    lengths = np.linalg.norm(local, axis=1)
    if not np.all(lengths > 1e-10):
        return {}, {"status": "degenerate_local_tangent"}
    local /= lengths[:, None]
    pivot_axis = np.zeros_like(axis)
    pivot_axis[pivot] = 1.
    seed_direction = span_orthogonal_seed((*local, pivot_axis), rest, seed, np)
    static = shell_options(native, axis, local[1], tangents[1], pivot, rest, anchor, nominal, grid, np)
    sham = shell_options(native, axis, seed_direction, None, pivot, rest, anchor, nominal, grid, np)
    chosen = match_options(((anchor,), static, sham), nominal, radius, np)
    counts = {"learned": 1, "cal_static": len(static), "seeded": len(sham)}
    receipt = {"eligible_candidates": counts, "target_radial_share": radial / anchor[1]}
    if chosen is None:
        return {}, receipt | {"status": "anchored_shell_unqualified"}
    remainders = np.stack([entry[0] - native for entry in chosen])
    remainders[:, pivot] = 0.
    remainders -= (remainders @ rest)[:, None] * rest
    lengths = np.linalg.norm(remainders, axis=1)
    if not np.all(lengths > 1e-12):
        return {}, {"status": "degenerate_reached_tangent", "eligible_candidates": counts}
    remainders /= lengths[:, None]
    cosines = [float(remainders[-1] @ arm) for arm in remainders[:2]]
    if max(map(abs, cosines)) > MAX_TANGENT_COSINE:
        return {}, {"status": "seed_orientation_unqualified", "eligible_candidates": counts}
    outputs = {arm: choice[0] for arm, choice in zip(ARMS, chosen, strict=True)}
    if len({v.tobytes() for v in outputs.values()}) != len(ARMS):
        return {}, receipt | {"status": "arm_alias"}
    dose_gap, radial_gap, norm_gap = joint(chosen, nominal, radius)
    measured = {arm: {"realized_l2": mass, "radial_l2": rd, "relative_reached_norm": norm / radius,
                      "planned_direction_cosine": angle}
                for arm, (_, mass, rd, norm, angle, _) in zip(ARMS, chosen, strict=True)}
    return outputs, receipt | {"status": "matched", "measured": measured, "nominal_l2": nominal,
                               "dose_gap_fraction": dose_gap, "radial_gap_fraction": radial_gap,
                               "relative_norm_gap": norm_gap, "realized_sham_remainder_cosines": cosines}


def geometry(native, projected, key, fraction, grid, np):
    axis = native / np.linalg.norm(native)
    source = projected / np.linalg.norm(projected, axis=1)[:, None]
    tangents = source - (source @ axis)[:, None] * axis
    lengths = np.linalg.norm(tangents, axis=1)
    if not np.all(lengths > 1e-10):
        return {}, {"status": "degenerate_tangent"}
    tangents /= lengths[:, None]
    seed = int.from_bytes(sha256(f"NAN-ROUTEMATCH-86|{key}|phase1".encode()).digest()[:8], "big")
    outputs, receipt = reached_shell(native, axis, tangents, seed, fraction, grid, np)
    if fraction == DOSE:
        reference, old = anchored_tangent(native, axis, tangents, seed, np)
        if (set(outputs) != set(reference) or any(
                not np.array_equal(outputs[a], reference[a]) for a in outputs)
                or any(abs(old[k] - receipt[k]) > 1e-12 for k in (
                    "dose_gap_fraction", "radial_gap_fraction", "relative_norm_gap") if outputs)):
            raise ValueError("The generalized planner changed the 2% shell")
    if not outputs:
        cfg = {"phase1_dose_fraction": fraction, "dose_relative_tolerance": DOSE_TOLERANCE,
               "minimum_direction_cosine": COSINE}
        excluded = []
        for arm, tangent in zip(ARMS[:2], tangents, strict=True):
            upper, scale = screen(native, tangent, cfg, np)
            if upper < COSINE and exact_infeasible(native, tangent, scale, cfg, np):
                excluded.append(arm)
        return {}, receipt | {"exact_angle_excluded": excluded}
    for i, arm in enumerate(ARMS[:2]):
        delta = outputs[arm] - native
        receipt["measured"][arm]["full_recipe_cosine"] = float(delta @ source[i] / np.linalg.norm(delta))
    return outputs, receipt

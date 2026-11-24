"""Repair the learned ray at its dose jump, then match the frozen control shell."""

from hashlib import sha256
from math import sqrt

from experiments.nan_delay39.instrument import Patch, prepare
from experiments.nan_dose38.run import MAX_MULTIPLE, STEPS, displacement
from experiments.nan_readpath37.run import bf16
from experiments.nan_routematch86.geometry import (
    ARMS, COSINE, DOSE_TOLERANCE, MAX_TANGENT_COSINE, joint, match_options, span_orthogonal_seed,
)
from experiments.nan_shelldark89.geometry import geometry as original_geometry
from experiments.nan_shelldark89.geometry import shell_options


FRACTION = .005


def repaired_candidates(native, direction, original, np):
    target = original.requested_l2
    unit = direction / np.linalg.norm(direction)

    def at(scale):
        return displacement(native, unit[None], np.array([scale]), np)[0]

    left, right = 0., MAX_MULTIPLE * target
    if np.linalg.norm(at(right)) < target:
        return [], 0
    for _ in range(STEPS):
        midpoint = (left + right) / 2
        if np.linalg.norm(at(midpoint)) < target:
            left = midpoint
        else:
            right = midpoint
    endpoints = (at(left), at(right))
    fixed = endpoints[0] != endpoints[1]
    count = int(fixed.sum())
    rest = ~fixed
    residual_norm = float(np.linalg.norm(native[rest]))
    if not count or not residual_norm:
        return [], count
    candidates = []
    for endpoint in endpoints:
        locked = endpoint[fixed]
        remaining = target ** 2 - float(locked @ locked)
        if remaining <= 0:
            continue
        patch = prepare(native[rest], unit[rest], sqrt(remaining) / residual_norm,
                        DOSE_TOLERANCE, 0., np)
        delta = endpoint.copy()
        delta[rest] = patch.after - patch.before
        mass = float(np.linalg.norm(delta))
        alignment = float(delta @ unit / mass)
        error = abs(mass / target - 1)
        after = native + delta
        if not np.array_equal(after, bf16(after, np)):
            raise ValueError("Compensated coordinates left the BF16 lattice")
        if error <= DOSE_TOLERANCE and alignment >= COSINE:
            candidates.append((error, -alignment, Patch(native, after, target, mass, alignment, True)))
    return [candidate[2] for candidate in sorted(candidates, key=lambda item: item[:2])], count


def shell(native, axis, tangents, seed, candidates, grid, np):
    radius = float(np.linalg.norm(native))
    pivot = int(np.argmax(abs(native)))
    rest = axis.copy()
    rest[pivot] = 0.
    norm = float(np.linalg.norm(rest))
    if norm <= 1e-10:
        return {}, {}
    rest /= norm
    local = tangents.copy()
    local[:, pivot] = 0.
    local -= (local @ rest)[:, None] * rest
    lengths = np.linalg.norm(local, axis=1)
    if not np.all(lengths > 1e-10):
        return {}, {}
    local /= lengths[:, None]
    pivot_axis = np.zeros_like(axis)
    pivot_axis[pivot] = 1.
    sham = span_orthogonal_seed((*local, pivot_axis), rest, seed, np)
    for rank, learned in enumerate(candidates, start=1):
        delta = learned.after - native
        radial = float(delta @ axis)
        nominal = learned.requested_l2
        anchor = (learned.after, learned.realized_l2, radial, float(np.linalg.norm(learned.after)),
                  learned.cosine, 0)
        static = shell_options(native, axis, local[1], tangents[1], pivot, rest, anchor, nominal, grid, np)
        seeded = shell_options(native, axis, sham, None, pivot, rest, anchor, nominal, grid, np)
        chosen = match_options(((anchor,), static, seeded), nominal, radius, np)
        if chosen is None:
            continue
        outputs = {arm: option[0] for arm, option in zip(ARMS, chosen, strict=True)}
        remainders = np.stack([write - native for write in outputs.values()])
        remainders[:, pivot] = 0.
        remainders -= (remainders @ rest)[:, None] * rest
        lengths = np.linalg.norm(remainders, axis=1)
        if not np.all(lengths > 1e-12) or len({w.tobytes() for w in outputs.values()}) != len(ARMS):
            continue
        remainders /= lengths[:, None]
        cosines = [float(remainders[-1] @ arm) for arm in remainders[:2]]
        if max(map(abs, cosines)) > MAX_TANGENT_COSINE:
            continue
        dose, rd, norm = joint(chosen, nominal, radius)
        geometry = {"repaired_candidate_rank": rank, "dose_gap_fraction": dose,
                    "radial_gap_fraction": rd, "relative_norm_gap": norm,
                    "realized_sham_remainder_cosines": cosines, "nominal_l2": nominal,
                    "measured": {arm: {"realized_l2": v[1], "planned_direction_cosine": v[4]}
                                 for arm, v in zip(ARMS, chosen, strict=True)}}
        return outputs, geometry
    return {}, {}


def plan(native, projected, key, grid, np):
    outputs, original = original_geometry(native, projected, key, FRACTION, grid, np)
    if outputs:
        return outputs, original, {"mode": "original_shell"}
    if original["status"] != "learned_tangent_unqualified" or original["ray_cosine"] < COSINE:
        return {}, original, {"mode": "unqualified"}
    axis = native / np.linalg.norm(native)
    directions = projected / np.linalg.norm(projected, axis=1)[:, None]
    tangents = directions - (directions @ axis)[:, None] * axis
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]
    ray = prepare(native, tangents[0], FRACTION, DOSE_TOLERANCE, COSINE, np)
    if ray.eligible:
        raise ValueError("A qualified learned ray was presented for repair")
    candidates, count = repaired_candidates(native, tangents[0], ray, np)
    seed = int.from_bytes(sha256(f"NAN-ROUTEMATCH-86|{key}|phase1".encode()).digest()[:8], "big")
    repaired, details = shell(native, axis, tangents, seed, candidates, grid, np)
    return repaired, original, details | {"mode": "compensated_shell" if repaired else "unqualified",
                                          "jump_coordinates": count, "eligible_anchors": len(candidates)}


def fixture(np):
    from experiments.nan_cibracket75.repair import compensate

    native = np.array([13., .25, .25, .25])
    direction = np.array([1., .1, .1, .1])
    before = prepare(native, direction, .02, DOSE_TOLERANCE, COSINE, np)
    expected, mode, count = compensate(native, direction, np)
    actual, changed = repaired_candidates(native, direction, before, np)
    assert mode == "compensated" and changed == count == 1
    assert np.array_equal(expected.after, actual[0].after)
    return {"compensation_fixture_matches_CIBRACKET75": True}

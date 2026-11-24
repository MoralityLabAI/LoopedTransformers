"""Search exact BF16 endpoint swaps using a fixed first-pass design vocabulary."""

from hashlib import sha256
from math import sqrt

from experiments.nan_chordsham106.geometry import (
    MAX_ABS_COSINE, MAX_ENDPOINT_RADIUS_ERROR, TRIALS,
)


DESIGN_ROWS = 512
DESIGN_OFFSET = 17
STARTS = 3
STEPS = 48
SHORTLIST = 16


def design_ids(vocab, np):
    if vocab // DESIGN_ROWS <= DESIGN_OFFSET:
        raise ValueError("The checkpoint vocabulary is too small for the fixed grid")
    return np.arange(DESIGN_ROWS) * vocab // DESIGN_ROWS + DESIGN_OFFSET


def head_jacobian(weighted_rows, midpoint, chord, live, eps, np):
    """Columns map signed chord coordinates to linearized half-differences."""
    scale = sqrt(float(np.mean(midpoint**2)) + eps)
    radial = (weighted_rows @ midpoint)[:, None] * midpoint[live]
    derivative = weighted_rows[:, live] / scale - radial / (len(midpoint) * scale**3)
    return np.asarray(derivative * (chord[live] / 2), dtype=np.float32, order="F")


def radial_bounds(radii, np):
    low, high = [radii**2 * (1 + sign * MAX_ENDPOINT_RADIUS_ERROR)**2 for sign in (-1, 1)]
    total = float(sum(radii**2))
    return max(total - 2*high[0], 2*low[1] - total), min(total - 2*low[0], 2*high[1] - total)


def flip_gain(matrix, response, signs, norms):
    return 4 * (norms - signs * (matrix.T @ response))


def candidates(matrix, weights, mix, radii, key, target, np):
    """Return a bounded shortlist and the number of geometry-feasible seeds."""
    energy = float(weights.sum())
    limit = MAX_ABS_COSINE * energy
    low, high = radial_bounds(radii, np)
    seed = int.from_bytes(sha256(key.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    signs = 1 - 2*rng.integers(0, 2, size=(TRIALS, len(weights)), dtype=np.int8)
    align, radii_product = signs @ weights, signs @ mix
    good = np.flatnonzero((abs(align) <= limit) & (radii_product >= low) & (radii_product <= high))
    if not len(good):
        return [], 0
    responses = (matrix @ signs[good].T).T
    scores = np.sqrt(np.mean(responses**2, axis=1))
    # Keep initial candidates as well as each step along the strongest trajectories.
    pool = [(float(abs(score-target)), signs[i].copy()) for i, score in zip(good, scores, strict=True)]
    starts = np.argsort(-scores, kind="stable")[:STARTS]
    norms = np.einsum("ij,ij->j", matrix, matrix)
    for k in starts:
        i = good[k]
        s, z = signs[i].copy(), responses[k].copy()
        u, v = align[i], radii_product[i]
        for _ in range(STEPS):
            next_u, next_v = u - 2*s*weights, v - 2*s*mix
            gain = flip_gain(matrix, z, s, norms)
            allowed = (abs(next_u) <= limit) & (next_v >= low) & (next_v <= high)
            gain[~allowed] = -np.inf
            j = int(gain.argmax())
            if gain[j] <= 1e-12:
                break
            u, v = next_u[j], next_v[j]
            z -= 2*s[j]*matrix[:, j]
            s[j] = -s[j]
            pool.append((abs(sqrt(float(np.mean(z**2))) - target), s.copy()))
    shortlisted = {}
    for _, s in sorted(pool, key=lambda item: item[0]):
        shortlisted.setdefault(s.tobytes(), s)
        if len(shortlisted) == SHORTLIST:
            break
    return list(shortlisted.values()), len(good)


def pair_rms(logits, clean, np):
    logits = np.asarray(logits, dtype=np.float64)
    half = (logits[:, 1] - logits[:, 0]) / 2
    shared = (logits[:, 1] + logits[:, 0]) / 2 - clean
    return np.sqrt(np.mean(np.stack((half**2, shared**2), axis=1), axis=2))


def select(before, left, right, key, head, ids, weighted_rows, np):
    chord = right - left
    live = chord != 0
    weights = chord[live]**2
    energy = float(weights.sum())
    if energy <= 0:
        raise ValueError("A swap requires a nonzero attained chord")
    floor = max(0., 2*float(weights.max())/energy - 1)
    diagnostics = {"orientation_cosine_floor": floor}
    if floor > MAX_ABS_COSINE:
        return None, diagnostics | {"status": "dominant_coordinate_obstruction"}
    radii = np.array([np.linalg.norm(left-before), np.linalg.norm(right-before)])
    if min(radii) <= 0:
        return None, diagnostics | {"status": "zero_active_endpoint_radius"}
    clean, *active = head.logits([before, left, right], ids)
    target = pair_rms([active], clean, np)[0]
    midpoint = (left + right) / 2
    matrix = head_jacobian(weighted_rows, midpoint, chord, live, float(head.eps), np)
    mix = (2*(midpoint-before))[live] * chord[live]
    signs, eligible = candidates(matrix, weights, mix, radii, key, target[0], np)
    if not signs:
        return None, diagnostics | {"status": "search_refused"}
    swap = np.zeros((len(signs), len(before)), dtype=bool)
    swap[:, live] = np.stack(signs) < 0
    pairs = np.stack((np.where(swap, right, left), np.where(swap, left, right)), axis=1)
    logits = head.logits(pairs.reshape(-1, len(before)), ids).reshape(len(pairs), 2, len(ids))
    rms = pair_rms(logits, clean, np)
    gaps = abs(rms - target)
    j = min(range(len(signs)), key=lambda i: (gaps[i, 0], gaps[i, 1], i))
    a, b = pairs[j]
    cosine = abs(float(chord @ (b-a))) / energy
    error = float(abs(np.linalg.norm(pairs[j]-before, axis=1)/radii - 1).max())
    if (not np.array_equal(a+b, left+right) or not np.array_equal(abs(b-a), abs(chord))
            or cosine > MAX_ABS_COSINE + 1e-12 or error > MAX_ENDPOINT_RADIUS_ERROR + 1e-12):
        raise ValueError("The selected swap missed an exact geometry gate")
    return (a, b), diagnostics | {
        "status": "matched", "reached_abs_chord_cosine": cosine, "max_endpoint_radius_relative_error": error,
        "eligible_seeds": eligible, "shortlist_size": len(signs), "selected_shortlist_index": j,
        "sample_active_rms": target.tolist(), "sample_sham_rms": rms[j].tolist(),
        "selected_signs_sha256": sha256(signs[j].tobytes()).hexdigest(),
    }

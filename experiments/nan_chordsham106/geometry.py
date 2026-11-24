"""Construct physically matched BF16 shams without a second rounding step."""

from hashlib import sha256


TRIALS = 256
MAX_ABS_COSINE = .10
MAX_ENDPOINT_RADIUS_ERROR = .03
MAX_RMS_GAP = .005


def swap_sham(before, left, right, key, np):
    d = right - left
    live = d != 0
    weights = d[live]**2
    energy = float(weights.sum())
    if energy <= 0.:
        raise ValueError("A coordinate-swap sham needs a nonzero reached chord")
    floor = max(0., 2*float(weights.max()) / energy - 1)
    if floor > MAX_ABS_COSINE + 1e-12:
        return None, {"status": "dominant_coordinate_obstruction", "orientation_cosine_floor": floor}
    radii = np.asarray([np.linalg.norm(left-before), np.linalg.norm(right-before)])
    if min(radii) <= 0.:
        return None, {"status": "zero_active_endpoint_radius", "orientation_cosine_floor": floor}
    centered_sum = (left + right) - 2*before
    mix = centered_sum[live] * d[live]
    energy_sum = float(np.sum(radii**2))
    seed = int.from_bytes(sha256(key.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    signs = 1 - 2*rng.integers(0, 2, size=(TRIALS, len(weights)), dtype=np.int8)
    alignment = abs(signs @ weights) / energy
    product = signs @ mix
    sham_radii = np.sqrt(np.maximum(0., (energy_sum + np.stack((-product, product), axis=1))/2))
    errors = abs(sham_radii/radii - 1).max(axis=1)
    candidate = np.flatnonzero((alignment <= MAX_ABS_COSINE) & (errors <= MAX_ENDPOINT_RADIUS_ERROR))
    if not len(candidate):
        return None, {"status": "search_refused", "orientation_cosine_floor": floor,
                      "minimum_sampled_abs_cosine": float(alignment.min()),
                      "within_cosine_trials": int((alignment <= MAX_ABS_COSINE).sum())}
    i = min(candidate, key=lambda k: (errors[k], alignment[k], k))
    swap = np.zeros_like(live)
    swap[live] = signs[i] < 0
    a, b = np.where(swap, right, left), np.where(swap, left, right)
    reached_cos = abs(float(d @ (b-a)) / energy)
    reached_error = float(abs(np.asarray([np.linalg.norm(a-before), np.linalg.norm(b-before)])/radii - 1).max())
    if (not np.array_equal(a+b, left+right) or not np.array_equal(abs(b-a), abs(d))
            or not np.isclose(reached_error, errors[i], rtol=0., atol=1e-12)
            or not np.isclose(reached_cos, alignment[i], rtol=0., atol=1e-12)
            or reached_error > MAX_ENDPOINT_RADIUS_ERROR + 1e-12 or reached_cos > MAX_ABS_COSINE + 1e-12):
        raise ValueError("The reached swap disagrees with its exact midpoint, chord or radius")
    return (a, b), {"status": "matched", "orientation_cosine_floor": floor,
                    "reached_abs_chord_cosine": reached_cos, "max_endpoint_radius_relative_error": reached_error,
                    "candidate_index": int(i), "eligible_trials": len(candidate), "nonzero_chord_coordinates": len(weights)}

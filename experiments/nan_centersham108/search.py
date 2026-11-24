"""Construct a new BF16 chord on the same midpoint as the attained edit."""

from hashlib import sha256
from math import sqrt

from experiments.nan_bankgap100.rounding import adjacent
from experiments.nan_bankdose99.geometry import CURRENT_DRIFT_LIMIT
from experiments.nan_chordsham106.geometry import MAX_ABS_COSINE, MAX_ENDPOINT_RADIUS_ERROR, MAX_RMS_GAP
from experiments.nan_headswap107.search import pair_rms
from experiments.nan_readpath37.run import bf16


ANGLES = 24
SCALES = (.84, .88, .92, .96, 1., 1.04, 1.08, 1.12, 1.16)
POWER_STEPS = 16
RANDOM_AXES = 2


def midpoint_round(mid, half, left, right, np):
    """Minimize half-chord error over mirrored nearest and adjacent BF16 rows."""
    total = left + right
    best_a = np.where(half >= 0., np.minimum(left, right), np.maximum(left, right))
    best_b = total - best_a
    best_error = abs((mid - best_a) - half)
    a_near, b_near = [bf16(v, np).astype(np.float64) for v in (mid-half, mid+half)]
    for reverse, near in ((False, a_near), (True, b_near)):
        for x in (near, *adjacent(near, np)):
            y = total - x
            a, b = (y, x) if reverse else (x, y)
            error = abs((mid - a) - half)
            admissible = (bf16(y, np) == y) & (a+b == total)
            take = admissible & (error < best_error)
            best_a = np.where(take, a, best_a)
            best_b = np.where(take, b, best_b)
            best_error = np.where(take, error, best_error)
    return best_a, best_b


def projector(mid, weighted_rows, eps, np):
    scale = sqrt(float(np.mean(mid**2)) + eps)
    radial = (weighted_rows @ mid)[:, None] * mid
    return weighted_rows / scale - radial / (len(mid) * scale**3)


def proposal_planes(active, carry, jac, rng, np):
    """Generate rings in the chord tangent and, when reachable, its radius-split slice."""
    radius = np.linalg.norm(active)
    active_u = active / radius
    side = carry - (carry @ active_u) * active_u
    side_norm = np.linalg.norm(side)
    width = len(active)
    tangent = np.stack([active_u])
    families = [(np.zeros(width), tangent, radius)]
    if side_norm > 1e-10 * radius:
        side_u = side / side_norm
        fixed = side_u * ((carry @ active) / side_norm)
        remainder = radius**2 - fixed @ fixed
        if remainder > 1e-12 * radius**2:
            families.append((fixed, np.stack([active_u, side_u]), sqrt(remainder)))
    for fixed, rows, reach in families:
        projected = jac - (jac @ rows.T) @ rows
        high = rng.standard_normal(width)
        for _ in range(POWER_STEPS):
            high = projected.T @ (projected @ high)
            length = np.linalg.norm(high)
            if length <= 0.:
                raise ValueError("Projected design head has no response")
            high /= length
        normal = np.vstack([rows, high])
        for _ in range(RANDOM_AXES):
            random = rng.standard_normal(width)
            random -= (normal @ random) @ normal
            random /= np.linalg.norm(random)
            yield fixed, reach * high, reach * random


def geometry(before, left, right, a, b, np):
    original = right - left
    new = b - a
    norms = np.linalg.norm(new, axis=1)
    chord_error = abs(norms/np.linalg.norm(original) - 1.)
    radii = np.array([np.linalg.norm(left-before), np.linalg.norm(right-before)])
    if min(radii) <= 0:
        raise ValueError("The active pair has a zero endpoint radius")
    new_radii = np.stack([np.linalg.norm(v-before, axis=1) for v in (a, b)], axis=1)
    radius_error = abs(new_radii/radii - 1.).max(axis=1)
    cosine = abs(new @ original) / (np.maximum(norms, np.finfo(float).tiny) * np.linalg.norm(original))
    return chord_error, radius_error, cosine


def select(before, left, right, key, head, ids, weighted_rows, np):
    active = (right-left)/2
    if np.linalg.norm(active) == 0:
        raise ValueError("The original chord is zero")
    mid = (right+left)/2
    jac = projector(mid, weighted_rows, float(head.eps), np)
    baseline, *endpoints = head.logits([before, left, right], ids)
    target = pair_rms([endpoints], baseline, np)[0]
    seed = int.from_bytes(sha256(key.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    theta = np.arange(ANGLES) * (2*np.pi/ANGLES)
    n = 0
    admissible = 0
    best = None
    for plane, (fixed, high, random) in enumerate(proposal_planes(active, mid-before, jac, rng, np)):
        ring = np.cos(theta[:, None])*high + np.sin(theta[:, None])*random
        for scale in SCALES:
            a, b = midpoint_round(mid, fixed + scale*ring, left, right, np)
            dose, radius, cos = geometry(before, left, right, a, b, np)
            eligible = np.flatnonzero((dose <= MAX_ENDPOINT_RADIUS_ERROR) & (radius <= MAX_ENDPOINT_RADIUS_ERROR)
                                      & (cos <= MAX_ABS_COSINE))
            n += len(ring)
            admissible += len(eligible)
            if not len(eligible):
                continue
            pairs = np.stack([a[eligible], b[eligible]], axis=1)
            logits = head.logits(pairs.reshape(-1, len(before)), ids).reshape(len(pairs), 2, len(ids))
            rms = pair_rms(logits, baseline, np)
            gaps = abs(rms - target)
            peaks = abs(logits - baseline).max(axis=(1, 2))
            for k, j in enumerate(eligible):
                # A design row already above the full-head cap can never pass that gate.
                score = (bool(peaks[k] > CURRENT_DRIFT_LIMIT), float(max(gaps[k])),
                         float(gaps[k, 0]), float(peaks[k]))
                if best is None or score < best[0]:
                    info = {
                        "status": "matched", "reached_abs_chord_cosine": float(cos[j]),
                        "max_endpoint_radius_relative_error": float(radius[j]),
                        "chord_relative_error": float(dose[j]),
                        "selected_plane": plane, "selected_scale": scale, "selected_angle_index": int(j),
                        "sample_active_rms": target.tolist(), "sample_sham_rms": rms[k].tolist(),
                        "sample_peak_drift": float(peaks[k]),
                    }
                    best = score, pairs[k].copy(), info
    trials = {"geometry_eligible_candidates": admissible, "candidates": n}
    if best is None:
        return None, {"status": "search_refused"} | trials
    _, (a, b), info = best
    if (not np.array_equal(a+b, left+right) or not np.array_equal(bf16(a, np), a)
            or not np.array_equal(bf16(b, np), b)):
        raise ValueError("Mirrored candidates violate the exact BF16 midpoint")
    active_half, sham_half = active, (b-a)/2
    info.update(
        changed_coordinate_magnitudes=int(np.sum(abs(active_half) != abs(sham_half))),
        selected_pair_sha256=sha256(np.array([a, b], dtype="<f4").tobytes()).hexdigest(),
        design_rms_matched=bool(max(abs(np.array(info["sample_sham_rms"])-target)) <= MAX_RMS_GAP),
    )
    return (a, b), info | trials

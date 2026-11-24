"""Float32 BPD write arithmetic and matched, quantized signed residual edits."""
from itertools import combinations

from experiments.nan_delay39.contrasts import dose_status, qualified_pair
from experiments.nan_delay39.instrument import prepare
from experiments.nan_readpath37.run import bf16
from experiments.nan_route50.masks import mass_matched


ARMS = ("learned", "static")
SIGNS = {"positive": 1, "negative": -1}


def writes(native, reads, masks, bank, eps, np):
    """T2 pass-mean, delta gate zero, followed by the registered sphere rule."""
    masks = np.asarray(masks, dtype=np.float32)
    means = bank["mean_comp_acts"][0]
    component = masks * reads + (1 - masks) * means
    out = native + (component @ bank["U"] + bank["mean_delta_out"][0] - native)
    fraction = (1 - masks).mean(axis=-1, keepdims=True, dtype=np.float32)
    power = (out**2).mean(axis=-1, keepdims=True, dtype=np.float32)
    rms = np.sqrt(power + np.float32(eps))
    desired = (1 - fraction) * rms + fraction * np.maximum(bank["mean_write_rms"][0], np.float32(eps))
    out *= np.where(power > eps, desired / rms, 1.)
    return bf16(native + (out - native), np)


def directions(x, y, learned, anchor, bank, eps, np):
    control = mass_matched(anchor, float(learned.sum()), np)
    masks = np.stack((learned, control))
    reached = writes(y, x @ bank["V"], masks, bank, eps, np)
    effects = reached.astype(np.float64) - y
    if not np.isfinite(effects).all():
        raise ValueError("Nonfinite post-rounding routes")
    return dict(zip(ARMS, effects, strict=True)), float(abs(control.sum() - learned.sum()))


def patches(residual, directions, cfg, np):
    choices = {f"{arm}/{tag}": prepare(residual, d * sign, cfg["phase1_dose_fraction"],
                                      cfg["dose_relative_tolerance"], cfg["minimum_direction_cosine"], np)
               for arm, d in directions.items() for tag, sign in SIGNS.items()}
    summaries = {key: dose_status(p) for key, p in choices.items()}
    admitted = all(qualified_pair(a, b, cfg) for a, b in combinations(summaries.values(), 2))
    return choices, summaries, admitted


def separation(vectors, first, second, np):
    a, b = (vectors[key] for key in (first, second))
    n = float(np.linalg.norm(a) * np.linalg.norm(b))
    return {"cosine": float(np.clip(float(a @ b) / n, -1, 1)) if n > 0 else None,
            "l2_distance": float(np.linalg.norm(a - b)), "exactly_equal": bool(np.array_equal(a, b))}

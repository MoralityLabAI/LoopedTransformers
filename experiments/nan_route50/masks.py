"""Capped soft budget and the IO18 T2 physical write calculation."""
from itertools import product

from experiments.nan_readpath37.run import bf16


def mass_matched(template, mass, np):
    """Euclidean projection onto the unit box with a fixed row sum."""
    source = np.asarray(template, dtype=np.float64)
    target = np.asarray(mass, dtype=np.float64)
    if (not np.isfinite(source).all() or not np.isfinite(target).all()
            or (source < 0).any() or (source > 1).any() or (target < 0).any() or (target > source.shape[-1]).any()):
        raise ValueError("Unexpected continuous-mask budget")
    shape = np.broadcast_shapes(source.shape[:-1], target.shape)
    low, high = np.full(shape, -1.), np.full(shape, 1.)
    for _ in range(50):
        mid = (low + high) / 2
        below = np.clip(source + mid[..., None], 0., 1.).sum(axis=-1) < target
        low = np.where(below, mid, low)
        high = np.where(below, high, mid)
    out = np.clip(source + ((low + high) / 2)[..., None], 0., 1.)
    if not np.allclose(out.sum(axis=-1), target, rtol=0, atol=1e-12):
        raise ValueError("A local control did not preserve the exact soft budget")
    return out


def relative_write_error(mask, native, reads, bank, phase, eps, np):
    """Delta gate is zero here. The native projection was recorded in BF16."""
    mask = np.asarray(mask, dtype=np.float32)
    mu = bank["mean_comp_acts"][phase]
    masked_acts = mask * reads + (1 - mask) * mu
    out = native + (masked_acts @ bank["U"] + bank["mean_delta_out"][phase] - native)
    frac = (1 - mask).mean(axis=-1, keepdims=True, dtype=np.float32)
    power = (out**2).mean(axis=-1, keepdims=True, dtype=np.float32)
    rms = np.sqrt(power + np.float32(eps))
    desired = (1 - frac) * rms + frac * np.maximum(bank["mean_write_rms"][phase], np.float32(eps))
    out = out * np.where(power > eps, desired / rms, 1.)
    reached = bf16(native + (out - native), np)
    den = np.linalg.norm(native.astype(np.float64), axis=-1)
    if not np.isfinite(reached).all() or (den <= 1e-12).any():
        raise ValueError("Archived native write normalization is undefined")
    return np.linalg.norm(reached.astype(np.float64) - native, axis=-1) / den


def best_cal_anchor(learned, native, reads, bank, phase, eps, archive, batch, np):
    mean = learned.mean(axis=0)
    anchors = [*product((0., 1.), repeat=learned.shape[1]), mean, archive]
    labels = [*("binary:" + "".join(str(int(v)) for v in a) for a in anchors[:-2]),
              "CAL_last_token_mean", "archived_CAL_all_token_mean"]
    scores = []
    for start in range(0, len(anchors), batch):
        selected = np.asarray(anchors[start:start + batch])
        maps = mass_matched(selected[:, None, :], learned.sum(axis=1)[None, :], np)
        score = relative_write_error(maps, native, reads, bank, phase, eps, np).mean(axis=1)
        scores.extend(map(float, score))
    winner = min(range(len(scores)), key=lambda i: (scores[i], i))
    return np.asarray(anchors[winner]), {"selected_anchor": labels[winner], "candidate_count": len(scores),
                                        "selected_CAL_mean_error": scores[winner],
                                        "last_token_mean_CAL_error": scores[-2], "archived_mean_CAL_error": scores[-1]}

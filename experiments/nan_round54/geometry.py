"""Build a BF16-aware, margin-null physical contrast and its matched sham."""
from hashlib import sha256

from experiments.nan_dose51.route import SIGNS, patches
from experiments.nan_lattice53.run import next_neighbors

ARMS = ("contrast", "sham")


def unit(v, np):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else None


def make_pair(h, raw, gradient, radius, threshold, tag, cfg, np):
    raw = unit(raw, np)
    if raw is None:
        return None, None, {"support": 0}
    down, up = next_neighbors(h.astype(np.float32), np)
    gaps = np.maximum(h - down.astype(np.float64), up.astype(np.float64) - h)
    if not np.isfinite(gaps).all() or min(gaps) <= 0:
        raise ValueError("Invalid symmetric BF16 spacing")
    support = np.ones(len(h), dtype=bool) if threshold is None else gaps <= threshold * radius * abs(raw)
    metrics = {"support": int(sum(support)), "retained_before_margin": float(np.linalg.norm(raw[support]))}
    if metrics["support"] < 3:
        return None, raw, metrics
    normal = unit(np.where(support, gradient, 0.), np)
    active = np.where(support, raw, 0.)
    if normal is not None:
        active -= float(active @ normal) * normal
    active = unit(active, np)
    if active is None:
        return None, raw, metrics
    metrics["continuous_contrast_cosine"] = float(active @ raw)
    seed = sha256(f"{cfg['sham_seed_namespace']}|{tag}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(seed[:8], "big"))
    sham = np.zeros_like(active)
    for spacing in np.unique(gaps[support]):
        ids = np.flatnonzero(support & (gaps == spacing))
        sham[ids] = rng.permutation(active[ids]) * rng.choice([-1., 1.], size=len(ids))
    if normal is not None:
        sham -= float(sham @ normal) * normal
    sham -= float(sham @ active) * active
    sham = unit(sham, np)
    if sham is None:
        return None, raw, metrics
    if (abs(float(sham @ active)) > 1e-10 or abs(float(sham @ raw)) > 1e-10
            or (normal is not None and max(abs(float(v @ normal)) for v in (active, sham)) > 1e-10)
            or np.any(active[~support] != 0.) or np.any(sham[~support] != 0.)):
        raise ValueError("Continuous matched support or projection failed")
    return dict(zip(ARMS, [active, sham])), raw, metrics


def match(h, directions, original, cfg, dose, np):
    edits, records, all_dose = patches(h, directions, dose, np)
    for key, edit in edits.items():
        if edit.realized_l2 > 0:
            signed = SIGNS[key.split("/")[1]]
            delta = edit.after - edit.before
            records[key]["cosine_with_original_contrast"] = float(delta @ original) * signed / edit.realized_l2
    separations = {
        sign: float(np.linalg.norm(edits[f"contrast/{sign}"].after - edits[f"sham/{sign}"].after))
        / edits[f"contrast/{sign}"].requested_l2 for sign in SIGNS
    }
    aligned = all(records[f"contrast/{sign}"].get("cosine_with_original_contrast", -1.)
                  >= cfg["minimum_reached_contrast_cosine"] for sign in SIGNS)
    sham_ok = all(abs(records[f"sham/{sign}"].get("cosine_with_original_contrast", 1.))
                  <= cfg["maximum_reached_sham_cosine"] for sign in SIGNS)
    distinct = min(separations.values()) >= cfg["minimum_contrast_sham_distance_in_nominal_radii"]
    return edits, {"dose_matched": all_dose, "contrast_retained": aligned,
                   "sham_specific": sham_ok, "far_enough": distinct, "arm_dose": records,
                   "distance_in_nominal_radii": separations,
                   "physical_gate": all_dose and aligned and sham_ok and distinct}


def verify(cfg, dose, np):
    rng = np.random.default_rng(54)
    h = (rng.integers(75, 250, size=3072) / 256).astype(np.float64)
    g = rng.normal(size=len(h))
    d = rng.normal(size=len(h))
    toy = dose | {"phase1_dose_fraction": .035}
    radius = np.linalg.norm(h) * toy["phase1_dose_fraction"]
    values, raw, features = make_pair(h, d, g, radius, .8, "fixture", cfg, np)
    assert features["support"] < len(h) and features["continuous_contrast_cosine"] > .5
    np.testing.assert_allclose([g @ v for v in values.values()], [0., 0.], atol=1e-10, rtol=0)
    np.testing.assert_allclose([raw @ values["sham"], values["contrast"] @ values["sham"]], [0., 0.], atol=1e-10)
    _, record = match(h, values, raw, cfg, toy, np)
    assert record["physical_gate"]
    other, _, _ = make_pair(h, d, g, radius, .8, "fixture", cfg, np)
    assert all(np.array_equal(other[k], v) for k, v in values.items())

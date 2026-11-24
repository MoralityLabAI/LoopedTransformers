"""Compare the same donor input under the fitted and initial physical operators."""

from hashlib import sha256
from math import isfinite

from experiments.nan_delay39.contrasts import dose_status
from experiments.nan_delay39.instrument import prepare
from experiments.nan_refit49.geometry import cos, difference


METHODS = ("svd", "t1_fit", "t2_fit", "t1_update", "t2_update", "orthogonal_sham")
SIGN = (("positive", 1), ("negative", -1))
CONTRASTS = (
    ("t2_update", "t1_update"), ("t2_fit", "t1_fit"),
    ("t2_fit", "svd"), ("t1_fit", "svd"),
    ("t2_update", "orthogonal_sham"), ("t1_update", "orthogonal_sham"),
)
CURRENT_DRIFT_LIMIT = 0.125


def bind_banks(t1, t2, np):
    initial, fit1 = t1
    peer, fit2 = t2
    if any(not np.array_equal(a, b) for a, b in zip(initial, peer, strict=True)):
        raise ValueError("The two IO18 fits require the same SVD initializer")
    return dict(zip(METHODS[:-1], (initial, fit1, fit2, difference(fit1, initial, np),
                                  difference(fit2, initial, np)), strict=True))


def materialize(delta, banks, root, world, np):
    if world not in ("A", "B") or delta.ndim != 1 or not np.isfinite(delta).all():
        raise ValueError("A finite, world-bound native input contrast is required")
    writers = {name: (delta @ v) @ u for name, (v, u) in banks.items()}
    span = np.stack([writers[name] for name in METHODS[:3]])
    if not np.isfinite(span).all() or not all(np.isfinite(u).all() for u in writers.values()):
        raise ValueError("The physical operator action must be finite")
    _, singular, right = np.linalg.svd(span, full_matrices=False)
    basis = right[singular > singular[0] * np.finfo(span.dtype).eps * max(span.shape)]
    seed = int.from_bytes(sha256(root.encode("utf-8")).digest()[:8], "big")
    seed_vector = np.random.default_rng(seed).normal(size=span.shape[-1])
    sham = seed_vector - (seed_vector @ basis.T) @ basis
    norm = np.linalg.norm(sham)
    if norm < 1e-12:
        raise ValueError("The approved physical bank leaves no seeded sham direction")
    writers["orthogonal_sham"] = (1 if world == "A" else -1) * sham / norm
    return writers


def panels(cases, cfg):
    def admit(methods):
        cells = [cases[method][sign] for method in methods for sign, _ in SIGN]
        requested = {p["requested_l2"] for p in cells}
        doses = [p["realized_l2"] for p in cells]
        nominal = max(requested)
        return all(p["eligible"] for p in cells) and len(requested) == 1 and nominal > 0 and (
            max(doses) - min(doses) <= nominal * cfg["max_contrast_dose_gap_fraction"])

    return {"all_arms": bool(admit(METHODS))} | {
        f"{a}_vs_{b}": bool(admit((a, b))) for a, b in CONTRASTS
    }


def plan(before, directions, cfg, np):
    cases, patches, unique = {}, [], {}
    for method in METHODS:
        vector = directions[method]
        natural = float(np.linalg.norm(vector))
        cases[method] = {}
        for sign, factor in SIGN:
            p = prepare(before, factor * vector, cfg["phase1_dose_fraction"], cfg["dose_relative_tolerance"],
                        cfg["minimum_direction_cosine"], np)
            index = None
            if p.eligible:
                raw = p.after.tobytes()
                if raw not in unique:
                    unique[raw] = len(patches)
                    patches.append(p)
                index = unique[raw]
            gain = p.requested_l2 / natural if method != "orthogonal_sham" and natural > 0 else None
            cases[method][sign] = dose_status(p) | {"physical_edit": index,
                    "natural_l2": natural if method != "orthogonal_sham" else None,
                    "nominal_over_natural": gain if gain is None or isfinite(gain) else None}
    return {"cases": cases, "qualified_panels": panels(cases, cfg), "distinct_rounded_edits": len(patches),
            "full_fit_direction_cosine": cos(directions["t1_fit"], directions["t2_fit"], np),
            "learned_direction_cosine": cos(directions["t1_update"], directions["t2_update"], np)}, patches

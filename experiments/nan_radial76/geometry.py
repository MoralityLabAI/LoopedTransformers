"""Match a randomized sham to the reached BF16 tangent edit."""
from math import sqrt

from experiments.nan_carry63.algebra import round_bf16
from experiments.nan_cibracket75.repair import compensate
from experiments.nan_cidose74.geometry import COSINE, DOSE, DOSE_TOLERANCE


ARMS = ("active", "original_seeded", "matched_seeded")
MAX_RADIAL_FRACTION = .01
MAX_NORM_GAP = .0002
MAX_TANGENT_COSINE = .05
RADIAL_GRID = (0., -.01, .01, -.02, .02, -.04, .04, -.08, .08)
TANGENT_GRID = (1., .995, 1.005, .99, 1.01, .98, 1.02, .96, 1.04)


def measure(after, native, axis, intended, np):
    change = after - native
    norm = float(np.linalg.norm(change))
    return {"l2": norm, "radial_l2": float(change @ axis),
            "direction_cosine": float(change @ intended / (norm * np.linalg.norm(intended))) if norm else None,
            "relative_norm_drift": float(np.linalg.norm(after)/np.linalg.norm(native) - 1)}


def plan(native, want, seeded, np):
    radius = float(np.linalg.norm(native))
    if radius == 0:
        return {}, {"status": "zero_native_write"}
    axis = native / radius
    active, active_method, _ = compensate(native, want, np)
    legacy, legacy_method, _ = compensate(native, seeded, np)
    result = {"active_planner": active_method, "original_sham_planner": legacy_method}
    if not active.eligible or not legacy.eligible:
        return {}, result | {"status": "baseline_dose_unqualified"}

    settled = active.after - native
    dose = active.realized_l2
    r = float(settled @ axis)
    anchored = settled - r*axis
    anchored_norm = float(np.linalg.norm(anchored))
    if anchored_norm == 0:
        return {}, result | {"status": "active_tangent_degenerate"}
    anchor = anchored / anchored_norm
    complement = seeded - (seeded @ axis)*axis - (seeded @ anchor)*anchor
    complement_norm = float(np.linalg.norm(complement))
    if complement_norm == 0:
        return {}, result | {"status": "sham_tangent_degenerate"}
    complement /= complement_norm

    ideal = r*axis + sqrt(max(0., dose**2-r**2))*complement
    nominal = DOSE * radius
    grid = [(delta_r, factor) for delta_r in RADIAL_GRID for factor in TANGENT_GRID]
    candidates = round_bf16(np.stack([native + (r+dr*nominal)*axis + factor*anchored_norm*complement
                                       for dr, factor in grid]), np).astype(np.float64)
    changes = candidates - native
    norms = np.linalg.norm(changes,axis=1)
    radial = changes @ axis
    tangent = changes - radial[:, None]*axis
    tangent_norm = np.linalg.norm(tangent,axis=1)
    complete = (norms > 0) & (tangent_norm > 0)
    cos = np.divide(changes@ideal, norms*np.linalg.norm(ideal), out=np.zeros_like(norms), where=complete)
    orth = np.divide(tangent@anchor,tangent_norm,out=np.zeros_like(norms),where=complete)
    norm_gap = abs(np.linalg.norm(candidates,axis=1)-np.linalg.norm(active.after))/radius
    band = np.maximum(norms, max(dose,legacy.realized_l2)) - np.minimum(norms, min(dose,legacy.realized_l2))
    limits = np.stack((abs(norms/nominal - 1)/DOSE_TOLERANCE, band/(nominal*DOSE_TOLERANCE),
                       abs(radial-r)/(nominal*MAX_RADIAL_FRACTION),
                       norm_gap/MAX_NORM_GAP, abs(orth)/MAX_TANGENT_COSINE))
    worst = np.max(limits,axis=0)
    valid = complete & (cos >= COSINE) & (worst <= 1.)
    original = {"active": active.after, "original_seeded": legacy.after}
    reports = {"active": measure(active.after,native,axis,want,np),
               "original_seeded": measure(legacy.after,native,axis,seeded,np)}
    result.update(nominal_l2=nominal, candidate_count=len(grid), qualified_candidates=int(valid.sum()), arms=reports)
    if not valid.any():
        return {}, result | {"status": "radial_sham_unqualified", "best_gate_usage": float(worst.min())}
    ix = int(np.argmin(np.where(valid,worst,np.inf)))
    selected = candidates[ix]
    reports["matched_seeded"] = measure(selected,native,axis,ideal,np)
    return original | {"matched_seeded": selected}, result | {
        "status": "matched", "joint_dose_spread_fraction": float(band[ix]/nominal),
        "matched_radial_gap_fraction": float(abs(radial[ix]-r)/nominal),
        "matched_relative_reached_norm_gap": float(norm_gap[ix]),
        "realized_tangent_cosine": float(orth[ix]), "chosen_grid": list(grid[ix])}


def verify(np):
    rng = np.random.default_rng(76019)
    native = round_bf16(rng.normal(size=768),np).astype(np.float64)
    want, sham = rng.normal(size=(2,len(native)))
    axis = native/np.linalg.norm(native)
    want -= float(want@axis)*axis
    sham -= float(sham@axis)*axis
    changes, rec = plan(native,want,sham,np)
    assert set(changes)==set(ARMS) and rec["status"]=="matched"
    for candidate in changes.values():
        np.testing.assert_array_equal(candidate,round_bf16(candidate,np))
    assert abs(rec["realized_tangent_cosine"])<=MAX_TANGENT_COSINE
    assert rec["matched_radial_gap_fraction"]<=MAX_RADIAL_FRACTION
    assert rec["matched_relative_reached_norm_gap"]<=MAX_NORM_GAP
    L, r = rec["arms"]["matched_seeded"]["l2"], rec["arms"]["matched_seeded"]["radial_l2"]
    R = np.linalg.norm(native)
    np.testing.assert_allclose(np.linalg.norm(changes["matched_seeded"])**2, R*R + 2*R*r + L*L,
                               atol=1e-12, rtol=1e-12)
    return rec

"""LS-SAE01: portable SAE causal observer geometry for looped hidden states.

The planner accepts already frozen physical encoder/decoder weights and a
clean hidden-state vector. All admission and sham selection is outcome blind.
The caller is responsible for source/target feature matching, clean rival
selection and model-specific continuation.

Requires NumPy. Independent LoopedTransformers instrumentation, not a
bitwise replay of the Torch SMC-16/17 random control generators.
"""
from __future__ import annotations

import hashlib
from typing import Callable

import numpy as np

PROTOCOL = "LS-SAE01/96-feature-active4-v1"
FEATURES = 96
ACTIVE = 4
GAUSSIAN_CANDIDATES = 128
BASELINE_TRIES = 4
TAU = 0.02
CODE_TOL = 1e-4
ORTHO_TOL = 1e-8
MIN_NORM = 1e-8
MAX_NATIVE_MULTIPLE = 6.0


def _seed(*parts: object) -> int:
    material = "|".join(map(str, (PROTOCOL,) + parts))
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big") & ((1 << 63) - 1)


def _as64(x: object) -> np.ndarray:
    v = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(v)):
        raise ValueError("nonfinite hidden state, dictionary or bias")
    return v


def encode_top4(h: np.ndarray, rows: np.ndarray, biases: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Physical affine SAE encoder followed by ReLU/top-four support."""
    pre = rows @ h + biases
    if pre.shape != (FEATURES,) or not np.all(np.isfinite(pre)):
        raise ValueError("expected 96 finite preactivations")
    order = np.argsort(-pre, kind="stable")[:ACTIVE]
    z = np.zeros(FEATURES, dtype=np.float64)
    z[order] = np.maximum(pre[order], 0.0)
    return z, pre


def _project(vec: np.ndarray, rows: np.ndarray) -> np.ndarray | None:
    """Orthogonal projection into the kernel of rows."""
    if len(rows) == 0:
        return vec.copy()
    _, singular, vh = np.linalg.svd(rows, full_matrices=False)
    if singular[-1] <= 1e-10:
        return None
    v = vec - vh.T @ (vh @ vec)
    if np.max(np.abs(rows @ v)) > ORTHO_TOL:
        return None
    return v


def _tangent_basis(rows: np.ndarray) -> np.ndarray:
    _, singular, vh = np.linalg.svd(rows, full_matrices=True)
    if singular[-1] <= 1e-10:
        raise ValueError("four active physical encoder covectors rank deficient")
    u = vh[ACTIVE:].T.copy()
    if np.max(np.abs(rows @ u)) > ORTHO_TOL:
        raise ArithmeticError("SVD tangent basis failed orthogonality")
    return u


def plan_edit(h: object, encoder_rows: object, encoder_bias: object,
              decoder_columns: object, matched_feature: int,
              *, case_key: str) -> dict:
    """Plan a matched selective kill and 131 candidate tangent shams.

    Physical encoder/decoder vectors and h use identical coordinates.
    The immutable case_key uses calibration/family/puzzle/goal metadata.
    Heldout answers and task outputs are excluded from its construction.
    """
    h = _as64(h)
    W = _as64(encoder_rows)
    bias = _as64(encoder_bias)
    D = _as64(decoder_columns)
    if h.ndim != 1 or len(h) <= ACTIVE or W.shape != (FEATURES, len(h)):
        raise ValueError("bad physical SAE encoder shape")
    if bias.shape != (FEATURES,) or D.shape != (FEATURES, len(h)):
        raise ValueError("bad physical SAE bias/decoder shape")
    if not isinstance(matched_feature, int) or not 0 <= matched_feature < FEATURES:
        raise ValueError("matched feature index outside frozen dictionary")
    if not case_key or not isinstance(case_key, str):
        raise ValueError("case_key must be a nonempty fixed visible identifier")
    status = {"protocol": PROTOCOL, "eligible": False,
              "matched_feature": matched_feature, "case_key": case_key}

    def abstain(reason: str) -> dict:
        return {**status, "reason": reason}

    clean_code, _ = encode_top4(h, W, bias)
    active = np.flatnonzero(clean_code > 0).tolist()
    z = float(clean_code[matched_feature])
    if z <= 0:
        return abstain("matched_inactive")
    if len(active) != ACTIVE or matched_feature not in active:
        return abstain("missing_strict_active_four")
    other = [i for i in active if i != matched_feature]
    aj = W[matched_feature]
    protected = W[other]
    v = _project(aj, protected)
    if v is None or np.linalg.norm(v) <= MIN_NORM:
        return abstain("protected_rank_or_target_span_obstruction")
    delta = (z + TAU) * v / np.dot(v, v)
    dose = float(np.linalg.norm(delta))
    native = z * float(np.linalg.norm(D[matched_feature]))
    if native <= 1e-12 or dose > MAX_NATIVE_MULTIPLE * native:
        return abstain("native_decoder_dose_cap")
    if abs(float(np.dot(aj, delta)) - (z + TAU)) > ORTHO_TOL:
        return abstain("target_affine_constraint")
    killed, _ = encode_top4(h - delta, W, bias)
    drift_k = max(abs(float(killed[i] - clean_code[i])) /
                  (1 + float(clean_code[i])) for i in other)
    if (float(killed[matched_feature]) > 1e-7 or
        any(float(killed[i]) <= 0 for i in other) or drift_k > CODE_TOL):
        return abstain("kill_reencoding_failed")

    A = W[active]
    try:
        U = _tangent_basis(A)
    except (ValueError, ArithmeticError):
        return abstain("four_active_rank_or_tangent_failure")
    inact = [j for j in range(FEATURES) if j not in active]
    B = W[inact]
    target_inactive = B @ delta
    C = B @ U
    coeff = np.linalg.lstsq(C, target_inactive, rcond=None)[0]
    s0 = U @ coeff
    residual_floor = float(np.linalg.norm(C @ coeff - target_inactive))
    inactive_rank = int(np.linalg.matrix_rank(B))
    candidate_receipts = []
    candidates: list[tuple[str, np.ndarray, float]] = []

    def offer(name: str, candidate: np.ndarray | None) -> None:
        item = {"candidate": name, "accepted": False, "epsilon": None}
        if candidate is None:
            candidate_receipts.append(item)
            return
        size = float(np.linalg.norm(candidate))
        if not np.isfinite(size) or size <= MIN_NORM:
            candidate_receipts.append(item)
            return
        sham = candidate * (dose / size)
        if (abs(float(np.linalg.norm(sham)) - dose) > ORTHO_TOL
            or np.max(np.abs(A @ sham)) > ORTHO_TOL):
            candidate_receipts.append(item)
            return
        recode, _ = encode_top4(h - sham, W, bias)
        drift = max(abs(float(recode[i] - clean_code[i])) /
                    (1 + float(clean_code[i])) for i in active)
        if any(float(recode[i]) <= 0 for i in active) or drift > CODE_TOL:
            candidate_receipts.append(item)
            return
        epsilon = float(np.linalg.norm(B @ (delta - sham)))
        item.update({"accepted": True, "epsilon": epsilon,
                     "max_active_relative_drift": drift})
        candidate_receipts.append(item)
        candidates.append((name, sham, epsilon))

    baseline = None
    for t in range(BASELINE_TRIES):
        rng = np.random.default_rng(_seed(case_key, "baseline", t))
        vec = _project(rng.normal(size=len(h)), A)
        prev = len(candidates)
        offer("baseline", vec)
        if len(candidates) > prev:
            baseline = candidates[-1]
            baseline_trials = t + 1
            break
        # The 131-candidate comparison counts the first VALID frozen
        # baseline, rather than failed attempts in its four-seed search.
        candidate_receipts.pop()
    if baseline is None:
        return abstain("baseline_sham_reencoding_failed")

    offer("least_squares_positive", s0)
    offer("least_squares_negative", -s0)
    for t in range(GAUSSIAN_CANDIDATES):
        rng = np.random.default_rng(_seed(case_key, "candidate", t))
        offer(f"random_{t:03d}", _project(rng.normal(size=len(h)), A))
    if len(candidate_receipts) != 1 + 2 + GAUSSIAN_CANDIDATES:
        raise AssertionError("frozen candidate set must contain 131 entries")

    selected = min(candidates, key=lambda item: item[2])
    if selected[2] > baseline[2] + 1e-8 or selected[2] + 1e-7 < residual_floor:
        raise AssertionError("inactive disturbance floor or control ordering violated")
    base_sham, best_sham = baseline[1], selected[1]
    return {**status, "eligible": True, "reason": "eligible",
        "kill": delta, "baseline": base_sham, "best": best_sham,
        "diagnostics": {
            "active_ids": active, "protected_ids": other,
            "inactive_ids": inact, "clean_code": clean_code.tolist(),
            "physical_norm": dose, "native_decoder_norm": native,
            "max_kill_protected_drift": drift_k,
            "inactive_encoder_rank": inactive_rank,
            "tangent_dimension": U.shape[1],
            "inactive_residual_floor": residual_floor,
            "inactive_residual_baseline": baseline[2],
            "inactive_residual_best": selected[2],
            "relative_inactive_residual": (selected[2] / baseline[2]
                if baseline[2] > 0 else None),
            "selected_candidate": selected[0],
            "baseline_seed_attempts": baseline_trials,
            "candidates": candidate_receipts,
            "all96_kill_preactivation_shift": (W @ delta).tolist(),
            "all96_baseline_preactivation_shift": (W @ base_sham).tolist(),
            "all96_best_preactivation_shift": (W @ best_sham).tolist(),
        }}


def score_margin(plan: dict, h: object, continuation: Callable[[np.ndarray], float]) -> dict:
    """Score a frozen scalar margin on clean and edited single-site states.

    The caller constructs the model-specific FP64 continuation and fixes the
    clean correct/rival digit before invoking this evaluation.
    """
    if not plan.get("eligible"):
        raise ValueError("must abstain on ineligible feature episodes")
    state = _as64(h)
    clean = float(continuation(state))
    kill = float(continuation(state - plan["kill"]))
    baseline = float(continuation(state - plan["baseline"]))
    best = float(continuation(state - plan["best"]))
    if not np.all(np.isfinite([clean, kill, baseline, best])):
        raise ArithmeticError("nonfinite frozen-model continuation")
    return {
        "clean_margin": clean,
        "kill_margin": kill,
        "baseline_sham_margin": baseline,
        "best_sham_margin": best,
        "kill_minus_baseline_margin_loss": baseline - kill,
        "kill_minus_best_margin_loss": best - kill,
    }

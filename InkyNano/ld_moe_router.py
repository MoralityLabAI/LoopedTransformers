"""NumPy reference for native selection and experimental ranking corrections.

Not a native runtime or a trained predictor. Scores are the native sigmoid
outputs, bias affects ranking only. Costs are in explicit caller-supplied units.
Stable ties are a deterministic reference convention; native tie parity must
be checked separately. Cache residency is deliberately not a hard route mask.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Route:
    ids: tuple
    weights: tuple
    native_ids: tuple
    margin: float
    correction_inf: float
    membership_certified: bool
    miss_ids: tuple


def _vec(x, n, name):
    v = np.asarray(x, dtype=np.float64)
    if v.shape != (n,) or not np.isfinite(v).all():
        raise ValueError("invalid " + name)
    return v


def choose(scores, bias, *, k=8, scale=8.0, delta=None, cost=None,
           penalty=0.0, resident=(), exact=False, numeric_slack=0.0):
    """Select over full expert universe; never silently substitute on a miss.

    exact=True disables correction and cost, including when some selected
    experts are absent. Returned misses require a fetch or CPU execution.
    Otherwise native sigmoid weights are normalized over the selected IDs.
    """
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 1 or not np.isfinite(s).all() or np.any(s < 0) or np.any(s > 1):
        raise ValueError("expected finite native sigmoid scores")
    n = len(s)
    if type(k) is not int or not 1 <= k < n:
        raise ValueError("invalid k")
    if not np.isfinite([scale, penalty, numeric_slack]).all() or scale <= 0 or penalty < 0 or numeric_slack < 0:
        raise ValueError("invalid scale, penalty or numerical slack")
    b = _vec(bias, n, "bias")
    d = np.zeros(n) if delta is None else _vec(delta, n, "delta")
    c = np.zeros(n) if cost is None else _vec(cost, n, "cost")
    if np.any(c < 0):
        raise ValueError("negative transfer cost")
    old = frozenset(resident)
    if any(not isinstance(e, (int, np.integer)) or isinstance(e, (bool, np.bool_)) or not 0 <= e < n for e in old):
        raise ValueError("invalid resident ID")
    native = s+b
    if not np.isfinite(native).all():
        raise ValueError("invalid native rank")
    order = np.argsort(-native, kind="stable")
    native_ids = tuple(map(int, order[:k]))
    margin = float(native[order[k-1]] - native[order[k]])
    perturb = np.zeros(n) if exact else d-penalty*c
    adjusted = native + perturb
    if not np.isfinite(adjusted).all() or not np.isfinite(perturb).all():
        raise ValueError("invalid adjusted rank")
    inf = float(np.max(np.abs(perturb)))
    certified = margin > 2*(inf+numeric_slack)
    ids = tuple(map(int, np.argsort(-adjusted, kind="stable")[:k]))
    w = s[list(ids)]
    if w.sum() <= 0:
        raise ValueError("zero selected mixture mass")
    weights = scale*w/(w.sum()+1e-20)
    if certified and set(ids) != set(native_ids):
        raise AssertionError("strict margin certificate contradicted")
    return Route(ids, tuple(map(float, weights)), native_ids, margin, inf,
                 certified, tuple(e for e in ids if e not in old))


def substitution_bound(outputs, replacements, weights):
    """Local triangle bound for a specified fixed-weight expert mapping.

    outputs[e] and replacements[e] are corresponding evaluated vectors,
    not predicted similarity scores. No bound for downstream model accuracy.
    """
    a, b = np.asarray(outputs, dtype=np.float64), np.asarray(replacements, dtype=np.float64)
    q = np.asarray(weights, dtype=np.float64)
    if a.ndim != 2 or a.shape != b.shape or q.shape != (len(a),) or not all(np.isfinite(x).all() for x in (a,b,q)):
        raise ValueError("invalid expert output shapes/values")
    d = a-b
    return {"actual_l2": float(np.linalg.norm(q @ d)),
            "triangle_bound": float(np.abs(q) @ np.linalg.norm(d, axis=1))}

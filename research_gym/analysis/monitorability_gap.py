"""RMP-MG-01: local observation-null directions and measured finite effects.

NumPy-only core. This module does not infer intent, train an actor, or equate
small observation drift with indistinguishability for arbitrary monitors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] == 0 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite matrix with a positive state width")
    return value


@dataclass(frozen=True)
class NullDirection:
    direction: np.ndarray
    sensitivity: float
    observation_derivative_norm: float
    future_derivative_norm: float
    observation_rank: int
    nullity: int
    rank_tolerance: float
    observation_singular_values: np.ndarray
    future_singular_values_on_nullspace: np.ndarray

    def receipt(self) -> dict:
        return {
            "sensitivity": self.sensitivity,
            "observation_derivative_norm": self.observation_derivative_norm,
            "future_derivative_norm": self.future_derivative_norm,
            "observation_rank": self.observation_rank,
            "nullity": self.nullity,
            "rank_tolerance": self.rank_tolerance,
            "observation_singular_values": self.observation_singular_values.tolist(),
            "future_singular_values_on_nullspace": self.future_singular_values_on_nullspace.tolist(),
            "direction": self.direction.tolist(),
        }


def null_project(rows: np.ndarray, observation_jacobian: np.ndarray,
                 *, rtol: float = 1e-10, atol: float = 1e-12) -> tuple[np.ndarray, dict]:
    """Right-project rows into the numerical kernel, without forming a d-by-d matrix.

    Thresholds define a NUMERICAL nullspace. Small retained residuals and a
    threshold sweep belong in every empirical report.
    """
    rows = _matrix(rows, "rows")
    c = _matrix(observation_jacobian, "observation_jacobian")
    if rows.shape[1] != c.shape[1]:
        raise ValueError("rows and observation_jacobian must have the same state width")
    if not np.isfinite([rtol, atol]).all() or rtol < 0 or atol < 0:
        raise ValueError("rank tolerances must be finite and nonnegative")
    if c.shape[0]:
        _, singular, right = np.linalg.svd(c, full_matrices=False)
    else:
        singular, right = np.empty(0), np.empty((0, c.shape[1]))
    threshold = max(atol, rtol * (float(singular[0]) if singular.size else 0.0))
    rank = int(np.count_nonzero(singular > threshold))
    basis = right[:rank]
    result = rows - (rows @ basis.T) @ basis
    return result, {"rank": rank, "nullity": c.shape[1] - rank,
                    "threshold": threshold, "singular_values": singular}


def strongest_null_direction(observation_jacobian: np.ndarray,
                              future_jacobian: np.ndarray, *,
                              rtol: float = 1e-10, atol: float = 1e-12) -> NullDirection:
    """Compute ||B P_ker(C)||_2 and a unit right-singular direction.

    A zero vector is returned when there is no numerically meaningful effect.
    """
    c = _matrix(observation_jacobian, "observation_jacobian")
    b = _matrix(future_jacobian, "future_jacobian")
    if b.shape[0] == 0:
        raise ValueError("future_jacobian needs at least one output")
    projected, meta = null_project(b, c, rtol=rtol, atol=atol)
    _, singular, right = np.linalg.svd(projected, full_matrices=False)
    beta = float(singular[0]) if singular.size else 0.0
    effect_tol = max(atol, rtol * float(np.linalg.norm(b, 2)))
    if meta["nullity"] == 0 or beta <= effect_tol:
        direction = np.zeros(c.shape[1])
        beta = 0.0
    else:
        direction = right[0]
        direction, _ = null_project(direction[None], c, rtol=rtol, atol=atol)
        direction = direction[0]
        norm = float(np.linalg.norm(direction))
        if norm == 0:
            raise RuntimeError("nonzero projected sensitivity produced a zero direction")
        direction /= norm
        # Deterministic orientation for portable receipts. Both signs are tested.
        if direction[int(np.argmax(np.abs(direction)))] < 0:
            direction = -direction
        beta = float(np.linalg.norm(b @ direction))
    return NullDirection(direction=direction, sensitivity=beta,
                         observation_derivative_norm=float(np.linalg.norm(c @ direction)),
                         future_derivative_norm=float(np.linalg.norm(b @ direction)),
                         observation_rank=meta["rank"], nullity=meta["nullity"],
                         rank_tolerance=meta["threshold"],
                         observation_singular_values=meta["singular_values"],
                         future_singular_values_on_nullspace=singular)


def taylor_bounds(radius: float, sensitivity: float, observation_curvature: float,
                  future_curvature: float, observation_linear_residual: float = 0.0) -> dict:
    """Bounds conditional on valid directional Hessian bounds over the WHOLE segment.

    The curvature constants are caller-supplied assumptions; this function does
    not estimate or certify them from a finite radius sweep.
    """
    values = np.asarray([radius, sensitivity, observation_curvature,
                         future_curvature, observation_linear_residual], dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("radius, sensitivity, curvature, and residual must be finite and nonnegative")
    return {
        "observation_drift_upper": radius * observation_linear_residual + .5 * observation_curvature * radius ** 2,
        "future_drift_lower": max(0.0, radius * sensitivity - .5 * future_curvature * radius ** 2),
        "status": "CONDITIONAL_ON_SEGMENT_CURVATURE_BOUNDS",
    }


def discrete_cmi(joint: np.ndarray) -> float:
    """I(A;B|C) in nats for an exact or empirical finite joint table [A,B,C]."""
    p = np.asarray(joint, dtype=np.float64)
    if p.ndim != 3 or not np.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
        raise ValueError("joint must be a finite nonnegative 3-D table with positive mass")
    p = p / p.sum()
    ac = p.sum(axis=1, keepdims=True)
    bc = p.sum(axis=0, keepdims=True)
    c = p.sum(axis=(0, 1), keepdims=True)
    ratio = np.divide(p * c, ac * bc, out=np.ones_like(p), where=(ac * bc) > 0)
    positive = p > 0
    return float(np.sum(p[positive] * np.log(ratio[positive])))


def log_loss(labels: Sequence[int], probabilities: np.ndarray) -> float:
    labels = np.asarray(labels)
    p = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must be a one-dimensional integer array")
    if p.ndim != 2 or len(p) != len(labels) or len(labels) == 0 or p.shape[1] < 2:
        raise ValueError("probabilities must have shape [nonempty items, classes>=2]")
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-9, rtol=1e-9):
        raise ValueError("probability rows must be finite, nonnegative, and normalized")
    if (labels < 0).any() or (labels >= p.shape[1]).any():
        raise ValueError("label index out of range")
    selected = p[np.arange(len(labels)), labels]
    if (selected == 0).any():
        return float("inf")
    return float(-np.log(selected).mean())


def balanced_bayes_accuracy(p0: np.ndarray, p1: np.ndarray) -> float:
    """Optimal equal-prior accuracy for known finite observation distributions."""
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    if p0.ndim != 1 or p0.shape != p1.shape or len(p0) == 0:
        raise ValueError("distributions must be equally sized nonempty vectors")
    if not np.isfinite(p0).all() or not np.isfinite(p1).all() or (p0 < 0).any() or (p1 < 0).any():
        raise ValueError("distributions must be finite and nonnegative")
    if not np.isclose(p0.sum(), 1) or not np.isclose(p1.sum(), 1):
        raise ValueError("distributions must sum to one")
    return .5 + .25 * float(np.abs(p0 - p1).sum())


def finite_effects(observation_before: np.ndarray, observation_after: np.ndarray,
                   future_before: np.ndarray, future_after: np.ndarray) -> dict:
    arrays = [np.asarray(x, dtype=float).reshape(-1) for x in
              [observation_before, observation_after, future_before, future_after]]
    a, aa, b, bb = arrays
    if a.shape != aa.shape or b.shape != bb.shape or not all(np.isfinite(x).all() for x in arrays):
        raise ValueError("before/after vectors must match and contain finite values")
    return {"observation_l2": float(np.linalg.norm(aa - a)),
            "future_l2": float(np.linalg.norm(bb - b))}


def torch_jacobians(state, observation_fn, future_fn) -> tuple[np.ndarray, np.ndarray]:
    """Differentiate two flat, differentiable observation functions at a state.

    Functions must return finite tensors and preserve their input's dtype/device.
    Dropout and sampling should be disabled by the caller. PyTorch is optional
    until this function is called.
    """
    import torch
    if not torch.is_tensor(state) or not state.is_floating_point() or not bool(torch.isfinite(state).all()):
        raise ValueError("state must be a finite floating-point PyTorch tensor")
    shape = state.shape
    flat = state.detach().reshape(-1).requires_grad_(True)
    def wrap(fn):
        def run(z):
            out = fn(z.reshape(shape))
            if not torch.is_tensor(out) or out.numel() == 0 or not bool(torch.isfinite(out).all()):
                raise ValueError("observation functions must return nonempty finite tensors")
            return out.reshape(-1)
        return run
    c = torch.autograd.functional.jacobian(wrap(observation_fn), flat, vectorize=False)
    b = torch.autograd.functional.jacobian(wrap(future_fn), flat, vectorize=False)
    return c.detach().cpu().double().numpy(), b.detach().cpu().double().numpy()

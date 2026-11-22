"""Numerical checks for Inkling's two-source cross-pass reader.

All indices here are zero-based. Goodfire's layer 13 / coordinate 2027
correspond to Hugging Face decoder layer 12 / hidden coordinate 2027.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

WIDTH = 2304
TARGET_COORD = 2027
HF_READER_LAYER = 12
EPSILON = 1e-6
ARMS = ("zero_2027", "l2_sham", "norm_preserving_zero", "swap", "reader_rescue")


def rms_normalize(vector: np.ndarray) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float64)
    if v.ndim != 1 or not np.isfinite(v).all():
        raise ValueError("expected a finite 1D vector")
    return v / np.sqrt(np.mean(v * v) + EPSILON)


def sham_coordinate(root_id: str, width: int = WIDTH, target: int = TARGET_COORD) -> int:
    if width <= 1 or not 0 <= target < width:
        raise ValueError("invalid width or target")
    digest = hashlib.sha256(f"INK-ROUTE-01|{root_id}|sham-v1".encode()).digest()
    candidate = int.from_bytes(digest[:8], "big") % (width - 1)
    return candidate if candidate < target else candidate + 1


def patch_vector(vector: np.ndarray, arm: str, sham: int, target: int = TARGET_COORD) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float64)
    if v.ndim != 1 or not np.isfinite(v).all():
        raise ValueError("expected a finite 1D vector")
    if arm not in ARMS or not 0 <= target < len(v) or not 0 <= sham < len(v) or sham == target:
        raise ValueError("invalid arm or indices")
    patched = v.copy()
    if arm in ("zero_2027", "reader_rescue", "norm_preserving_zero"):
        patched[target] = 0.0
        if arm == "norm_preserving_zero":
            denominator = float(np.linalg.norm(patched))
            if denominator == 0:
                raise ValueError("cannot preserve norm of an all-target vector")
            patched *= float(np.linalg.norm(v)) / denominator
    elif arm == "l2_sham":
        patched[sham] -= v[target]  # same exact L2 dose as removing target
    elif arm == "swap":
        patched[target], patched[sham] = v[sham], v[target]
    return patched


@dataclass(frozen=True)
class ReaderFit:
    beta3: float
    reconstruction_relative_error: float
    gap_logit: float


def reconstruct_reader(block2: np.ndarray, block3: np.ndarray, message: np.ndarray) -> list[ReaderFit]:
    """Recover beta_3 from y=(1-beta)*norm(b2)+beta*norm(b3).

    This is an output-side projection, not a read of secret query weights.
    With nondegenerate source blocks it reconstructs all four Q/K/V/MoE readers.
    """
    b2, b3 = rms_normalize(block2), rms_normalize(block3)
    messages = np.asarray(message, dtype=np.float64)
    if messages.shape != (4, b2.size) or not np.isfinite(messages).all():
        raise ValueError("message must have shape (4, hidden_size)")
    direction = b3 - b2
    denom = float(direction @ direction)
    if denom < 1e-12:
        raise ValueError("reader sources are indistinguishable")
    fits = []
    for y in messages:
        beta = float((y - b2) @ direction / denom)
        fitted = b2 + beta * direction
        relative_error = float(np.linalg.norm(y - fitted) / max(np.linalg.norm(y), 1e-12))
        clipped = min(max(beta, 1e-12), 1 - 1e-12)
        fits.append(ReaderFit(beta, relative_error, float(np.log(clipped / (1 - clipped)))))
    return fits


def vector_metrics(before: np.ndarray, after: np.ndarray, coord: int = TARGET_COORD) -> dict[str, float]:
    before, after = np.asarray(before, dtype=np.float64), np.asarray(after, dtype=np.float64)
    if before.shape != after.shape or before.ndim != 1:
        raise ValueError("shape mismatch")
    normalized = rms_normalize(before)
    return {
        "raw_coordinate": float(before[coord]),
        "rms_normalized_coordinate": float(normalized[coord]),
        "coordinate_squared_norm_fraction": float(normalized[coord] ** 2 / np.dot(normalized, normalized)),
        "raw_l2_delta": float(np.linalg.norm(after - before)),
        "raw_l2_before": float(np.linalg.norm(before)),
        "raw_l2_after": float(np.linalg.norm(after)),
    }

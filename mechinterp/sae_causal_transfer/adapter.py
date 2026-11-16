"""A source-pinned input contract for SAE probing of looped-model states.

Extract the arrays from a model's own cached hidden states and SAE weights.
The probe itself imports no Torch and never fits weights on heldout states.
"""
from __future__ import annotations

import hashlib
import numpy as np


def to_physical_dictionary(encoder_weight, encoder_bias, decoder_weight,
                           state_mean, state_scale) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert an SAE trained on (h-mean)/scale into physical hidden coordinates.

    encoder_weight: features x dim
    encoder_bias: features
    decoder_weight: dim x features (PyTorch Linear module convention)
    output decoder columns: features x dim
    """
    E = np.asarray(encoder_weight, dtype=np.float64)
    b = np.asarray(encoder_bias, dtype=np.float64)
    D = np.asarray(decoder_weight, dtype=np.float64)
    mean = np.asarray(state_mean, dtype=np.float64)
    scale = np.asarray(state_scale, dtype=np.float64)
    if E.ndim != 2 or b.shape != (E.shape[0],):
        raise ValueError("encoder weights or bias shape wrong")
    if mean.shape != (E.shape[1],) or scale.shape != mean.shape or D.shape != (E.shape[1], E.shape[0]):
        raise ValueError("state normalization or decoder shape wrong")
    if (not np.all(np.isfinite(E)) or not np.all(np.isfinite(b)) or
        not np.all(np.isfinite(D)) or not np.all(np.isfinite(mean)) or
        not np.all(np.isfinite(scale)) or np.any(scale <= 0)):
        raise ValueError("SAE tensor is nonfinite or scale nonpositive")
    physical_rows = E / scale[None, :]
    physical_bias = b - physical_rows @ mean
    physical_columns = (D * scale[:, None]).T
    return physical_rows, physical_bias, physical_columns


def fingerprint_arrays(*arrays) -> str:
    """SHA256 of a canonical, little-endian float64 numerical-array transcript.

    This is a new LoopedTransformers manifest hash, not the original PyTorch
    checkpoint-file SHA. Retain original file-byte SHA separately.
    """
    h = hashlib.sha256()
    for a in arrays:
        arr = np.asarray(a, dtype="<f8")
        if not np.all(np.isfinite(arr)):
            raise ValueError("nonfinite array cannot be source locked")
        h.update(repr(tuple(arr.shape)).encode() + b"\0")
        h.update(arr.tobytes(order="C"))
    return h.hexdigest()

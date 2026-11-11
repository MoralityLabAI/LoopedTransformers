"""Feature-variable association with select-then-confirm, used by controls 1, 2, 7 and 8."""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from .analysis import AnalysisModel
from .bundle import TraceExample
from .dictionaries import TopKDictionary, model_states, rms


FOUND_MARGIN = 0.35
FOUND_P = 0.001


def hash_bit(example_ids: Sequence[str]) -> np.ndarray:
    """Control 8's variable: bit 0 of the SHA-256 of each example id. No model can compute it."""

    return np.array([hashlib.sha256(identifier.encode()).digest()[0] & 1 for identifier in example_ids])


def token_parity(rows: Sequence[TraceExample], position: int = 4) -> np.ndarray:
    """Control 7's planted variable: the parity of one input token."""

    return np.array([row.tokens[position] % 2 for row in rows])


@torch.no_grad()
def answer_codes(model: AnalysisModel, dictionary: TopKDictionary, tokens: Tensor, *, batch: int = 250) -> Tensor:
    """Dictionary codes of the answer-position state after every visit: (N, visits, width), on CPU."""

    device = next(model.model.parameters()).device
    parts = []
    for start in range(0, tokens.shape[0], batch):
        states = model_states(model, tokens[start : start + batch].to(device))[:, 1:, -1]
        parts.append(dictionary.encode(states / rms(states)).cpu())
    return torch.cat(parts)


def auroc_columns(activations: Tensor, labels: np.ndarray) -> Tensor:
    """AUROC of every column for a binary label, ties counted as half. activations: (N, F)."""

    labels_t = torch.as_tensor(labels.astype(bool), device=activations.device)
    positive = activations[labels_t].T.contiguous()
    negative = activations[~labels_t].T.contiguous().sort(dim=1).values
    below = torch.searchsorted(negative, positive, right=False).float()
    at_or_below = torch.searchsorted(negative, positive, right=True).float()
    wins = (below + 0.5 * (at_or_below - below)).sum(dim=1)
    return wins / (positive.shape[1] * negative.shape[1])


def select_then_confirm(
    select_codes: Tensor, select_labels: np.ndarray, report_codes: Tensor, report_labels: np.ndarray,
    *, rounds: int = 1000, seed: int = 0, device: str | torch.device = "cpu",
) -> dict:
    """Pick the (site, feature) with the largest |AUROC - 0.5| on one half; test that one on the other."""

    best = (-1.0, 0, 0, 0.5)
    for site in range(select_codes.shape[1]):
        auroc = auroc_columns(select_codes[:, site].to(device), select_labels)
        deviation = (auroc - 0.5).abs()
        feature = int(deviation.argmax())
        if float(deviation[feature]) > best[0]:
            best = (float(deviation[feature]), site, feature, float(auroc[feature]))
    _, site, feature, selected_auroc = best
    column = report_codes[:, site, feature : feature + 1].to(device)
    observed = float(auroc_columns(column, report_labels)[0])
    generator = np.random.default_rng(seed)
    exceed = sum(
        abs(float(auroc_columns(column, generator.permutation(report_labels))[0]) - 0.5) >= abs(observed - 0.5)
        for _ in range(rounds)
    )
    p_value = (exceed + 1) / (rounds + 1)
    return {
        "site": site + 1, "feature": feature, "selection_auroc": selected_auroc, "report_auroc": observed,
        "p": p_value, "found": bool(abs(observed - 0.5) >= FOUND_MARGIN and p_value < FOUND_P),
    }

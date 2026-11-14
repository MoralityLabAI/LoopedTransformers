"""Parameter-free recurrent feedback schedules (RLT, arXiv:2611.16265, Eq. 4)."""
from __future__ import annotations

import math

MODES = ("direct", "rlt", "matched_damping")


def weight(mode: str, step: int, budget: int) -> float:
    """Zero-based step. Damping matches RLT's coefficient sum and terminal replacement."""
    if mode not in MODES:
        raise ValueError(f"unknown feedback {mode!r}; expected one of {MODES}")
    if budget < 1 or not 0 <= step < budget:
        raise ValueError(f"invalid feedback step={step}, budget={budget}")
    if mode == "direct" or step == budget - 1:
        return 1.0
    if mode == "rlt":
        return 1.0 / (budget - step)
    return math.fsum(1.0 / j for j in range(2, budget + 1)) / (budget - 1)


def advance(current, endpoint, mode: str, step: int, budget: int):
    eta = weight(mode, step, budget)
    # Preserve the original direct-feedback computation exactly, including at K=1.
    return endpoint if eta == 1.0 else current + eta * (endpoint - current)

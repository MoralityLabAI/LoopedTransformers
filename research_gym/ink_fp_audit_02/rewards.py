"""Rewards, group-relative advantages and the GSPO sequence ratio (pure Python).

Reasoning effort is an observable inference control. Nothing here models internal depth.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

EFFORT_ORDER = {"low": 0, "medium": 1, "high": 2}


@dataclass(frozen=True)
class RewardConfig:
    correct: float = 1.0
    rescue_bonus: float = 0.35      # high-effort correct where every low-effort sample failed
    repetition: float = -0.50
    incomplete: float = -0.25
    token_cost_per_1k: float = 0.0  # >0 gives the cost-sensitive variant (arm G)
    shuffle_rescue: bool = False    # arm E control: bonus assigned to a random sample


def rescue_flags(samples: list[dict]) -> list[bool]:
    """A rescue is a correct high-effort sample on a problem where all low-effort samples failed.

    Length plays no role: a long correct medium/low sample, or a long high-effort sample on a
    problem low effort already solved, gets no bonus.
    """
    lows = [s for s in samples if s["effort"] == "low"]
    low_failed = bool(lows) and not any(s["correct"] for s in lows)
    return [low_failed and s["effort"] == "high" and bool(s["correct"]) for s in samples]


def rewards(samples: list[dict], cfg: RewardConfig = RewardConfig(), rng: random.Random | None = None) -> list[float]:
    """samples: one problem's group, each {"effort","correct","completed","repetition","completion_tokens"}."""
    flags = rescue_flags(samples)
    if cfg.shuffle_rescue:
        k = sum(flags)
        idx = list(range(len(samples)))
        (rng or random.Random(0)).shuffle(idx)
        chosen = set(idx[:k])
        flags = [i in chosen for i in range(len(samples))]
    out = []
    for s, rescued in zip(samples, flags):
        r = 0.0
        if s["correct"]:
            r += cfg.correct
        if rescued:
            r += cfg.rescue_bonus
        if s["repetition"]:
            r += cfg.repetition
        if not s["completed"]:
            r += cfg.incomplete
        r -= cfg.token_cost_per_1k * s.get("completion_tokens", 0) / 1000.0
        out.append(r)
    return out


def group_advantages(rs: list[float], normalize_std: bool = True, eps: float = 1e-6) -> list[float]:
    """GRPO: A_i = (r_i - mean) / (std + eps); zero for degenerate groups."""
    n = len(rs)
    mu = sum(rs) / n
    if not normalize_std:
        return [r - mu for r in rs]
    sd = math.sqrt(sum((r - mu) ** 2 for r in rs) / n)
    if sd < eps:
        return [0.0] * n
    return [(r - mu) / (sd + eps) for r in rs]


def is_frontier(samples: list[dict]) -> bool:
    """Partially solved: at least one success and at least one failure."""
    c = [bool(s["correct"]) for s in samples]
    return any(c) and not all(c)


def gspo_ratio(new_logprobs: list[float], old_logprobs: list[float]) -> float:
    """GSPO sequence ratio s = exp(mean_t(log pi_new - log pi_old)), length-normalised."""
    if len(new_logprobs) != len(old_logprobs) or not new_logprobs:
        raise ValueError("token logprob sequences must match and be non-empty")
    return math.exp(sum(a - b for a, b in zip(new_logprobs, old_logprobs)) / len(new_logprobs))


def gspo_objective(ratio: float, advantage: float, eps_low: float = 3e-4, eps_high: float = 4e-4) -> float:
    """Per-sequence clipped surrogate min(s*A, clip(s, 1-eps_low, 1+eps_high)*A)."""
    clipped = min(max(ratio, 1 - eps_low), 1 + eps_high)
    return min(ratio * advantage, clipped * advantage)

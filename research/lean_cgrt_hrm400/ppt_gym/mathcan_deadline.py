"""Finite-horizon proof values from observed Lean transitions."""

from collections import defaultdict
from functools import lru_cache

import torch

from .mathcan_sparse import CHECKS, CHECK_COST


def supported_distances(tape):
    """Back up shortest closure on sampled edges, with censored edges omitted."""
    edges = defaultdict(dict)
    for before, action, status, after, _ in tape:
        if status == "censored":
            continue
        previous = edges[before.key].setdefault(action, (status, after))
        if previous != (status, after):
            raise ValueError("one sampled action produced conflicting verdicts")

    @lru_cache(maxsize=None)
    def distance(key):
        # A shortest feasible continuation never needs to waste a check rejecting a tactic.
        return min((action_distance(key, status, after)
                    for status, after in edges[key].values() if status != "rejected"), default=CHECKS + 1)

    def action_distance(key, status, after):
        if status == "archived_closure":
            return 1
        if status == "accepted_nonterminal" and (after is None or after[2][:-1] != key[2] or after[:2] != key[:2]):
            raise ValueError("an accepted continuation must append a tactic to the exact prefix")
        next_distance = distance(key if status == "rejected" else after)
        return min(CHECKS + 1, 1 + next_distance)

    return [action_distance(before.key, status, after) for before, _, status, after, _ in tape
            if status != "censored"]


def deadline_utility(curve, remaining):
    """Score on-time success minus capped check cost under the closure distribution."""
    cutoff = remaining.to(dtype=torch.long)[..., None, None] - 1
    cutoff = cutoff.expand(*curve.shape[:-1], 1)
    probability = curve.gather(-1, cutoff).squeeze(-1)
    # A search that has yet to finish spends its next available check.
    survival = torch.cat((torch.ones_like(curve[..., :1]), 1. - curve[..., :-1]), dim=-1)
    available = torch.arange(CHECKS, device=curve.device) <= cutoff
    return probability - CHECK_COST * (survival * available).sum(dim=-1)

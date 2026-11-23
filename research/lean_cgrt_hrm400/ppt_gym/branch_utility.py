"""LSA-02J public-only branch viability and finite recovery-cost ranking head.

The frozen 57,475-param Plan B backbone supplies candidate embeddings. A
separate small learned head sees those embeddings and eight transparent
features of the *public Lean context*. It never sees native graph entries,
proof costs, theorem family, shortest reference actions or oracle labels at
inference.

A separate exact public Horn fixed-point scorer provides a strong model-free
structural control. Unlike the learned head, it solves an explicit graph
over public hypothesis types. It must be evaluated as an algorithmic control,
rather than being credited as model learning.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict

import torch
from torch import nn

from .typed_state import _public, _type_entries, _formula


SCHEMA = "lsa02j.branch_utility_head.v1"
PUBLIC_FEATURES = 8
BASE_FEATURES = 32
HIDDEN = 24
HEAD_SCALE = 5.0
COST_SCALE = 0.5
SOURCE_MAX_COST = 16


def _public_entries(packet: str, candidates):
    goals, remaining, _ = _public(packet)
    hypotheses = _type_entries(goals[0])
    by_name = {name: (pre, head) for name, pre, head in hypotheses}
    if len(by_name) != len(hypotheses):
        raise ValueError("public Lean premise names conflict")
    for action in candidates:
        if action == "constructor":
            if "and" not in goals[0]["target"]:
                raise ValueError("constructor cannot apply to this public target")
        elif not action.startswith("apply ") or action[6:] not in by_name:
            raise ValueError("public tactic is absent from hypothesis context")
    return goals, remaining, hypotheses, by_name


def action_features(packet: str, candidates) -> torch.Tensor:
    """Eight normalized features from model-visible public type declarations."""
    goals, remaining, hypotheses, by_name = _public_entries(packet, candidates)
    facts = {head for _, pre, head in hypotheses if not pre}
    rules = len(hypotheses) - sum(not pre for _, pre, _ in hypotheses)
    rows = []
    for action in candidates:
        if action == "constructor":
            inputs = tuple(_formula(g) for g in goals[0]["target"]["and"])
        else:
            inputs = by_name[action[6:]][0]
        supplied = sum(p in facts for p in inputs)
        unseen = len(inputs) - supplied
        rows.append([
            min(len(inputs), 8) / 8,
            supplied / max(1, len(inputs)),
            min(unseen, 8) / 8,
            min(len(facts), 24) / 24,
            min(rules, 24) / 24,
            min(len(candidates), 32) / 32,
            remaining / 8,
            min(len(goals)-1, 8) / 8,
        ])
    return torch.tensor(rows, dtype=torch.float32)


class BranchUtility(nn.Module):
    """Frozen-backbone feature probe: finite viability and recovery cost."""
    def __init__(self, *, width=BASE_FEATURES):
        super().__init__()
        self.width = width
        self.net = nn.Sequential(
            nn.Linear(width + PUBLIC_FEATURES, HIDDEN),
            nn.Tanh(),
            nn.Linear(HIDDEN, 2),
        )

    def forward(self, features, public):
        if features.ndim != 2 or features.shape[1] != self.width:
            raise ValueError("frozen Plan B candidate feature width changed")
        if public.shape != (features.shape[0], PUBLIC_FEATURES):
            raise ValueError("public tactic feature shape does not match menu")
        return self.net(torch.cat([features, public.to(features.device)], -1))

    def additive_scores(self, features, packet: str, candidates):
        outcomes = self(
            features,
            action_features(packet, candidates),
        )
        viability = outcomes[:, 0].sigmoid()
        normalized_cost = outcomes[:, 1].sigmoid()
        return HEAD_SCALE * (viability - COST_SCALE * normalized_cost)


def public_horn_costs(packet: str, candidates) -> list[int | None]:
    """Strong no-model public structural fixed-point baseline.

    Computes minimum tactic counts by forward-chaining each *public* Horn rule
    and conjunction constructor. No Lean queries, generator fields or graph
    oracle are consulted. This solver is an intentionally strong classical
    algorithmic control for the small synthetic Horn grammar.
    """
    goals, _, entries, by_name = _public_entries(packet, candidates)
    known: dict[str, int] = {}
    for _, antecedents, output in entries:
        if not antecedents:
            known[output] = min(known.get(output, 10**6), 1)
    # The least positive solution converges because every rule costs +1.
    # Use a bounded monotone iteration; if the cap is reached, fail closed.
    for _ in range(max(16, len(entries) * len(entries) + 1)):
        changed = False
        for _, antecedents, output in entries:
            if all(p in known for p in antecedents):
                n = 1 + sum(known[p] for p in antecedents)
                if n < known.get(output, 10**6):
                    known[output] = n
                    changed = True
        if not changed:
            break
    else:
        raise ValueError("public Horn fixed-point did not converge")

    def size(formula):
        if "a" in formula:
            return known.get(formula["a"])
        if "and" in formula:
            xs = [size(x) for x in formula["and"]]
            return 1 + sum(xs) if all(x is not None for x in xs) else None
        raise ValueError("unsupported public goal formula")

    pending = [size(g["target"]) for g in goals[1:]]
    if any(s is None for s in pending):
        return [None for _ in candidates]
    remaining = sum(pending)
    costs = []
    for action in candidates:
        if action == "constructor":
            needed = [size(x) for x in goals[0]["target"]["and"]]
        else:
            needed = [known.get(x) for x in by_name[action[6:]][0]]
        costs.append(
            1 + sum(needed) + remaining
            if all(x is not None for x in needed) else None
        )
    return costs


def public_horn_order(packet: str, candidates) -> int:
    """Model-free cheapest finite public branch, deterministic fallback."""
    costs = public_horn_costs(packet, candidates)
    indexed = [
        (cost, action, idx)
        for idx, (action, cost) in enumerate(zip(candidates, costs))
        if cost is not None
    ]
    if indexed:
        return min(indexed)[2]
    # Deterministic public syntactic priority avoids consulting the native
    # graph when no finite proof is derivable by the public Horn closure.
    return min(range(len(candidates)), key=lambda i: candidates[i])

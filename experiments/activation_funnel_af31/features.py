"""AF-31 public Horn action features, never using generator IDs or oracle labels.

The two reverse messages are graph relaxations, not forward least-fixed-point
reachability. Every examination is charged to the controller's work receipt.
"""
from __future__ import annotations

import math

DIM = 8
FEATURE_NAMES = (
    "negative_unconditional_goal_hops",
    "negative_blocked_reverse_cost",
    "goal_ancestor_indicator",
    "direct_downstream_rule_count",
    "immediate_goal_head",
    "newly_enabled_after_action",
    "negative_unproducible_downstream_companions",
    "rule_arity",
)
MISSING = 30.
BLOCK_PENALTY = 7.


def parse(public):
    return tuple((int(slot), tuple(int(x) for x in prem), int(head))
                 for slot, prem, head in public["rules_physical"])


def enabled(rules, state):
    state = frozenset(state)
    return tuple(sorted(
        slot for slot, prem, head in rules
        if head not in state and all(p in state for p in prem)
    ))


def reverse_relaxations(public, state):
    """Bipartite rule-to-premise relaxation; count all attempted edges."""
    rules = parse(public)
    producer = {head for _, _, head in rules}
    goal = int(public["goal_slot"])
    hops = {goal: 0.}
    blocked = {goal: 0.}
    work = 0
    for _ in range(len(rules)+1):
        changed = False
        for _, prem, head in rules:
            work += 1
            if head not in hops:
                continue
            for p in prem:
                h = 1. + hops[head]
                penalty = BLOCK_PENALTY * sum(
                    q not in producer and q not in state for q in prem if q != p
                )
                v = blocked[head] + 1. + penalty
                if h < hops.get(p, MISSING):
                    hops[p] = h
                    changed = True
                if v < blocked.get(p, MISSING):
                    blocked[p] = v
                    changed = True
        if not changed:
            break
    return hops, blocked, producer, work


def extract(public, state):
    """Produce 8 features for each currently legal physical rule slot.

    The extractor is deliberately public and oblivious to source pair IDs,
    train/test split, proof budget, STOP oracle and previous scored outcomes.
    """
    rules = parse(public)
    state = frozenset(state)
    hops, blocked, producer, work = reverse_relaxations(public, state)
    candidates = {}
    for slot, prem, head in rules:
        if head in state or any(p not in state for p in prem):
            continue
        onward = [(r, pre, h) for r, pre, h in rules if head in pre]
        downstream_dead = min((
            sum(other not in producer and other not in state
                for other in pre if other != head)
            for _, pre, _ in onward
        ), default=0)
        next_state = state | {head}
        new = sum(
            h not in next_state and all(p in next_state for p in pre)
            and not (h not in state and all(p in state for p in pre))
            for _, pre, h in rules
        )
        work += 2*len(rules)
        candidates[slot] = (
            -hops.get(head, MISSING)/6.,
            -blocked.get(head, MISSING)/10.,
            float(head in hops),
            min(5, len(onward))/5.,
            float(head == public["goal_slot"]),
            min(5, new)/5.,
            -float(downstream_dead),
            len(prem)/2.,
        )
    if not candidates and public["goal_slot"] not in state:
        # Empty legal frontiers are permitted for genuinely unreachable graphs.
        return {}, {"feature_rule_examinations": work,
                    "candidate_feature_rows": 0}
    if any(len(v) != DIM or not all(math.isfinite(f) for f in v)
           for v in candidates.values()):
        raise AssertionError("bad or nonfinite AF31 action feature")
    if tuple(sorted(candidates)) != enabled(rules, state):
        raise AssertionError("feature extractor disagrees with public Horn legality")
    return candidates, {
        "feature_rule_examinations": work,
        "candidate_feature_rows": len(candidates),
    }

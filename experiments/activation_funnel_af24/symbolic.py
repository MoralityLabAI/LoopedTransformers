"""AF-24: goal-aware symbolic Horn controllers with explicit inference work.

Only public physical rule incidence, root-known atoms, and the current proved
set enter these controllers. Ground truth BFS, proof labels, neural scores and
the shortest+1 episode budget are held by the experiment runner, not supplied
to the controllers. No Torch or learned weights are needed.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
import random
import time

CAPS = {"bfs_cap4": 4, "bfs_cap16": 16, "bfs_cap64": 64}
CORE = ("ancestor_greedy", "closure_direct", *CAPS)
RANDOM = tuple(f"ancestor_random_{i}" for i in range(1, 6))
ARMS = (*CORE, *RANDOM, "fixed", "shortest")
INF = 10**6


def hash_int(obj):
    return int(hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16], 16)


@dataclass(frozen=True)
class PublicHorn:
    """Role-bearing public graph, independent of logical generator names."""
    rules: tuple[tuple[int, tuple[int, ...], int], ...]
    initial: frozenset[int]
    goal: int

    @classmethod
    def from_public_features(cls, features):
        rules = tuple((int(i), tuple(int(p) for p in ps), int(h))
                      for i, ps, h in features["rules_physical"])
        if len(set(i for i, _, _ in rules)) != len(rules):
            raise ValueError("duplicate physical rule slot")
        return cls(
            rules=tuple(sorted(rules)),
            initial=frozenset(features["initial_physical"]),
            goal=int(features["goal_slot"]),
        )

    def enabled(self, known):
        return tuple((slot, head) for slot, prem, head in self.rules
                     if head not in known and all(p in known for p in prem))

    def fire(self, known, slot):
        legal = dict(self.enabled(known))
        if slot not in legal:
            raise ValueError("host rejected illegal physical Horn action")
        return frozenset((*known, legal[slot]))


def ancestry(graph):
    """Unconditional public reverse hypergraph relevance, not a truth oracle."""
    dist = {graph.goal: 0}
    checked = 0
    for _ in range(len(graph.rules) + 1):
        changed = False
        for _, pre, head in graph.rules:
            checked += 1
            if head not in dist:
                continue
            for atom in pre:
                if atom not in dist or dist[atom] > dist[head] + 1:
                    dist[atom] = dist[head] + 1
                    changed = True
        if not changed:
            break
    return dist, checked


def _relevance_move(graph, known, random_stream=None):
    distance, checks = ancestry(graph)
    enabled = graph.enabled(known)
    checks += len(graph.rules)
    if not enabled:
        return None, {"expanded": 0, "rule_examinations": checks,
                      "closure_passes": 0, "search_exhausted": True}
    ranked = [(distance.get(head, INF), slot) for slot, head in enabled]
    least = min(d for d, _ in ranked)
    choices = [slot for d, slot in ranked if d == least]
    if random_stream is None:
        action = min(choices)
    else:
        # Register a random, equal-relevance rule order; no oracle labels.
        action = random_stream.choice(choices)
    return action, {"expanded": 0, "rule_examinations": checks,
                    "closure_passes": 0, "search_exhausted": False}


def _closure_move(graph, known):
    """Relaxed additive-cost Horn derivation using the public incidence.

    This controller performs forward dynamic planning from public known atoms.
    It does not receive true proof cost or a hidden answer. For shared
    subproofs the additive relaxation may overcount; this is intentional.
    """
    if graph.goal in known:
        return None, {"expanded": 0, "rule_examinations": 0,
                      "closure_passes": 0, "search_exhausted": True}
    cost = {a: 0 for a in known}
    witness = {}
    checks = 0
    passes = 0
    for _ in range(len(graph.rules) + 1):
        passes += 1
        changed = False
        for slot, prem, head in graph.rules:
            checks += 1
            if not all(p in cost for p in prem):
                continue
            new = 1 + sum(cost[p] for p in prem)
            if head not in cost or new < cost[head] or (
                new == cost[head] and slot < witness.get(head, INF)
            ):
                cost[head] = new
                witness[head] = slot
                changed = True
        if not changed:
            break
    if graph.goal not in cost:
        return None, {"expanded": 0, "rule_examinations": checks,
                      "closure_passes": passes, "search_exhausted": True}

    by_slot = {slot: (pre, head) for slot, pre, head in graph.rules}
    goal_path_heads = set()
    todo = [graph.goal]
    traversed = set()
    while todo:
        atom = todo.pop()
        if atom in known or atom in traversed:
            continue
        traversed.add(atom)
        goal_path_heads.add(atom)
        slot = witness.get(atom)
        if slot is None:
            raise AssertionError("public closure cannot reconstruct an available goal")
        todo.extend(by_slot[slot][0])
    legal = graph.enabled(known)
    checks += len(graph.rules)
    moves = [(cost[head], slot) for slot, head in legal if head in goal_path_heads]
    if not moves:
        raise AssertionError("finite Horn goal derivation lacks a legal frontier step")
    action = min(moves)[1]
    return action, {"expanded": 0, "rule_examinations": checks,
                    "closure_passes": passes, "search_exhausted": False}


def _bfs_move(graph, known, cap):
    """Bounded on-line breadth-first planner with explicit node-expansion cap."""
    if not (isinstance(cap, int) and cap > 0):
        raise ValueError("positive BFS expansion cap required")
    root = frozenset(known)
    todo = deque([(root, None)])
    seen = {root}
    expanded = checks = 0
    while todo and expanded < cap:
        state, first = todo.popleft()
        expanded += 1
        if graph.goal in state:
            return first, {"expanded": expanded,
                           "rule_examinations": checks,
                           "closure_passes": 0, "search_exhausted": True}
        enabled = graph.enabled(state)
        checks += len(graph.rules)
        for slot, _ in enabled:
            next_state = graph.fire(state, slot)
            if next_state not in seen:
                seen.add(next_state)
                todo.append((next_state, slot if first is None else first))
    if not todo:   # Exhausted entire reachable host-state graph: can STOP.
        return None, {"expanded": expanded, "rule_examinations": checks,
                      "closure_passes": 0, "search_exhausted": True}
    # On cap exhaustion, use the public ancestry heuristic for a legal move.
    fallback, work = _relevance_move(graph, root)
    return fallback, {"expanded": expanded,
                      "rule_examinations": checks + work["rule_examinations"],
                      "closure_passes": work["closure_passes"],
                      "search_exhausted": False}


def choose(graph, known, arm, seed=17, case_id="", step=0):
    """Public graph controller; never consult true shortest distances."""
    if arm not in ARMS:
        raise ValueError("unregistered AF-24 arm")
    t = time.perf_counter_ns()
    if graph.goal in known:
        action, work = None, {"expanded": 0, "rule_examinations": 0,
                              "closure_passes": 0, "search_exhausted": True}
    elif arm == "fixed":
        legal = graph.enabled(known)
        action = legal[0][0] if legal else None
        work = {"expanded": 0, "rule_examinations": len(graph.rules),
                "closure_passes": 0, "search_exhausted": not legal}
    elif arm == "ancestor_greedy":
        action, work = _relevance_move(graph, known)
    elif arm.startswith("ancestor_random_"):
        arm_id = int(arm.rsplit("_", 1)[-1])
        rng = random.Random(hash_int([seed, case_id, step, arm_id]))
        action, work = _relevance_move(graph, known, rng)
    elif arm.startswith("bfs_cap"):
        action, work = _bfs_move(graph, known, CAPS[arm])
    elif arm == "closure_direct":
        action, work = _closure_move(graph, known)
    else:
        raise ValueError("shortest upper control is scorer-only, never inference")
    receipt = {"arm": arm, "action": action, **work,
               "wall_ns": time.perf_counter_ns() - t}
    return receipt

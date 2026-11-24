"""OLR-06 exact signed-receipt reference, with no neural or native-proof claims.

Known edge u--v carries w_v = sign*w_u, sign in {-1,+1}.
Anchors fix one w. Solve each component exactly; unanchored components
stay UNKNOWN and inconsistent components stay CONFLICT. These are
algebraic consequences in this grammar, not new Lean certificates.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass


UNKNOWN = "unknown"
CONFLICT = "conflict"


@dataclass(frozen=True)
class Component:
    nodes: tuple[int, ...]
    status: str
    # Relative signs to the least-addressed node; retain only if consistent.
    relative: tuple[tuple[int, int], ...]
    values: tuple[tuple[int, int], ...]


def _sign(value):
    if type(value) is not int or value not in (-1, 1):
        raise ValueError("signed receipts must be integer -1 or +1")
    return value


def solve(nodes, edges, anchors):
    """Return immutable components and per-node resolved/unknown/conflict status.

    Edges and anchors may repeat; incompatible receipts flag the entire
    component. No hidden branch label or generator truth is an input.
    Time O(V+E+A) excluding deterministic sorting.
    """
    ids = tuple(nodes)
    if any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("distinct nonnegative integer nodes required")
    adjacency = {i: [] for i in ids}
    evidence = {i: [] for i in ids}
    for u, v, sign in edges:
        if u not in adjacency or v not in adjacency:
            raise ValueError("receipt references inactive node")
        sign = _sign(sign)
        adjacency[u].append((v, sign))
        adjacency[v].append((u, sign))
    for u, sign in anchors:
        if u not in evidence:
            raise ValueError("anchor references inactive node")
        evidence[u].append(_sign(sign))
    seen = set()
    components = []
    output = {}
    for start in sorted(ids):
        if start in seen:
            continue
        relative = {start: 1}
        queue = deque([start])
        seen.add(start)
        conflict = False
        possible_anchor = set()
        while queue:
            u = queue.popleft()
            possible_anchor.update(relative[u] * s for s in evidence[u])
            for v, sign in adjacency[u]:
                candidate = relative[u] * sign
                if v in relative:
                    conflict |= relative[v] != candidate
                else:
                    relative[v] = candidate
                    seen.add(v)
                    queue.append(v)
        members = tuple(sorted(relative))
        conflict |= len(possible_anchor) > 1
        if conflict:
            component = Component(members, CONFLICT, (), ())
            output.update((u, CONFLICT) for u in members)
        elif not possible_anchor:
            component = Component(members, UNKNOWN, tuple(sorted(relative.items())), ())
            output.update((u, UNKNOWN) for u in members)
        else:
            a = next(iter(possible_anchor))
            values = tuple((u, a * relative[u]) for u in members)
            component = Component(members, "resolved", tuple(sorted(relative.items())), values)
            output.update(values)
        components.append(component)
    return tuple(components), output


def pilot_labels(rows, parents):
    """Adapter for Fable's seven-field OLR-06 feasibility pilot.

    Input only: active, has_parent, type_sign, has_relative, relative_sign,
    has_absolute, absolute_sign; and public parent pointers. The target
    class encoding is 0=-1, 1=+1, 2=unknown, 3=padding.
    """
    if len(rows) != len(parents) or any(len(row) != 7 for row in rows):
        raise ValueError("pilot ledger shape differs")
    if any(row[j] not in (0, 1) for row in rows for j in (0, 1, 3, 5)):
        raise ValueError("malformed public presence bit")
    active = [i for i, row in enumerate(rows) if row[0]]
    active_set = set(active)
    edges = []
    anchors = []
    for i in active:
        _, hasparent, t, hasrel, r, hasabs, a = rows[i]
        if hasparent and (type(parents[i]) is not int or parents[i] not in active_set):
            raise ValueError("public parent pointer outside active ledger")
        if hasrel:
            if not hasparent or t not in (-1, 1) or r not in (-1, 1):
                raise ValueError("invalid relative receipt")
            edges.append((parents[i], i, int(t * r)))
        if hasabs:
            if a not in (-1, 1):
                raise ValueError("invalid absolute receipt")
            anchors.append((i, int(a)))
    _, labels = solve(active, edges, anchors)
    if CONFLICT in labels.values():
        raise ValueError("consistent feasibility-pilot schema contains conflict")
    return [3 if i not in labels else 2 if labels[i] == UNKNOWN
            else (labels[i] + 1) // 2 for i in range(len(rows))]


def optimal_anchor_plan(components, offers, budget):
    """Exact static anchor-probe planning under a restricted cost model.

    Each offered native probe is guaranteed to supply one consistent absolute
    anchor for its component, with a positive integral cost. No new edges,
    stochastic probe failures or revisions. Any anchor resolves the entire
    previously unknown component. The objective is count of newly resolved
    nodes; each component is selected at most once. This is a multiple-choice
    knapsack reduced to its cheapest offer per component.
    """
    if type(budget) is not int or not 0 <= budget <= 10000:
        raise ValueError("bounded nonnegative integer budget required")
    owner = {}
    for idx, component in enumerate(components):
        for u in component.nodes:
            if u in owner:
                raise ValueError("overlapping component partition")
            owner[u] = idx
    cheapest = {}
    for u, cost in offers:
        if u not in owner or type(cost) is not int or cost <= 0:
            raise ValueError("invalid anchor offer")
        idx = owner[u]
        if components[idx].status != UNKNOWN:
            continue
        key = (cost, u)
        if idx not in cheapest or key < cheapest[idx]:
            cheapest[idx] = key
    # dp[spent] -> (number newly resolved, chosen addresses)
    dp = {0: (0, ())}
    for idx in sorted(cheapest):
        cost, u = cheapest[idx]
        gain = len(components[idx].nodes)
        for spent, (score, chosen) in list(dp.items()):
            new_spent = spent + cost
            if new_spent > budget:
                continue
            candidate = (score + gain, tuple(sorted((*chosen, u))))
            current = dp.get(new_spent)
            if (current is None or candidate[0] > current[0]
                    or candidate[0] == current[0] and candidate[1] < current[1]):
                dp[new_spent] = candidate
    spent, (gain, plan) = min(dp.items(), key=lambda item: (
        -item[1][0], item[0], item[1][1]))
    return {"anchors": plan, "cost": spent, "newly_resolved": gain}

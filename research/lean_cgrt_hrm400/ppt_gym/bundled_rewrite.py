"""Public equality facts exposed by a bounded Lean conjunction opening."""

from collections import defaultdict, deque
import re

from .rewrite import BPD_REWRITE_FAMILY, EQUALITY, rewrite_problem


BPD_BUNDLE_FAMILY = "bpd-equality-bundle-opening"
REWRITE_FAMILIES = frozenset((BPD_REWRITE_FAMILY, BPD_BUNDLE_FAMILY))
BUNDLE = re.compile(r"rcases (h[01]) with ⟨r[0-9]{1,4}(?:, r[0-9]{1,4})+⟩")


def bundle_hypothesis(action):
    match = BUNDLE.fullmatch(action)
    return match[1] if match else None


def bundle_views(goal):
    """Pair each allowed opening with the publicly declared equalities it exposes."""
    views = {}
    local_nat = {hyp["name"] for hyp in goal["hypotheses"] if hyp["type"] == {"raw": "Nat"}}
    names = {hyp["name"] for hyp in goal["hypotheses"]}
    for hyp in goal["hypotheses"]:
        name, tail = hyp["name"], hyp["type"]
        if name not in ("h0", "h1") or "and" not in tail:
            continue
        equalities = []
        while "and" in tail:
            left, tail = tail["and"]
            equalities.append(left)
        equalities.append(tail)
        if not all((match := EQUALITY.fullmatch(formula.get("raw", "")))
                   and set(match.groups()) <= local_nat for formula in equalities):
            continue
        start = 100 * (1 + int(name[1]))
        bound = [f"r{start + index}" for index in range(len(equalities))]
        if len(bound) >= 100 or set(bound) & names:
            continue
        action = f"rcases {name} with ⟨{', '.join(bound)}⟩"
        views[action] = [{"name": key, "type": formula} for key, formula in zip(bound, equalities, strict=True)]
    return views


def bundle_continuation(goal, menu):
    """Choose the next shortest public rewrite after a bundle has been opened."""
    if "rfl" in menu:
        return menu.index("rfl")
    _, target, edges = rewrite_problem(goal)
    neighbors = defaultdict(list)
    for _, left, right in edges:
        neighbors[right].append(left)
    distance = {target: 0}
    queue = deque([target])
    while queue:
        current = queue.popleft()
        for prior in neighbors[current]:
            if prior not in distance:
                distance[prior] = distance[current] + 1
                queue.append(prior)
    endpoints = {action: right for action, _, right in edges}
    reachable = [i for i, action in enumerate(menu) if endpoints.get(action) in distance]
    if reachable:
        return min(reachable, key=lambda i: distance[endpoints[menu[i]]])
    return next((i for i, action in enumerate(menu) if bundle_hypothesis(action)), 0)

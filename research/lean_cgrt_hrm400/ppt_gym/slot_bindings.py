"""Describe how a public tactic links equality goals to their local facts."""

from .rewrite import EQUALITY, rewrite_hypothesis


BINDING_CHANNELS = 7


def equality(target):
    match = EQUALITY.fullmatch(target.get("raw", ""))
    return match.groups() if match else None


def binding_features(goal, action=None):
    """Local incidences only. No proof search, return labels or path distances."""
    hypotheses = {item["name"]: item["type"] for item in goal["hypotheses"]}
    target = goal["target"]
    affected = False
    if action and action.startswith("apply "):
        implication = hypotheses.get(action.removeprefix("apply "), {}).get("imp", ())
        if len(implication) == 2 and implication[1] == target:
            target = implication[0]
            affected = True
    targets = [equality(part) for part in target.get("and", [target])]
    targets = [pair for pair in targets if pair is not None]
    if not targets:
        return (0.,) * BINDING_CHANNELS

    rewrite = action.removeprefix("all_goals ") if action else ""
    name = rewrite_hypothesis(rewrite)
    edge = equality(hypotheses.get(name, {}))
    if edge is not None:
        source, dest = edge[::-1] if rewrite.startswith("rw [← ") else edge
        projected = [tuple(dest if term == source else term for term in pair) for pair in targets]
        affected = projected != targets
        targets = projected

    edges = {fact for value in hypotheses.values() if (fact := equality(value)) is not None}
    fraction = 1. / len(targets)
    return (
        1., float(affected), float(len(targets) == 2),
        fraction * sum(left == right for left, right in targets),
        fraction * sum((left, right) in edges for left, right in targets),
        fraction * sum((right, left) in edges for left, right in targets),
        fraction * sum(any(source == left and dest != right for source, dest in edges) for left, right in targets),
    )


def action_bindings(goals, action):
    """Use the same focused-goal convention as the Lean runner."""
    return [binding_features(goal, action if index == 0 or action.startswith("all_goals ") else None)
            for index, goal in enumerate(goals)]

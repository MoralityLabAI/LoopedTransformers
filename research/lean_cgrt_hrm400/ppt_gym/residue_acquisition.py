"""Price a Lean residue query before buying its one-bit observation."""

from math import fsum

from .failed_prefix import CHARGE, GUARD_CHECK, WORLDS, legal, planned_values
from .paired_recovery import solve_target
from .residue_recovery import MODULUS, world_weights


COMMIT = 0
QUERY_COST = 1


def queries(pair):
    first, target = pair["first"], pair["second"]
    expression = pair["certificate"].rsplit(" = ", 1)[0]
    groups = [tuple(route["residues"]) for route in target["routes"]]
    ordered = sorted(range(MODULUS), key=lambda value: (
        next(i for i, route in enumerate(first["routes"]) if value in route["residues"]),
        next(i for i, group in enumerate(groups) if value in group), value,
    ))
    memberships = [*groups, tuple(sorted(ordered[::2])), tuple(sorted(
        value for index, value in enumerate(ordered) if index % 3 != 1))]
    return [{"members": list(group), "tactic": "have hPptQuery : (" + " ∨ ".join(
        f"{expression} = {value}" for value in group) + f") := by {GUARD_CHECK}"} for group in memberships]


def split(prior, query, bit):
    members = set(query["members"])
    mass = tuple(weight if (value in members) == bit else 0.
                 for value, weight in enumerate(prior))
    return fsum(mass), mass


def forecast(target, allowance, menu, prior, support):
    """Return expected post-prefix utility, including fees and option to stop."""
    def value(weights, remaining):
        planner = planned_values(target["routes"], world_weights(target, weights, WORLDS))
        return max((0., *planner(remaining, 0).values()))

    scores = {COMMIT: value(prior, allowance)}
    remaining = allowance - QUERY_COST
    if not legal(target["routes"], remaining, 0):
        return scores
    for action, query in enumerate(menu, 1):
        if len({value in query["members"] for value, ok in enumerate(support) if ok}) < 2:
            continue
        scores[action] = fsum(mass * value(weights, remaining) for bit in (False, True)
                              for mass, weights in [split(prior, query, bit)] if mass) - CHARGE * QUERY_COST
    return scores


def execute(target, allowance, menu, prior, action, outcomes):
    """Buy the selected check, then restore the native saved proof state."""
    if action == COMMIT:
        row = solve_target(target, allowance, world_weights(target, prior, WORLDS))
        return {"second": row, "query": None, "utility": row["utility"]}
    query, bit = menu[action - 1], outcomes[action - 1]
    mass, weights = split(prior, query, bit)
    if not mass:
        raise ValueError("a paid observation is incompatible with the policy prior")
    row = solve_target(target, allowance - QUERY_COST, world_weights(target, weights, WORLDS))
    return {"second": row, "query": {"action": action, "passed": bit, "charge": QUERY_COST},
            "utility": row["utility"] - CHARGE * QUERY_COST}

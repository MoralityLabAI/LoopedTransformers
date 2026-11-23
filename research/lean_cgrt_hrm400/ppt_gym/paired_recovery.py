"""Transfer paid native guard evidence between paired Lean proofs."""

from math import fsum

from .failed_prefix import CHARGE, PREFIX_COST, WORLDS, legal, planned_values, trace_case


BUDGETS = ((3, 5), (5, 3), (5, 5), (7, 5))
STOP = 3


def order(source, target):
    guards = [route["guard"] for route in source["routes"]]
    shifted = [route["guard"] for route in target["routes"]]
    if len(set(guards)) != 3 or set(guards) != set(shifted):
        raise ValueError("transfer requires three unambiguous public guard identities")
    return tuple(guards.index(guard) for guard in shifted)


def target_prior(weights, alias, observed=None):
    adjusted = []
    for bits, weight in zip(WORLDS, weights, strict=True):
        if observed is not None and observed[1] != bits[observed[0]]:
            weight = 0.
        adjusted.append((tuple(bits[index] for index in alias), weight))
    by_truth = dict(adjusted)
    return [by_truth[bits] for bits in WORLDS]


def stop_or_pick(values):
    best = max(values, key=values.__getitem__, default=None)
    return best if best is not None and values[best] > 0. else None


def solve_target(target, allowance, weights):
    scores = planned_values(target["routes"], weights)
    return trace_case(target, allowance, lambda remaining, failed, _menu: stop_or_pick(scores(remaining, failed)))


def one_trial(source, allowance, choice):
    events, proved, spent = [], False, 0
    if choice != STOP:
        if choice not in legal(source["routes"], allowance, 0):
            raise ValueError("the first proof chose an unaffordable continuation")
        route = source["routes"][choice]
        proved = source["truth"][choice]
        spent = route["pass_cost" if proved else "fail_cost"]
        events.append({"arm": choice, "before": allowance, "passed": proved, "charge": spent})
    return {"allowance": allowance, "trace": events, "checks": PREFIX_COST + spent, "proved": proved,
            "utility": float(proved) - CHARGE * (PREFIX_COST + spent)}


def run_pair(source, target, budgets, weights, choice, *, remember=True):
    first, second = budgets
    source_trace = one_trial(source, first, choice)
    observed = (choice, source_trace["proved"]) if remember and choice != STOP else None
    after = solve_target(target, second, target_prior(weights, order(source, target), observed))
    return {"allowance": first * 10 + second, "first": source_trace, "second": after, "choice": choice,
            "proofs": int(source_trace["proved"]) + int(after["proved"]),
            "checks": source_trace["checks"] + after["checks"], "utility": source_trace["utility"] + after["utility"]}


def prospect(source, target, budgets, weights):
    """Compare one immediate proof attempt with its value for the second theorem."""
    first, second = budgets
    alias = order(source, target)
    base = planned_values(target["routes"], target_prior(weights, alias))
    reset = max((0., *base(second, 0).values()))
    joint, immediate = {STOP: reset}, {STOP: 0.}
    total = fsum(weights)
    for arm in legal(source["routes"], first, 0):
        route = source["routes"][arm]
        fail = fsum(w for bits, w in zip(WORLDS, weights, strict=True) if not bits[arm]) / total
        passed = 1 - fail
        after = []
        for bit in (False, True):
            continuation = planned_values(target["routes"], target_prior(weights, alias, (arm, bit)))
            after.append(max((0., *continuation(second, 0).values())))
        immediate[arm] = passed * (1 - CHARGE * route["pass_cost"]) - fail * CHARGE * route["fail_cost"]
        joint[arm] = immediate[arm] + fail * after[0] + passed * after[1]
    return joint, immediate

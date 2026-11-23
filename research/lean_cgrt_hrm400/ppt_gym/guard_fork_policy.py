"""Role-preserving observations for computational witness and condition goals."""

import re

import torch

from .computational_guards import GUARD_CHECK
from .live_quote_policy import pad_features


def formula_chain(formula):
    premises = []
    while "imp" in formula:
        premise, formula = formula["imp"]
        premises.append(premise)
    return premises, formula


def guard_tokens(raw):
    family = "geom" if "geom" in raw else "maskAt" if "maskAt" in raw else None
    if family is None or " = " not in raw:
        return None
    numbers = tuple(int(match) for match in re.findall("[0-9]+", raw))
    if family == "geom" and len(numbers) == 4:
        return (*numbers, 1)
    if family == "maskAt" and len(numbers) == 3:
        return (0, *numbers, 0)
    return None


def numeric_description(token):
    if token is None:
        return [0.] * 20
    base, depth, mod, rhs, family = token
    bits = [float(value >> shift & 1) for value, width in ((base, 4), (depth, 6), (rhs, 5)) for shift in range(width)]
    return [float(family), base / 8, depth / 64, mod / 19, rhs / 19, *bits]


def features(public, menu, ticks, warm):
    goal, *pending = public["goals"]
    hypotheses = {hyp["name"]: hyp["type"] for hyp in goal["hypotheses"]}
    rows, guards = [], []
    for action in menu:
        tactic, _, name = action.partition(" ")
        formula = hypotheses.get(name, goal["target"])
        premises, target = formula_chain(formula)
        known = [part for premise in premises if (part := guard_tokens(premise.get("raw", ""))) is not None]
        late = known[-1] if known else guard_tokens(target.get("raw", ""))
        early = known[0] if len(known) > 1 else None
        guards.append(late)
        context = [len(pending) / 4, len(premises) / 4, len(known) / 4, ticks / 8, tactic == "apply",
                   tactic == "exact", action == GUARD_CHECK, tactic == "constructor", late is not None]
        rows.append([pad_features(numeric_description(late) + context[:4]),
                     pad_features(numeric_description(early) + context[:4]),
                     pad_features(context + list(warm.get(action, (0.,) * 8)))])
    return torch.tensor(rows, dtype=torch.float32), guards

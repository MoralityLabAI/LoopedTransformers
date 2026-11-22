"""Expose local Nat definitions and their single-name normalization tactics."""

import re

from .conjunctive_rewrite import proposal_menu
from .rewrite import EQUALITY, NAME


ALIAS = re.compile(rf"({NAME}) : Nat := ({NAME})")
TOKEN = re.compile(NAME)


def local_definitions(goal):
    local_nats = {hyp["name"] for hyp in goal["hypotheses"] if hyp["type"] == {"raw": "Nat"}}
    definitions = {}
    for line in goal.get("pp", "").splitlines():
        binding = ALIAS.fullmatch(line.strip())
        if binding and set(binding.groups()) <= local_nats:
            name, value = binding.groups()
            definitions[name] = value
    return definitions


def alias_menu(state):
    goal = state["goals"][0]
    definitions = local_definitions(goal)
    menu = list(proposal_menu(state))
    target = goal["target"]
    formulas = target.get("and", [target])
    active = {token for formula in formulas for token in TOKEN.findall(formula.get("raw", ""))}
    menu.extend(f"dsimp only [{name}]" for name in definitions if name in active)
    equation = EQUALITY.fullmatch(target.get("raw", ""))
    if equation and "rfl" not in menu:
        left, right = equation.groups()
        if definitions.get(left, left) == definitions.get(right, right):
            menu.append("rfl")
    return tuple(menu)

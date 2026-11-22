"""Public tactic proposals for rewriting before or after splitting a conjunctive guard."""

import re

from .dependent_rewrite import term_rewrites
from .rewrite import EQUALITY, HYPOTHESIS, NAME


CONJUNCTIVE_RULE = re.compile(
    rf"(∀ \({NAME} {NAME} : Nat\), )({NAME} = {NAME}) ∧ ({NAME} = {NAME}) → (.+)"
)


def proposal_menu(state):
    goal = state["goals"][0]
    target = goal["target"]
    hypotheses = []
    for hyp in goal["hypotheses"]:
        match = CONJUNCTIVE_RULE.fullmatch(hyp["type"].get("raw", ""))
        if match:
            intro, first, _, conclusion = match.groups()
            # Reuse the binding-aware action grammar. Lean still sees the original two-part premise.
            hyp = {**hyp, "type": {"raw": f"{intro}{first} → {conclusion}"}}
        hypotheses.append(hyp)
    term = term_rewrites({**goal, "hypotheses": hypotheses})
    if term is not None:
        return term

    conjuncts = target.get("and")
    formulas = conjuncts if conjuncts is not None else [target]
    equations = [EQUALITY.fullmatch(formula.get("raw", "")) for formula in formulas]
    if not equations or not all(equations):
        return ()
    atoms = {atom for equation in equations for atom in equation.groups()}
    local_nat = {hyp["name"] for hyp in hypotheses if hyp["type"] == {"raw": "Nat"}}
    if not atoms <= local_nat:
        return ()
    if conjuncts is None and len(atoms) == 1:
        return ("rfl",)
    tactics = ["constructor"] if conjuncts is not None else []
    for hyp in hypotheses:
        equality = EQUALITY.fullmatch(hyp["type"].get("raw", ""))
        if equality is None or not HYPOTHESIS.fullmatch(hyp["name"]):
            continue
        left, right = equality.groups()
        if left == right or not {left, right} <= local_nat:
            continue
        if left in atoms:
            tactics.append(f"rw [{hyp['name']}]")
        if right in atoms:
            tactics.append(f"rw [← {hyp['name']}]")
    return tuple(tactics)

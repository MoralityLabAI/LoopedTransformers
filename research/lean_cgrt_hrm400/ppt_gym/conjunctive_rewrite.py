"""Public tactic proposals for rewriting before or after splitting a conjunctive guard."""

import re

from .dependent_rewrite import term_rewrites
from .rewrite import EQUALITY, HYPOTHESIS, NAME


CONJUNCTIVE_RULE = re.compile(
    rf"(∀ \({NAME} {NAME} : Nat\), )({NAME} = {NAME}) ∧ ({NAME} = {NAME}) → (.+)"
)


def project_rules(goal):
    hypotheses = []
    guards = {}
    for hyp in goal["hypotheses"]:
        match = CONJUNCTIVE_RULE.fullmatch(hyp["type"].get("raw", ""))
        if match:
            intro, first, second, conclusion = match.groups()
            # Reuse the binding-aware action grammar. Lean still sees the original two-part premise.
            hyp = {**hyp, "type": {"raw": f"{intro}{first} → {conclusion}"}}
            guards[hyp["name"]] = EQUALITY.fullmatch(second).groups()
        hypotheses.append(hyp)
    return {**goal, "hypotheses": hypotheses}, guards


def nat_equalities(goal):
    names = tuple(sorted(hyp["name"] for hyp in goal["hypotheses"] if hyp["type"] == {"raw": "Nat"}))
    known = set(names)
    edges = []
    for hyp in goal["hypotheses"]:
        equality = EQUALITY.fullmatch(hyp["type"].get("raw", ""))
        if equality is None or not HYPOTHESIS.fullmatch(hyp["name"]):
            continue
        left, right = equality.groups()
        if left == right or not {left, right} <= known:
            continue
        edges.extend(((f"rw [{hyp['name']}]", left, right), (f"rw [← {hyp['name']}]", right, left)))
    return names, tuple(edges)


def proposal_menu(state):
    goal = state["goals"][0]
    target = goal["target"]
    term = term_rewrites(project_rules(goal)[0])
    if term is not None:
        return term

    conjuncts = target.get("and")
    formulas = conjuncts if conjuncts is not None else [target]
    equations = [EQUALITY.fullmatch(formula.get("raw", "")) for formula in formulas]
    if not equations or not all(equations):
        return ()
    atoms = {atom for equation in equations for atom in equation.groups()}
    local_nat, edges = nat_equalities(goal)
    if not atoms <= set(local_nat):
        return ()
    if conjuncts is None and len(atoms) == 1:
        return ("rfl",)
    tactics = ["constructor"] if conjuncts is not None else []
    tactics.extend(action for action, source, _ in edges if source in atoms)
    return tuple(tactics)

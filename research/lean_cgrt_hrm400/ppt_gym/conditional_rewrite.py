"""Public rule views for equality rewrites that create proposition goals."""

import json

from .hybrid_rl.graph_model import MAX_PREMISES, GraphProofPolicy, atom, indexed_graph, target_atoms
from .hybrid_rl.model import ProofPolicy
from .rewrite import EQUALITY, rewrite_hypothesis, rewrite_problem


def project_equalities(goal):
    hypotheses, guards = [], {}
    for hyp in goal["hypotheses"]:
        formula = hyp["type"]
        implication = formula.get("imp")
        if implication is not None and atom(implication[0]) is not None and EQUALITY.fullmatch(implication[1].get("raw", "")):
            guards[hyp["name"]] = atom(implication[0])
            formula = implication[1]
        hypotheses.append({**hyp, "type": formula})
    return {**goal, "hypotheses": hypotheses}, guards


def conditional_graph(packet, actions):
    goals = json.loads(packet)["state"]["goals"]
    primary = goals[0]
    projected, guards = project_equalities(primary)
    problem = rewrite_problem(projected)
    if not guards or problem is None:
        return None
    left, right, edges = problem
    pending = set().union(*(target_atoms(goal["target"]) for goal in goals[1:]))
    facts, known = {right}, {left, right} | pending
    rules = []
    for action, start, end in edges:
        condition = guards.get(rewrite_hypothesis(action))
        sources = [end, condition] if condition is not None else [end]
        known.update((start, *sources))
        rules.append((action, sources, start))
    for hyp in primary["hypotheses"]:
        formula, sources = hyp["type"], []
        while "imp" in formula:
            antecedent, formula = formula["imp"]
            sources.append(atom(antecedent))
        head = atom(formula)
        if head is None or None in sources or len(sources) > MAX_PREMISES:
            continue
        known.update((*sources, head))
        if not sources:
            facts.add(head)
        rules.append((hyp["name"], sources, head))
    return indexed_graph({left}, pending, known, facts, rules, actions)


class ConditionalRewritePolicy(GraphProofPolicy):
    """Let the recurrent proof graph carry reachability and pending conditions together."""

    def forward(self, packet, actions, passes, **kwargs):
        graph = conditional_graph(packet, actions)
        if graph is None:
            return super().forward(packet, actions, passes, **kwargs)
        logits, value = ProofPolicy.forward(self, packet, actions, passes, **kwargs)
        rule_logits, rule_value = self.graph(graph, passes)
        return logits + rule_logits, value + rule_value

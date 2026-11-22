"""Bind public two-argument rewrite rules to terms in the proof workspace."""

from dataclasses import dataclass
import json
import re

from .conditional_rewrite import ConditionalRewritePolicy
from .hybrid_rl.graph_model import indexed_graph
from .hybrid_rl.model import ProofPolicy
from .rewrite import EQUALITY, HYPOTHESIS, NAME


TWO_NATS = {"imp": [{"raw": "Nat"}, {"imp": [{"raw": "Nat"}, {"raw": "Nat"}]}]}
TERM_EQUALITY = re.compile(rf"({NAME}) ({NAME}) ({NAME}) = ({NAME}) ({NAME}) ({NAME})")
QUANTIFIED = re.compile(rf"∀ \(({NAME}) ({NAME}) : Nat\), (?:(?:({NAME}) = ({NAME})) → )?(.+)")


@dataclass(frozen=True)
class Rule:
    name: str
    source: tuple[str, str, str]
    destination: tuple[str, str, str]
    guard: tuple[str, str] | None

    def apply(self, term, *, reverse=False):
        origin, target = (self.destination, self.source) if reverse else (self.source, self.destination)
        if term[0] != origin[0]:
            return None
        values = dict(zip(origin[1:], term[1:], strict=True))
        following = (target[0], values[target[1]], values[target[2]])
        condition = (values[self.guard[0]], self.guard[1]) if self.guard else None
        name = f"rw [← {self.name}]" if reverse else f"rw [{self.name}]"
        return name, following, condition


def term_problem(goal):
    equation = TERM_EQUALITY.fullmatch(goal["target"].get("raw", ""))
    if equation is None:
        return None
    left, right = equation.groups()[:3], equation.groups()[3:]
    local_types = {hyp["name"]: hyp["type"] for hyp in goal["hypotheses"]}
    if (any(local_types.get(term[0]) != TWO_NATS for term in (left, right))
            or any(local_types.get(name) != {"raw": "Nat"} for name in (*left[1:], *right[1:]))
            or sorted(left[1:]) != sorted(right[1:])):
        return None
    rules = []
    for name, formula in local_types.items():
        match = QUANTIFIED.fullmatch(formula.get("raw", "")) if HYPOTHESIS.fullmatch(name) else None
        if not match:
            continue
        x, y, premise, anchor, body = match.groups()
        eq = TERM_EQUALITY.fullmatch(body)
        if eq is None or x == y:
            continue
        source, destination = eq.groups()[:3], eq.groups()[3:]
        if any(local_types.get(term[0]) != TWO_NATS or set(term[1:]) != {x, y} for term in (source, destination)):
            continue
        if premise and (premise not in (x, y) or local_types.get(anchor) != {"raw": "Nat"}):
            continue
        rules.append(Rule(name, source, destination, (premise, anchor) if premise else None))
    return left, right, rules


def term_rewrites(goal):
    problem = term_problem(goal)
    if problem is None:
        return None
    left, right, rules = problem
    if left == right:
        return ("rfl",)
    return tuple(result[0] for rule in rules for reverse in (False, True)
                 if (result := rule.apply(left, reverse=reverse)) is not None)


def dependent_graph(packet, actions):
    goals = json.loads(packet)["state"]["goals"]
    problem = term_problem(goals[0])
    if problem is None:
        return None
    left, right, rules = problem
    arguments = left[1:]
    bindings = tuple(dict.fromkeys((arguments, arguments[::-1])))

    def term_key(term):
        return "term:" + ":".join(term)

    def guard_key(pair):
        return "guard:" + ":".join(pair)

    known, facts = {term_key(left), term_key(right)}, {term_key(right)}
    edges, anchors = [], set()
    for rule in rules:
        for reverse in (False, True):
            function = rule.destination[0] if reverse else rule.source[0]
            for args in bindings:
                current = (function, *args)
                action, following, guard = rule.apply(current, reverse=reverse)
                origins = [term_key(following)]
                if guard is not None:
                    anchors.add(guard[1])
                    origins.append(guard_key(guard))
                edges.append((f"{action}:{term_key(current)}", origins, term_key(current)))
                known.update((term_key(current), *origins))
    pending = {guard_key(match.groups()) for goal in goals[1:]
               if (match := EQUALITY.fullmatch(goal["target"].get("raw", ""))) is not None}
    nat_names = {hyp["name"] for hyp in goals[0]["hypotheses"] if hyp["type"] == {"raw": "Nat"}}
    nat_edges = [edge for hyp in goals[0]["hypotheses"] if HYPOTHESIS.fullmatch(hyp["name"])
                 and (edge := EQUALITY.fullmatch(hyp["type"].get("raw", ""))) and set(edge.groups()) <= nat_names]
    for anchor in sorted(anchors):
        facts.add(guard_key((anchor, anchor)))
        for number, edge in enumerate(nat_edges):
            start, end = edge.groups()
            source, head = guard_key((end, anchor)), guard_key((start, anchor))
            known.update((source, head))
            edges.append((f"guard:{anchor}:{number}", [source], head))
    return indexed_graph({term_key(left)}, pending, known | facts | pending, facts, edges,
                         [f"{action}:{term_key(left)}" for action in actions])


class DependentRewritePolicy(ConditionalRewritePolicy):
    """Share the recurrent graph across all public argument bindings."""

    def forward(self, packet, actions, passes, **kwargs):
        graph = dependent_graph(packet, actions)
        if graph is None:
            return super().forward(packet, actions, passes, **kwargs)
        logits, value = ProofPolicy.forward(self, packet, actions, passes, **kwargs)
        evidence, graph_value = self.graph(graph, passes)
        return logits + evidence, value + graph_value

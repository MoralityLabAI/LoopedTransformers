"""OLR-01A structural AND-OR oracle and diagnostic, not a Lean proof checker.

The ledger uses tree-proof costs for a *fully specified* Horn hypergraph.
Public facts and allowed rule applications must come from a trustworthy caller.
No forward-chained state is promoted to a Lean kernel fact by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import inf, isfinite


@dataclass(frozen=True)
class Rule:
    """A hyperedge: all premises are needed to establish the head."""

    head: int
    premises: tuple[int, ...]
    cost: float = 1.0
    enabled: bool = True


@dataclass(frozen=True)
class Hypergraph:
    nodes: int
    facts: frozenset[int]
    rules: tuple[Rule, ...]

    def __post_init__(self) -> None:
        if self.nodes <= 0:
            raise ValueError("positive node count required")
        if any(not 0 <= v < self.nodes for v in self.facts):
            raise ValueError("fact vertex outside graph")
        for rule in self.rules:
            if (not 0 <= rule.head < self.nodes or
                    any(not 0 <= v < self.nodes for v in rule.premises)):
                raise ValueError("rule vertex outside graph")
            if not isfinite(rule.cost) or rule.cost <= 0:
                raise ValueError("rule cost must be finite and strictly positive")


@dataclass(frozen=True)
class LedgerTrace:
    costs: tuple[float, ...]
    # A real source fact is different from a graph-derived reachable node.
    source_facts: tuple[bool, ...]
    structurally_reachable: tuple[bool, ...]
    steps: int
    converged: bool
    # State after each synchronous pass; useful for depth/overthinking audits.
    trajectory: tuple[tuple[float, ...], ...]


def propagate(graph: Hypergraph, max_passes: int = 64, *, fact_cost: float = 0.0) -> LedgerTrace:
    """Run monotone min-plus AND-OR propagation with synchronous updates.

    The result is the minimum derivation-TREE cost for the given fragment
    when converged. Shared dependencies may be charged more than once.
    A positive fact_cost counts the action required to discharge each fact.
    """
    if max_passes < 0:
        raise ValueError("max_passes must be nonnegative")
    if not isfinite(fact_cost) or fact_cost < 0:
        raise ValueError("fact_cost must be finite and nonnegative")
    # Fact_cost=1 matches a separate "apply h" tactic in the restricted
    # backward-chaining Lean fragment. Zero gives logical derivation edge cost.
    costs = tuple(float(fact_cost) if v in graph.facts else inf for v in range(graph.nodes))
    trajectory = [costs]
    converged = False
    steps = 0
    for _ in range(max_passes):
        updated = list(costs)
        for edge in graph.rules:
            if not edge.enabled:
                continue
            inputs = (costs[node] for node in edge.premises)
            # Short-circuit unreachable prerequisites, including cycles.
            if all(isfinite(d) for d in inputs):
                candidate = edge.cost + sum(costs[node] for node in edge.premises)
                updated[edge.head] = min(updated[edge.head], candidate)
        next_costs = tuple(updated)
        steps += 1
        trajectory.append(next_costs)
        if next_costs == costs:
            converged = True
            break
        costs = next_costs
    return LedgerTrace(
        costs=costs,
        source_facts=tuple(v in graph.facts for v in range(graph.nodes)),
        structurally_reachable=tuple(isfinite(d) for d in costs),
        steps=steps,
        converged=converged,
        trajectory=tuple(trajectory),
    )


def needed_obligations(graph: Hypergraph, goal: int, trace: LedgerTrace) -> frozenset[int]:
    """Mark outstanding goals along the cheapest known derivation.

    If a goal is still unreachable, conservatively include premises of all
    its available producing rules. This is a potential obligation frontier,
    not a claim that Lean supplied these as active goals.
    """
    if not 0 <= goal < graph.nodes:
        raise ValueError("goal vertex outside graph")
    needed: set[int] = set()

    def walk(node: int) -> None:
        if node in graph.facts or node in needed:
            return
        needed.add(node)
        choices = [
            (rule.cost + sum(trace.costs[p] for p in rule.premises),
             len(rule.premises), rule.premises, rule)
            for rule in graph.rules
            if rule.enabled and rule.head == node and
               all(isfinite(trace.costs[p]) for p in rule.premises)
        ]
        if choices and isfinite(trace.costs[node]):
            selected = (min(choices, key=lambda item: item[:3])[-1],)
        else:
            selected = tuple(r for r in graph.rules if r.enabled and r.head == node)
        for rule in selected:
            for premise in rule.premises:
                walk(premise)

    walk(goal)
    return frozenset(needed)


def mean_premise_cue(graph: Hypergraph, head: int) -> float:
    """Deliberately lossy diagnostic used for a mean-pooling alias test."""
    incoming = [p for e in graph.rules if e.head == head and e.enabled for p in e.premises]
    return sum(p in graph.facts for p in incoming) / len(incoming) if incoming else 0.0

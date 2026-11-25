"""Finite contextual Horn planning with proof-tree and rule-support costs.

All atoms/rules are public. Labels are sets of ground rule INSTANCE IDs.
No native verdicts, generator witnesses or trainable parameters enter planning.
"""
from __future__ import annotations
from dataclasses import dataclass
import itertools
import random
import time


@dataclass(frozen=True)
class Edge:
    head: object
    body: tuple
    rule: int
    args: tuple = ()


def insert(antichain, mask):
    if any(m & mask == m for m in antichain):
        return False
    antichain.difference_update({m for m in antichain if m & mask == mask})
    antichain.add(mask)
    return True


def join(a, b):
    out = set()
    for m in sorted(a):
        for n in sorted(b):
            insert(out, m | n)
    return out


def masks_for(edges, facts, *, cap=5000):
    """Least fixed point of monotone upward-closed support families."""
    atoms = set(facts)
    for e in edges:
        atoms.add(e.head)
        atoms.update(e.body)
    table = {a: ({0} if a in facts else set()) for a in atoms}
    rounds = 0
    while True:
        rounds += 1
        changed = False
        for i, e in enumerate(edges):
            supports = {0}
            for a in e.body:
                supports = join(supports, table[a])
                if not supports:
                    break
            for m in sorted(supports):
                changed |= insert(table[e.head], m | (1 << i))
            if len(table[e.head]) > cap:
                raise RuntimeError("exact support antichain cap exceeded")
        if not changed:
            return table, rounds


def closure(edges, facts, mask):
    """Independent forward chaining on a selected ground-instance subset."""
    known = set(facts)
    deriv = []
    while True:
        old = len(known)
        for i, e in enumerate(edges):
            if mask & (1 << i) and e.head not in known and all(a in known for a in e.body):
                known.add(e.head)
                deriv.append(i)
        if len(known) == old:
            return known, deriv


def audit_table(edges, facts, table):
    for a in facts:
        assert table[a] == {0}
    checked = 0
    for a, labels in table.items():
        for m in labels:
            assert a in closure(edges, facts, m)[0], (a, m)
            assert not any(n != m and n & m == n for n in labels)
            checked += 1
    for i, e in enumerate(edges):
        supports = {0}
        for a in e.body:
            supports = join(supports, table[a])
        for m in supports:
            wanted = m | (1 << i)
            assert any(n & wanted == n for n in table[e.head]), (i, wanted)
    return checked


def tree_cost(edges, facts):
    """Positive unit rule cost. Facts cost zero, tree subderivations add."""
    dist = {a: 0 for a in facts}
    choice = {}
    while True:
        change = False
        for i, e in enumerate(edges):
            if all(a in dist for a in e.body):
                c = 1 + sum(dist[a] for a in e.body)
                if c < dist.get(e.head, float("inf")):
                    dist[e.head] = c
                    choice[e.head] = i
                    change = True
        if not change:
            return dist, choice


def tree_witness(edges, facts, dist, choice, a):
    if a in facts:
        return ()
    assert a in choice
    i = choice[a]
    e = edges[i]
    assert dist[a] == 1 + sum(dist[b] for b in e.body)
    return (i,) + tuple(j for b in e.body for j in tree_witness(edges, facts, dist, choice, b))


def min_dag(edges, facts, table, goal):
    """Minimum number of distinct fired instances in this finite Horn graph."""
    labels = table.get(goal, set())
    if not labels:
        return None
    mask = min(labels, key=lambda m: (m.bit_count(), m))
    known, plan = closure(edges, facts, mask)
    assert goal in known
    # Remove irrelevant firings, keeping only backwards dependencies.
    produced = {edges[i].head: i for i in plan}
    need = set()
    def visit(a):
        if a not in facts:
            i = produced[a]
            if i not in need:
                need.add(i)
                for b in edges[i].body:
                    visit(b)
    visit(goal)
    plan = tuple(i for i in plan if i in need)
    # Exact cardinality minimality implies no redundant selected instance.
    assert len(plan) == mask.bit_count(), (goal, mask, plan)
    return plan


def audit():
    rng = random.Random(2026112513)
    counts = {"systems": 96, "contexts": 0, "frontier_comparisons": 0,
              "subset_queries": 0, "witness_masks": 0, "dag_checks": 0,
              "max_frontier": 0, "max_rounds": 0, "mismatches": 0}
    start = time.perf_counter()
    for i in range(counts["systems"]):
        edges = tuple(Edge(rng.randrange(5), tuple(rng.sample(range(5), rng.randrange(0, 3))), j)
                      for j in range(7))
        for facts in ({0}, {1}):
            table, rounds = masks_for(edges, facts)
            counts["contexts"] += 1
            counts["max_rounds"] = max(counts["max_rounds"], rounds)
            counts["witness_masks"] += audit_table(edges, facts, table)
            # Exhaustive subset integration, independently derive all reachable atoms.
            exact = {a: [] for a in range(5)}
            for m in range(1 << len(edges)):
                for a in closure(edges, facts, m)[0]:
                    exact[a].append(m)
            dist, choice = tree_cost(edges, facts)
            for a in range(5):
                vals = exact[a]
                want = {m for m in vals if not any(n != m and n & m == n for n in vals)}
                got = table.get(a, set())
                assert want == got, (i, facts, a, want, got)
                counts["frontier_comparisons"] += 1
                counts["max_frontier"] = max(counts["max_frontier"], len(got))
                for m in range(1 << len(edges)):
                    assert any(n & m == n for n in got) == (m in vals)
                    counts["subset_queries"] += 1
                assert bool(got) == (a in dist)
                if got:
                    plan = min_dag(edges, facts, table, a)
                    assert len(plan) == min(m.bit_count() for m in vals)
                    assert len(tree_witness(edges, facts, dist, choice, a)) == dist[a]
                    assert len(plan) <= dist[a]
                    counts["dag_checks"] += 1
    # Strict tree/DAG tradeoff: shared chain A->C0->C used by two branches,
    # then D,E->F and F->G. Alternative simple chain has seven edges.
    shared = [(1,(0,)),(2,(1,)),(3,(2,)),(4,(2,)),(5,(3,4)),(6,(5,))]
    chain = [(7,(0,))] + [(j+1,(j,)) for j in range(7,12)] + [(6,(12,))]
    edges = tuple(Edge(h,b,i) for i,(h,b) in enumerate(shared+chain))
    table,_ = masks_for(edges,{0})
    dist,choice = tree_cost(edges,{0})
    assert dist[6]==7 and len(min_dag(edges,{0},table,6))==6
    assert len(set(tree_witness(edges,{0},dist,choice,6)))==7
    counts["strict_tradeoff"] = {"min_tree": 7, "min_dag": 6}
    counts["seconds"] = time.perf_counter()-start
    return counts

"""Exact oracle tests for the proposed OLR-06 control, not Lean certificates."""
import itertools
import random
import unittest

from ppt_gym.olr06_exact_ledger import (
    CONFLICT, UNKNOWN, optimal_anchor_plan, pilot_labels, solve,
)


def exhaustive(nodes, edges, anchors):
    result = {}
    components, _ = solve(nodes, edges, [])
    for component in components:
        group = component.nodes
        g = set(group)
        assignments = []
        for bits in itertools.product((-1, 1), repeat=len(group)):
            w = dict(zip(group, bits))
            if (all(w[v] == sign*w[u] for u,v,sign in edges if u in g)
                    and all(w[u] == sign for u,sign in anchors if u in g)):
                assignments.append(w)
        for u in group:
            vals = {w[u] for w in assignments}
            result[u] = next(iter(vals)) if len(vals)==1 else (
                UNKNOWN if vals else CONFLICT)
    return result


class ExactLedgerTests(unittest.TestCase):
    def test_bidirectional_anchor_and_unresolved_component(self):
        components, labels = solve(range(5),[(0,1,-1),(1,2,1),(3,4,-1)],[(2,1)])
        self.assertEqual(labels, {0:-1,1:1,2:1,3:UNKNOWN,4:UNKNOWN})
        self.assertEqual([c.status for c in components],["resolved",UNKNOWN])

    def test_inconsistent_receipts_do_not_certify(self):
        _, labels = solve(range(4),[(0,1,1),(1,2,1),(2,0,-1)],[(3,-1)])
        self.assertEqual(labels,{0:CONFLICT,1:CONFLICT,2:CONFLICT,3:-1})
        _, labels = solve([0,1],[(0,1,1)],[(0,1),(1,-1)])
        self.assertEqual(labels,{0:CONFLICT,1:CONFLICT})

    def test_random_against_exhaustive_assignments(self):
        rng=random.Random(20261124)
        for _ in range(400):
            n=rng.randrange(1,8)
            edges=[(u,v,rng.choice((-1,1))) for u in range(n)
                   for v in range(u+1,n) if rng.random()<.3]
            anchors=[(u,rng.choice((-1,1))) for u in range(n) if rng.random()<.35]
            expected=exhaustive(range(n),edges,anchors)
            components, labels=solve(range(n),edges,anchors)
            self.assertEqual(labels,expected)
            rng.shuffle(edges); rng.shuffle(anchors)
            self.assertEqual(solve(reversed(range(n)),edges,anchors),
                             (components,labels))

    def test_pilot_adapter_uses_relative_and_absolute_signs(self):
        x=[[1,0,0,0,0,0,0],
           [1,1,-1,1,1,1,1],
           [1,1,1,1,-1,0,0],
           [1,0,0,0,0,0,0],
           [0,0,0,0,0,0,0]]
        self.assertEqual(pilot_labels(x,[0,0,1,3,0]),[0,1,0,2,3])
        with self.assertRaises(ValueError):
            pilot_labels(x,[0,4,1,3,0])

    def test_query_plan_against_exhaustive_subsets(self):
        rng=random.Random(610)
        for _ in range(150):
            n=rng.randrange(1,8)
            edges=[(u-1,u,1) for u in range(1,n) if rng.random()<.5]
            anchors=[(u,1) for u in range(n) if rng.random()<.2]
            comps,_=solve(range(n),edges,anchors)
            offers=[(u,rng.randrange(1,6)) for u in range(n)]
            budget=rng.randrange(0,10)
            owner={u:c for c in comps for u in c.nodes}
            best=0
            for mask in range(1<<n):
                choice=[o for j,o in enumerate(offers) if mask>>j&1]
                if sum(c for u,c in choice)<=budget:
                    touched={owner[u] for u,c in choice if owner[u].status==UNKNOWN}
                    best=max(best,sum(len(c.nodes) for c in touched))
            plan=optimal_anchor_plan(comps,offers,budget)
            self.assertEqual(plan["newly_resolved"],best)
            self.assertLessEqual(plan["cost"],budget)
            self.assertEqual(sum(dict(offers)[u] for u in plan["anchors"]),plan["cost"])


if __name__=="__main__":
    unittest.main()

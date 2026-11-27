"""OLR-01A: seeded acyclic oracle property check over 400 graphs."""
import random
import unittest
from math import inf

from ppt_gym.obligation_ledger import Hypergraph, Rule, propagate


class RandomAcyclicOracleTest(unittest.TestCase):
    def test_400_random_graphs_against_topological_oracle(self):
        rng = random.Random(20261123)
        for _ in range(400):
            count = rng.randrange(2, 8)
            facts = frozenset(v for v in range(count) if rng.random() < .3)
            rules = []
            for head in range(count):
                for _j in range(rng.randrange(4)):
                    inputs = tuple(v for v in range(head) if rng.random() < .4)
                    rules.append(Rule(head, inputs, rng.randrange(1, 5)))
            graph = Hypergraph(count, facts, tuple(rules))
            # Separate topological solver; every premise index precedes its head.
            exact = [inf] * count
            for node in range(count):
                if node in facts:
                    exact[node] = 0.
                for rule in rules:
                    if rule.head == node and all(exact[p] < inf for p in rule.premises):
                        exact[node] = min(exact[node], rule.cost + sum(exact[p] for p in rule.premises))
            actual = propagate(graph, count + 1)
            self.assertTrue(actual.converged)
            self.assertEqual(actual.costs, tuple(exact))


if __name__ == "__main__":
    unittest.main()

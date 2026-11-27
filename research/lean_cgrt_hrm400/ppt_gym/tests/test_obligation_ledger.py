"""Deterministic correctness tests for the OLR-01A structural oracle."""
import unittest
from math import inf

from ppt_gym.obligation_ledger import Hypergraph, Rule, mean_premise_cue, needed_obligations, propagate


class ObligationLedgerTests(unittest.TestCase):
    def test_serial_depth_exact_and_early_halt(self):
        g = Hypergraph(5, frozenset([0]), tuple(Rule(i + 1, (i,)) for i in range(4)))
        result = propagate(g, 12)
        self.assertTrue(result.converged)
        self.assertEqual(result.costs, (0., 1., 2., 3., 4.))
        self.assertEqual(result.steps, 5)  # 4 new edges + convergence check
        self.assertEqual(propagate(g, 2).costs[4], inf)
        self.assertEqual(needed_obligations(g, 4, result), frozenset([1, 2, 3, 4]))
        self.assertEqual(result.source_facts, (True, False, False, False, False))

    def test_and_join_requires_every_premise(self):
        g = Hypergraph(3, frozenset([0]), (Rule(2, (0, 1)),))
        self.assertEqual(propagate(g).costs[2], inf)
        both = Hypergraph(3, frozenset([0, 1]), (Rule(2, (0, 1)),))
        self.assertEqual(propagate(both).costs[2], 1.)

    def test_unresolved_backward_frontier_exposes_missing_premise(self):
        g = Hypergraph(3, frozenset((0,)), (Rule(2, (0, 1)),))
        self.assertEqual(needed_obligations(g, 2, propagate(g)), frozenset((1, 2)))

    def test_or_alternative_and_costs(self):
        g = Hypergraph(4, frozenset([0, 1]), (
            Rule(2, (0, 1), 4.), Rule(2, (1,), 2.), Rule(3, (2,), 1.)))
        self.assertEqual(propagate(g).costs[3], 3.)

    def test_no_spontaneous_cycle_and_positive_seeded_cycle(self):
        no_seed = Hypergraph(2, frozenset(), (Rule(1, (0,)), Rule(0, (1,))))
        self.assertEqual(propagate(no_seed, 30).costs, (inf, inf))
        seeded = Hypergraph(2, frozenset([0]), no_seed.rules)
        self.assertEqual(propagate(seeded, 30).costs, (0., 1.))

    def test_disabled_rule_is_not_a_verified_fact(self):
        g = Hypergraph(2, frozenset([0]), (Rule(1, (0,), 1., False),))
        self.assertEqual(propagate(g).costs, (0., inf))

    def test_mean_pooling_collision_different_truth(self):
        # Same facts, same rule count, same arity, same mean premise cue.
        # A,B are true, C,D are false. One pairing includes a complete rule;
        # another distributes the true premises across incomplete rules.
        facts = frozenset((0, 1))
        with_witness = Hypergraph(5, facts, (Rule(4, (0, 1)), Rule(4, (2, 3))))
        no_witness = Hypergraph(5, facts, (Rule(4, (0, 2)), Rule(4, (1, 3))))
        self.assertEqual(mean_premise_cue(with_witness, 4), 0.5)
        self.assertEqual(mean_premise_cue(no_witness, 4), 0.5)
        self.assertEqual(propagate(with_witness).costs[4], 1.)
        self.assertEqual(propagate(no_witness).costs[4], inf)

    def test_shared_dependency_tree_accounting(self):
        # Two branches both need node 1; tree-cost counts that derivation twice.
        g = Hypergraph(5, frozenset((0,)), (
            Rule(1, (0,)), Rule(2, (1,)), Rule(3, (1,)), Rule(4, (2, 3))))
        self.assertEqual(propagate(g).costs[4], 5.)
        # A DAG with memoized proof of node 1 can use four distinct rules.

    def test_permutation_invariance_of_costs(self):
        edges = (Rule(2, (0, 1), 2), Rule(3, (2,), 1), Rule(3, (1,), 5))
        g1 = Hypergraph(4, frozenset([0, 1]), edges)
        g2 = Hypergraph(4, frozenset([0, 1]), tuple(reversed(edges)))
        self.assertEqual(propagate(g1).costs, propagate(g2).costs)

    def test_zero_budget_and_input_guards(self):
        g = Hypergraph(1, frozenset((0,)), tuple())
        r = propagate(g, 0)
        self.assertFalse(r.converged)
        self.assertEqual(r.steps, 0)
        with self.assertRaises(ValueError):
            Hypergraph(2, frozenset(), (Rule(2, tuple()),))
        with self.assertRaises(ValueError):
            Hypergraph(2, frozenset(), (Rule(1, (0,), -1),))


if __name__ == '__main__':
    unittest.main()

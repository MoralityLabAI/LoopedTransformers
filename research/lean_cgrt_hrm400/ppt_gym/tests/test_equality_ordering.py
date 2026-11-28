"""Symbolic and model engineering checks; native certification is a separate gate."""
from collections import deque
import math
import unittest

import torch

from ppt_gym.data import digest
from ppt_gym.equality_suite import EqCase, neighbors, system_lineage
from ppt_gym.equality_search import plan, path_moves
from ppt_gym.equality_ordering import (
    DIM, Encoder, Net, best_first, pair_bound, positional_bound, hamming, edit,
)
from ppt_gym.tests.test_equality_preflight import fixture


class OrderingTests(unittest.TestCase):
    def test_alpha_orientation_and_order_encoding(self):
        c = fixture()
        t = str.maketrans("abcd", "dcab")
        rules = tuple((b.translate(t), a.translate(t)) for a, b in reversed(c.equations))
        source, target = c.source.translate(t), c.target.translate(t)
        root = digest([c.lineage, c.split, rules, source, target])[:24]
        other = EqCase(root, system_lineage(rules), c.split, c.family, rules, source, target).validate()
        a, b = Encoder(c), Encoder(other)
        self.assertEqual(len(a(c.target)), DIM)
        for w in [c.source, c.target, *(m.next_word for m in neighbors(c.target, c.equations))]:
            self.assertEqual(a(w), b(w.translate(t)))

    def test_admissible_relaxations(self):
        c = fixture()
        pos = positional_bound(c)
        pair, expansions, edges = pair_bound(c)
        self.assertGreater(expansions, 0)
        self.assertGreater(edges, 0)
        q = deque([c.source])
        distances = {c.source: 0}
        while q:
            w = q.popleft()
            d = distances[w]
            for h in (pos(w), pair(w), math.ceil(hamming(w, c.source)/2),
                      math.ceil(edit(w, c.source)/2)):
                self.assertLessEqual(h, d)
            self.assertGreaterEqual(pair(w), pos(w))
            if d >= 4:
                continue
            for m in neighbors(w, c.equations):
                if m.next_word not in distances:
                    distances[m.next_word] = d+1
                    q.append(m.next_word)
        self.assertGreater(len(distances), 10)

    def test_best_first_budget_and_valid_paths(self):
        c = fixture()
        exact = plan(c)
        for arm in ("hamming_greedy", "hamming_astar", "edit_astar",
                    "positional_astar", "pair_astar"):
            for cap in (1, 16, 1024):
                r = best_first(c, cap, arm)
                self.assertLessEqual(r["expansions"], cap)
                if r["found"]:
                    self.assertLessEqual(len(r["words"]), 16)
                    path_moves(c, r["words"])
            r = best_first(c, 1024, arm)
            self.assertTrue(r["found"])
            if arm != "hamming_greedy":
                self.assertEqual(len(r["words"]), len(exact["words"]))

    def test_weight_tying_and_gradient(self):
        c = fixture()
        a, b = Net(4), Net(1)
        self.assertEqual(sum(p.numel() for p in a.parameters()),
                         sum(p.numel() for p in b.parameters()))
        x = torch.tensor([Encoder(c)(c.target), Encoder(c)(c.source)])
        y = a(x)
        self.assertEqual(tuple(y.shape), (2,))
        y.sum().backward()
        self.assertTrue(torch.isfinite(a.loop.weight.grad).all())
        self.assertGreater(float(a.loop.weight.grad.abs().sum()), 0)
        r = best_first(c, 16, "learned", a.eval())
        self.assertGreater(r["model_states_scored"], 0)
        self.assertGreater(r["model_batches"], 0)


if __name__ == "__main__":
    unittest.main()

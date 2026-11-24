"""Prospective guard regression; does not deserialize any archived split."""
from __future__ import annotations

import random
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lbf18_root_guard as g


class RootGuardTests(unittest.TestCase):
    def test_referee_counterexample(self):
        w, rules = g.REFEREE_ROOT, g.REFEREE_RULES
        exact = g.exact_ball(w, rules, 3)
        self.assertEqual(exact, frozenset(("abc", "dec", "def", "abf", "acc")))
        masks = g.first_hop_masks(w, rules)
        self.assertEqual(masks[0], 0b111)
        self.assertTrue(g.legacy_suppresses(w, 0, "ab", 0))
        self.assertFalse(g.safe_suppresses(w, 0, "ab", 0, masks))
        self.assertTrue(g.safe_suppresses(w, 0, "abc", 0, masks))
        self.assertNotIn("abf", g.fiber_ball(w, rules, 2, mode="legacy"))
        self.assertNotIn("acc", g.fiber_ball(w, rules, 3, mode="legacy"))
        self.assertTrue(exact <= g.fiber_ball(w, rules, 3))
        self.assertEqual(exact, g.fiber_ball(w, rules, 3, abstract=False))

    def test_original_chimera_fixture_retains_precision(self):
        w = "abc"
        rules = (("ab", "de"), ("bc", "fa"), ("df", "cc"))
        for k in range(1, 5):
            self.assertEqual(g.fiber_ball(w, rules, k), g.exact_ball(w, rules, k))

    def test_modulo_collision_is_conservative(self):
        w = "abccab"
        rules = (("ab", "de"),)
        masks = g.first_hop_masks(w, rules)
        self.assertEqual(masks[0], 0b110011)
        self.assertFalse(g.safe_suppresses(w, 0, "ab", 0, masks))
        self.assertFalse(g.safe_suppresses(w, 4, "ab", 0, masks))

    def test_generated_fixtures_second_pass_guard_and_horizon_recall(self):
        rng = random.Random(20261124)
        checked = legacy_bad = safe_drops = 0
        for _ in range(200):
            length = rng.randrange(3, 8)
            root = "".join(rng.choices("abcdef", k=length))
            rs = set()
            for j in range(6):
                n = rng.randrange(2, 4)
                p = rng.randrange(length-n+1)
                a = root[p:p+n] if j < 3 else "".join(rng.choices("abcdef", k=n))
                b = "".join(rng.choices("abcdef", k=n))
                if a != b:
                    rs.add((a,b))
            rules = tuple(sorted(rs))
            masks = g.first_hop_masks(root, rules)
            for first in g.moves(root, rules):
                f = first.pos % 4
                for m in g.moves(first.successor, rules):
                    checked += 1
                    if g.legacy_suppresses(root, m.pos, m.rhs, f) and m.successor != root:
                        legacy_bad += 1
                    if g.safe_suppresses(root, m.pos, m.rhs, f, masks):
                        safe_drops += 1
                        self.assertEqual(m.successor, root)
            for k in range(4):
                exact = g.exact_ball(root, rules, k)
                self.assertEqual(exact, g.fiber_ball(root, rules, k, abstract=False))
                self.assertTrue(exact <= g.fiber_ball(root, rules, k))
        self.assertGreater(checked, 0)
        self.assertGreater(safe_drops, 0)
        print(f"guard audit: {checked} second-hop transitions; "
              f"{legacy_bad} unsafe legacy suppressions; {safe_drops} safe root returns suppressed")


if __name__ == "__main__":
    unittest.main()

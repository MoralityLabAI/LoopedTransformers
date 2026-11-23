"""Source invariants and exact rational policy DP, without simulated Lean credit."""
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest

from lcgrt import fragment as F
from ppt_gym import lbf10_source
from ppt_gym.lbf10_uniform import coefficients
from ppt_gym.live_suite import create_suite, load_suite


class StructuralSourceTest(unittest.TestCase):
    def test_independent_equal_cost_live_and_dead_groups(self):
        for seed in range(5):
            for family in F.FAMILIES:
                base = F.generate(seed, 3, family, tid="audit")
                t = lbf10_source.generate(seed, 3, family, tid="audit")
                self.assertEqual(t.to_json(),
                                 lbf10_source.generate(seed, 3, family, tid="audit").to_json())
                size, old = F.reference_sizes(t), F.reference_sizes(base)
                self.assertEqual([size[g] for g in t.goal], [old[g] for g in base.goal])
                self.assertEqual(len(t.facts), len(set(t.facts)))
                all_fresh = set()
                edges = {(tuple(pre), head) for pre, head in t.rules}
                for group in t.gen["lbf10"]["groups"]:
                    head, d = group["head"], group["cost"]
                    self.assertEqual(size[head], d)
                    self.assertEqual(len(group["live"]), d-1)
                    self.assertEqual(len(group["dead"]), d-1)
                    for kind in ("live", "dead"):
                        levels = group[kind]
                        for i, level in enumerate(levels):
                            self.assertEqual(len(set(level)), 2)
                            self.assertFalse(all_fresh & set(level))
                            all_fresh.update(level)
                            self.assertNotIn(head, level)
                            self.assertTrue(all(size[a] == (i+1 if kind=="live" else F.INF)
                                                for a in level))
                            for a in level:
                                self.assertEqual(a in t.facts, kind=="live" and i==0)
                        for lo, hi in zip(levels, levels[1:]):
                            self.assertTrue(all(((a,), b) in edges for a in lo for b in hi))
                        self.assertTrue(all(((a,), head) in edges for a in levels[-1]))
                self.assertTrue(all_fresh)

    def test_suite_custody_and_legacy_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            structural = create_suite(p/"s", seed=20261201, per_split=4, structural_alt=True)
            legacy = create_suite(p/"l", seed=20261202, per_split=4)
            self.assertTrue(structural["structural_alt"])
            self.assertNotIn("structural_alt", legacy)
            cases, src = load_suite(p/"s", "calibration", 4)
            self.assertEqual(len(cases), 4)
            self.assertEqual(src["source_family"], lbf10_source.FAMILY)
            self.assertTrue(all("lbf10" in x.theorem().gen for x in cases))
            cases, _ = load_suite(p/"l", "calibration", 4)
            self.assertTrue(all("lbf10" not in x.theorem().gen for x in cases))
            with self.assertRaises(ValueError):
                create_suite(p/"mixed", per_split=4, structural_alt=True, equal_alt_p=1.0)


class UniformDPTest(unittest.TestCase):
    def test_cycle(self):
        p = coefficients({"g": {"finish": [], "loop": ["g"]}}, 8)["g"]
        self.assertEqual(p[1:], [Fraction(1, 2**n) for n in range(1,9)])
        self.assertEqual(sum(p), Fraction(255,256))

    def test_pending_goals_and_dead_end(self):
        table = {"g":{"and":["a","b"],"dead":["d"]},
                 "a":{"fact":[]}, "b":{"fact":[],"loop":["b"]}, "d":{}}
        p = coefficients(table, 4)["g"]
        self.assertEqual(p, [0,0,0,Fraction(1,4),Fraction(1,8)])
        self.assertEqual(sum(p), Fraction(3,8))

    def test_equal_tactic_witnesses(self):
        p = coefficients({"g":{"a":[],"b":[],"fail":["d"]},"d":{}}, 8)["g"]
        self.assertEqual(sum(p), Fraction(2,3))
        with self.assertRaises(ValueError):
            coefficients({"g":{"missing":["x"]}}, 8)

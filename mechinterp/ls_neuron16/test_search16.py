"""Check the selection semantics that distinguish breadth from greedy depth."""
import unittest

from search16 import select


def node(identity, code, bound):
    return {"id": identity, "codes": [code] * 52, "scores": {"bound": bound}}


class SelectionTests(unittest.TestCase):
    def test_keep_best_scale_variant_before_width_truncation(self):
        pool = [node("old_a", 1, 2.), node("b", 2, 3.), node("new_a", 1, 1.), node("c", 3, 4.)]
        self.assertEqual([n["id"] for n in select(pool, 3)], ["new_a", "b", "c"])

    def test_exact_tie_preserves_incumbent_and_insertion_order(self):
        pool = [node("parent", 1, 1.), node("child", 1, 1.), node("alternate", 2, 1.)]
        self.assertEqual([n["id"] for n in select(pool, 2)], ["parent", "alternate"])

    def test_raw_minimum_applies_to_both_widths(self):
        pool = [node("parent", 1, 1.), node("slightly_better", 2, 1. - 5e-13), node("third", 3, 2.)]
        self.assertEqual(select(pool, 1)[0], select(pool, 3)[0])
        self.assertEqual(select(pool, 1)[0]["id"], "slightly_better")

    def test_worse_child_can_survive_without_worsening_best(self):
        parents = [node("a", 1, .1), node("b", 2, .4), node("c", 3, .8)]
        child = node("child_of_a", 4, .2)
        self.assertEqual([n["id"] for n in select(parents + [child], 3)], ["a", "child_of_a", "b"])
        self.assertEqual(select(parents + [child], 1)[0]["id"], "a")

    def test_short_pool_fails(self):
        with self.assertRaises(ValueError):
            select([node("a", 1, .1), node("same_codes", 1, .2)], 2)


if __name__ == "__main__":
    unittest.main()

import random
import unittest

from research_gym.ink_fp_audit_02 import curriculum as c
from research_gym.ink_fp_audit_02 import gyms  # noqa: F401
from research_gym.ink_fp_audit_02 import stats as s02
from research_gym.ink_fp_audit_03 import frontier as F


def first(fam, pred, n=40):
    for t in c.generate(fam, "dev", n):
        if pred(t):
            return t
    raise AssertionError(f"no {fam} task matching predicate")


class FrontierReferenceTests(unittest.TestCase):
    def test_every_family_reference_verifies_and_perturbation_fails(self):
        for fam in F.FAMILIES:
            for t in c.generate(fam, "dev", 8):
                self.assertTrue(c.verify(t, "work\nANSWER: " + t.answer)["correct"], t.task_id)
                self.assertFalse(c.verify(t, "ANSWER: " + t.wrong_answer(random.Random(1)))["correct"], t.task_id)
                self.assertEqual(c.world_partition(t.source_world), "dev")

    def test_partitions_disjoint(self):
        P = c.build_partitions(list(F.FAMILIES), {"sealed": 3, "accept": 3, "dev": 3, "train": 3})
        self.assertTrue(c.leakage_report(P["partitions"])["ok"])
        self.assertFalse({t.task_id for t in P["partitions"]["train"]} & set(P["sealed_task_ids"]))


class CertificateTests(unittest.TestCase):
    def test_unreachability_certificate_must_be_exact(self):
        t = first("storyworld_v2", lambda t: not t.certificate["reachable"])
        var, val = t.answer.split()[1].split("=")
        self.assertFalse(F.storyworld_v2_check(t, f"NONE {var}={int(val) + 1}"))
        self.assertFalse(F.storyworld_v2_check(t, "NONE"))
        self.assertTrue(F.storyworld_v2_check(t, t.answer))

    def test_plan_rejected_on_unreachable_and_alarm(self):
        t = first("storyworld_v2", lambda t: t.certificate["reachable"])
        self.assertFalse(F.storyworld_v2_check(t, "investigate,investigate,investigate,investigate"))

    def test_odd_cycle_must_use_edges(self):
        t = first("graph_certificate", lambda t: not t.certificate["bipartite"])
        cyc = t.answer.split()[1].split(",")
        self.assertTrue(len(cyc) % 2 == 1)
        es = {frozenset(e) for e in t.certificate["edges"]}
        for tri in ((0, 1, 2), (0, 2, 4), (1, 3, 5)):
            is_cycle = all(frozenset((tri[i], tri[(i + 1) % 3])) in es for i in range(3))
            self.assertEqual(F.graph_check(t, "ODD " + ",".join(map(str, tri))), is_cycle)
        self.assertFalse(F.graph_check(t, "ODD " + ",".join(cyc[:-1])))  # even-length or broken cycle
        self.assertFalse(F.graph_check(t, "BIPARTITE " + ",".join("0" * t.certificate["n"])))

    def test_sorting_network_exhaustive(self):
        t = first("sorting_network", lambda t: t.certificate["n"] == 5)
        self.assertTrue(F.network_check(t, t.answer))
        self.assertFalse(F.network_check(t, ",".join(t.answer.split(",")[:-1])))  # one comparator short
        self.assertFalse(F.network_check(t, F.REFERENCE_NETWORK[5]))  # sorts, but ignores the required prefix
        pre = ",".join(f"{i}-{j}" for i, j in t.certificate["prefix"])
        bubble = pre + "," + ",".join(f"{j}-{j + 1}" for i in range(5) for j in range(4 - i))  # prefix + 10, sorts
        self.assertEqual(F.network_check(t, bubble), t.certificate["max_comparators"] >= len(bubble.split(",")))

    def test_untried_order_insensitive(self):
        t = first("hillclimb_diagnosis", lambda t: t.answer.startswith("UNTRIED") and "," in t.answer)
        items = t.answer.split(" ", 1)[1].split(",")
        self.assertTrue(c.VERIFIERS["hillclimb_diag.v1"](t, "UNTRIED " + ",".join(reversed(items))))
        self.assertFalse(c.VERIFIERS["hillclimb_diag.v1"](t, "UNTRIED " + ",".join(items[:-1])))

    def test_power_matches_audit02_and_minimality(self):
        self.assertAlmostEqual(F.mcnemar_power(199, 0.2, 0.1, 0.05), s02.mcnemar_power(199, 0.2, 0.1, 0.05), places=9)
        t = next(c.generate("power_design", "dev", 1))
        self.assertGreaterEqual(t.certificate["power_at_n"], 0.7)
        self.assertLess(t.certificate["power_at_n_minus_1"], t.certificate["power_at_n"] + 1e-12)

    def test_experiment_choice_margin(self):
        for t in c.generate("experiment_choice", "dev", 6):
            sc = t.certificate["eig_per_cost"]
            self.assertEqual(t.answer, max(sc, key=sc.get))
            self.assertGreaterEqual(t.certificate["margin"], 0.02)


if __name__ == "__main__":
    unittest.main()

import unittest

from research_gym.ink_fp_audit_03 import budget_policies as bp


def cell(ok, ct, fin=True, sec=10.0):
    return {"ok": ok, "ct": ct, "fin": fin, "sec": sec}


GRID = {
    # easy: low solves in 300 tokens
    "t1": {"family": "a", "low": cell(True, 300), "medium": cell(True, 900), "high": cell(True, 2000)},
    # needs a long budget at low effort
    "t2": {"family": "a", "low": cell(True, 3000), "medium": cell(True, 5000), "high": cell(False, 8192, fin=False)},
    # unsolvable
    "t3": {"family": "b", "low": cell(False, 8192, fin=False), "medium": cell(False, 8192, fin=False),
           "high": cell(False, 8192, fin=False)},
    # only medium solves
    "t4": {"family": "b", "low": cell(False, 700), "medium": cell(True, 1500), "high": cell(False, 8192, fin=False)},
}


class GridTests(unittest.TestCase):
    def test_outcome_cap_semantics(self):
        self.assertEqual(bp.outcome(GRID["t2"]["low"], 2048)[:2], (False, 2048))
        self.assertEqual(bp.outcome(GRID["t2"]["low"], 4096)[:2], (True, 3000))
        self.assertEqual(bp.outcome(GRID["t4"]["low"], 1024)[:2], (False, 700))  # finished but wrong

    def test_static(self):
        s = bp.summarize(bp.run_static(GRID, "low", 1024))
        self.assertEqual((s["success"], s["tokens"]), (1, 300 + 1024 + 1024 + 700))

    def test_ladder_charges_failed_rungs(self):
        rows = bp.run_ladder(GRID, [("low", 1024), ("medium", 4096), ("high", 8192)])
        self.assertTrue(rows["t2"]["ok"] is False or rows["t2"]["tokens"] >= 1024)
        self.assertEqual(rows["t4"]["tokens"], 700 + 1500)
        self.assertTrue(rows["t4"]["ok"])
        self.assertEqual(rows["t3"]["tokens"], 1024 + 4096 + 8192)
        s = bp.summarize(rows)
        self.assertEqual(s["escalated"], 3)

    def test_oracle_is_ceiling(self):
        o = bp.summarize(bp.run_oracle(GRID))
        for e in bp.EFFORTS:
            for c in bp.CAPS:
                self.assertGreaterEqual(o["success"], bp.summarize(bp.run_static(GRID, e, c))["success"])
        self.assertEqual(o["success"], 3)
        self.assertEqual(bp.run_oracle(GRID)["t1"]["tokens"], 300)

    def test_router_is_leave_one_family_out(self):
        prompts = {t: ("x " * (10 + i * 50)) for i, t in enumerate(GRID)}
        rows, choices = bp.run_router(GRID, prompts, lam=0.5)
        self.assertEqual(set(rows), set(GRID))
        self.assertEqual(sum(choices.values()), len(GRID))


if __name__ == "__main__":
    unittest.main()

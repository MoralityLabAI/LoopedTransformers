import unittest
from public_smoke import ARMS, ROOT_NAME, render, route


class PublicSmokeTests(unittest.TestCase):
    def test_scope(self):
        for arm in ARMS:
            source, term, _ = render(4, 0, arm)
            before = source.split("theorem pdiTarget")[0]
            self.assertEqual(ROOT_NAME in before, arm == "verified_reuse")
            self.assertEqual(ROOT_NAME in term, arm == "verified_reuse")
            self.assertNotIn("pdiTarget", before)
            self.assertIn("#print axioms PDIPublicSmoke.pdiTarget", source)

    def test_sentinels(self):
        source, term, _ = render(2, 0, "revoked_version", probe="stale")
        self.assertIn(ROOT_NAME, term)
        self.assertNotIn("theorem " + ROOT_NAME, source)
        self.assertEqual(render(2, 0, "revoked_version", probe="self")[1], "pdiTarget")

    def test_route(self):
        for depth in range(2, 10):
            baseline, expansions = route(depth, 0, "no_reuse")
            self.assertEqual(expansions, depth + 1)
            self.assertEqual(route(depth, 0, "matched_retrieval"), (baseline, expansions))
            self.assertLessEqual(route(depth, 0, "verified_reuse")[1], expansions)


if __name__ == "__main__":
    unittest.main()

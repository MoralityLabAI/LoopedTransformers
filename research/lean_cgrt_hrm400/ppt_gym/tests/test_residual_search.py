from unittest import TestCase

from ppt_gym.residual_search import PROBE, STOP, TARGET, ResidualSearch, State
from ppt_gym.residual_search_policy import opaque_context


class ResidualSearchTests(TestCase):
    def setUp(self):
        child = {"actions": {"apply r": {"status": "closed"}}}
        self.search = ResidualSearch({
            "target": {"actions": {"omega": {"status": "rejected"},
                                   "apply r": {"status": "progress", "child": child}}},
            "sibling": {"actions": {"omega": {"status": "closed"}}},
        })

    def test_sibling_closure_is_paid_evidence_only(self):
        actions = iter(((PROBE, "omega"), STOP))
        trial = self.search.run(2, lambda _: next(actions))
        self.assertFalse(trial["proof"])
        self.assertEqual(trial["script"], [])
        self.assertAlmostEqual(trial["utility"], -.05)

    def test_progress_keeps_sibling_feedback_and_reuses_tactic_on_new_goal(self):
        initial = State(2)
        _, after_probe, _ = self.search.transition(initial, (PROBE, "omega"))
        self.assertEqual(after_probe.left, 2)
        self.assertNotIn((PROBE, "omega"), self.search.menu(after_probe))
        _, after_progress, _ = self.search.transition(after_probe, (TARGET, "apply r"))
        self.assertEqual(after_progress.probe, ("omega", "closed"))
        self.assertIn((TARGET, "apply r"), self.search.menu(after_progress))
        actions = iter(((PROBE, "omega"), (TARGET, "apply r"), (TARGET, "apply r")))
        trial = self.search.run(2, lambda _: next(actions))
        self.assertTrue(trial["proof"])
        self.assertAlmostEqual(trial["utility"], .90)

    def test_rejected_target_attempt_blocks_only_that_tactic(self):
        _, after, _ = self.search.transition(State(2), (TARGET, "omega"))
        self.assertEqual(after.branch, "")
        self.assertNotIn((TARGET, "omega"), self.search.menu(after))
        self.assertIn((TARGET, "apply r"), self.search.menu(after))

    def test_opaque_aliases_preserve_types_and_action_binding(self):
        original = {"goals": [{"pp": "h1 : True\n⊢ h1 = Foo.h1", "target": {"raw": "h1 = Foo.h1"},
                               "hypotheses": [{"name": "h1", "type": {"raw": "True"}}]}]}
        changed, actions = opaque_context(original, ("apply h1", "omega"), "case")
        alias = changed["goals"][0]["hypotheses"][0]["name"]
        self.assertNotEqual(alias, "h1")
        self.assertEqual(actions, (f"apply {alias}", "omega"))
        self.assertEqual(changed["goals"][0]["target"], {"raw": f"{alias} = Foo.h1"})
        self.assertEqual(original["goals"][0]["hypotheses"][0]["name"], "h1")

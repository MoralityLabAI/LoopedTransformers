from unittest import TestCase

import torch

from configs.run_ppt_mathcan_sparse import Transition, observed_support, play
from ppt_gym.mathcan_deadline import deadline_utility, supported_distances
from ppt_gym.mathcan_sparse import Frame, RecordedSearch


class Archive:
    key = ("MathCan/Example.lean", "example", ())
    states = {key: {name: {"state_sha256": "state", "source_sha256": "source"}
                    for name in ("reject", "unknown", "close")}}

    def step(self, key, action):
        return {"reject": ("rejected", key),
                "unknown": ("censored", None),
                "close": ("archived_closure", None)}[action]


class SparseSearchTests(TestCase):
    def test_censor_has_no_return_and_rejection_preserves_the_prefix(self):
        search = RecordedSearch(Archive())
        initial = Frame(Archive.key, 3)
        after = search.step(initial, "reject")["after"]
        self.assertEqual(after.key, initial.key)
        self.assertEqual(after.left, 2)
        self.assertNotIn("reject", search.menu(after))
        censored = play(search, initial.key, 3, lambda _: "unknown")
        self.assertTrue(censored["censored"])
        self.assertIsNone(censored["observed_return"])
        failed = play(search, initial.key, 1, lambda _: "reject")
        self.assertFalse(failed["censored"])
        self.assertEqual(failed["observed_return"], -.05)

    def test_support_transfers_across_histories_only_at_same_prefix(self):
        search = RecordedSearch(Archive())
        initial = Frame(Archive.key, 3)
        after = search.step(initial, "reject")["after"]
        tape = [Transition(initial, "reject", after, "rejected", False, None, False),
                Transition(initial, "close", None, "archived_closure", True, 1, True),
                Transition(Frame(("another", "example", ()), 3), "unknown", None,
                           "archived_closure", True, 1, True)]
        self.assertEqual(observed_support(search, tape)[after], {"close"})

    def test_deadline_backup_joins_sampled_continuations_and_charges_retries(self):
        initial = Frame(Archive.key, 8)
        next_key = (*Archive.key[:2], ("intro",))
        next_state = Frame(next_key, 8)
        tape = [(initial, "reject", "rejected", initial.key, None),
                (initial, "intro", "accepted_nonterminal", next_key, None),
                (next_state, "close", "archived_closure", None, None),
                (initial, "unknown", "censored", None, None)]
        self.assertEqual(supported_distances(tape), [3, 2, 1])
        # The slow tactic can have stronger eventual success and still miss the deadline.
        curves = torch.tensor([[[.7] * 8, [0., 0.] + [1.] * 6]])
        self.assertEqual(int(deadline_utility(curves, torch.tensor([2])).argmax()), 0)
        self.assertEqual(int(deadline_utility(curves, torch.tensor([8])).argmax()), 1)

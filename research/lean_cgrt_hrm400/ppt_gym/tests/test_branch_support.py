from itertools import product
from unittest import TestCase

import torch

from ppt_gym.branch_support import BRANCHES, OPEN, PROBES, State, Support, legal, observe, play
from ppt_gym.branch_support_policy import raw_features, tokens


def worlds():
    return [{"observations": [{"value": sum(bit for i, bit in enumerate(bits) if key & (1 << i)) % 2}
                               for key in range(1, 8)]} for bits in product((0, 1), repeat=3)]


class BranchSupportTests(TestCase):
    def test_two_purchases_resolve_both_contexts_in_the_novel_deck(self):
        bank = worlds()
        support = Support(bank)
        for allowance, expected in ((1, .825), (2, .90)):
            start = State((3, 7), allowance)
            outcomes = []
            for world in bank:
                for key in start.deck:
                    record = play(world, start, key, lambda state: max(
                        support.exact(state), key=support.exact(state).__getitem__))
                    kept = record["branch"] ^ world["observations"][key - 1]["value"]
                    outcomes.append(1 - .5 * kept - .05 * len(record["probes"]))
                    self.assertLessEqual(len(record["probes"]), allowance)
            self.assertAlmostEqual(sum(outcomes) / len(outcomes), expected)

    def test_composition_and_expiry_use_only_paid_native_values(self):
        start = State((3, 7), 2)
        first = observe(start, PROBES.index(3), 1)
        both = observe(first, PROBES.index(4), 0)
        self.assertEqual(legal(both), (OPEN,))
        query = State(start.deck, 0, both.observed, 7)
        items = torch.tensor(tokens(query))
        for slot in (0, 1):
            self.assertEqual(float(items[BRANCHES[0], slot, 6]), 0. if slot == 0 else -1.)
        features = raw_features(items)
        self.assertEqual(float(features[BRANCHES[0], -4]), -1.)
        self.assertEqual(float(features[BRANCHES[1], -4]), 1.)

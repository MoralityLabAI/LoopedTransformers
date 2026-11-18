"""Independent source tests for LRM-REV-02; only fixture sources are opened."""
from dataclasses import replace
from fractions import Fraction as Q
from itertools import combinations, product
from math import comb
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gym import MATCH, PRIORS, Episode
from lrm_rev02_fixture import (
    CONTROL_COUNT, DEVELOPMENT_EPISODES, DOMAIN, FIXTURE_EPISODES, INPUT_DIM,
    J, N, ROOT_COUNT, TRAIN_EPISODES, control_slots, encode_prefix,
    encode_stream, initial_posterior, make_pilot_episode, model_seed,
    training_slots,
)


def named_initial_posterior(old, j):
    """Sum raw named-bit likelihoods over every hidden schedule."""
    n = len(old)
    weights = []
    for prior, t in zip(PRIORS, MATCH):
        weight = Q(0)
        for slots in combinations(range(n), j):
            p = prior / comb(n, j)
            for i, bit in enumerate(old):
                p *= Q(1, 2) if i in slots else (t if bit else 1 - t)
            weight += p
        weights.append(weight)
    total = sum(weights, Q(0))
    return tuple(weight / total for weight in weights)


def observed_posterior(bits):
    weights = []
    for prior, t in zip(PRIORS, MATCH):
        weight = prior
        for bit in bits:
            weight *= t if bit else 1 - t
        weights.append(weight)
    total = sum(weights, Q(0))
    return tuple(weight / total for weight in weights)


def decode(token):
    return (token[:3].index(1.0), token[3:15].index(1.0),
            token[15:].index(1.0))


class PilotFixtureTests(unittest.TestCase):
    def test_initial_posterior_matches_hidden_schedule_enumeration(self):
        for n in range(1, 6):
            for j in range(n + 1):
                for old in product((0, 1), repeat=n):
                    self.assertEqual(initial_posterior(sum(old), n, j),
                                     named_initial_posterior(old, j))
        # All provisional bits carry zero initial information about the regime.
        for count in range(5):
            self.assertEqual(initial_posterior(count, 4, 4), PRIORS)

    def test_fixture_ids_roles_and_reproducibility(self):
        identifiers = set()
        choices = []
        for root in range(ROOT_COUNT):
            for index in range(8):
                e = make_pilot_episode(root, index, "fixture")
                self.assertEqual(e, make_pilot_episode(root, index, "fixture"))
                self.assertNotIn(e.source_id, identifiers)
                identifiers.add(e.source_id)
                self.assertEqual(len(e.source_id), 64)
                int(e.source_id, 16)
                self.assertEqual(len(e.old), N)
                self.assertEqual(len(e.correction_slots), J)
                self.assertEqual(tuple(sorted(e.correction_slots)), e.correction_slots)
                self.assertIn(e.true_regime, range(3))
                choices.append(training_slots(e) == e.correction_slots)
                self.assertEqual(training_slots(e), training_slots(e))
        self.assertIn(True, choices)
        self.assertIn(False, choices)
        self.assertEqual(len({model_seed(root) for root in range(ROOT_COUNT)}), ROOT_COUNT)
        self.assertNotIn("lrm-rev01/", DOMAIN)

    def test_public_tokens_truth_and_equal_input_dose(self):
        for root in range(ROOT_COUNT):
            e = make_pilot_episode(root, root, "fixture")
            prefix = encode_prefix(e)
            self.assertEqual(len(prefix), 18)
            changed_private = replace(e, source_id="private-source-changed",
                                      true_regime=(e.true_regime + 1) % 3)
            self.assertEqual(prefix, encode_prefix(changed_private))
            changed_future = replace(
                e, correction_slots=tuple(range(6)),
                replacements=tuple(1 - bit for bit in e.replacements))
            self.assertEqual(prefix[:12], encode_prefix(changed_future)[:12])
            self.assertEqual([decode(row) for row in prefix[:12]],
                             [(0, i, bit) for i, bit in enumerate(e.old)])
            accepted = list(e.old)
            for row in prefix[12:]:
                event_type, i, bit = decode(row)
                self.assertEqual(event_type, 1)
                self.assertIn(i, e.correction_slots)
                accepted[i] = bit
            self.assertEqual(tuple(accepted), e.accepted())
            self.assertEqual(observed_posterior(accepted), e.full_reference())
            for slots in (e.correction_slots,
                          *(control_slots(e, c) for c in range(CONTROL_COUNT))):
                stream = encode_stream(e, slots)
                self.assertEqual(len(stream), 24)
                self.assertEqual(stream[:18], prefix)
                self.assertEqual([decode(row) for row in stream[18:]],
                                 [(2, i, e.old[i]) for i in sorted(slots)])
                for row in stream:
                    self.assertEqual(len(row), INPUT_DIM)
                    self.assertTrue(all(type(value) is float for value in row))
                    self.assertTrue(all(value in (0.0, 1.0) for value in row))
                    self.assertEqual(sum(value * value for value in row), 3.0)

    def test_control_and_training_choices_ignore_outcomes(self):
        e = make_pilot_episode(0, 3, "fixture")
        changed = replace(e, old=tuple(1 - bit for bit in e.old),
                          replacements=tuple(1 - bit for bit in e.replacements),
                          true_regime=(e.true_regime + 2) % 3)
        for c in range(CONTROL_COUNT):
            slots = control_slots(e, c)
            self.assertEqual(slots, control_slots(changed, c))
            self.assertEqual(len(slots), J)
            self.assertEqual(len(set(slots)), J)
            self.assertEqual(tuple(sorted(slots)), slots)
            self.assertTrue(all(0 <= i < N for i in slots))
        self.assertEqual(training_slots(e), training_slots(changed))
        # A control that coincides with S is preserved without resampling.
        coincident = replace(e, correction_slots=control_slots(e, 0))
        self.assertEqual(control_slots(coincident, 0), coincident.correction_slots)

    def test_invalid_phase_root_index_and_metadata_rejected_before_generation(self):
        cases = [
            (True, 0, "fixture"), (-1, 0, "fixture"), (ROOT_COUNT, 0, "fixture"),
            (0, True, "fixture"), (0, 0.0, "fixture"), (0, -1, "fixture"),
            (0, FIXTURE_EPISODES, "fixture"), (0, TRAIN_EPISODES, "train"),
            (0, DEVELOPMENT_EPISODES, "development"), (0, 0, "confirmation"),
            (0, 0, "test"), (0, 0, None), (0, 0, True),
        ]
        with patch("lrm_rev02_fixture._rng", side_effect=AssertionError("RNG reached")):
            for args in cases:
                with self.subTest(args=args), self.assertRaises(ValueError):
                    make_pilot_episode(*args)
        # Cached valid calls must not admit bool or float metadata aliases.
        initial_posterior(1, 4, 2)
        for args in ((True, 4, 2), (1.0, 4, 2), (1, True, 0),
                     (1, 4, True), (1, 0, 0), (1, 13, 6),
                     (-1, 4, 2), (5, 4, 2), (1, 4, 5)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                initial_posterior(*args)
        e = make_pilot_episode(0, 0, "fixture")
        for control in (-1, CONTROL_COUNT, True, 0.0):
            with self.assertRaises(ValueError):
                control_slots(e, control)
        for slots in ((0,) * J, tuple(range(5)), (0, 1, 2, 3, 4, 12),
                      (False, 1, 2, 3, 4, 5)):
            with self.assertRaises(ValueError):
                encode_stream(e, slots)
        with self.assertRaises(ValueError):
            encode_prefix(replace(e, old=(0,) * 11))
        with self.assertRaises(ValueError):
            encode_prefix(replace(e, replacements=(0, 0, 0, 0, 0, True)))


if __name__ == "__main__":
    unittest.main()

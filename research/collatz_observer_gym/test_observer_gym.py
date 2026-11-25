import itertools
import unittest
from collections import Counter, defaultdict

from .collatz_core import analyze, branch_history, full_step, modular_labels, odd_step
from .observer_gym import DEPTHS, ceiling, matched_shuffle, observer, recover_seed, run


class ObserverChecks(unittest.TestCase):
    def test_acceleration_matches_unaccelerated_map(self):
        for d in (1, 5, 13, 29, 313):
            for n in range(1, 2 * d + 2, 2):
                nxt, a, _ = odd_step(n, d)
                full = full_step(n, d)
                for _ in range(a):
                    self.assertEqual(full % 2, 0)
                    full = full_step(full, d)
                self.assertEqual(full, nxt)

    def test_ceiling_against_all_decoders(self):
        labels = {0: 0, 1: 1, 2: 1, 3: 0, 4: 2}
        observations = {0: "a", 1: "a", 2: "b", 3: "b", 4: "b"}
        brute = max(sum(mapping[0 if observations[n] == "a" else 1] == y
                        for n, y in labels.items())
                    for mapping in itertools.product(range(3), repeat=2))
        self.assertEqual(ceiling(labels, observations)["correct"], brute)

    def test_invariant_and_history_refinement(self):
        for d in (13, 29, 313, 2021):
            core, q = analyze(d), modular_labels(d)
            histories = {n: branch_history(n, d, 16) for n in core.states}
            baseline = ceiling(core.basin, {n: q[n % d] for n in core.states})
            last = baseline["correct"]
            for depth in DEPTHS:
                repeated = observer(core, q, histories, "quotient_repeat", depth)
                self.assertEqual(ceiling(core.basin, repeated), baseline)
                ordered = observer(core, q, histories, "branch_exponent_prefix", depth)
                score = ceiling(core.basin, ordered)["correct"]
                self.assertGreaterEqual(score, last)
                bag = observer(core, q, histories, "branch_exponent_bag", depth)
                self.assertGreaterEqual(score, ceiling(core.basin, bag)["correct"])
                last = score
            for n in core.states:
                self.assertEqual(q[n % d], q[core.successors[n] % d])

    def test_sham_preserves_cells_within_quotient(self):
        core, q = analyze(2021), modular_labels(2021)
        histories = {n: branch_history(n, 2021, 4) for n in core.states}
        obs = observer(core, q, histories, "branch_exponent_prefix", 4)
        shuffled = matched_shuffle(core, q, obs, 11)
        before, after = defaultdict(Counter), defaultdict(Counter)
        for n in core.states:
            before[q[n % core.d]][obs[n]] += 1
            after[q[n % core.d]][shuffled[n]] += 1
        self.assertEqual(before, after)

    def test_collision_has_different_exact_cycles(self):
        core, q = analyze(13), modular_labels(13)
        self.assertEqual(q[1], q[5])
        self.assertNotEqual(core.basin[1], core.basin[5])
        self.assertEqual(ceiling(core.basin, {n: q[n % 13] for n in core.states})["correct"], 4)

    def test_affine_prefix_recovers_seed(self):
        for d in (1, 13, 313, 1025, 2021):
            for n in analyze(d).states:
                history = branch_history(n, d, 5)
                self.assertEqual(recover_seed(d, history), n)
        self.assertIsNone(recover_seed(13, ()))
        with self.assertRaises(ValueError):
            recover_seed(13, ((0, 1),))

    def test_repeat_is_deterministic_and_invalid_inputs_fail(self):
        self.assertEqual(run(29, 13), run(29, 13))
        for n, d in ((2, 5), (1, 3), (0, 5), (1, -3)):
            with self.assertRaises(ValueError):
                odd_step(n, d)
        with self.assertRaises(ValueError):
            modular_labels(9)


if __name__ == "__main__":
    unittest.main()

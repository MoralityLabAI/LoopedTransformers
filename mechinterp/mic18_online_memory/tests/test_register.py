"""Independent truth-table and interface checks for the exact neural port."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from register import MemoryProgram, ReluNetwork, make_cell, observer_before_query, pack, word_bits


class RegisterTests(unittest.TestCase):
    def test_overwrite_truth_table_and_repeated_writes(self):
        for width in range(1, 5):
            program = MemoryProgram.build(width)
            for word in range(2**width):
                state = word_bits(word, width)
                for phase in range(width):
                    for bit in (0, 1):
                        expected = list(state)
                        expected[phase] = bit
                        updated = program.update(state, bit, phase)
                        self.assertEqual(updated, tuple(expected))
                        self.assertEqual(program.update(updated, 1-bit, phase)[phase], 1-bit)
                        for query in range(width):
                            self.assertEqual(program.query(updated, query), expected[query])

    def test_delayed_query_receives_only_committed_state(self):
        program = MemoryProgram.build(3)
        a = program.commit((1, 0, 1))
        b = program.commit((0, 1, 1))
        self.assertEqual(observer_before_query(a), observer_before_query(b))
        self.assertEqual(program.query(a, 0), 1)
        self.assertEqual(program.query(b, 0), 0)
        self.assertEqual(program.query(a, 2), program.query(b, 2))

    def test_parity_recurrent_truth_table(self):
        program = MemoryProgram.build(4, "parity", "parity")
        for word in range(16):
            observations = word_bits(word, 4)
            self.assertEqual(program.query(program.commit(observations)), sum(observations) % 2)
        compact = MemoryProgram.build(1, "parity", "parity")
        for parity in (0, 1):
            for bit in (0, 1):
                self.assertEqual(compact.update((parity,), bit, 0), (parity ^ bit,))

    def test_matched_arithmetic_and_state_allocation(self):
        for width in range(1, 9):
            reports = [MemoryProgram.build(width, mode).resource_account()
                       for mode in ("register", "parity", "last_bit")]
            for key in ("allocated_binary_state_bits", "episode_dense_products", "episode_relu_evaluations"):
                self.assertEqual(len({report[key] for report in reports}), 1)
            self.assertEqual(reports[0]["episode_relu_evaluations"], 2*width*width+width)
            self.assertEqual(reports[0]["episode_dense_products"], 6*width**3+4*width**2+width)

    def test_mutated_weight_is_detected_by_truth_table(self):
        program = MemoryProgram.build(2)
        weights = [list(row) for row in program.cell.w_in]
        weights[1][2] = 0
        damaged = replace(program, cell=replace(program.cell, w_in=tuple(map(tuple, weights))))
        self.assertNotEqual(damaged.update((0, 0), 1, 0), (1, 0))

    def test_invalid_domains_and_shapes_fail(self):
        program = MemoryProgram.build(2)
        for args in [((0, 2), 1, 0), ((0, 0), True, 0), ((0,), 1, 0), ((0, 0), 1, 2)]:
            with self.assertRaises(ValueError):
                program.update(*args)
        with self.assertRaises(ValueError):
            program.query((0, 1), -1)
        with self.assertRaises(ValueError):
            ReluNetwork(((1,), (1, 0)), (0, 0), ((1, 1),), (0,))
        with self.assertRaises(ValueError):
            make_cell(0)

    def test_word_order_and_compaction(self):
        for width in range(1, 9):
            for word in range(2**width):
                self.assertEqual(pack(word_bits(word, width)), word)
        self.assertEqual(word_bits(5, 3), (1, 0, 1))


if __name__ == "__main__":
    unittest.main()

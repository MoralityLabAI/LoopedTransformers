import unittest
from experiments.nan_prime20.common import public_rows
from experiments.nan_prime20.audit import coverage_groups


class MissingCoverage(unittest.TestCase):
    def test_partial_world_is_not_a_complete_root_pair(self):
        _, public = public_rows(); source = public[0]
        groups = coverage_groups(public, {source['id']: {'correct': True, 'valid': True}})
        focused = next(r for r in groups if r['arm'] == 'focused')
        self.assertEqual(focused['graded_contexts'], 1); self.assertEqual(focused['complete_world_pairs'], 0)
        self.assertEqual(focused['both_world_roots_correct'], 0)
        self.assertEqual(focused['whole_registered_correctness_lower_bound'], 1/32)
        self.assertEqual(focused['whole_registered_correctness_upper_bound'], 1.)

    def test_unbound_response_does_not_create_a_grade(self):
        _, public = public_rows()
        groups = coverage_groups(public, {'not-an-official-source': {'correct': True, 'valid': True}})
        for row in groups:
            self.assertEqual(row['graded_contexts'], 0); self.assertEqual(row['missing_graded_contexts'], 32)
            self.assertEqual(row['whole_registered_correctness_lower_bound'], 0.)
            self.assertEqual(row['whole_registered_correctness_upper_bound'], 1.)


if __name__ == '__main__':
    unittest.main()

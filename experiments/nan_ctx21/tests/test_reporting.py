"""Missing-world bounds and independent bounded grammar regressions."""
import unittest
from experiments.nan_ctx21.audit import grade
from experiments.nan_ctx21.report import groups


class ReportingContracts(unittest.TestCase):
    def roster(self):
        return [dict(id=root+world, root=root, world=world, split='development',context_budget=8192,position='late',arm='main')
            for root in ('r0','r1') for world in ('A','B')]

    def test_pending_is_unknown_not_zero_accuracy(self):
        for g in groups(self.roster(), []):
            self.assertIsNone(g['correct_worlds']); self.assertIsNone(g['both_world_roots_correct'])
            self.assertEqual(g['whole_roster_correct_bounds'],[0,4]); self.assertEqual(g['whole_roster_both_world_bounds'],[0,2])

    def test_partial_pairs_keep_known_wrong_worlds_out_of_upper_bound(self):
        source = self.roster()
        records = [source[0] | dict(correct=[1,1],format_valid=[1,1],token_ids=[1,1]),
                   source[3] | dict(correct=[0,0],format_valid=[1,1],token_ids=[2,2])]
        for g in groups(source,records):
            self.assertEqual((g['observed_contexts'],g['missing_contexts'],g['complete_world_pairs']),(2,2,0))
            self.assertEqual(g['whole_roster_correct_bounds'],[1,3]); self.assertEqual(g['whole_roster_both_world_bounds'],[0,1])
            self.assertIsNone(g['both_world_roots_correct'])

    def test_anchored_grammar_accepts_no_prose_or_unclosed_wrapper_rescue(self):
        for text in ('A',' A\n','[P]','**B**','*C*.','`D`'):
            self.assertIn(grade(text),'ABCDEFGHIJKLMNOP')
        for text in ('Answer: A','A or B','**A','```A```','Q','A B','[]','a','A'*65):
            self.assertIsNone(grade(text))


if __name__=='__main__':
    unittest.main()

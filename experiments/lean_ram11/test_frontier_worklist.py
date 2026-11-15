"""RAM11F worklist is source-only, deterministic and never certificate-credited."""
from __future__ import annotations
import unittest
from test_prefix_audit import row
from frontier_worklist import worklist

class FrontierTests(unittest.TestCase):
    def test_unexecuted_goal_alias_queued_as_unverified(self):
        rows=[row([],'A','x',['B'],2),
              row([],'A','y',['C'],2),
              row(['x'],'B','done',[],3),
              row(['y'],'C','q',['B'],2)]
        selected,summary=worklist(rows,4)
        self.assertEqual(len(selected),1)
        self.assertEqual(selected[0]['proposed_new_actions'],['y','q','done'])
        self.assertEqual(selected[0]['alias_recorded_prefix'],['x'])
        self.assertEqual(summary['full_source_proofs_certified'],0)
        self.assertEqual(summary['new_compiler_queries'],0)
        again,_=worklist(rows,4)
        self.assertEqual(selected,again)

    def test_no_alias_followup_is_censored(self):
        rows=[row([],'A','go',['C'],2)]
        proposals,receipt=worklist(rows,4)
        self.assertEqual(proposals,[])
        self.assertEqual(receipt['candidate_complete_scripts_selected'],0)

    def test_exact_prefix_already_recorded_is_excluded(self):
        rows=[row([],'A','x',['B'],2),row(['x'],'B','done',[],3)]
        proposals,_=worklist(rows,4)
        self.assertEqual(proposals,[])

if __name__=='__main__':unittest.main()

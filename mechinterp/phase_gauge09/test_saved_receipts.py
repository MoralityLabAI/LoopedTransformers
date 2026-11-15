"""Immutable synthetic receipt cross-checks. Original compressed sources are
verified separately by replay_coe04_synthetic and replay_coe05_synthetic.
"""
from __future__ import annotations
import json
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent

class SavedSyntheticReceipts(unittest.TestCase):
    def test_first_cohort(self):
        r=json.loads((HERE/'COE09_SYNTHETIC_RECEIPT.json').read_text())
        self.assertTrue(r['explicitly_not_pretrained'])
        self.assertEqual(r['fit_contexts'],[0,1,2,3])
        self.assertEqual(r['heldout_contexts'],[4,5,6,7])
        self.assertEqual([x['seed'] for x in r['results']],[2406,2407,2408])
        for mode,expected in {'ordered':[15,11,16], 'cyclic':[12,11,15],
                              'count':[12,11,15],'final':[8,5,8]}.items():
            self.assertEqual([x['label_alignment']['modes'][mode]['observed_correct']
                              for x in r['results']],expected)
        self.assertEqual(r['source_activation_file_sha256'],
                         'faba1776a02c3b747f9999247095a12bb51f7b8d167d2251c62ff25d575c49a0')

    def test_second_cohort(self):
        r=json.loads((HERE/'COE09_SYNTHETIC_REPLICATION_RECEIPT.json').read_text())
        self.assertTrue(r['explicitly_not_pretrained'])
        self.assertEqual([x['seed'] for x in r['rows']],[2411,2412,2413])
        self.assertEqual([x['original_heldout_pair_hits'] for x in r['rows']],[82,70,91])
        self.assertEqual([x['original_count_only_pair_hits'] for x in r['rows']],[78,69,86])
        for mode,expected in {'ordered':[11,9,14], 'cyclic':[8,8,11],
                              'count':[8,8,11],'final':[8,8,8]}.items():
            self.assertEqual([x['label_alignment']['modes'][mode]['observed_correct']
                              for x in r['rows']],expected)

if __name__=='__main__':unittest.main()

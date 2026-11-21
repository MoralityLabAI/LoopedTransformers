import copy
import re
import unittest
from collections import Counter
from experiments.nan_comp14.common import ROOT, read
from experiments.nan_comp14.parser import label
from experiments.nan_comp14.sources import materialize


class Contracts(unittest.TestCase):
    def test_bounded_grammar_has_no_prose_or_unclosed_wrapper_rescue(self):
        for text in ('A', ' A\n', '[P]', '**B**', '*C*.', '`D`'):
            self.assertIn(label(text), 'ABCDEFGHIJKLMNOP')
        for text in ('Answer: A', 'A or B', '**A', '```A```', 'Q', 'A B', '[]', 'a', 'A' * 65):
            self.assertIsNone(label(text))

    def test_every_pair_changes_the_query_answer_and_only_two_sources(self):
        cfg = read(ROOT / 'config.json'); public, gold = materialize(cfg)
        self.assertEqual(len(public), 240); self.assertEqual(len(gold), 240)
        self.assertEqual(len({r['root'] for r in public}), 24)
        self.assertTrue(all('answer' not in r for r in public))
        by_id = {r['id']: r for r in gold}
        for r in public:
            g = by_id[r['id']]
            if r['arm'] in cfg['templates']:
                pairs = dict(re.findall(r'([A-P])->([A-P])', r['prompt'].split('\n\n')[-1]))
                self.assertEqual(pairs[g['query']], g['answer'])
            if r['world'] == 'A':
                b = by_id[r['id'].replace(':A:', ':B:')]
                self.assertNotEqual(g['answer'], b['answer'])
                self.assertEqual(sum(g['edges'][k] != b['edges'][k] for k in g['edges']), 2)

    def test_erased_pairs_are_identical_and_split_targets_are_balanced(self):
        cfg = read(ROOT / 'config.json'); public, gold = materialize(cfg)
        by_id = {r['id']: r for r in public}
        for r in public:
            if r['world'] == 'A' and r['arm'].endswith('_erased'):
                self.assertEqual(r['prompt'], by_id[r['id'].replace(':A:', ':B:')]['prompt'])
        for split in cfg['roots']:
            counts = Counter(r['answer'] for r in gold if r['split'] == split and r['arm'] == 'focused')
            self.assertEqual(len(counts), 16); self.assertEqual(len(set(counts.values())), 1)


if __name__ == '__main__':
    unittest.main()

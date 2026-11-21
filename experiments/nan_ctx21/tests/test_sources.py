"""Full source coverage plus adversarial oracle, erasure and length mutations."""
import copy
import unittest
from experiments.nan_ctx21.common import ROOT, read, rows
from experiments.nan_ctx21.source_audit import parse, validate, old_graphs


class SourceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.public = rows(ROOT/'fixtures/public.jsonl')
        cls.host = rows(ROOT/'fixtures/host_labels.jsonl')
        cls.tokens = read(ROOT/'fixtures/tokenizer_receipt.json')
        cls.cfg = read(ROOT/'config.json')

    def test_whole_fresh_source_roster(self):
        result = validate(self.public, self.host, self.tokens, self.cfg, old_graphs())
        self.assertEqual((result['contexts'], result['fresh_target_graphs']), (624, 40))

    def test_missing_record_is_not_zero(self):
        with self.assertRaises(AssertionError):
            validate(self.public[:-1], self.host[:-1], self.tokens, self.cfg)

    def test_host_answer_must_match_textual_edge(self):
        host = copy.deepcopy(self.host); host[0]['answer'] = 'P' if host[0]['answer'] != 'P' else 'A'
        with self.assertRaises(AssertionError):
            validate(self.public, host, self.tokens, self.cfg)

    def test_every_target_destination_is_erased(self):
        row = copy.deepcopy(next(r for r in self.public if r['arm'] == 'erased'))
        name = self.host[next(i for i,r in enumerate(self.public) if r['id'] == row['id'])]['target_graph_id']
        before, after = row['prompt'].split('Graph '+name+': ', 1)
        row['prompt'] = before+'Graph '+name+': '+after.replace('->?', '->A', 1)
        with self.assertRaises(AssertionError):
            parse(row)

    def test_long_context_band_is_enforced(self):
        tokens = copy.deepcopy(self.tokens); tokens['contexts'][self.public[0]['id']]['tokens'] = 4097
        with self.assertRaises(AssertionError):
            validate(self.public, self.host, tokens, self.cfg)

    def test_exposed_graph_overlap_rejects_not_resamples(self):
        old = {tuple(sorted(self.host[0]['target_edges'].items()))}
        with self.assertRaises(AssertionError):
            validate(self.public, self.host, self.tokens, self.cfg, old)


if __name__ == '__main__':
    unittest.main()

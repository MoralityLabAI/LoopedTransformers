import unittest
from experiments.nan_comp14.audit import answer
from experiments.nan_comp14.common import ROOT, read
from experiments.nan_comp14.report import grade_groups, gates
from experiments.nan_comp14.sources import materialize


class IndependentContracts(unittest.TestCase):
    def test_independent_grammar_refuses_embedded_or_multiple_answers(self):
        for text, expected in [(' A ', 'A'), ('[P].', 'P'), ('**B**', 'B'), ('`C`', 'C'),
                               ('A or B', None), ('Answer: A', None), ('**A', None), ('```A```', None),
                               ('***A***', None), ('A .', None), ('**[A]**', None)]:
            self.assertEqual(answer(text), expected)

    def test_bounded_pass_cannot_promote_failed_first_token_gate(self):
        cfg = read(ROOT / 'config.json'); public, _ = materialize(cfg)
        records = []
        for row in public:
            erased = row['arm'].endswith('_erased')
            records.append({k: row[k] for k in ('id', 'root', 'split', 'world', 'arm')} | dict(
                first_token_correct=[0, int(row['arm'] == 'copy')], first_token_valid=[0, 1],
                bounded_correct=int(not erased), bounded_valid=1, generated_ids_sha256='a' * 64))
        result = gates({'records': records, 'worker_status': 'completed'}, grade_groups(records), cfg)
        self.assertTrue(all(v['passed'] for v in result if v['endpoint'] == 'bounded_loop2'))
        self.assertTrue(all(not v['passed'] for v in result if v['endpoint'] == 'first_token_loop2'))
        failed = gates({'records': records, 'worker_status': 'failed'}, grade_groups(records), cfg)
        self.assertTrue(all(not v['passed'] for v in failed))


if __name__ == '__main__':
    unittest.main()

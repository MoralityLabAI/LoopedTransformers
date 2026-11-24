import unittest
from experiments.nan_grad15.common import ROOT, read, rows, eligible
from experiments.nan_grad15.sources import select


class Contracts(unittest.TestCase):
    def test_selection_is_fixed_CAL_both_worlds_not_outcomes(self):
        public = rows('experiments/nan_comp14/fixtures/public.jsonl'); chosen = select(public)
        self.assertEqual([v['upstream_ordinal'] for v in chosen], [1, 6, 11, 16])
        self.assertEqual(len({v['root'] for v in chosen}), 2)
        self.assertTrue(all(v['split'] == 'calibration' and v['arm'] == 'focused' for v in chosen))
        self.assertEqual(chosen, read(ROOT / 'fixtures/public.json'))

    def test_budget_and_zero_training_are_not_silently_expanded(self):
        cfg = read(ROOT / 'config.json')
        self.assertEqual(cfg['optimizer_steps'], 0); self.assertEqual(cfg['passes'], [1, 2])
        self.assertEqual(cfg['maximum_added_saved_tensor_CPU_bytes'], 2 * 1024**3)
        self.assertEqual(cfg['rank'], 8); self.assertFalse(cfg['BPD_donor_transfer_enabled'])

    def test_bounded_pass_does_not_qualify_first_token_preflight(self):
        public = rows('experiments/nan_comp14/fixtures/public.jsonl'); records = []
        for r in public:
            erased = r['arm'].endswith('_erased')
            records.append({k: r[k] for k in ('id', 'root', 'split', 'world', 'arm')} | dict(
                first_token_correct=[0, int(r['arm'] == 'copy')], first_token_valid=[0, 1],
                bounded_correct=int(not erased), bounded_valid=1, generated_ids_sha256='a' * 64))
        audit = dict(status='passed', source_manifest_sha256=read(ROOT / 'config.json')['upstream_comp14_seal'],
                     contexts_checked=240, missing_contexts=0, worker_status='completed', records=records)
        self.assertFalse(eligible({'nf4': audit, 'bf16': audit})['passed'])


if __name__ == '__main__':
    unittest.main()

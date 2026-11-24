from pathlib import Path
import unittest
from experiments.nan_xref19.common import ROOT, read, qualified


class ReplacementBinding(unittest.TestCase):
    def test_analysis_inventory_and_numeric_configuration_unchanged(self):
        parent = Path('experiments/nan_xref17')
        self.assertEqual((ROOT / 'CACHE_INVENTORY.json').read_bytes(), (parent / 'CACHE_INVENTORY.json').read_bytes())
        a, b = read(ROOT / 'config.json'), read(parent / 'config.json')
        for values in (a, b):
            values.pop('experiment'); values.pop('FIT16_seal')
        self.assertEqual(a, b)
        for name in ('cache.py', 'worker.py', 'audit.py'):
            expected = (parent / name).read_text(encoding='utf-8').replace('nan_xref17', 'nan_xref19').replace('runs/nan_fit16', 'runs/nan_fit16_io18')
            self.assertEqual((ROOT / name).read_text(encoding='utf-8'), expected)

    def test_original_or_partial_fit_cannot_qualify(self):
        audit = dict(status='passed', source_manifest_sha256=read(ROOT / 'config.json')['FIT16_seal'],
            cells=[dict(passes=1, status='passed', training={'optimizer_steps': 16})])
        self.assertTrue(qualified(audit, 1)); self.assertFalse(qualified(audit, 2))
        audit['cells'][0]['training']['optimizer_steps'] = 2
        self.assertFalse(qualified(audit, 1))
        audit['source_manifest_sha256'] = '35371bbb5323be88ae4eca4cb43677c2f583494b8a46a5f7b7aa89c2fc710139'
        with self.assertRaises(AssertionError):
            qualified(audit, 1)


if __name__ == '__main__':
    unittest.main()

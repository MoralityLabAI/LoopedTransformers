import copy
import json
import subprocess
import unittest
from mechinterp.nanbeige_bpd_gym.ranksite_results import project


class RankSiteResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(subprocess.check_output(['git', 'show', 'HEAD:reports/nanbeige_bpd_gym/nan_ranksite13_progress.json']))

    def test_complete_cannot_hide_a_missing_cache_audit(self):
        data = copy.deepcopy(self.data)
        cell = next(c for c in data['cells'] if c['status'] == 'completed_and_audited')
        cell['cached_suite_probe'] = None
        with self.assertRaises(AssertionError):
            project(data)

    def test_missing_control_is_rejected(self):
        data = copy.deepcopy(self.data)
        cell = next(c for c in data['cells'] if c['fit_independent_audit'])
        cell['fit_independent_audit']['final_groups'].pop()
        with self.assertRaises(AssertionError):
            project(data)

    def test_pending_cell_has_missing_metrics_and_no_invented_correctness(self):
        data = copy.deepcopy(self.data)
        cell = data['cells'][-1]
        cell.update(status='queued', fit_independent_audit=None, physical_write_SAE_probe=None, cached_suite_probe=None)
        data['learned_VPD_BPD_pairs'] = [p for p in data['learned_VPD_BPD_pairs'] if (p['module'], p['rank']) != (cell['module'], cell['rank'])]
        result = project(data)
        self.assertIsNone(result['cells'][-1]['factor_update_l2'])
        self.assertIsNone(result['cells'][-1]['cached_vectors'])
        self.assertFalse(any(r['condition'] == cell['condition'] for r in result['correctness']))


if __name__ == '__main__':
    unittest.main()

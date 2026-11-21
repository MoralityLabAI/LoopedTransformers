"""Presentation refuses omitted routes, fabricated pending grades and hidden failures."""
import copy
import json
import subprocess
import unittest
from mechinterp.nanbeige_bpd_gym.io18_results import SOURCE,DEFAULT_COMMIT,validate,render


class AuditPresentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=json.loads(subprocess.check_output(['git','show',DEFAULT_COMMIT+':'+SOURCE]))

    def test_full_audit_snapshot_is_presented_with_pending_backend_unknown(self):
        text=render(self.data,DEFAULT_COMMIT,'immutable-source-sha')
        self.assertIn('| bf16 | pending | — | unknown | — | — | — |',text)
        self.assertIn('cumulative within the backend',text)

    def test_omitted_routing_group_cannot_disappear(self):
        data=copy.deepcopy(self.data); data['backends']['nf4']['cells'][1]['final_groups'].pop()
        with self.assertRaises(AssertionError):
            validate(data)

    def test_pending_backend_cannot_receive_a_successful_cell(self):
        data=copy.deepcopy(self.data); data['backends']['bf16']['cells']=[data['backends']['nf4']['cells'][0]]
        with self.assertRaises(AssertionError):
            validate(data)

    def test_undefined_mask_gap_cannot_be_replaced_by_zero(self):
        data=copy.deepcopy(self.data)
        data['backends']['nf4']['cells'][0]['final_groups'][0]['reached_vs_clean_mask_max_abs']['equal_root_mean']=0.
        with self.assertRaises(AssertionError):
            validate(data)

    def test_original_partial_failure_cannot_be_relabeled(self):
        data=copy.deepcopy(self.data); data['superseded_attempt']['NF4_cells'][0]['status']='passed'
        with self.assertRaises(AssertionError):
            validate(data)


if __name__=='__main__':
    unittest.main()

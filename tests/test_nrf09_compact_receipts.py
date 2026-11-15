"""NRF-09 compact receipts and direction-gate integrity (standard library)."""
import csv
import json
import math
import unittest
from pathlib import Path

class NRF09ReceiptTests(unittest.TestCase):
    def test_compact_evaluation_receipts(self):
        root=Path(__file__).resolve().parents[1]/"experiments/numerical_range_fixed_point_v1"
        rows=list(csv.DictReader((root/"nrf09_matched_step_receipts.csv").open()))
        self.assertEqual(len(rows),24)
        self.assertEqual(len({(r['seed'],r['rounds'],r['optimizer_steps']) for r in rows}),24)
        for r in rows:
            self.assertEqual(int(r['exposure']),8*int(r['rounds'])*int(r['optimizer_steps']))
            self.assertEqual(len(r['checkpoint_sha256']),64)
            mse, leak = float(r['supervised_mse']),float(r['supervised_leakage'])
            self.assertAlmostEqual(mse+leak*leak,float(r['supervised_J']),places=9)
            self.assertEqual(r['supervised_wins']=='true',float(r['supervised_J'])<float(r['random_complete_J_median']))
        wins=lambda R,u:sum(r['supervised_wins']=='true' for r in rows if int(r['rounds'])==R and int(r['optimizer_steps'])==u)
        self.assertEqual(wins(16,32),6)
        self.assertEqual(wins(32,32),1)

    def test_summary_path_identity(self):
        root=Path(__file__).resolve().parents[1]/"experiments/numerical_range_fixed_point_v1"
        data=json.loads((root/"nrf09_matched_step_summary.json").read_text())
        self.assertTrue(data['primary_gate']['joint_pass'])
        for row in data['paired_path']['per_seed']:
            self.assertAlmostEqual(row['equal_exposure_depth_gap'],row['equal_updates_depth_gap']+row['R32_update_lag'],places=12)

if __name__=="__main__":
    unittest.main()

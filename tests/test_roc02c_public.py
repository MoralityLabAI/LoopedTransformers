"""ROC-02C pure-stdlib public evidence checks, no sealed source access."""
import csv
import json
import math
import os
import unittest
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT=Path(os.environ.get('ROC02C_PUBLIC_DIR',Path(__file__).resolve().parents[1]/'experiments/roc02_delayed_rule_v1/roc02c_calibration'))

class PublicROC02C(unittest.TestCase):
    def test_source_receipt_complete(self):
        rows=[json.loads(line) for line in (ROOT/'roc02c_full_receipts.jsonl').read_text().splitlines() if line.strip()]
        self.assertEqual(len(rows),12)
        self.assertEqual({p['depth'] for p in rows},{8,16})
        self.assertEqual({p['model_seed'] for p in rows},{2026111601,2026111603,2026111607,2026111609,2026111613,2026111619})
        for p in rows:
            self.assertIn('sealed withheld',p['source_split'])
            self.assertEqual(len(p['rows']),8)
            self.assertEqual(len(p['null_medians']),12)
            self.assertTrue(all(0. < r['median_b'] < 1. and r['null_128_distinct'] for r in p['null_medians']))
            for rank in (7,8):
                arms=[a for a in p['rows'] if a['rank']==rank]
                self.assertEqual(sorted(a['gamma'] for a in arms),[0,.05,.2,1.])
                self.assertEqual(sum(bool(a['selected']) for a in arms),1)

    def test_published_numerical_summary(self):
        s=json.loads((ROOT/'roc02c_summary.json').read_text())
        with (ROOT/'roc02c_all_arms.csv').open(newline='') as f:rows=list(csv.DictReader(f))
        self.assertTrue(s['calibration_only']);self.assertFalse(s['sealed_test_evaluated'])
        self.assertEqual(len(rows),96)
        for summary in s['by_depth_rank']:
            d,k=summary['depth'],summary['rank']
            chosen=[r for r in rows if int(r['depth'])==d and int(r['rank'])==k and r['selected']=='True']
            tasks=[r for r in rows if int(r['depth'])==d and int(r['rank'])==k and r['gamma']=='0.0']
            self.assertEqual(len(chosen),6);self.assertEqual(len(tasks),6)
            for x,y,m in [('mean_joint_MSE','selection_calibration_MSE',chosen),('mean_task_only_MSE','selection_calibration_MSE',tasks),('mean_joint_normalized_B_squared','selection_B_null_squared',chosen),('mean_task_only_normalized_B_squared','selection_B_null_squared',tasks),('mean_joint_future_logit_capture','selection_H_future_logit_capture',chosen)]:
                self.assertAlmostEqual(summary[x],mean(float(r[y]) for r in m),places=12)
            self.assertEqual(summary['joint_J_better_count'],6)
            self.assertEqual(summary['joint_MSE_better_count'],6)

    def test_null_median_cardinality(self):
        with (ROOT/'roc02c_null_medians.csv').open(newline='') as f:rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),144)
        self.assertEqual(len({(r['depth'],r['seed'],r['rank'],r['state_index']) for r in rows}),144)
        self.assertTrue(all(r['null_128_distinct']=='True' for r in rows))

if __name__=='__main__':unittest.main()

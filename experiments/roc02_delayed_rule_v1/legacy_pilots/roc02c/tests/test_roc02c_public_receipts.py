"""Stdlib public receipt verification of ROC-02C calibration-only results."""
from __future__ import annotations
import csv
import hashlib
import io
import json
import os
import statistics
import unittest
import zipfile
from pathlib import Path

ROOT=Path(os.environ.get('ROC02C_RECEIPT_DIR',Path(__file__).resolve().parents[1]/'experiments/roc02_delayed_rule_v1/roc02c_calibration'))
SEEDS=(2026111601,2026111603,2026111607,2026111609,2026111613,2026111619)
DEPTHS=(8,16)

def read_csv(name):
    with (ROOT/name).open(newline='') as f:return list(csv.DictReader(f))

class ROC02CPublic(unittest.TestCase):
    def test_full_cardinality_and_provenance(self):
        s=json.loads((ROOT/'roc02c_summary.json').read_text())
        assert s['n_model_cells']==12 and s['n_candidate_rows']==72
        assert s['calibration_only'] and not s['sealed_test_opened']
        assert s['trained_observer_steps_per_arm']==80 and s['rank']==8
        assert s['penalty_grid']==[0.,.05,.2,1.,2.]
        self.assertEqual(len(s['adapter_receipts']),12)
        self.assertTrue(s['registered_calibration_gate_passed'])
        self.assertEqual(len(read_csv('roc02c_candidates.csv')),72)
        self.assertEqual(len(read_csv('roc02c_paired_by_seed.csv')),12)

    def test_task_error_and_penalty_arithmetic(self):
        rows=read_csv('roc02c_candidates.csv')
        for r in rows:
            E=float(r['calibration_mse']);P=float(r['B2_null_penalty']);J=float(r['calibration_J_lambda1'])
            self.assertAlmostEqual(J,E+P,places=11)
            self.assertLess(float(r['orthogonality_error']) if 'orthogonality_error' in r else 0.,1e-8)
            self.assertTrue(r['arm']=='baseline' or float(r['gamma']) in (0.,.05,.2,1.,2.))
        for d in DEPTHS:
            v=[x for x in rows if int(x['depth'])==d]
            self.assertEqual(len(v),36)
            self.assertEqual(sum(x['is_selected']=='True' for x in v),6)

    def test_paired_budget_and_equal_updates(self):
        p=read_csv('roc02c_paired_by_seed.csv')
        for r in p:
            self.assertLessEqual(float(r['joint_mse']),1.1*float(r['task_only_mse'])+1e-12)
            self.assertLess(float(r['joint_J']),float(r['task_only_J']))
            self.assertEqual(r['joint_win'],'True')
            self.assertEqual(r['joint_task_budget'],'True')
        s=json.loads((ROOT/'roc02c_summary.json').read_text())
        for depth in DEPTHS:
            x=s['by_depth'][str(depth)]
            self.assertEqual(x['joint_objective_wins'],6)
            self.assertEqual(x['strict_task_budget_passes'],6)
            self.assertEqual(x['MSE_improves'],6)
            self.assertEqual(x['H_capture_improves'],6)
            for metric,field in [('task_only_mse','task_only_mse'),('joint_J','joint_J')]:
                self.assertAlmostEqual(x['means'][metric],statistics.mean(float(r[field]) for r in p if int(r['depth'])==depth),places=12)

    def test_source_archive_and_adapter_hashes(self):
        expected=json.loads((ROOT/'roc02c_summary.json').read_text())['adapter_receipts']
        path=ROOT/'roc02c_receipts_adapters.zip'
        with zipfile.ZipFile(path) as z:
            self.assertEqual(z.testzip(),None)
            filenames=set(z.namelist())
            self.assertEqual(sum(x.startswith('receipts/') for x in filenames),12)
            self.assertEqual(sum(x.startswith('adapters/') for x in filenames),12)
            self.assertFalse(any('sealed_test' in x for x in filenames))
            for e in expected:
                for prefix,key,filekey in [('receipts/','receipt_sha256','receipt_file'),('adapters/','adapter_sha256','adapter_file')]:
                    name=prefix+e[filekey]
                    self.assertIn(name,filenames)
                    self.assertEqual(hashlib.sha256(z.read(name)).hexdigest(),e[key])
                record=json.loads(z.read('receipts/'+e['receipt_file']))
                self.assertEqual(record['source_checkpoint_sha256'],e['source_checkpoint_sha256'])
                self.assertEqual(record['observer_adapter_sha256'],e['adapter_sha256'])
                self.assertTrue(record['calibration_only'])

if __name__=='__main__':unittest.main()

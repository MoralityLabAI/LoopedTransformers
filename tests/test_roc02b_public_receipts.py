"""ROC-02B public stdlib receipt and claim-boundary tests."""
import csv
import gzip
import json
import math
import os
import unittest
from collections import defaultdict
from pathlib import Path

ROOT=Path(os.environ.get('ROC02B_RECEIPT_DIR', Path(__file__).resolve().parents[1]/'experiments'/'roc02_delayed_rule_v1'/'roc02b_calibration'))
SEEDS=(2026111601,2026111603,2026111607,2026111609,2026111613,2026111619)
RANKS=(1,2,3,4,7,8,12,16,32)


def read_csv(name,gzip_input=False):
    path=ROOT/name
    with (gzip.open(path,'rt',newline='') if gzip_input else path.open(newline='')) as f:
        return list(csv.DictReader(f))

class PublicReceipts(unittest.TestCase):
    def test_counts_hashes_and_no_test_outputs(self):
        summary=json.loads((ROOT/'roc02b_summary.json').read_text())
        self.assertTrue(summary['calibration_only'] if 'calibration_only' in summary else summary['no_test_claim'])
        self.assertEqual(summary['n_models'],12)
        self.assertEqual(summary['n_null_rows'],27648)
        self.assertEqual(summary['n_cards'],216)
        self.assertEqual(len(summary['artifact_hashes']),12)
        for r in summary['artifact_hashes']:
            self.assertEqual(len(r['checkpoint_sha256']),64)
            self.assertEqual(len(r['receipt_sha256']),64)
        self.assertTrue(summary['no_equal_budget_80_step_optimizer'])

    def test_rank_sweep_and_pareto(self):
        cells=read_csv('roc02b_rank_by_seed.csv')
        frontier=read_csv('roc02b_3d_pareto.csv')
        self.assertEqual(len(cells),216)
        self.assertEqual(len(frontier),216)
        for depth in (8,16):
            for seed in SEEDS:
                g=[r for r in cells if int(r['depth'])==depth and int(r['seed'])==seed]
                self.assertEqual(len(g),18)
                self.assertEqual(sorted({int(r['rank']) for r in g}),list(RANKS))
                for r in g:
                    key=(r['seed'],r['depth'],r['rank'],r['arm'])
                    x=next(f for f in frontier if (f['seed'],f['depth'],f['rank'],f['arm'])==key)
                    xyz=(int(r['rank']),float(r['calibration_mse']),float(r['B_to_null']))
                    def better(q):
                        v=(int(q['rank']),float(q['calibration_mse']),float(q['B_to_null']))
                        return all(a<=b for a,b in zip(v,xyz)) and v!=xyz
                    self.assertEqual(x['pareto']=='True',not any(better(q) for q in g if q is not r))

    def test_128_distinct_nulls_and_raw_med_points(self):
        nulls=read_csv('roc02b_nulls.csv.gz',True)
        self.assertEqual(len(nulls),27648)
        d=defaultdict(list)
        for r in nulls:
            d[(r['seed'],r['depth'],r['state_index'],r['rank'])].append(r)
        self.assertEqual(len(d),216)
        for k,group in d.items():
            self.assertEqual(len(group),128)
            self.assertEqual(len({r['null_seed'] for r in group}),128)
            self.assertEqual(len({r['subspace_sha256'] for r in group}),128)
            for key in ('a','b','c','s'):
                vals=sorted(float(r[key]) for r in group)
                self.assertTrue(math.isfinite((vals[63]+vals[64])/2))
                self.assertTrue(all(0<=x<=1 for x in vals))

    def test_card_integrity_and_tag_scope(self):
        rows=read_csv('roc02b_observer_states.csv')
        self.assertEqual(len(rows),432)
        for r in rows:
            k=int(r['rank'])
            self.assertAlmostEqual(float(r['B']),float(r['B_per_sqrtk'])*math.sqrt(k),places=10)
            self.assertAlmostEqual(float(r['a'])**2+float(r['b'])**2,
                                   (float(r['A'])**2+float(r['B'])**2)/float(r['D_frob'])**2,places=12)
            self.assertEqual(r['tags'],"['unclassified']")
            for source in ('observed','dark','isotropic'):
                self.assertGreaterEqual(float(r[source+'_delta_logits']),0)
                self.assertGreaterEqual(float(r[source+'_tangent_residual_logits']),0)

    def test_summary_means_and_calibration_only(self):
        cells=read_csv('roc02b_rank_by_seed.csv')
        grouped=read_csv('roc02b_rank_summary.csv')
        for r in grouped:
            group=[x for x in cells if x['depth']==r['depth'] and x['arm']==r['arm'] and x['rank']==r['rank']]
            self.assertEqual(len(group),6)
            for metric,measure in [('mean_calibration_mse','calibration_mse'),('mean_B_to_rank_null','B_to_null'),('mean_causal_future_logit_capture','H_capture')]:
                expect=sum(float(x[measure]) for x in group)/6
                self.assertAlmostEqual(float(r[metric]),expect,places=12)

if __name__=='__main__':unittest.main()
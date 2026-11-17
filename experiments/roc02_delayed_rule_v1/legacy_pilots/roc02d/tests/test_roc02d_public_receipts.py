"""Independent standard-library checks for the frozen ROC-02D calibration pilot."""
import csv, gzip, hashlib, json, os, unittest, zipfile
from pathlib import Path
P=Path(os.environ.get('ROC02D_RECEIPT_DIR','experiments/roc02_delayed_rule_v1/roc02d_matched_input'))
SEEDS=(2026111601,2026111603,2026111607,2026111609,2026111613,2026111619)
ARMS=('full_legal_flip','input_soft_radius','input_induced_midpoint_radius','hidden_observed_radius','hidden_dark_radius','hidden_isotropic_radius')
class Public(unittest.TestCase):
    def test_cardinality_and_scope(self):
        a=json.loads((P/'roc02d_summary.json').read_text());self.assertEqual((a['model_cells'],a['unique_case_model_pairs'],a['observer_case_rows'],a['arm_effect_rows']),(12,72,144,864));self.assertTrue(a['calibration_only']);self.assertFalse(a['sealed_test_opened']);self.assertEqual(a['soft_input_radius_success'],72)
    def test_manifest_and_zip(self):
        m=json.loads((P/'roc02d_manifest.json').read_text())
        for path,sha in m.items():self.assertEqual(hashlib.sha256((P/path).read_bytes()).hexdigest(),sha)
        a=json.loads((P/'roc02d_summary.json').read_text())
        with zipfile.ZipFile(P/'roc02d_raw_receipts.zip') as z:
            self.assertIsNone(z.testzip());self.assertEqual(len(z.namelist()),12)
            for r in a['source_receipts']:
                v=z.read('receipts/'+r['file']);self.assertEqual(hashlib.sha256(v).hexdigest(),r['sha256']);self.assertFalse(json.loads(v)['sealed_test_opened'])
    def test_all_radii_and_oracles(self):
        with gzip.open(P/'roc02d_per_case_effects.csv.gz','rt',newline='') as f:rr=list(csv.DictReader(f))
        self.assertEqual(len(rr),864);self.assertEqual({r['arm'] for r in rr},set(ARMS))
        self.assertEqual(len({(r['depth'],r['seed'],r['case_index'],r['observer_arm'],r['arm']) for r in rr}),864)
        for r in rr:
            self.assertNotEqual(r['label_original'],r['label_flipped']);self.assertNotEqual(r['source_family'],r['flipped_family'])
            if r['arm']!='full_legal_flip':self.assertAlmostEqual(float(r['midpoint_shift_norm']),.2,delta=2e-5)
    def test_paired_seed_signs(self):
        with (P/'roc02d_paired_by_seed.csv').open(newline='') as f:rr=list(csv.DictReader(f))
        self.assertEqual(len(rr),12)
        for dep in (8,16):
            group=[r for r in rr if int(r['depth'])==dep];self.assertEqual(len(group),6)
            for arm in ('full_legal_flip','input_soft_radius'):
                wins=sum(float(r[f'joint_{arm}_observer_mid_readout_change_norm'])>float(r[f'task_only_{arm}_observer_mid_readout_change_norm']) for r in group)
                self.assertEqual(wins,1 if dep==8 else 0)
if __name__=='__main__':unittest.main()

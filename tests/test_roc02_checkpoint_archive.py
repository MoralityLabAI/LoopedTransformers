"""Source-locked ROC-02 training cohort, public archive and receipt tests."""
import csv,json,os,tempfile,unittest
from pathlib import Path
try:
    from research_gym.analysis.roc02_checkpoint_archive import verify
except ImportError:
    from roc02_checkpoint_archive import verify

ROOT=Path(os.environ.get('ROC02_TRAINED_DIR','experiments/roc02_delayed_rule_v1'))

class ROC02CohortReceipts(unittest.TestCase):
    def test_train_calibration_summary(self):
        report_path=ROOT/'roc02_train_calibration_summary.json'
        if not report_path.is_file():self.skipTest('published summaries not mounted')
        report=json.loads(report_path.read_text())
        self.assertEqual(report['batch_size'],8)
        self.assertEqual(report['updates_per_model'],240)
        self.assertEqual(len(report['rows']),12)
        for depth in (8,16):
            rows=[r for r in report['rows'] if r['depth']==depth]
            self.assertEqual(len(rows),6)
            self.assertEqual(len({r['seed'] for r in rows}),6)
            for r in rows:
                self.assertEqual(len(r['checkpoint_sha256']),64)
                self.assertEqual(r['train']['cases'],1024)
                self.assertEqual(r['calibration']['cases'],256)
                self.assertTrue(0<=r['calibration']['accuracy']<=1)
                self.assertTrue(0<=r['train']['accuracy']<=1)
        self.assertEqual(report['by_depth']['8']['n_models'],6)
        self.assertEqual(report['by_depth']['16']['n_models'],6)

    def test_all_original_pytorch_checkpoint_bytes(self):
        index=ROOT/'checkpoints/roc02_checkpoint_archive_index.json'
        report=ROOT/'roc02_train_calibration_summary.json'
        if not index.is_file():self.skipTest('binary packages unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            counts=verify(index.parent,index,report,Path(tmp))
        self.assertEqual(counts['verified_archives'],6)
        self.assertEqual(counts['unique_models'],12)
        self.assertFalse(counts['sealed_test_evaluated'])

if __name__=='__main__':unittest.main()

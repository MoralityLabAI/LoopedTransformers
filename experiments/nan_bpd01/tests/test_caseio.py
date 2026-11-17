import copy
import json
from pathlib import Path
import tempfile
import unittest
from experiments.nan_bpd01.caseio import load_case

class ImmutableTapeTest(unittest.TestCase):
    def test_overlay_retains_initial_bytes_and_marks_incoherent_controls(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as t:
            root=Path(t);(root/'bf16').mkdir();repair=root/'prefix_repair/bf16';repair.mkdir(parents=True)
            rec={'operator_logits':[[[0.]*4 for _ in range(3)] for _ in range(16)],'log_normalizers':{'1':[[1.]*3 for _ in range(16)]}}
            bad=copy.deepcopy(rec);bad['operator_logits'][1][0][0]=2.
            initial={'case_id':'r','interventions':{'random_dense_dark_741':{'delayed':bad}}}
            file=root/'bf16/r.json';file.write_text(json.dumps(initial));before=file.read_bytes()
            (repair/'receipt.json').write_text(json.dumps({'status':'completed'}))
            (repair/'r.json').write_text(json.dumps({'case_id':'r','interventions':{'random_shared_dark_741':{'delayed':rec}}}))
            merged=load_case(root,'bf16','r')
            self.assertEqual(file.read_bytes(),before)
            self.assertFalse(merged['interventions']['random_dense_dark_741']['canonical_prefix_consistent'])
            self.assertTrue(merged['interventions']['random_shared_dark_741']['canonical_prefix_consistent'])

if __name__=='__main__':unittest.main()

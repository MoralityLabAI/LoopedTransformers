"""RAM17 finalized negative result must preserve admission failures and TEST seal."""
import json
from pathlib import Path
import unittest
FILE=Path(__file__).resolve().parent/"RAM17_RESULT_20261116.json"

class NegativeSourceGate(unittest.TestCase):
    def test_negative_gate_is_an_explicit_stopping_rule(self):
        rec=json.loads(FILE.read_text())
        self.assertEqual(rec["status"],"passed_compiler_replay_gate_failed")
        self.assertEqual(rec["all_triple_compiled_positives"],20)
        self.assertEqual(rec["audited_train"]["triply_compiled_positives"],18)
        self.assertEqual(rec["audited_train"]["nat_induction"]["actions"],0)
        self.assertEqual(rec["audited_train"]["rewrite_prior"]["modules"],1)
        self.assertEqual(rec["first_collection"]["induction_candidate_roots"],5)
        self.assertEqual(rec["first_collection"]["induction_full_source_closures"],0)
        self.assertEqual(rec["training_fits"],0)
        self.assertEqual(rec["prospective_test_cases_scored"],0)
        self.assertIs(rec["source_test_accessed"],False)
    def test_hash_receipt_fingerprints(self):
        rec=json.loads(FILE.read_text())["source_evidence"]
        self.assertTrue(all(len(x)==64 for x in rec.values()))

if __name__=="__main__":unittest.main()
